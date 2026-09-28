"""邮件 workflow 的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `LinkJudge`（零 token）：文本里有没有链接，必须与注入的链接维一致。抓
  「提示词说了没有链接，模型还是塞了个网址」这类硬伤；
- `LinkTextJudge`（零 token）：链接文字那一族的真值，数链接行里的地址就得出来
  （0 / 1 / ≥2 个 → `no_link` / `matches` / `mismatched`）——这一族问的是**结构事实**，
  能数就不要问模型（见 `render.link_text_of`）；
- `TraceJudge`（一次后端调用）：把写好的邮件**交回后端**，把 `TRACE_FAMILIES` 里的
  原子题原样再问一遍，答案必须与注入特征一致。这是「渲染必须可反解」的落地
  （PLAN 3.4 第 ③ 档）。能纯代码反解的族（链接文字、发件人身份）不在这里，见上。

输入/输出：只出 `Verdict`，不改样本。反解失败一律判不通过——宁可少要几条，
也不要一条标签和文本对不上的数据。但**端点故障不算数据不合格**：重试与降级都试过之后
还拿不到草稿，只能记成 `skipped=True`（没判成），否则端点抖动会被写成合成质量问题。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from dataclasses import dataclass
from typing import Final, Literal

from jc.backend.types import BackendError, GenerationBackend, GenerationTask
from jc.judge import Verdict
from pydantic import BaseModel

from . import features, locales, render
from .spec import MailSpec

#: 反解时重问的原子题族（题族名与数据集里完全一致，选项也用同一份 locale）。
#: `link_text` **不在这里**：它的真值是结构事实（链接行里有几个地址），交给零 token 的
#: `LinkTextJudge` 数一遍就够了——问模型只会白搭一次调用、还多一处误拒（2026-09-26 实测，
#: 那一族的失败占了全部失败的一半）。
#: `sender_identity` 同理：交给零 token 的 `SenderIdentityJudge` 读「官方域名」行有没有、
#: 与发件地址域名同不同（2026-09-26 在 167/168 条真端点产物上与注入标签一致，
#: 唯一那条例外是真的缺那一行——它本来就该被装配时的 `check_official_domain` 拦下）。
TRACE_FAMILIES: Final[tuple[str, ...]] = (
    "requested",
    "pressure",
)

#: 单独成行的 JSON 残留：模型偶尔把收尾括号 / 代码围栏写进落款，这是实打实的数据损坏
#: （实跑抓到过落款末尾多一个 `}`，而四个原子题全都答对了，单靠反解看不出来）。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset(
    {"}", "]", "},", "```", "```json", "```jsonl"}
)

#: 混进 state 的 HTML 换行标签：模型想在 JSON 字符串里换行时会写 `<br>`，
#: 但邮件正文是纯文本，这属于格式损坏（实跑抓到过落款里连着一串 `<br>`）。
LEFTOVER_TAGS: Final[tuple[str, ...]] = ("<br>", "<br/>", "<br />", "</br>")


def owned_domain_prefix(loc: dict[str, JsonValue]) -> str:
    """输入：语言字典。输出：「官方域名」那一行的行首标记（模板里 `{domain}` 之前的部分）。

    预期行为：复用 `render.domain_line_prefix`——同一份 locale 模板只能有一个解析口径，
    两处各写一份迟早会漂移（这一行归渲染器写，渲染器那边才是权威）。
    """
    return render.domain_line_prefix(loc)


def duplicated_owned_lines(state: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：state、语言字典。输出：出现了两次以上的「渲染器自有的行」行首标记。

    预期行为：那一行由渲染器写（`render.to_state` 负责追加），模型再写一遍就成了重复行。
    实跑抓到过（`nim-smoke-13` / `13b` 的 `mail-train-0000001`，落款里连着两个「官方域名：」）。
    只认行首、只数次数，不做模糊匹配——误判会直接拒掉好数据。
    """
    prefix = owned_domain_prefix(loc)
    hits = [line for line in state.splitlines() if line.strip().startswith(prefix)]
    return [prefix] if len(hits) > 1 else []


class TraceDraft(BaseModel):
    """反解草稿：反解要问的原子特征取值。

    取值域必须与 `features.ASK` / `features.LURE` 以及 locale 里 `criteria` 的 key 一致；
    这条由 `tests/workflows/mail/test_locales.py` 锁住，不靠人记。
    链接文字与发件人身份两族不在这里——它们分别由零 token 的 `LinkTextJudge` /
    `SenderIdentityJudge` 从 state 的结构事实里读出来（数链接行里的地址 / 读「官方域名」行）。

    `extra_requests` 是**清单之外的索取**：数据里那道题是单选，文本却夹带第二种索取时，
    它就有两个说得通的答案（实跑抓到过：清单只要求打开附件，正文却写「请按提示输入密码」）。
    单选判别器看不出来，所以单独问一遍，有夹带就整条丢掉。
    """

    requested: Literal["none", "credentials", "payment", "click_link", "attachment"]
    extra_requests: list[Literal["credentials", "payment", "click_link", "attachment"]]
    pressure: Literal["none", "urgency", "reward", "fear", "curiosity"]


def build_verify_prompt(state: str, loc: dict[str, JsonValue]) -> str:
    """组装反解提示词。

    输入：渲染好的邮件文本、语言字典。
    输出：提示词。
    预期行为：把 `TRACE_FAMILIES` 里的原子题连同选项原样再问一遍——反解用的选项与数据集里的一致，
    否则「答对了」与「标签对了」不是一回事。
    """
    lines = [
        loc["verify"]["intro"],
        "",
        loc["verify"]["state_header"],
        state,
        "",
        loc["verify"]["questions_header"],
    ]
    for index, family in enumerate(TRACE_FAMILIES, start=1):
        lines.append(f"{index}. {loc['fam'][family]['q'][0]}")
        for name, desc in loc["fam"][family]["criteria"].items():
            lines.append(loc["verify"]["option_line"].format(name=name, desc=desc))
    lines.append(f"{len(TRACE_FAMILIES) + 1}. {loc['verify']['extra_question']}")
    for name in features.ASK:
        if name == features.ASK_NONE:
            continue
        lines.append(
            loc["verify"]["option_line"].format(
                name=name, desc=loc["fam"]["requested"]["criteria"][name]
            )
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class LinkJudge:
    """纯规则判别器：链接的有无必须与注入的链接维一致。"""

    name: str = "link"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只看链接有无，不看内容。"""
        if not isinstance(spec, MailSpec):
            return Verdict(True, reason="没有 spec，跳过链接判别", skipped=True)
        parsed = render.parse_state(sample["state"])
        expected = spec.tags.link != features.LINK_NONE
        if parsed.has_url == expected:
            return Verdict(True, reason="链接有无与注入特征一致")
        return Verdict(
            False,
            reason=(
                f"注入的是{'有' if expected else '无'}链接，"
                f"文本里却{'有' if parsed.has_url else '没有'}链接"
            ),
            raw={"urls": list(parsed.urls)},
        )


@dataclass(frozen=True)
class LinkTextJudge:
    """零 token 判别器：`link_text` 这一族的真值，纯代码就能反解。

    字段：`name`。

    预期行为：数 state **链接行**（最后一行）里的不同网址——0 个 → `no_link`、
    1 个 → `matches`、2 个及以上 → `mismatched`，再与注入的标签比。
    口径来自渲染规则「链接文字与真实目标不一致时两个地址都写」（见 `render.link_text_of`）。

    为什么不让模型判（2026-09-26）：这一族问的是「链接文字与真实目标是否一致」，
    答案其实是**结构事实**。在 141 条真端点产物上核对，纯代码规则与注入标签 **141/141**
    一致（其中 2 条模型把显示地址写成了不带 scheme 的裸域名，所以有链接时裸域名也算地址），
    而 LLM 反解在这一族上的失败占了全部失败的一半——同一个判断，能数地址就不要问模型。
    """

    name: str = "link_text"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只数链接行里的地址，不看语义。"""
        if not isinstance(spec, MailSpec):
            return Verdict(True, reason="没有 spec，跳过链接文字判别", skipped=True)
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过链接文字判别", skipped=True)
        expected = spec.labels.link_text
        # 清单说有链接时，链接行里的裸域名也算地址（模型偶尔把显示的文字写成不带 scheme 的
        # 裸域名）；没有链接的邮件则不能这么算，否则「官方域名」那一行会被数成一个地址。
        bare = spec.tags.link != features.LINK_NONE
        urls = render.link_line_urls(state, bare_domains=bare)
        found = render.link_text_of(state, bare_domains=bare)
        if found == expected:
            return Verdict(True, reason=f"链接行里的地址写法与注入的 {expected!r} 一致")
        return Verdict(
            False,
            reason=(
                f"注入的是 {expected!r}，链接行里反解成 {found!r}"
                f"（最后一行有 {len(urls)} 个不同网址）"
            ),
            raw={"urls": list(urls)},
        )


@dataclass(frozen=True)
class SenderIdentityJudge:
    """零 token 判别器：`sender_identity` 这一族的真值，纯代码就能反解。

    字段：`locale` 语言（要按语言认「官方域名」那一行的行首标记）。

    预期行为：读两个**结构事实**——落款有没有「官方域名」行、那一行的域名与 `From:` 里
    发件地址的域名是不是同一个站点：

    - 没有那一行 → `no_claim`（不声称机构身份）；
    - 有、且域名一致 → `matches`；
    - 有、但域名不同 → `forged`。

    为什么不让模型判（2026-09-26）：装配时的 `render.check_official_domain` 已经把
    「声称机构 ⟺ 那一行非空」结构上强制了，所以这一族的答案就是结构事实。在 168 条
    真端点产物上核对：纯代码规则与注入标签 **167/168** 一致，唯一那条例外
    （`mail-train-0000007`）是模型真的漏写了那一行——它本来就该被装配时拦下。
    """

    locale: str
    name: str = "sender_identity"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只读「官方域名」行与 `From:` 行。"""
        if not isinstance(spec, MailSpec):
            return Verdict(True, reason="没有 spec，跳过发件人身份判别", skipped=True)
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过发件人身份判别", skipped=True)
        official = render.official_domain_of(state, locales.load(self.locale))
        address = render.sender_domain_of(state)
        if not official:
            found = "no_claim"
        elif render.host_of(official) == render.host_of(address):
            found = "matches"
        else:
            found = "forged"
        expected = spec.labels.sender_identity
        if found == expected:
            return Verdict(True, reason=f"「官方域名」行与发件地址读出来就是 {expected!r}")
        return Verdict(
            False,
            reason=(
                f"注入的是 {expected!r}，文本里读出来是 {found!r}"
                f"（官方域名行 {official!r}，发件地址 {address!r}）"
            ),
            raw={"official_domain": official, "from_address": address},
        )


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：state 里不许出现格式残留。

    查两类：**单独成行的 JSON 残留 / HTML 换行标签**，以及**渲染器自有的行被写了两遍**
    （模型把「官方域名」那一行又写进了 `signature`）。

    字段：`locale` 语言（要按语言认那一行的行首标记）。
    输入：样本、真值规格（本判别器用不到）。输出：Verdict。
    预期行为：口径压到最窄——宁可漏检也不误判，因为误判会直接拒掉好数据。实跑抓到过
    落款末尾多一个 `}`、落款里连着一串 `<br>`，两种情况下四个原子题都答对了，
    单靠反解看不出来。
    """

    locale: str
    name: str = "hygiene"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只看残留行与重复行，不看语义。"""
        del spec
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过卫生检查")
        leftovers = sorted(
            {line.strip() for line in state.splitlines() if line.strip() in LEFTOVER_LINES}
        )
        lowered = state.lower()
        tags = [tag for tag in LEFTOVER_TAGS if tag in lowered]
        duplicated = duplicated_owned_lines(state, locales.load(self.locale))
        if not leftovers and not tags and not duplicated:
            return Verdict(True, reason="没有格式残留")
        return Verdict(
            False,
            reason=(
                "state 里混进了格式残留："
                + "、".join([*leftovers, *tags, *(f"{name}×2" for name in duplicated)])
            ),
            raw={"leftovers": leftovers, "tags": tags, "duplicated": duplicated},
        )


@dataclass(frozen=True)
class TraceJudge:
    """模型判别器：把邮件交回后端反解，`TRACE_FAMILIES` 的原子特征必须与注入的一致。

    字段：`backend` 生成后端（由 `bind_backend` 注入）；`locale` 语言。
    """

    backend: GenerationBackend
    locale: str
    name: str = "trace"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：不一致时把「注入的」与「反解出来的」逐条列在理由里，方便人看是哪一维写歪了。
        端点故障（重试与降级都试过）与草稿缺字段都只算「没判成」（`skipped=True`）——
        记成「不通过」会把端点抖动写成数据质量问题（实测踩过，见 PLAN 10.10）。
        """
        if not isinstance(spec, MailSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        if self.backend is None:
            return Verdict(True, reason="没有后端，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="text",
            domain="mail",
            prompt=build_verify_prompt(sample["state"], loc),
            response_model=TraceDraft,
            spec={"tags": spec.tags.as_dict()},
        )
        try:
            draft = self.backend.generate(task)
        except BackendError as exc:
            if getattr(exc, "fatal_for_run", False):
                raise  # 全部 provider 都挂了：整批该停下（PLAN 3.1.4）
            return Verdict(
                True,
                reason=f"反解没跑成（端点故障，不是数据问题）：{exc}",
                raw={"error": type(exc).__name__},
                skipped=True,
            )
        data = draft.data if isinstance(draft.data, dict) else {}
        broken = [name for name in (*TRACE_FAMILIES, "extra_requests") if name not in data]
        if broken:
            return Verdict(
                True,
                reason=f"反解草稿缺字段，判不了：{'、'.join(broken)}",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        if not isinstance(data["extra_requests"], list):
            return Verdict(
                True,
                reason="反解草稿的 extra_requests 不是列表，判不了夹带",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        labels = spec.labels
        problems = [
            f"{family}: 注入 {getattr(labels, family)!r}，反解成 {data[family]!r}"
            for family in TRACE_FAMILIES
            if data[family] != getattr(labels, family)
        ]
        # 夹带检测：反解出的「额外索取」里，除了本来就该有的那一档之外还有别的就判不通过
        extra = [name for name in data["extra_requests"] if name != labels.requested]
        if extra:
            problems.append(f"夹带了清单之外的索取：{'、'.join(extra)}")
        return Verdict(
            not problems,
            reason="; ".join(problems),
            raw={"traced": {family: data[family] for family in TRACE_FAMILIES}},
        )
