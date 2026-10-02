"""聊天回复门控的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `HygieneJudge`（零 token）：产物得**兑现注入特征**——消息非空、长度在区间内、按字面算出来的
  命中组合与计划逐条相同；三条候选各自的毛病与计划逐条相同（含「该发的那条一个毛病都不踩」）；
  而且**没有把判据 / 毛病原文整段贴进去**；整段 state 不许有格式残留；
- `FrameJudge`（零 token）：state 的骨架必须完整且规范——判据表里 C1…C4 一个不少、毛病表里
  F1…F3 一个不少、顺序与 `features.TAG_IDS` / `FLAW_IDS` 一致，消息与三条候选都解析得出来。
  任何一处对不上都说明渲染挂错了 case、或者语言文件缺了一条，标签会跟着一起错；
- `TraceJudge`（一次后端调用）：把整段 state 交回后端**读一遍**，答「这条消息踩了哪几条」
  与「三条候选里哪条干净」，必须与计划算出来的结论一致。这是「渲染必须可反解」的落地：题面上的
  答案就是从 state 里读出来的，所以这段 state 必须真的读得出那个结论。

为什么「踩了哪几条」既机械查、又问模型：机械那一遍是**口径的权威**（渲染期还能据此重试），模型
那一遍是**读者视角的复核**——机械口径只认字符，消息要是写成「本消息踩的判据是：把决定推给我」
这种自指句子，机械口径照样通过，而读者读出来的东西没有意义，这种样本该丢。两遍问的不是同一件事。

输入/输出：只出 `Verdict`，不改样本。反解失败一律判不通过——宁可少要几条，也不要一条标签和文本
对不上的数据。但**端点故障不算数据不合格**：重试与降级都试过之后还拿不到草稿，只能记成
`skipped=True`（没判成），否则端点抖动会被写成合成质量问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel, Field

from jc.judge import quote_hit as _quote_hit
from jc.backend.types import BackendError, GenerationBackend, GenerationTask
from jc.judge import Verdict, verify_prompt
from jc.jsontypes import JsonValue

from . import features, locales, render
from .spec import ChatGateSpec

#: 格式残留：单独成行的围栏 / HTML 换行标签。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset(
    {"```", "```json", "```toml", "```text", "```markdown", "```md", "```yaml"}
)
LEFTOVER_TAGS: Final[tuple[str, ...]] = ("<br>", "<br/>", "<br />", "</br>")


#: 反解草稿里「编号」字段的取值域：**从 `features.TAG_IDS` 现取**，不手抄（手抄会漂）。
#: 为什么用 `Literal` 而不是裸 `str`：2026-10-02 真机复验，读者会把正文里的词（HTML 标签名、
#: 字段名、中文词）当成编号报回来，按既有口径那一条只算「没判成」= 闸门静默少一道；
#: `Literal` 进 JSON schema 的 `enum`，支持结构化输出的端点会约束生成。范式 §9.96。
TagId = Literal[*features.TAG_IDS]


class TraceHit(BaseModel):
    """反解草稿里的一条命中：判据编号 + 让它命中的那段原文。

    为什么要连原文一起要：模型报「踩了 C1」却抄不出那段话，是**误报**，不是数据错——消息里到底
    有没有那个形状是机械可查的（`features.PATTERNS`），所以让它把证据抄出来，代码再拿正则复核
    一遍，抄不出来的那条直接不算。
    """

    tag: TagId = Field(description="命中的判据编号（C1 / C2 / C3 / C4）")
    quote: str = Field(description="从消息里**原样抄出来**的那一段——就是它满足了这条判据")


class TraceDraft(BaseModel):
    """反解草稿：这条消息踩了哪几条判据 + 三条候选里哪一条干净。

    `clean` 是**候选的编号**（1 / 2 / 3），0 表示「说不出来」（干净的不止一条或一条都不干净）。
    取值域由 `features.TAG_IDS` 与 `features.CANDIDATE_SLOTS` 定，认不出来的编号一律算
    「没判成」（`skipped=True`）而不是「不通过」——那是反解格式没对齐，不是数据本身错。
    """

    tags: list[TraceHit] = Field(description="命中的判据（每条带编号与引文）；一条都没踩就给空数组")
    clean: int = Field(description="三条候选里一个毛病都不踩的那一条的编号（1 / 2 / 3）；说不出来填 0")


#: 转出去：口径只有一份，住在 `jc.judge.verify_prompt`（18 家逐字同构，只留那一份）。
build_trace_prompt = verify_prompt


def quote_holds(tag: str, quote: str, body: str) -> bool:
    """输入：判据编号、反解抄出来的引文、消息正文。输出：这条命中站不站得住。

    预期行为：两件事同时成立才算数——引文**原样出现在消息里**（空白折叠后比对，模型重排空白不算
    抄错），且引文本身满足该判据的机械判据（`features.PATTERNS`）。抄不出原文、或抄出来的那段
    按正则根本不算命中，一律不算数。
    """
    if tag not in features.TAG_IDS:
        return False
    hit = _quote_hit(quote, body)
    if not hit:
        return False
    return features.PATTERNS[tag].search(hit) is not None


def _dropped_note(dropped: list[str]) -> str:
    """输入：被丢掉的误报编号。输出：附在理由后面的那句话。预期行为：空列表给空串。"""
    if not dropped:
        return ""
    return f"（另有 {len(dropped)} 条报出来的判据抄不出原文或引文不合判据，已按没命中处理：{dropped}）"


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：消息与三条候选必须兑现注入特征。

    字段：`locale` 语言（判据文案按它取，报错信息才读得懂）。
    """

    locale: str
    name: str = "hygiene"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：没有规格或 state 不是字符串时记「没判成」；有规格就按字面重算命中组合与每条候选
        的毛病，与计划逐条比。规格类型不对（串了别的 workflow）也算没判成。
        """
        locales.load(self.locale)
        if not isinstance(spec, ChatGateSpec):
            return Verdict(True, reason="没有 spec，跳过兑现检查", skipped=True)
        state = sample.get("state")
        if not isinstance(state, str):
            return Verdict(True, reason="样本里没有 state，跳过兑现检查", skipped=True)
        parsed = render.parse_state(state)
        if parsed.message is None or not parsed.message.strip():
            return Verdict(False, reason="消息那一块解析不出来或为空")
        if len(parsed.replies) != len(features.CANDIDATE_SLOTS):
            return Verdict(
                False,
                reason=f"候选回复解析出 {len(parsed.replies)} 条，应为 {len(features.CANDIDATE_SLOTS)} 条",
            )
        # 「抄原文」排在命中组合之前：贴了判据 / 毛病的原文，命中组合**必然**跟着错，先报组合
        # 的话理由会指向一个次要症状，而真正的毛病是这条样本在抄规格（与 `render.check_draft`
        # 的顺序一致，范式 §9.45）。
        lang = render.detect_locale(state) or self.locale
        loc = locales.load(lang)
        bodies = [parsed.message, *parsed.replies]
        pasted = sorted({rule for body in bodies for rule in render.pasted_rules(body, loc)})
        if pasted:
            return Verdict(False, reason=f"消息或候选把判据 / 毛病原文原样贴了进去：{pasted}")
        wanted = "+".join(spec.plan.hits)
        found = features.hits_of(parsed.message)
        if found != wanted:
            return Verdict(
                False,
                reason=(
                    f"消息按字面算出来的命中组合是 {found or '无'}，计划要的是 {wanted or '无'}"
                ),
            )
        for index, body in enumerate(parsed.replies):
            got = features.flaw_set(body)
            want = spec.plan.reply.flaws[index]
            if got != want:
                return Verdict(
                    False,
                    reason=(
                        f"第 {index + 1} 条候选的毛病是 {list(got) or '无'}，"
                        f"计划要的是 {list(want) or '无'}"
                    ),
                )
        leftovers = [
            line for line in (raw.strip() for raw in state.splitlines()) if line in LEFTOVER_LINES
        ]
        if leftovers:
            return Verdict(False, reason=f"state 里有格式残留：{sorted(set(leftovers))}")
        if any(tag in state for tag in LEFTOVER_TAGS):
            return Verdict(False, reason="state 里混进了 HTML 换行标签")
        return Verdict(True, reason="消息与三条候选兑现了计划，且没有格式残留")


@dataclass(frozen=True)
class FrameJudge:
    """零 token 判别器：state 的骨架必须完整且规范。

    字段：`locale` 语言。
    """

    locale: str
    name: str = "frame"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：判据表里的编号必须是 `C1…C4` 且顺序一致，毛病表里必须是 `F1…F3` 且顺序一致，
        消息与三条候选都必须解析得出来。
        """
        locales.load(self.locale)
        if not isinstance(spec, ChatGateSpec):
            return Verdict(True, reason="没有 spec，跳过骨架检查", skipped=True)
        state = sample.get("state")
        if not isinstance(state, str):
            return Verdict(True, reason="样本里没有 state，跳过骨架检查", skipped=True)
        parsed = render.parse_state(state)
        problems: list[str] = []
        if parsed.tags != features.TAG_IDS:
            problems.append(
                f"判据表列出的编号是 {list(parsed.tags)}，应为 {list(features.TAG_IDS)}"
            )
        if parsed.flaws != features.FLAW_IDS:
            problems.append(
                f"毛病表列出的编号是 {list(parsed.flaws)}，应为 {list(features.FLAW_IDS)}"
            )
        if parsed.message is None or not parsed.message.strip():
            problems.append("消息那一块解析不出来或为空")
        if len(parsed.replies) != len(features.CANDIDATE_SLOTS):
            problems.append(
                f"候选回复解析出 {len(parsed.replies)} 条，应为 {len(features.CANDIDATE_SLOTS)} 条"
            )
        if not problems:
            return Verdict(True, reason="三张表与消息 / 候选的骨架都与规范形态一致")
        return Verdict(False, reason="；".join(problems))


@dataclass(frozen=True)
class TraceJudge:
    """模型判别器：把整段 state 交回后端读一遍，读出来的结论必须与计划一致。

    字段：`backend` 生成后端（由 `bind_backend` 注入）；`locale` 语言。
    """

    backend: GenerationBackend
    locale: str
    name: str = "trace"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：不一致时把「计划的」与「反解出来的」列在理由里，方便人看是哪里写歪了。端点故障、
        草稿缺字段与不认识的编号都只算「没判成」（`skipped=True`）——记成「不通过」会把端点抖动或
        格式漂移写成数据质量问题。但「反解说不出唯一的干净候选」（`clean=0`）算**不通过**：
        state 里印着毛病表，读不出干净的那条，说明这条样本对一个读者来说不唯一。
        """
        if not isinstance(spec, ChatGateSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="choice",
            domain="chatgate",
            prompt=build_trace_prompt(str(sample["state"]), loc),
            response_model=TraceDraft,
            seeds=(),
            spec={
                "case": spec.case.as_dict(),
                "hits": list(spec.plan.hits),
                "clean": spec.plan.reply.clean_answer,
            },
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
        labels = spec.labels
        seen = draft.data.get("tags")
        if not isinstance(seen, list):
            return Verdict(
                True,
                reason=f"反解草稿缺 tags 字段，判不了：{seen!r}",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        pairs: list[tuple[str, str]] = []
        for item in seen:
            if not isinstance(item, dict):
                return Verdict(
                    True,
                    reason=f"反解草稿的 tags 项不是「编号 + 引文」对象，判不了：{item!r}",
                    raw={"raw_text": draft.raw_text[:500]},
                    skipped=True,
                )
            pairs.append((str(item.get("tag", "")), str(item.get("quote", ""))))
        unknown = [tag for tag, _ in pairs if tag not in features.TAG_IDS]
        if unknown:
            return Verdict(
                True,
                reason=f"反解草稿里出现不认识的判据编号 {unknown}，判不了",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        picked = draft.data.get("clean")
        if not isinstance(picked, int) or isinstance(picked, bool):
            return Verdict(
                True,
                reason=f"反解草稿的 clean 不是整数，判不了：{picked!r}",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        if picked not in (0, *(index + 1 for index in features.CANDIDATE_SLOTS)):
            return Verdict(
                True,
                reason=f"反解草稿的 clean 超出候选编号范围，判不了：{picked}",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        body = render.parse_state(str(sample["state"])).message or ""
        reported: list[str] = []
        dropped: list[str] = []
        for tag, quote in pairs:
            (reported if quote_holds(tag, quote, body) else dropped).append(tag)
        problems: list[str] = []
        if set(reported) != set(labels.hits):
            problems.append(
                f"计划里命中 {list(labels.hits) or '无'}，"
                f"反解读成 {reported or '无'}{_dropped_note(dropped)}"
            )
        if picked != labels.clean[0] + 1:
            want = labels.clean[0] + 1
            got = "说不出来" if picked == 0 else f"第 {picked} 条"
            problems.append(f"计划里该发的是第 {want} 条候选，反解读成 {got}")
        if problems:
            return Verdict(False, reason="；".join(problems), raw={"traced_hits": reported})
        return Verdict(
            True,
            reason=(
                f"反解与计划一致：命中 {list(labels.hits) or '无'}，"
                f"该发第 {labels.clean[0] + 1} 条{_dropped_note(dropped)}"
            ),
            raw={"traced_hits": reported},
        )
