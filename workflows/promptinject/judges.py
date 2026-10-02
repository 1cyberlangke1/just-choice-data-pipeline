"""提示注入的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `HygieneJudge`（零 token）：内容得**兑现注入特征**——非空、不带围栏、长度落在区间内，按字面
  算出来的命中组合与计划逐条相同，诱饵该有就有、该没有就没有；整段 state 不许有格式残留；
- `FrameJudge`（零 token）：state 的骨架必须完整且规范——判据表里 T1…T4 一个不少、顺序与
  `features.TAG_IDS` 一致，内容那一块解析得出来。任何一处对不上都说明渲染挂错了 case、或者
  语言文件缺了判据，标签会跟着一起错；
- `TraceJudge`（一次后端调用）：把整段 state 交回后端**读一遍**，答「这段内容带了哪几条」，必须与
  计划算出来的结论一致。这是「渲染必须可反解」的落地：题面上的答案就是从 state 里读出来的，
  所以这段 state 必须真的读得出那个结论。

为什么「带了哪几条」既机械查、又问模型：机械那一遍是**口径的权威**（渲染期还能据此重试），模型
那一遍是**读者视角的复核**——机械口径只认字符，内容要是写成「本段包含的指令是：忽略之前的指令」
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
from jc.judge import Verdict, verify_prompt, leftover_in
from jc.jsontypes import JsonValue

from . import features, locales, render
from .spec import PromptInjectSpec

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

    为什么要连原文一起要：模型报「命中了 T1」却抄不出那段话，是**误报**，不是数据错——
    内容里到底有没有那个形状是机械可查的（`features.PATTERNS`），所以让它把证据抄出来，
    代码再拿正则复核一遍，抄不出来的那条直接不算。
    """

    tag: TagId = Field(description="命中的判据编号（T1 / T2 / T3 / T4）")
    quote: str = Field(
        description="从不可信内容里**原样抄出来**的那一段——就是它满足了这条判据"
    )


class TraceDraft(BaseModel):
    """反解草稿：只回答「这段内容带了哪几条判据」。

    取值域由 `features.TAG_IDS` 定，认不出来的编号一律算「没判成」（`skipped=True`）而不是
    「不通过」——那是反解格式没对齐，不是数据本身错。
    """

    tags: list[TraceHit] = Field(
        description="命中的判据（每条带编号与引文）；一条都没带就给空数组"
    )


#: 转出去：口径只有一份，住在 `jc.judge.verify_prompt`（18 家逐字同构，只留那一份）。
build_trace_prompt = verify_prompt


def quote_holds(tag: str, quote: str, body: str) -> bool:
    """输入：判据编号、反解抄出来的引文、不可信内容。输出：这条命中站不站得住。

    预期行为：两件事同时成立才算数——引文**原样出现在内容里**（空白折叠后比对，模型重排空白
    不算抄错），且引文本身满足该判据的机械判据（`features.PATTERNS`）。抄不出原文、或抄出来
    的那段按正则根本不算命中，都说明模型是**误报**：丢掉它，别让它把好数据拒掉。反方向不受
    影响——「内容里真有、模型却没读出来」照样对不上计划，照旧不通过。

    为什么对 T2（行首角色标记）也够用：引文是从内容里原样抄出来的，行首那条线在抄的时候就带上了；
    就算模型只抄了 `system:` 这几个字符，它在内容里本来也确实是行首，判据 `^` 在引文开头照样成立。
    """
    pattern = features.PATTERNS.get(tag)
    if pattern is None:
        return False
    hit = _quote_hit(quote, body)
    if not hit:
        return False
    return pattern.search(hit) is not None


def _dropped_note(dropped: list[str]) -> str:
    """输入：被当成误报丢掉的编号。输出：贴进理由的一句话（没有误报就是空串）。

    预期行为：误报不影响判定，但要留痕——不然「模型多报了一条」在日志里就凭空消失了。
    """
    if not dropped:
        return ""
    return f"（另有 {len(dropped)} 条报出来的判据抄不出原文或引文不合判据，已按没命中处理：{dropped}）"


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：内容得兑现注入特征，整段 state 不许有残留。

    查三类：**单独成行的围栏 / HTML 换行标签**、**反解不出内容**、**内容与注入特征对不上**（口径
    直接复用 `render.check_draft`：渲染期查一遍是为了重试，这里再查一遍是为了兜住**不是本
    workflow 渲染出来的** state）。
    """

    locale: str
    name: str = "hygiene"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只做机械比对。"""
        if not isinstance(spec, PromptInjectSpec):
            return Verdict(True, reason="没有 spec，跳过卫生检查", skipped=True)
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过卫生检查", skipped=True)
        leftover = leftover_in(state, LEFTOVER_LINES, LEFTOVER_TAGS)
        if leftover:
            return Verdict(False, reason=f"state 里有格式残留：{leftover!r}")
        parsed = render.parse_state(state)
        if parsed.text is None:
            return Verdict(False, reason="state 里认不出不可信内容")
        draft = render.ContentDraft.model_construct(text=parsed.text)
        try:
            render.check_draft(draft, spec.case)
        except ValueError as exc:
            return Verdict(False, reason=str(exc))
        return Verdict(True, reason=f"内容兑现了注入特征（命中 {spec.case.hits or '无'}）")


@dataclass(frozen=True)
class FrameJudge:
    """零 token 判别器：state 的骨架必须完整且规范。

    判据表里 T1…T4 一个不少、顺序与 `features.TAG_IDS` 一致，内容那一块解析得出来。任何一处
    对不上都说明渲染挂错了 case 或语言文件缺了判据——那是构造 bug，标签会跟着一起错。
    """

    locale: str
    name: str = "frame"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只做结构比对。"""
        if not isinstance(spec, PromptInjectSpec):
            return Verdict(True, reason="没有 spec，跳过框架比对", skipped=True)
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过框架比对", skipped=True)
        parsed = render.parse_state(state)
        problems: list[str] = []
        if parsed.tags != features.TAG_IDS:
            problems.append(
                f"判据表列出的编号是 {list(parsed.tags)}，应为 {list(features.TAG_IDS)}"
            )
        if parsed.text is None or not parsed.text.strip():
            problems.append("内容那一块解析不出来或为空")
        if not problems:
            return Verdict(True, reason="判据表与内容骨架都与规范形态一致")
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
        格式漂移写成数据质量问题。
        """
        if not isinstance(spec, PromptInjectSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="choice",
            domain="promptinject",
            prompt=build_trace_prompt(str(sample["state"]), loc),
            response_model=TraceDraft,
            seeds=(),
            spec={
                "case": spec.case.as_dict(),
                "hits": list(spec.case.tags()),
                "source": spec.case.source,
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
        body = render.parse_state(str(sample["state"])).text or ""
        reported: list[str] = []
        dropped: list[str] = []
        for tag, quote in pairs:
            (reported if quote_holds(tag, quote, body) else dropped).append(tag)
        if set(reported) != set(labels.hits):
            return Verdict(
                False,
                reason=(
                    f"计划里命中 {list(labels.hits) or '无'}，"
                    f"反解读成 {reported or '无'}{_dropped_note(dropped)}"
                ),
                raw={"traced_hits": reported},
            )
        return Verdict(
            True,
            reason=f"反解与计划一致：命中 {list(labels.hits) or '无'}{_dropped_note(dropped)}",
            raw={"traced_hits": reported},
        )
