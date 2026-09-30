"""引文核查的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `HygieneJudge`（零 token）：文档结构与引文必须**兑现引文计划**——3 节、每节 2 句、每句一个
  具体数字、句子两两不同，引文与「按计划剪出来的那一句」逐字节相同，`altered` 的引文整篇
  文档一处都不出现，`exact` / `partial` 的引文只出在那一节；整段 state 不许有格式残留；
- `FrameJudge`（零 token）：节数、每节句数、声称出处必须逐格等于注入特征——渲染串味
  （挂错 case、声称出处抄错）全在这一关拦下；
- `TraceJudge`（一次后端调用）：把整段 state 交回后端**读一遍**，答「引文在不在文档里、
  出自第几节」，必须与计划算出来的结论一致。这是「渲染必须可反解」的落地：题面上的答案就是
  从 state 里读出来的，所以这段 state 必须真的读得出那个结论。

为什么「引文在不在文档里」既机械查、又问模型：机械那一遍是**口径的权威**（渲染期还能据此
重试），模型那一遍是**读者视角的复核**——机械口径只认字符，正文要是写成「本文引用的那句话
是：……」这种自指句子，机械口径照样通过，而读者读出来的东西没有意义，这种样本该丢。两遍
问的不是同一件事。

输入/输出：只出 `Verdict`，不改样本。反解失败一律判不通过——宁可少要几条，也不要一条标签
和文本对不上的数据。但**端点故障不算数据不合格**：重试与降级都试过之后还拿不到草稿，只能
记成 `skipped=True`（没判成），否则端点抖动会被写成合成质量问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import BackendError, GenerationBackend, GenerationTask
from jc.judge import Verdict, leftover_in
from jc.jsontypes import JsonValue

from . import features, locales, render
from .spec import CitationSpec

#: 格式残留：单独成行的围栏 / HTML 换行标签。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset(
    {"```", "```json", "```toml", "```text", "```markdown", "```md", "```yaml"}
)
LEFTOVER_TAGS: Final[tuple[str, ...]] = ("<br>", "<br/>", "<br />", "</br>")


class TraceDraft(BaseModel):
    """反解草稿：只回答「引文在不在文档里」与「出自第几节」。

    节号是 1 起的自然数（题面上也是 1 起），不在文档里时给 null；取值域由
    `tests/workflows/citation/test_judges.py` 锁住，不靠人记。
    """

    in_doc: bool = Field(description="引文是否原样出现在文档里")
    section: int | None = Field(
        default=None,
        ge=1,
        le=features.SECTIONS,
        description="引文出自第几节（1 起）；不在文档里时给 null",
    )


def build_trace_prompt(state: str, loc: dict[str, JsonValue]) -> str:
    """组装反解提示词。

    输入：渲染好的 state、语言字典。输出：提示词。
    预期行为：把整段 state（含文档、引文与声称出处）原样交给后端，只问「引文在不在、出自
    第几节」——反解用的口径与渲染期用的机械口径是同一份（都写在 state 里），否则「答对了」
    与「标签对了」不是一回事。
    """
    return "\n".join(
        [
            str(loc["verify"]["intro"]),
            "",
            str(loc["verify"]["state_header"]),
            state,
            "",
            str(loc["verify"]["q"]).format(n=features.SECTIONS),
        ]
    )


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：文档与引文得兑现引文计划，整段 state 不许有残留。

    查三类：**单独成行的围栏 / HTML 换行标签**、**反解不出文档或引文**、**引文与计划对不上**
    （口径直接复用 `render.check_draft`：渲染期查一遍是为了重试，这里再查一遍是为了兜住
    **不是本 workflow 渲染出来的** state）。
    """

    locale: str
    name: str = "hygiene"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只做机械比对。"""
        if not isinstance(spec, CitationSpec):
            return Verdict(True, reason="没有 spec，跳过卫生检查", skipped=True)
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过卫生检查", skipped=True)
        leftover = leftover_in(state, LEFTOVER_LINES, LEFTOVER_TAGS)
        if leftover:
            return Verdict(False, reason=f"state 里有格式残留：{leftover!r}")
        parsed = render.parse_state(state)
        if parsed.title is None:
            return Verdict(False, reason="state 里认不出文档抬头")
        if parsed.quote is None:
            return Verdict(False, reason="state 里认不出引文")
        lang = render.detect_locale(state) or self.locale
        loc = locales.load(lang)
        sentences = parsed.sentences
        draft = render.CitationDraft.model_construct(title=parsed.title, sentences=list(sentences))
        try:
            render.check_draft(draft, spec.plan, loc)
        except ValueError as exc:
            return Verdict(False, reason=str(exc))
        wanted = render.make_quote(
            sentences[spec.plan.sentence_index], spec.plan.form, sentences
        )
        if parsed.quote != wanted:
            return Verdict(
                False, reason=f"state 里的引文与按计划剪出来的不一致：{parsed.quote!r} != {wanted!r}"
            )
        return Verdict(True, reason=f"{len(sentences)} 句正文全部兑现了引文计划")


@dataclass(frozen=True)
class FrameJudge:
    """零 token 判别器：节数、每节句数、声称出处必须逐格等于注入特征与常量。

    任何一处对不上都判不通过——那是渲染挂错了 case，标签会跟着一起错。
    """

    locale: str
    name: str = "frame"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只做结构比对。"""
        if not isinstance(spec, CitationSpec):
            return Verdict(True, reason="没有 spec，跳过框架比对", skipped=True)
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过框架比对", skipped=True)
        parsed = render.parse_state(state)
        problems: list[str] = []
        if len(parsed.sections) != features.SECTIONS:
            problems.append(
                f"小节数对不上：解析出 {len(parsed.sections)} 节，应为 {features.SECTIONS} 节"
            )
        for index, group in enumerate(parsed.sections):
            if len(group) != features.SENTENCES_PER_SECTION:
                problems.append(
                    f"第 {index + 1} 节该有 {features.SENTENCES_PER_SECTION} 句，"
                    f"实际 {len(group)} 句"
                )
        if parsed.quote is None or not parsed.quote.strip():
            problems.append("引文解析不出来或为空")
        if parsed.claimed != spec.plan.claimed:
            problems.append(
                f"声称出处与注入特征对不上：解析出 "
                f"{'空' if parsed.claimed is None else f'第 {parsed.claimed + 1} 节'}，"
                f"应为第 {spec.plan.claimed + 1} 节"
            )
        if not problems:
            return Verdict(True, reason="节数、每节句数与声称出处都与注入特征一致")
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

        预期行为：不一致时把「计划的」与「反解出来的」列在理由里，方便人看是哪里写歪了。
        端点故障与草稿缺字段都只算「没判成」（`skipped=True`）——记成「不通过」会把端点抖动
        写成数据质量问题。
        """
        if not isinstance(spec, CitationSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="choice",
            domain="citation",
            prompt=build_trace_prompt(str(sample["state"]), loc),
            response_model=TraceDraft,
            spec={
                "case": spec.case.as_dict(),
                "form": spec.plan.form,
                "true_section": spec.plan.true_section,
                "claimed": spec.plan.claimed,
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
        seen_in_doc = draft.data.get("in_doc")
        seen_section = draft.data.get("section")
        if not isinstance(seen_in_doc, bool):
            return Verdict(
                True,
                reason=f"反解草稿缺 in_doc 字段，判不了：{seen_in_doc!r}",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        if isinstance(seen_section, bool) or not isinstance(seen_section, (int, type(None))):
            return Verdict(
                True,
                reason=f"反解草稿的 section 不是整数也不是空，判不了：{seen_section!r}",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        index = None if seen_section is None else int(seen_section) - 1
        if seen_in_doc != labels.in_doc:
            return Verdict(
                False,
                reason=(
                    f"计划里引文{'在' if labels.in_doc else '不在'}文档里，"
                    f"反解读成{'在' if seen_in_doc else '不在'}"
                ),
                raw={"traced_in_doc": seen_in_doc},
            )
        if index != labels.section:
            return Verdict(
                False,
                reason=(
                    f"计划里引文出自 "
                    f"{'无' if labels.section is None else f'第 {labels.section + 1} 节'}，"
                    f"反解读成 {'无' if index is None else f'第 {index + 1} 节'}"
                ),
                raw={"traced_section": index},
            )
        return Verdict(
            True,
            reason=f"反解与计划一致：{'在' if labels.in_doc else '不在'}文档里",
            raw={"traced_in_doc": seen_in_doc, "traced_section": index},
        )
