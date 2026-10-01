"""改动引入性判定的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `HygieneJudge`（零 token）：发现段与修法段必须存在、单行、非空，整段 state 不含围栏 /
  HTML 标签，改动记录与判定口径两段必须解析得出来；
- `FrameJudge`（零 token）：改动记录必须**逐行等于计划**（表长 + 每一格），判定口径
  必须逐行等于规范表——渲染串味（挂错计划、漏行、错格）全在这一关拦下；
- `TraceJudge`（一次后端调用）：把整段 state 交回后端反解**指向哪一行、引用哪个符号、
  修法点没点名**，三件都必须与注入的一致。这是「渲染必须可反解」的落地
  （PLAN 3.4 第 ③ 档）：那条发现与那句修法是模型写的，唯一由文本承载的标签就是它们，
  必须能从文本里读回来，否则这道题没人答得对。

输入/输出：只出 `Verdict`，不改样本。反解失败一律判不通过——宁可少要几条，
也不要一条标签和文本对不上的数据。但**端点故障不算数据不合格**：重试与降级都试过之后
还拿不到草稿，只能记成 `skipped=True`（没判成），否则端点抖动会被写成合成质量问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel

from jc.backend.types import BackendError, GenerationBackend, GenerationTask
from jc.judge import Verdict, leftover_in
from jc.jsontypes import JsonValue

from . import locales, render
from .spec import FindTriageSpec

#: 格式残留：单独成行的围栏 / HTML 换行标签。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset(
    {"```", "```json", "```toml", "```text", "```markdown"}
)
LEFTOVER_TAGS: Final[tuple[str, ...]] = ("<br>", "<br/>", "<br />", "</br>")


class TraceDraft(BaseModel):
    """反解草稿：指向哪一行（选项名）、引用哪个符号、修法点没点名。

    `cite` 的取值域随表长变化（`s1`…`sN` 或 `none`），所以这里只能是 `str`；
    合法取值由 `render.cite_options` 拼进提示词，比对时按选项名精确相等。
    """

    cite: str
    symbol: str
    actionable: bool


def build_trace_prompt(state: str, size: int, loc: dict[str, JsonValue]) -> str:
    """组装反解提示词。

    输入：渲染好的 state、记录表长、语言字典。
    输出：提示词。预期行为：把 `cite` 与 `actionable` 两族的原子题连同选项原样再问一遍
    ——反解用的选项与数据集里的一致，否则「答对了」与「标签对了」不是一回事；
    符号那一问没有选项，直接问「用反引号括起来的那个符号是哪个」。
    """
    verify = loc["verify"]
    cite_spec = loc["fam"]["cite"]
    lines = [
        str(verify["intro"]),
        "",
        str(verify["state_header"]),
        state,
        "",
        str(verify["questions_header"]),
        f"1. {cite_spec['q'][0]}",
    ]
    options = render_cite_options(size, loc)
    lines.extend(
        str(verify["option_line"]).format(name=name, desc=options[name]) for name in options
    )
    lines.append(str(verify["answer_rule"]))
    lines.append(f"2. {verify['symbol_q']}")
    lines.append(f"3. {loc['fam']['actionable']['q'][0]}")
    return "\n".join(lines)


def render_cite_options(size: int, loc: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """输入：表长、语言字典。输出：`cite` 那一族的选项（供反解提示词复用）。

    预期行为：与 `questions.cite_options` 同源——这里**再拼一遍**是为了不让判别器
    import 题族模块（题族 import 特征，判别器只该依赖特征与文案），两边靠
    `tests/workflows/findtriage/test_judges.py` 的等价断言钉住。
    """
    template = str(loc["meta"]["row_ref"])
    options: dict[str, JsonValue] = {
        f"s{index}": template.format(n=index) for index in range(1, size + 1)
    }
    options[str(loc["meta"]["no_cite"])] = str(loc["fam"]["cite"]["opt"]["none"])
    return options


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：state 里不许有格式残留，发现段与修法段还得有内容、是单行。

    查六类：**单独成行的围栏 / HTML 换行标签**、**发现段空的 / 多行的**、
    **修法段空的 / 多行的**、**改动记录解析不出来**、**判定口径解析不出来**、
    **归属口径解析不出来**。
    口径压到最窄——宁可漏检也不误判，因为误判会直接拒掉好数据。

    字段：`locale` 语言（要按语言认那几段的段落名）。
    """

    locale: str
    name: str = "hygiene"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只看残留与形状，不看语义。"""
        del spec
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过卫生检查")
        loc = locales.load(self.locale)
        leftover = leftover_in(state, LEFTOVER_LINES, LEFTOVER_TAGS)
        if leftover:
            return Verdict(False, reason=f"state 里有格式残留：{leftover!r}")
        problems: list[str] = []
        for key, label in (("finding_label", "发现"), ("fix_label", "修法")):
            lines = render.section_lines(state, loc, key)
            if not lines:
                problems.append(f"{label}那一段不见了")
            elif len(lines) > 1:
                problems.append(f"{label}是多行——它必须是一行")
            elif not lines[0].strip():
                problems.append(f"{label}是空的")
        if not render.rows_of(state, loc):
            problems.append("改动记录解析不出来")
        if not render.rules_of(state, loc):
            problems.append("判定口径解析不出来")
        if not render.owner_rules_of(state, loc):
            problems.append("归属口径解析不出来")
        if problems:
            return Verdict(False, reason="；".join(problems))
        return Verdict(True, reason="发现段与修法段都在，三张表也都解析得出来")


@dataclass(frozen=True)
class FrameJudge:
    """零 token 判别器：改动记录、判定口径与归属口径必须与代码算出来的一模一样。

    字段：`locale` 语言。预期行为：逐行比对改动记录（表长 + 每一格），再逐行比对判定口径
    与归属口径；任何一处对不上都判不通过——那是渲染挂错了计划，标签会跟着一起错。
    """

    locale: str
    name: str = "frame"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只做结构比对。"""
        if not isinstance(spec, FindTriageSpec):
            return Verdict(True, reason="没有 spec，跳过框架比对", skipped=True)
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过框架比对", skipped=True)
        loc = locales.load(self.locale)
        problems: list[str] = []
        expected = spec.plan.rows
        parsed = render.rows_of(state, loc)
        if parsed != expected:
            problems.append(f"改动记录与计划对不上：解析出 {len(parsed)} 行，应为 {len(expected)} 行")
        rules = render.section_lines(state, loc, "rules_label")
        canonical = render.rule_lines(loc)
        if rules != canonical:
            problems.append(f"判定口径与规范表对不上：解析出 {len(rules)} 行，应为 {len(canonical)} 行")
        owners = render.section_lines(state, loc, "owner_label")
        canonical_owners = render.owner_lines(loc)
        if owners != canonical_owners:
            problems.append(
                f"归属口径与规范表对不上：解析出 {len(owners)} 行，应为 {len(canonical_owners)} 行"
            )
        if not problems:
            return Verdict(True, reason="改动记录、判定口径与归属口径都与计划一致")
        return Verdict(False, reason="；".join(problems), raw={"rows": len(parsed)})


@dataclass(frozen=True)
class TraceJudge:
    """模型判别器：把整段 state 交回后端反解三件事，都必须与注入的一致。

    字段：`backend` 生成后端（由 `bind_backend` 注入）；`locale` 语言。
    """

    backend: GenerationBackend
    locale: str
    name: str = "trace"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：不一致时把「注入的」与「反解出来的」列在理由里，方便人看是哪里写歪了。
        端点故障与草稿缺字段都只算「没判成」（`skipped=True`）——记成「不通过」会把
        端点抖动写成数据质量问题。
        """
        if not isinstance(spec, FindTriageSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        if self.backend is None:
            return Verdict(True, reason="没有后端，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        plan = spec.plan
        labels = spec.labels
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="text",
            domain="finding-triage",
            prompt=build_trace_prompt(str(sample["state"]), plan.size, loc),
            response_model=TraceDraft,
            spec={
                "tags": spec.case.as_dict(),
                "truth": {
                    "cite": labels.cite_key,
                    "symbol": plan.evidence_symbol,
                    "actionable": labels.actionable,
                },
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
        data = draft.data if isinstance(draft.data, dict) else {}
        traced_cite = data.get("cite")
        traced_symbol = data.get("symbol")
        traced_actionable = data.get("actionable")
        if not isinstance(traced_cite, str) or not isinstance(traced_symbol, str):
            return Verdict(
                True,
                reason="反解草稿缺字段，判不了",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        if not isinstance(traced_actionable, bool):
            return Verdict(
                True,
                reason="反解草稿的 `actionable` 不是布尔，判不了",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        problems: list[str] = []
        if traced_cite != labels.cite_key:
            problems.append(f"cite: 注入 {labels.cite_key!r}，反解成 {traced_cite!r}")
        if traced_symbol != plan.evidence_symbol:
            problems.append(f"symbol: 注入 {plan.evidence_symbol!r}，反解成 {traced_symbol!r}")
        if traced_actionable is not labels.actionable:
            problems.append(f"actionable: 注入 {labels.actionable}，反解成 {traced_actionable}")
        if problems:
            return Verdict(False, reason="；".join(problems), raw={"traced": traced_cite})
        return Verdict(
            True,
            reason="反解出的指向、符号、修法三件都与注入的一致",
            raw={"traced": traced_cite},
        )
