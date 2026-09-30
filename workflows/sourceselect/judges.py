"""来源选择的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `HygieneJudge`（零 token）：查询那一段必须存在、单行、非空、不含围栏 / HTML 标签 /
  段落名残留，候选表与选源口径两段必须解析得出来；
- `FrameJudge`（零 token）：候选表必须**逐行等于计划**（表长 + 每一格的取值），选源口径
  必须逐行等于规范表——渲染串味（挂错计划、漏行、错格）全在这一关拦下；
- `TraceJudge`（一次后端调用）：把整段 state 交回后端反解**首要要求**，必须与注入的一致。
  这是「渲染必须可反解」的落地（PLAN 3.4 第 ③ 档）：那句查询是模型写的，唯一由文本承载的
  标签就是它，它必须能从文本里读回来，否则这道题没人答得对。

输入/输出：只出 `Verdict`，不改样本。反解失败一律判不通过——宁可少要几条，
也不要一条标签和文本对不上的数据。但**端点故障不算数据不合格**：重试与降级都试过之后
还拿不到草稿，只能记成 `skipped=True`（没判成），否则端点抖动会被写成合成质量问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from pydantic import BaseModel

from jc.backend.types import BackendError, GenerationBackend, GenerationTask
from jc.judge import Verdict, leftover_in
from jc.jsontypes import JsonValue

from . import locales, render
from .spec import SourceSpec

#: 格式残留：单独成行的围栏 / HTML 换行标签（围栏在查询里由 `render.check_draft` 与
#: 卫生判别器各查一次；这里再按「单独成行」查一遍整段 state）。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset(
    {"```", "```json", "```toml", "```text", "```markdown"}
)
LEFTOVER_TAGS: Final[tuple[str, ...]] = ("<br>", "<br/>", "<br />", "</br>")


class TraceDraft(BaseModel):
    """反解草稿：反解「这次查询对来源的首要要求是哪一项」。

    取值域必须与 `features.NEEDS` 以及 locale 里 `fam.need.criteria` 的 key 一致；
    这条由 `tests/workflows/sourceselect/test_locales.py` 锁住，不靠人记。
    """

    need: Literal["authority", "freshness", "coverage"]


def build_trace_prompt(state: str, loc: dict[str, JsonValue]) -> str:
    """组装反解提示词。

    输入：渲染好的 state、语言字典。
    输出：提示词。预期行为：把 `need` 那一族的原子题连同选项原样再问一遍——
    反解用的选项与数据集里的一致，否则「答对了」与「标签对了」不是一回事。
    """
    spec = loc["fam"]["need"]
    lines = [
        str(loc["verify"]["intro"]),
        "",
        str(loc["verify"]["state_header"]),
        state,
        "",
        str(loc["verify"]["questions_header"]),
        f"1. {spec['q'][0]}",
    ]
    lines.extend(
        str(loc["verify"]["option_line"]).format(name=name, desc=spec["criteria"][name])
        for name in spec["criteria"]
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：state 里不许有格式残留，查询那一段还得有内容、是单行。

    查四类：**单独成行的围栏 / HTML 换行标签**、**查询段是空的 / 多行的**、
    **候选表解析不出来**、**选源口径解析不出来**。口径压到最窄——宁可漏检也不误判，
    因为误判会直接拒掉好数据。

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
        query_lines = render.section_lines(state, loc, "query_label")
        if not query_lines:
            problems.append("查询那一段不见了")
        elif len(query_lines) > 1:
            problems.append("查询是多行——它必须是一行")
        elif not query_lines[0].strip():
            problems.append("查询是空的")
        if not render.rows_of(state, loc):
            problems.append("候选表解析不出来")
        if not render.rules_of(state, loc):
            problems.append("选源口径解析不出来")
        if not problems:
            return Verdict(True, reason="查询段与两张表都在，且没有格式残留")
        return Verdict(False, reason="；".join(problems))


@dataclass(frozen=True)
class FrameJudge:
    """零 token 判别器：候选表与选源口径必须与代码算出来的一模一样。

    字段：`locale` 语言。预期行为：逐行比对候选表（表长 + 每一格的取值），再逐行比对
    选源口径；任何一处对不上都判不通过——那是渲染挂错了计划，标签会跟着一起错。
    """

    locale: str
    name: str = "frame"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只做结构比对。"""
        if not isinstance(spec, SourceSpec):
            return Verdict(True, reason="没有 spec，跳过框架比对", skipped=True)
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过框架比对", skipped=True)
        loc = locales.load(self.locale)
        problems: list[str] = []
        expected = spec.plan.rows
        parsed = render.rows_of(state, loc)
        if parsed != expected:
            problems.append(
                f"候选表与计划对不上：解析出 {len(parsed)} 行，应为 {len(expected)} 行"
            )
        rules = render.section_lines(state, loc, "rules_label")
        canonical = render.rule_lines(loc)
        if rules != canonical:
            problems.append(
                f"选源口径与规范表对不上：解析出 {len(rules)} 行，应为 {len(canonical)} 行"
            )
        if not problems:
            return Verdict(True, reason="候选表与选源口径都与计划一致")
        return Verdict(False, reason="；".join(problems), raw={"rows": len(parsed)})


@dataclass(frozen=True)
class TraceJudge:
    """模型判别器：把整段 state 交回后端反解首要要求，必须与注入的一致。

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
        if not isinstance(spec, SourceSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        if self.backend is None:
            return Verdict(True, reason="没有后端，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="text",
            domain="source-selection",
            prompt=build_trace_prompt(str(sample["state"]), loc),
            response_model=TraceDraft,
            spec={"tags": spec.case.as_dict()},
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
        traced = data.get("need")
        if not isinstance(traced, str):
            return Verdict(
                True,
                reason="反解草稿缺 `need` 字段，判不了",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        expected = spec.labels.need
        if traced != expected:
            return Verdict(
                False,
                reason=f"need: 注入 {expected!r}，反解成 {traced!r}",
                raw={"traced": traced},
            )
        return Verdict(True, reason="反解出的首要要求与注入的一致", raw={"traced": traced})
