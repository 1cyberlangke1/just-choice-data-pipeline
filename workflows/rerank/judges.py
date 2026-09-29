"""检索重排的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `HygieneJudge`（零 token）：查询与候选必须**兑现将计划**（每条候选该沾上几个关键词就得
  沾上几个），关键词个数与两两不互为子串、查询里得出现每个词、候选正好 5 条且互不相同，
  整段 state 不许有格式残留；
- `FrameJudge`（零 token）：关键词个数、候选条数、排序口径那两行必须逐格等于注入特征与
  规范表——渲染串味（挂错 case、抄错口径）全在这一关拦下；
- `TraceJudge`（一次后端调用）：把整段 state 交回后端**读一遍**，数一遍「有几条候选沾上了
  至少一个关键词」，必须与计划里的相关条数一致。这是「渲染必须可反解」的落地：题面上的
  答案就是从 state 里读出来的，所以这段 state 必须真的读得出那个数。

为什么覆盖率既机械查、又问模型：机械那一遍是**口径的权威**（渲染期还能据此重试），模型
那一遍是**读者视角的复核**——关键词要是挑成 `the` 这类词，机械口径照样通过，而读者数出来
的东西没有意义，这种样本该丢。两遍问的不是同一件事。

输入/输出：只出 `Verdict`，不改样本。反解失败一律判不通过——宁可少要几条，也不要一条标签
和文本对不上的数据。但**端点故障不算数据不合格**：重试与降级都试过之后还拿不到草稿，只能
记成 `skipped=True`（没判成），否则端点抖动会被写成合成质量问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import BackendError, GenerationBackend, GenerationTask
from jc.judge import Verdict
from jc.jsontypes import JsonValue

from . import features, locales, render
from .spec import RerankSpec

#: 格式残留：单独成行的围栏 / HTML 换行标签。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset(
    {"```", "```json", "```toml", "```text", "```markdown", "```md", "```yaml"}
)
LEFTOVER_TAGS: Final[tuple[str, ...]] = ("<br>", "<br/>", "<br />", "</br>")


class TraceDraft(BaseModel):
    """反解草稿：只回答「有几条候选沾上了至少一个关键词」。

    取值域是 0~候选条数，与 `features.RELEVANT` 的上界一致；这条由
    `tests/workflows/rerank/test_judges.py` 锁住，不靠人记。
    """

    relevant: int = Field(ge=0, le=features.CANDIDATES, description="沾上至少一个关键词的候选条数")


def build_trace_prompt(state: str, loc: dict[str, JsonValue]) -> str:
    """组装反解提示词。

    输入：渲染好的 state、语言字典。输出：提示词。
    预期行为：把整段 state（含关键词表与排序口径）原样交给后端，只问「有几条相关」——反解
    用的口径与渲染期用的机械口径是同一份（都写在 state 的关键词与口径里），否则「答对了」
    与「标签对了」不是一回事。
    """
    return "\n".join(
        [
            str(loc["verify"]["intro"]),
            "",
            str(loc["verify"]["state_header"]),
            state,
            "",
            str(loc["verify"]["q"]).format(n=features.CANDIDATES),
        ]
    )


def _has_leftover(state: str) -> str:
    """输入：state 文本。输出：命中的残留（没有就空串）。"""
    for line in state.splitlines():
        if line.strip() in LEFTOVER_LINES:
            return line.strip()
    for tag in LEFTOVER_TAGS:
        if tag in state:
            return tag
    return ""


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：查询与候选得兑现将计划、条数得对、整段 state 不许有残留。

    查三类：**单独成行的围栏 / HTML 换行标签**、**查询或候选反解不出来**、**覆盖率与计划对
    不上**（口径直接复用 `render.check_parts`：渲染期查一遍是为了重试，这里再查一遍是为了
    兜住**不是本 workflow 渲染出来的** state）。
    """

    locale: str
    name: str = "hygiene"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只做机械比对。"""
        if not isinstance(spec, RerankSpec):
            return Verdict(True, reason="没有 spec，跳过卫生检查", skipped=True)
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过卫生检查", skipped=True)
        leftover = _has_leftover(state)
        if leftover:
            return Verdict(False, reason=f"state 里有格式残留：{leftover!r}")
        parsed = render.parse_state(state)
        if parsed.query is None:
            return Verdict(False, reason="state 里认不出查询抬头（渲染串味或格式坏了）")
        loc = locales.load(self.locale)
        try:
            render.check_parts(
                parsed.query, list(parsed.terms), list(parsed.candidates), spec.plan, loc
            )
        except ValueError as exc:
            return Verdict(False, reason=str(exc))
        return Verdict(True, reason=f"{len(parsed.candidates)} 条候选全部兑现了名次计划")


@dataclass(frozen=True)
class FrameJudge:
    """零 token 判别器：关键词个数、候选条数、排序口径必须逐格等于注入特征与规范表。

    任何一处对不上都判不通过——那是渲染挂错了 case，标签会跟着一起错。
    """

    locale: str
    name: str = "frame"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。预期行为：只做结构比对。"""
        if not isinstance(spec, RerankSpec):
            return Verdict(True, reason="没有 spec，跳过框架比对", skipped=True)
        state = sample["state"]
        if not isinstance(state, str):
            return Verdict(True, reason="state 不是文本，跳过框架比对", skipped=True)
        loc = locales.load(self.locale)
        parsed = render.parse_state(state)
        problems: list[str] = []
        if len(parsed.terms) != spec.plan.terms:
            problems.append(
                f"关键词个数对不上：解析出 {len(parsed.terms)} 个，应为 {spec.plan.terms} 个"
            )
        if len(parsed.candidates) != features.CANDIDATES:
            problems.append(
                f"候选条数对不上：解析出 {len(parsed.candidates)} 条，"
                f"应为 {features.CANDIDATES} 条"
            )
        canonical = render.rule_rows(loc)
        if parsed.rule != canonical:
            problems.append(
                f"排序口径与规范表对不上：解析出 {len(parsed.rule)} 行，应为 {len(canonical)} 行"
            )
        if not problems:
            return Verdict(True, reason="关键词个数、候选条数与排序口径都与注入特征一致")
        return Verdict(False, reason="；".join(problems))


@dataclass(frozen=True)
class TraceJudge:
    """模型判别器：把整段 state 交回后端读一遍，数出来的相关条数必须与计划一致。

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
        if not isinstance(spec, RerankSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="choice",
            domain="rerank",
            prompt=build_trace_prompt(str(sample["state"]), loc),
            response_model=TraceDraft,
            spec={"case": spec.case.as_dict(), "overlaps": list(spec.plan.overlaps)},
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
        seen = draft.data.get("relevant")
        if not isinstance(seen, int) or isinstance(seen, bool):
            return Verdict(
                True,
                reason=f"反解草稿缺 relevant 字段，判不了：{seen!r}",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        if seen != spec.plan.relevant:
            return Verdict(
                False,
                reason=f"计划里相关 {spec.plan.relevant} 条，反解读成 {seen} 条",
                raw={"traced": seen},
            )
        return Verdict(True, reason=f"反解与计划一致：相关 {seen} 条", raw={"traced": seen})


def option_names(question: dict[str, JsonValue]) -> list[str]:
    """输入：一道 choice 题。输出：按**产物里写的顺序**排好的选项名。

    预期行为：答案编号就是在这个顺序里的下标（见 `jc.assemble`）——审计要拿它把编号换回
    选项名与选项描述，才能与纯代码重算的结果比对。
    """
    criteria = question.get("criteria")
    return list(criteria) if isinstance(criteria, dict) else []
