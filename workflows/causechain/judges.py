"""根因定位的判别器：分工是「纯代码能查的」与「只能问模型的」。

- `HygieneJudge`（零 token）：每个服务那一行得**兑现现场计划**——六个服务一条不少、非空、
  不带围栏、长度落在区间内，按字面算出来的事件类型与计划逐条相同，而且**没有把事件判据原文
  整段贴进去**；整段 state 不许有格式残留；
- `FrameJudge`（零 token）：state 的骨架必须完整且规范——依赖图那一段还原出来的边表要与
  这张图逐条相同、现场日志的排列顺序要与计划一致、六个服务一条不少。任何一处对不上都说明
  渲染挂错了 case、或者语言文件缺了内容，标签会跟着一起错；
- `TraceJudge`（一次后端调用）：把整段 state 交回后端**读一遍**，答「根因是哪个服务」，必须与
  计划算出来的结论一致。这是「渲染必须可反解」的落地：题面上的答案就是从 state 里读出来的，
  所以这段 state 必须真的读得出那个结论。

为什么「谁出错」既机械查、又问模型：机械那一遍是**口径的权威**（渲染期还能据此重试），模型
那一遍是**读者视角的复核**——机械口径只认字符，日志要是写成「本行属于超时事件」这种自指
句子，机械口径照样通过，而读者读出来的东西没有意义，这种样本该丢。两遍问的不是同一件事。

**反解只问根因、不要引文**：答案是一个服务编号（从六个里挑一个），不是从正文里抄一段话，
所以这里没有「引文站不站得住」那一步（范式 §9.88 讲的是引文被装进 JSON 数组的情况）。
端点故障、草稿缺字段与不认识的编号都只算「没判成」——记成「不通过」会把端点抖动或格式漂移
写成数据质量问题。

输入/输出：只出 `Verdict`，不改样本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import BackendError, GenerationBackend, GenerationTask
from jc.judge import Verdict, verify_prompt
from jc.jsontypes import JsonValue

from . import features, locales, render
from .features import SERVICES
from .spec import CauseSpec

#: 格式残留：单独成行的围栏 / HTML 换行标签。
LEFTOVER_LINES: Final[frozenset[str]] = frozenset(
    {"```", "```json", "```toml", "```text", "```markdown", "```md", "```yaml"}
)
LEFTOVER_TAGS: Final[tuple[str, ...]] = ("<br>", "<br/>", "<br />", "</br>")


class TraceDraft(BaseModel):
    """反解草稿：只回答「这次故障的根因是哪个服务」。

    取值域由 `features.SERVICES` 定，认不出来的编号一律算「没判成」（`skipped=True`）而不是
    「不通过」——那是反解格式没对齐，不是数据本身错。
    """

    root: str = Field(description="根因服务的编号（S1 / S2 / S3 / S4 / S5 / S6）")


#: 转出去：口径只有一份，住在 `jc.judge.verify_prompt`（18 家逐字同构，只留那一份）。
build_trace_prompt = verify_prompt


@dataclass(frozen=True)
class HygieneJudge:
    """零 token 判别器：现场日志必须兑现现场计划。

    字段：`locale` 语言（只用来当场验一下这个语言存不存在）；`name` 判别器名。
    """

    locale: str
    name: str = "hygiene"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：没有规格或 state 不是字符串时记「没判成」；有规格就按字面重算每个服务的
        事件类型，与计划逐条比。规格类型不对（串了别的 workflow）也算没判成。
        """
        locales.load(self.locale)
        if not isinstance(spec, CauseSpec):
            return Verdict(True, reason="没有 spec，跳过兑现检查", skipped=True)
        state = sample.get("state")
        if not isinstance(state, str):
            return Verdict(True, reason="样本里没有 state，跳过兑现检查", skipped=True)
        parsed = render.parse_state(state)
        if parsed.text is None or not parsed.text.strip():
            return Verdict(False, reason="现场日志那一块解析不出来或为空")
        lang = render.detect_locale(state) or self.locale
        loc = locales.load(lang)
        logs = render.parse_logs(parsed.text, loc)
        missing = [service for service in SERVICES if service not in logs]
        if missing:
            return Verdict(False, reason=f"现场日志缺了这几个服务：{missing}")
        for service in SERVICES:
            found = features.event_of(logs[service])
            if found != spec.plan.events[service]:
                return Verdict(
                    False,
                    reason=(
                        f"{service} 那一行按字面算出来是 {found or '认不出的形状'}，"
                        f"计划要的是 {spec.plan.events[service]}"
                    ),
                )
            pasted = render.pasted_rules(logs[service], loc)
            if pasted:
                return Verdict(False, reason=f"{service} 那一行把判据原文原样贴了进去：{pasted}")
        leftovers = [
            line for line in (raw.strip() for raw in state.splitlines()) if line in LEFTOVER_LINES
        ]
        if leftovers:
            return Verdict(False, reason=f"state 里有格式残留：{sorted(set(leftovers))}")
        if any(tag in state for tag in LEFTOVER_TAGS):
            return Verdict(False, reason="state 里混进了 HTML 换行标签")
        return Verdict(True, reason="现场日志兑现了计划，且没有格式残留")


@dataclass(frozen=True)
class FrameJudge:
    """零 token 判别器：state 的骨架必须完整且规范。

    字段：`locale` 语言。
    """

    locale: str
    name: str = "frame"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：依赖图那一段还原出来的边表必须与这张图逐条相同、现场日志的排列顺序必须与
        计划一致、六个服务一条不少。
        """
        locales.load(self.locale)
        if not isinstance(spec, CauseSpec):
            return Verdict(True, reason="没有 spec，跳过骨架检查", skipped=True)
        state = sample.get("state")
        if not isinstance(state, str):
            return Verdict(True, reason="样本里没有 state，跳过骨架检查", skipped=True)
        parsed = render.parse_state(state)
        problems: list[str] = []
        edges = render.parse_edges(parsed.tags)
        if edges != features.GRAPHS[spec.plan.graph]:
            problems.append(
                f"依赖图还原出来是 {list(edges)}，应为 {list(features.GRAPHS[spec.plan.graph])}"
            )
        lang = render.detect_locale(state) or self.locale
        rows = render.parse_rows(parsed.text, locales.load(lang))
        if [service for _, service, _ in rows] != list(spec.plan.order):
            problems.append(
                f"现场日志的服务顺序是 {[service for _, service, _ in rows]}，"
                f"应为 {list(spec.plan.order)}"
            )
        if not problems:
            return Verdict(True, reason="依赖图与现场日志骨架都与规范形态一致")
        return Verdict(False, reason="；".join(problems))


@dataclass(frozen=True)
class TraceJudge:
    """模型判别器：把整段 state 交回后端读一遍，读出来的根因必须与计划一致。

    字段：`backend` 生成后端（由 `bind_backend` 注入）；`locale` 语言。
    """

    backend: GenerationBackend
    locale: str
    name: str = "trace"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。

        预期行为：不一致时把「计划的」与「反解出来的」列在理由里，方便人看是哪里写歪了。
        """
        if not isinstance(spec, CauseSpec):
            return Verdict(True, reason="没有 spec，跳过反解", skipped=True)
        loc = locales.load(self.locale)
        task = GenerationTask(
            task_id=spec.trace_task_id,
            kind="choice",
            domain="causechain",
            prompt=build_trace_prompt(str(sample["state"]), loc),
            response_model=TraceDraft,
            seeds=(),
            spec={"case": spec.case.as_dict(), "root": spec.plan.root},
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
        seen = draft.data.get("root")
        if not isinstance(seen, str) or seen not in SERVICES:
            return Verdict(
                True,
                reason=f"反解草稿给的根因认不出来（{seen!r}），判不了",
                raw={"raw_text": draft.raw_text[:500]},
                skipped=True,
            )
        if seen != spec.plan.root:
            return Verdict(
                False,
                reason=f"计划里的根因是 {spec.plan.root}，反解读成 {seen}",
                raw={"traced_root": seen},
            )
        return Verdict(
            True,
            reason=f"反解与计划一致：根因 {spec.plan.root}",
            raw={"traced_root": seen},
        )
