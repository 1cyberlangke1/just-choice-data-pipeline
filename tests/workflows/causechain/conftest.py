"""causechain 的假后端：按 `task.spec` 现造六行现场日志，不联网。

输入：`GenerationTask`（渲染任务带 `events`，反解任务带 `root`）。
输出：`Draft`。
预期行为：默认完全兑现计划——每个服务那行踩中 `plan.events` 点名的那一种形状；另外几个开关
各自破坏一件事，用来验「渲染期与判别器真的会拦下来」。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from tests.workflows._fakes import draft as _draft
from workflows.causechain import features, locales, render
from workflows.causechain.features import SERVICES, CauseCase, CausePlan
from workflows.causechain.spec import CauseSpec

#: 三种事件各一段片段：**只踩中自己那一种形状**，且都不等于语言文件里的判据原文
#: （整段照抄会被 `render.check_draft` 的照抄闸打回）。
FRAGMENTS: Final[dict[str, str]] = {
    features.EVENT_TIMEOUT: "上游连接超时，正在重试第 2 次",
    features.EVENT_ERROR: "写入订单失败：数据库返回 500，事务已回滚",
    features.EVENT_OK: "健康检查通过，请求处理完成，耗时 12ms",
}


def text_of(service: str, event: str) -> str:
    """输入：服务编号、事件类型。输出：一行兑现它的日志（六个服务那六行彼此不同）。"""
    return f"{service} 节点 req-{service[1:]}07 记录：{FRAGMENTS[event]}"


def logs_of(plan: CausePlan) -> dict[str, str]:
    """输入：现场计划。输出：{服务: 日志}。预期行为：完全兑现计划。"""
    return {service: text_of(service, plan.events[service]) for service in SERVICES}


def state_of(plan: CausePlan, logs: dict[str, str], lang: str = "zh") -> str:
    """输入：现场计划、日志、语言。输出：契约允许的 state。"""
    return render.to_state(render.ContentDraft(logs=logs), plan, locales.load(lang))


def spec_for(case: CauseCase, phrasing: int = 0, split: str = "train", seq: int = 1) -> CauseSpec:
    """输入：case、问法编号、切分、序号。输出：真值规格。"""
    return CauseSpec(
        case=case, phrasing=phrasing, split=split, seq=seq, plan=features.plan_for(case, phrasing)
    )


def faithful(
    case: CauseCase, lang: str = "zh", phrasing: int = 0
) -> tuple[CauseSpec, dict[str, JsonValue]]:
    """输入：case、语言、问法编号。输出：(规格, 样本)。预期行为：日志完全兑现计划。"""
    spec = spec_for(case, phrasing)
    logs = logs_of(spec.plan)
    render.check_draft(render.ContentDraft(logs=logs), spec.plan, locales.load(lang))
    return spec, {"state": state_of(spec.plan, logs, lang)}


class FakeBackend:
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：

    - `trace_lies` 反解时换一个服务（验 `trace` 判别器）；
    - `trace_raises` 反解时抛异常（验「端点故障只算没判成」）；
    - `trace_junk` 反解时给一个不存在的编号；
    - `flip_event` 把第一个服务那行换成另一种形状（验渲染期的兑现闸）；
    - `drop_service` 少写一行、`extra_service` 多写一行；
    - `pasted_rule` 把事件判据原文整段贴进日志；
    - `too_short` 把某一行缩到一个字符、`fence` 给某一行套上围栏。
    """

    def __init__(
        self,
        *,
        trace_lies: bool = False,
        trace_raises: Exception | None = None,
        trace_junk: bool = False,
        flip_event: bool = False,
        drop_service: bool = False,
        extra_service: bool = False,
        pasted_rule: bool = False,
        too_short: bool = False,
        fence: bool = False,
    ) -> None:
        """输入：各开关。输出：实例。"""
        self.trace_lies = trace_lies
        self.trace_raises = trace_raises
        self.trace_junk = trace_junk
        self.flip_event = flip_event
        self.drop_service = drop_service
        self.extra_service = extra_service
        self.pasted_rule = pasted_rule
        self.too_short = too_short
        self.fence = fence
        self.tasks: list[str] = []

    def logs_for(self, events: dict[str, str]) -> dict[str, str]:
        """输入：{服务: 事件类型}。输出：按开关写歪之后的六行日志。"""
        head = SERVICES[0]
        logs = {service: text_of(service, events[service]) for service in SERVICES}
        if self.flip_event:
            other = next(event for event in features.EVENTS if event != events[head])
            logs[head] = text_of(head, other)
        if self.drop_service:
            logs.pop(SERVICES[-1])
        if self.extra_service:
            logs["S9"] = text_of("S9", features.EVENT_OK)
        if self.pasted_rule:
            rule = str(render.event_copy(locales.load("zh"), events[head])["rule"])
            logs[head] = f"{logs[head]} {rule}"
        if self.too_short:
            logs[head] = "短"
        if self.fence:
            logs[head] = f"```{logs[head]}```"
        return logs

    def generate(self, task: object) -> Draft:
        """输入：`GenerationTask`。输出：草稿。预期行为：按 `task.spec` 现造，不发网络。"""
        spec = getattr(task, "spec")
        task_id = str(getattr(task, "task_id"))
        self.tasks.append(task_id)
        if task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            root = str(spec["root"])
            if self.trace_lies:
                root = next(service for service in SERVICES if service != root)
            if self.trace_junk:
                root = "S9"
            return _draft({"root": root}, f"root={root}")
        events = {str(key): str(value) for key, value in spec["events"].items()}
        logs = self.logs_for(events)
        return _draft({"logs": dict(logs)}, "logs=" + ",".join(sorted(logs)))

    def generate_many(self, tasks: list[object]) -> list[Draft]:
        """输入：任务列表。输出：草稿列表。预期行为：逐个走 `generate`。"""
        return [self.generate(task) for task in tasks]


@pytest.fixture
def backend() -> FakeBackend:
    """输入：无。输出：一个完全兑现计划的假后端。"""
    return FakeBackend()


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(trace_lies=True)` 造一个假后端。"""
    return FakeBackend
