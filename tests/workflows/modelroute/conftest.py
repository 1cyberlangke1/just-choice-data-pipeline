"""模型路由 workflow 测试的公共夹具：一个**不发网络请求**的假后端。

假后端按 `task.spec["tags"]` 造草稿，所以它天然「知道」注入特征：
- 渲染任务 → 按（任务类型 × 规模）挑一句与之相称的用户请求；
- 反解任务 → 直接把注入的任务类型当成反解结果（也就是「模型完全答对」）。

要验「判别器能抓错」时，用 `FakeBackend(trace_lies=True)` / `FakeBackend(fence=True)`
之类的开关故意说反话；要验「端点故障不算数据问题」时，用
`FakeBackend(trace_raises=TransportError(...))`。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from tests.workflows import _fakes
from tests.workflows._fakes import draft
from workflows.modelroute import features

#: 每个（任务类型 × 规模）一句与特征相称、且不含任何模型名的用户请求。
#: 请求里体现规模的那半句必须跟着规模变——`judges.TraceJudge` 只反解任务类型，
#: 但人看数据时要能一眼看出规模注入生效了。
REQUESTS: Final[dict[str, dict[str, str]]] = {
    "code_gen": {
        "small": "帮我把这个 20 行的工具函数改成异步的",
        "medium": "帮我把这个模块里的十几个函数都改成异步的",
        "huge": "帮我把整个仓库里的同步调用全改成异步的",
    },
    "long_summary": {
        "small": "把这段几百字的说明压成三句话",
        "medium": "把这份十几页的事故复盘压成要点",
        "huge": "把上百页的会议记录压成一份摘要",
    },
    "extraction": {
        "small": "从这段两行的订单里抽出金额和收货人",
        "medium": "从这几页合同里抽出签约方和金额",
        "huge": "从整库的历史工单里抽出设备编号和故障码",
    },
    "chat": {
        "small": "陪我聊聊今天午饭该吃什么",
        "medium": "陪我聊聊这几页产品说明该怎么写",
        "huge": "陪我聊聊这上百页技术方案该怎么取舍",
    },
    "math_reason": {
        "small": "帮我算一下这道两步就能解出来的方程",
        "medium": "帮我算一下这道要好几步的数列题",
        "huge": "帮我算一下这道要几十步推导的证明题",
    },
    "vision": {
        "small": "看看这张小截图里按钮上写的是什么字",
        "medium": "看看这几页图表里哪条线在往上走",
        "huge": "看看这上百张图纸里哪几张标错了尺寸",
    },
}


class FakeBackend(_fakes.ScriptedBackend):
    """脚本化假后端：草稿由 `task.spec["tags"]` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：
    - `trace_lies` 反解时把任务类型说成另一个取值（验 `trace` 判别器）；
    - `multi_line_request` 把请求写成两行（验 `check_draft` 与 `hygiene` 判别器）；
    - `fence_request` 把请求包进 Markdown 围栏（验 `check_draft` 与 `hygiene` 判别器）；
    - `empty_request` 把请求写成空白（验 `check_draft` 与 `hygiene` 判别器）；
    - `label_request` 在请求开头再写一遍「请求：」（验渲染期的段落名残留检查）；
    - `name_request` 在请求里点名候选表里的模型（验渲染期的答案泄漏检查）；
    - `trace_raises` 反解时直接抛异常（验端点故障被记成跳过而不是拒绝）；
    - `trace_data` 反解时原样吐这份草稿（验缺字段 / 字段类型不对也记成跳过）。
    """

    name = "fake"
    capabilities = None

    def __init__(
        self,
        *,
        trace_lies: bool = False,
        multi_line_request: bool = False,
        fence_request: bool = False,
        empty_request: bool = False,
        label_request: bool = False,
        name_request: bool = False,
        trace_raises: Exception | None = None,
        trace_data: dict[str, JsonValue] | None = None,
    ) -> None:
        """输入：九个作弊开关。输出：假后端实例。"""
        self.trace_lies = trace_lies
        self.multi_line_request = multi_line_request
        self.fence_request = fence_request
        self.empty_request = empty_request
        self.label_request = label_request
        self.name_request = name_request
        self.trace_raises = trace_raises
        self.trace_data = trace_data
        self.calls: list[str] = []

    def generate(self, task: object) -> Draft:
        """输入：一个任务。输出：草稿。

        预期行为：按任务 id 后缀区分「渲染」与「反解」；任务里没带 tags 就报错，
        免得测试因为夹具写错而静默通过。
        """
        self.calls.append(task.task_id)
        tags = task.spec.get("tags")
        assert tags, f"假后端要求任务带上 tags：{task.task_id}"
        if task.task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            return self._draft(self._trace(tags))
        return self._draft(self._render(tags))

    def _render(self, tags: dict[str, str]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`RouteDraft` 形态的字典。

        预期行为：请求必须与「任务类型」一致（`trace` 判别器就靠这个得出答案），
        并且真的体现「材料规模」；默认写法单行、无围栏、不点名任何模型。
        """
        request = REQUESTS[tags["task_kind"]][tags["size"]]
        if self.name_request:
            request = f"{request}，用 atlas-xl 那个就行"
        if self.label_request:
            request = f"请求：{request}"
        if self.fence_request:
            request = f"```\n{request}\n```"
        if self.multi_line_request:
            request = f"{request}\n（材料在附件里）"
        if self.empty_request:
            request = "   "
        return {"request": request}

    def _trace(self, tags: dict[str, str]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`TraceDraft` 形态的字典（默认完全答对）。"""
        if self.trace_data is not None:
            return dict(self.trace_data)
        kind = tags["task_kind"]
        if self.trace_lies:
            order = list(features.TASK_KINDS)
            kind = order[(order.index(kind) + 1) % len(order)]
        return {"kind": kind}

    @staticmethod
    def _draft(data: dict[str, JsonValue]) -> Draft:
        """输入：草稿数据。输出：`Draft`。预期行为：记账字段填假值（共用工厂 `_fakes.draft`）。"""
        return draft(data)


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(trace_lies=True)` 造一个假后端。"""
    return FakeBackend
