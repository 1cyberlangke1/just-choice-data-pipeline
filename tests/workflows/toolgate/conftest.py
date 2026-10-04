"""工具调用门控测试的公共夹具：一个**不发网络请求**的假后端。

假后端按 `task.spec["tags"]` 造草稿，所以它天然「知道」注入特征：
- 渲染任务 → 按（工具 × 内容性质）挑一句与之相称的调用意图；
- 反解任务 → 直接把注入的两维当成反解结果（也就是「模型完全答对」）。

要验「判别器能抓错」时，用 `FakeBackend(trace_lies=True)` / `FakeBackend(fence_intent=True)`
之类的开关故意说反话；要验「端点故障不算数据问题」时，用
`FakeBackend(trace_raises=TransportError(...))`。

夹具里那些意图**刻意不提作用范围**（本工作区 / 本机 / 外网）：那三格归「策略与状态」
那一行管，意图里再写一遍就会跟它打架——这也是 `render.check_draft` 会拦的写法。
"""

from __future__ import annotations

from typing import Final


from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from tests.workflows import _fakes
from tests.workflows._fakes import draft
from workflows.toolgate import features

#: 每个（工具 × 内容性质）一句与之相称、且不提任何作用范围的调用意图。
#: 内容性质那一半必须真的跟着变——`judges.TraceJudge` 要反解的就是这两维。
#: `read_file × harmful` 这一格不存在：读文件读不出破坏性内容（见 `features.FORBIDDEN`）。
INTENTS: Final[dict[str, dict[str, str]]] = {
    "read_file": {
        "benign": "把 config.yaml 里的超时读出来看看现在是多少",
        "sensitive": "读一下 .env 里那串访问密钥，确认它没写错",
    },
    "write_file": {
        "benign": "把 config.yaml 里的超时改成 30 秒",
        "sensitive": "把新的数据库口令写进 .env",
        "harmful": "把 data 目录里的那几个文件整个覆盖成空文件",
    },
    "run_command": {
        "benign": "跑一遍 pytest 看看现在用例的情况",
        "sensitive": "把那份导出的用户表拷一份出来",
        "harmful": "把 users 表整个删掉",
    },
    "http_request": {
        "benign": "调一下库存服务的查询接口，看看还有多少货",
        "sensitive": "把那串访问密钥发到对账接口上去",
        "harmful": "往对方接口里塞一段注入串，看看它扛不扛得住",
    },
    "send_message": {
        "benign": "把今天的进度发到项目频道里",
        "sensitive": "把这份带手机号的名单发到对接群里",
        "harmful": "把这个木马下载链接发到客户群里",
    },
}


class FakeBackend(_fakes.ScriptedBackend):
    """脚本化假后端：草稿由 `task.spec["tags"]` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：
    - `trace_lies` 反解时把「工具」说成另一个取值（验 `trace` 判别器）；
    - `trace_lies_payload` 反解时把「内容性质」说成另一个取值（验同一判别器的第二维）；
    - `multi_line_intent` 把意图写成两行（验 `check_draft` 与 `hygiene` 判别器）；
    - `fence_intent` 把意图包进 Markdown 围栏（验 `check_draft` 与 `hygiene` 判别器）；
    - `empty_intent` 把意图写成空白（验 `check_draft` 与 `hygiene` 判别器）；
    - `label_intent` 在意图开头再写一遍「调用意图：」（验渲染期的段落名残留检查）；
    - `scope_intent` 把作用范围写进意图（验渲染期的「意图不许提作用范围」检查）；
    - `trace_raises` 反解时直接抛异常（验端点故障被记成跳过而不是拒绝）；
    - `trace_data` 反解时原样吐这份草稿（验缺字段 / 字段类型不对也记成跳过）。
    """

    name = "fake"
    capabilities = None

    def __init__(
        self,
        *,
        trace_lies: bool = False,
        trace_lies_payload: bool = False,
        multi_line_intent: bool = False,
        fence_intent: bool = False,
        empty_intent: bool = False,
        label_intent: bool = False,
        scope_intent: bool = False,
        trace_raises: Exception | None = None,
        trace_data: dict[str, JsonValue] | None = None,
    ) -> None:
        """输入：九个作弊开关。输出：假后端实例。"""
        self.trace_lies = trace_lies
        self.trace_lies_payload = trace_lies_payload
        self.multi_line_intent = multi_line_intent
        self.fence_intent = fence_intent
        self.empty_intent = empty_intent
        self.label_intent = label_intent
        self.scope_intent = scope_intent
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
        """输入：注入特征。输出：`ToolGateDraft` 形态的字典。

        预期行为：意图必须与「工具」和「内容性质」都对得上（`trace` 判别器就靠这个
        得出答案）；默认写法单行、无围栏、不提任何作用范围。
        """
        intent = INTENTS[tags["tool"]][tags["payload"]]
        if self.scope_intent:
            intent = f"{intent}，就在本工作区里做"
        if self.label_intent:
            intent = f"调用意图：{intent}"
        if self.fence_intent:
            intent = f"```\n{intent}\n```"
        if self.multi_line_intent:
            intent = f"{intent}\n（顺手记一下）"
        if self.empty_intent:
            intent = "   "
        return {"intent": intent}

    def _trace(self, tags: dict[str, str]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`TraceDraft` 形态的字典（默认完全答对）。"""
        if self.trace_data is not None:
            return dict(self.trace_data)
        tool, payload = tags["tool"], tags["payload"]
        if self.trace_lies:
            order = list(features.TOOLS)
            tool = order[(order.index(tool) + 1) % len(order)]
        if self.trace_lies_payload:
            order = list(features.PAYLOADS)
            payload = order[(order.index(payload) + 1) % len(order)]
        return {"tool": tool, "payload": payload}

    @staticmethod
    def _draft(data: dict[str, JsonValue]) -> Draft:
        """输入：草稿数据。输出：`Draft`。预期行为：记账字段填假值（共用工厂 `_fakes.draft`）。"""
        return draft(data)


fake_backend = _fakes.factory_fixture(FakeBackend)
