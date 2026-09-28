"""完成声明核验测试的公共夹具：一个**不发网络请求**的假后端。

假后端按 `task.spec["tags"]` 造草稿，所以它天然「知道」注入特征：
- 渲染任务 → 按（声称做完的事 × 口气）挑一句与之相称的收尾话；
- 反解任务 → 直接把注入的两维当成反解结果（也就是「模型完全答对」）。

要验「判别器能抓错」时，用 `FakeBackend(trace_lies=True)` / `FakeBackend(fence_claim=True)`
之类的开关故意说反话；要验「端点故障不算数据问题」时，用
`FakeBackend(trace_raises=TransportError(...))`。

夹具里那些收尾话**刻意不提证据**（检查结果 / 覆盖率 / 是否提交）：证据归「自报事实」
那一行管，声明里再写一遍就会跟它打架——这也是 `render.check_draft` 会拦的写法。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from workflows.verify import features

#: 每个（声称做完的事 × 口气）一句与之相称、且不含任何证据话术的收尾话。
#: 口气那一半必须真的跟着变——`judges.TraceJudge` 要反解的就是这两维。
CLAIMS: Final[dict[str, dict[str, str]]] = {
    "tests_added": {
        "plain": "给登录模块补了三个用例",
        "hedged": "给登录模块大概补了三个用例",
        "absolute": "给登录模块补了三个用例，肯定够用了，绝对不会有漏网的",
    },
    "bug_fixed": {
        "plain": "把订单列表偶发的空指针修掉了",
        "hedged": "订单列表那个空指针看上去修掉了",
        "absolute": "订单列表那个空指针肯定修掉了，绝不可能再复现",
    },
    "perf_improved": {
        "plain": "把列表渲染的热路径重写了一遍，快了不少",
        "hedged": "列表渲染的热路径重写了一遍，应该快了一些",
        "absolute": "列表渲染的热路径重写了一遍，肯定快了一倍不止",
    },
    "refactor_done": {
        "plain": "把订单模块拆成了三个小文件，对外行为没变",
        "hedged": "订单模块应该算是拆干净了，对外行为大概没变",
        "absolute": "订单模块彻底拆干净了，对外行为绝对没变",
    },
    "api_added": {
        "plain": "给订单服务加了一个批量查询的入口",
        "hedged": "给订单服务大概加了一个批量查询的入口",
        "absolute": "给订单服务加好了批量查询的入口，肯定能直接用",
    },
}


class FakeBackend:
    """脚本化假后端：草稿由 `task.spec["tags"]` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：
    - `trace_lies` 反解时把「声称做完的事」说成另一个取值（验 `trace` 判别器）；
    - `trace_lies_tone` 反解时把「口气」说成另一个取值（验同一判别器的第二维）；
    - `multi_line_claim` 把声明写成两行（验 `check_draft` 与 `hygiene` 判别器）；
    - `fence_claim` 把声明包进 Markdown 围栏（验 `check_draft` 与 `hygiene` 判别器）；
    - `empty_claim` 把声明写成空白（验 `check_draft` 与 `hygiene` 判别器）；
    - `label_claim` 在声明开头再写一遍「声明：」（验渲染期的段落名残留检查）；
    - `evidence_claim` 把证据写进声明（验渲染期的「声明不许提证据」检查）；
    - `trace_raises` 反解时直接抛异常（验端点故障被记成跳过而不是拒绝）；
    - `trace_data` 反解时原样吐这份草稿（验缺字段 / 字段类型不对也记成跳过）。
    """

    name = "fake"
    capabilities = None

    def __init__(
        self,
        *,
        trace_lies: bool = False,
        trace_lies_tone: bool = False,
        multi_line_claim: bool = False,
        fence_claim: bool = False,
        empty_claim: bool = False,
        label_claim: bool = False,
        evidence_claim: bool = False,
        trace_raises: Exception | None = None,
        trace_data: dict[str, JsonValue] | None = None,
    ) -> None:
        """输入：九个作弊开关。输出：假后端实例。"""
        self.trace_lies = trace_lies
        self.trace_lies_tone = trace_lies_tone
        self.multi_line_claim = multi_line_claim
        self.fence_claim = fence_claim
        self.empty_claim = empty_claim
        self.label_claim = label_claim
        self.evidence_claim = evidence_claim
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

    def generate_many(self, tasks: list[object]) -> list[Draft]:
        """输入：任务列表。输出：与入参顺序一致的草稿列表。"""
        return [self.generate(task) for task in tasks]

    def _render(self, tags: dict[str, str]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`VerifyDraft` 形态的字典。

        预期行为：声明必须与「声称做完的事」和「口气」都对得上（`trace` 判别器就靠这个
        得出答案）；默认写法单行、无围栏、不提任何证据。
        """
        claim = CLAIMS[tags["claim_kind"]][tags["tone"]]
        if self.evidence_claim:
            claim = f"{claim}，测试全过了"
        if self.label_claim:
            claim = f"声明：{claim}"
        if self.fence_claim:
            claim = f"```\n{claim}\n```"
        if self.multi_line_claim:
            claim = f"{claim}\n（顺手把注释也补了）"
        if self.empty_claim:
            claim = "   "
        return {"claim": claim}

    def _trace(self, tags: dict[str, str]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`TraceDraft` 形态的字典（默认完全答对）。"""
        if self.trace_data is not None:
            return dict(self.trace_data)
        kind, tone = tags["claim_kind"], tags["tone"]
        if self.trace_lies:
            order = list(features.CLAIM_KINDS)
            kind = order[(order.index(kind) + 1) % len(order)]
        if self.trace_lies_tone:
            order = list(features.TONES)
            tone = order[(order.index(tone) + 1) % len(order)]
        return {"kind": kind, "tone": tone}

    @staticmethod
    def _draft(data: dict[str, JsonValue]) -> Draft:
        """输入：草稿数据。输出：`Draft`。预期行为：填满必填的记账字段。"""
        return Draft(
            data=data,
            raw_text="<fake>",
            backend="fake",
            provider="fake",
            model="fake-model",
            mode="json_schema",
            usage={},
        )


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(trace_lies=True)` 造一个假后端。"""
    return FakeBackend
