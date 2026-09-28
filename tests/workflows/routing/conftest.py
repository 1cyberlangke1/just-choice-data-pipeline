"""工单分诊测试的公共夹具：一个**不发网络请求**的假后端。

假后端按 `task.spec["tags"]` 造草稿，所以它天然「知道」注入特征：
- 渲染任务 → 按（是什么问题 × 有多急）挑一段与之相称的正文；
- 反解任务 → 直接把注入的两维当成反解结果（也就是「模型完全答对」）。

要验「判别器能抓错」时，用 `FakeBackend(trace_lies_issue=True)` / `FakeBackend(fence_body=True)`
之类的开关故意说反话；要验「端点故障不算数据问题」时，用
`FakeBackend(trace_raises=TransportError(...))`。

夹具里那些正文**刻意不提客户等级 / 合规标记 / 产品域**：那三格归「工单属性」那一行管，
正文里再写一遍就会跟它打架——这也是 `render.check_draft` 会拦的写法。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from workflows.routing import features

#: （是什么问题 × 有多急）→ 一段相称的正文。三对被禁的组合不在这里（`build_cases` 不会产）。
BASE: Final[dict[tuple[str, str], str]] = {
    ("outage", "blocked"): "我们这条线从今早开始就传不上数据了，整条业务停在这里。",
    ("outage", "deadline"): "我们这条线从今早开始就传不上数据了，今天下班前必须恢复。",
    ("data_loss", "blocked"): "我们发现少了一批记录，现在这条流程完全走不下去。",
    ("data_loss", "deadline"): "我们发现少了一批记录，今天下班前得给个说法。",
    ("bug", "can_wait"): "它导出的和文档写的不一样，不过不急，等有空看看。",
    ("bug", "blocked"): "它导出的和文档写的不一样，整条流程卡住了。",
    ("bug", "deadline"): "它导出的和文档写的不一样，今天下班前要出结论。",
    ("howto", "can_wait"): "想确认一下这个接口的正确用法，不急。",
    ("howto", "deadline"): "想确认一下这个接口的正确用法，今天下班前要用到。",
    ("feature", "can_wait"): "希望你们能支持一种新的导出方式，不急。",
    ("feature", "blocked"): "希望你们能支持一种新的导出方式，没有它我们交付不了。",
    ("feature", "deadline"): "希望你们能支持一种新的导出方式，今天下班前要定。",
}


def body_for(tags: dict[str, str]) -> str:
    """输入：注入特征。输出：与之相称的正文。

    预期行为：夹具里没有这一格时直接抛 KeyError——静默兜底会让「文本承载两维」这条
    前提在测试里悄悄失效。
    """
    return BASE[(tags["issue"], tags["urgency"])]


class FakeBackend:
    """脚本化假后端：草稿由 `task.spec["tags"]` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：
    - `trace_lies_issue` 反解时把「是什么问题」说成另一个取值（验 `trace` 判别器）；
    - `trace_lies_urgency` 反解时把「有多急」说成另一个取值（验同一判别器的第二维）；
    - `multi_line_body` 把正文写成两行（验 `check_draft` 与 `hygiene` 判别器）；
    - `fence_body` 把正文包进 Markdown 围栏（验 `check_draft` 与 `hygiene` 判别器）；
    - `empty_body` 把正文写成空白（验 `check_draft` 与 `hygiene` 判别器）；
    - `label_body` 在正文开头再写一遍「工单正文：」（验渲染期的段落名残留检查）；
    - `attrs_body` 把属性词写进正文（验渲染期的「正文不许提属性」检查）；
    - `trace_raises` 反解时直接抛异常（验端点故障被记成跳过而不是拒绝）；
    - `trace_data` 反解时原样吐这份草稿（验缺字段 / 字段类型不对也记成跳过）。
    """

    name = "fake"
    capabilities = None

    def __init__(
        self,
        *,
        trace_lies_issue: bool = False,
        trace_lies_urgency: bool = False,
        multi_line_body: bool = False,
        fence_body: bool = False,
        empty_body: bool = False,
        label_body: bool = False,
        attrs_body: bool = False,
        trace_raises: Exception | None = None,
        trace_data: dict[str, JsonValue] | None = None,
    ) -> None:
        """输入：九个作弊开关。输出：假后端实例。"""
        self.trace_lies_issue = trace_lies_issue
        self.trace_lies_urgency = trace_lies_urgency
        self.multi_line_body = multi_line_body
        self.fence_body = fence_body
        self.empty_body = empty_body
        self.label_body = label_body
        self.attrs_body = attrs_body
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
            if self.trace_data is not None:
                return self._draft(dict(self.trace_data))
            return self._draft(self._trace(tags))
        return self._draft(self._render(tags))

    def generate_many(self, tasks: list[object]) -> list[Draft]:
        """输入：任务列表。输出：与入参顺序一致的草稿列表。"""
        return [self.generate(task) for task in tasks]

    def _render(self, tags: dict[str, str]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`RoutingDraft` 形态的字典。

        预期行为：正文必须与「是什么问题」和「有多急」都对得上（`trace` 判别器就靠这个
        得出答案）；默认写法单行、无围栏、不提属性。
        """
        body = body_for(tags)
        if self.attrs_body:
            # 只收高精度禁用语，所以这里必须写**清单里那条**说法（`企业版`），
            # 写「我们是付费用户」这种不在清单里的说法，渲染期本来就不会拦。
            body = f"{body}（顺便说一句，我们是企业版）"
        if self.label_body:
            body = f"工单正文：{body}"
        if self.fence_body:
            body = f"```{body}```"
        if self.multi_line_body:
            body = f"{body}\n（后面还有一句）"
        if self.empty_body:
            body = "   "
        return {"body": body}

    def _trace(self, tags: dict[str, str]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`TraceDraft` 形态的字典（默认完全答对）。"""
        out: dict[str, JsonValue] = {"issue": tags["issue"], "urgency": tags["urgency"]}
        if self.trace_lies_issue:
            out["issue"] = next(value for value in features.ISSUES if value != tags["issue"])
        if self.trace_lies_urgency:
            out["urgency"] = next(
                value for value in features.URGENCIES if value != tags["urgency"]
            )
        return out

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
    """工厂夹具：`fake_backend(trace_lies_issue=True)` 造一个假后端。"""
    return FakeBackend
