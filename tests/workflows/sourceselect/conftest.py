"""来源选择测试的公共夹具：一个**不发网络请求**的假后端。

假后端按 `task.spec["tags"]` 造草稿，所以它天然「知道」注入特征：
- 渲染任务 → 按首要要求挑一句与之相称的查询；
- 反解任务 → 直接把注入的首要要求当成反解结果（也就是「模型完全答对」）。

要验「判别器能抓错」时，用 `FakeBackend(trace_lies_need=True)` / `FakeBackend(fence_query=True)`
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
from workflows.sourceselect import features

#: 每个首要要求一句查询底稿。刻意不写「候选来源 / 来源表 / 序号」也不含竖线——
#: 那几条是渲染期禁用语，撞上就会白拒。
QUERIES: Final[dict[str, str]] = {
    "authority": "帮我找这份协议的原始标准文本",
    "freshness": "这个库最新版本改了什么",
    "coverage": "把这件事讲全的完整清单在哪",
}


def query_for(tags: dict[str, JsonValue]) -> str:
    """输入：注入特征。输出：与首要要求相称的那句查询。

    预期行为：夹具里没有这一格时直接抛 KeyError——静默兜底会让「文本承载 need」这条
    前提在测试里悄悄失效。
    """
    return QUERIES[str(tags["need"])]


class FakeBackend(_fakes.ScriptedBackend):
    """脚本化假后端：草稿由 `task.spec["tags"]` 推出，不联网。

    字段（都是「故意做坏事」的开关，默认全关）：
    - `trace_lies_need` 反解时把首要要求说成另一个取值（验 `trace` 判别器）；
    - `multi_line_query` 把查询写成两行（验 `check_draft` 与 `hygiene` 判别器）；
    - `fence_query` 把查询包进 Markdown 围栏（验同一对）；
    - `empty_query` 把查询写成空白（验同一对）；
    - `label_query` 在查询开头再写一遍「查询：」（验渲染期的段落名残留检查）；
    - `table_query` 在查询里谈候选表（验渲染期的禁用语检查）；
    - `trace_raises` 反解时直接抛异常（验端点故障被记成跳过而不是拒绝）；
    - `trace_data` 反解时原样吐这份草稿（验缺字段 / 字段类型不对也记成跳过）。
    """

    name = "fake"
    capabilities = None

    def __init__(
        self,
        *,
        trace_lies_need: bool = False,
        multi_line_query: bool = False,
        fence_query: bool = False,
        empty_query: bool = False,
        label_query: bool = False,
        table_query: bool = False,
        trace_raises: Exception | None = None,
        trace_data: dict[str, JsonValue] | None = None,
    ) -> None:
        """输入：八个作弊开关。输出：假后端实例。"""
        self.trace_lies_need = trace_lies_need
        self.multi_line_query = multi_line_query
        self.fence_query = fence_query
        self.empty_query = empty_query
        self.label_query = label_query
        self.table_query = table_query
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

    def _render(self, tags: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`SourceDraft` 形态的字典。"""
        query = query_for(tags)
        if self.multi_line_query:
            query = f"{query}\n第二行"
        if self.fence_query:
            query = f"```{query}```"
        if self.empty_query:
            query = "   "
        if self.label_query:
            query = f"查询：{query}"
        if self.table_query:
            query = f"请按候选来源表的序号挑一行"
        return {"query": query}

    def _trace(self, tags: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """输入：注入特征。输出：`TraceDraft` 形态的字典（默认完全答对）。"""
        need = str(tags["need"])
        if self.trace_lies_need:
            need = next(value for value in features.NEEDS if value != need)
        return {"need": need}

    @staticmethod
    def _draft(data: dict[str, JsonValue]) -> Draft:
        """输入：草稿数据。输出：`Draft`。预期行为：记账字段填假值（共用工厂 `_fakes.draft`）。"""
        return draft(data)


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(trace_lies_need=True)` 造一个假后端。"""
    return FakeBackend
