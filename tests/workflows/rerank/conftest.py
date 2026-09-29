"""rerank 的假后端与草稿构造器：脚本化出草稿，不联网。

假后端按 `task.spec["overlaps"]` 现造查询与候选——**默认完全兑现计划**，另外几个开关各自
破坏一件事，用来验「渲染期与判别器真的会拦下来」。
"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from workflows.rerank import features

#: 关键词池：两两不互为子串、都够长（`render.MIN_TERM_CHARS`）。
TERMS: Final[tuple[str, ...]] = ("alpha", "bravo", "charlie", "delta")

#: 干净候选的句子池（一条关键词都不含）。
CLEAN: Final[tuple[str, ...]] = (
    "the cafeteria menu changed again this week",
    "a raccoon was spotted near the loading dock",
    "the quarterly picnic moved to the west lawn",
    "someone left a kettle in the third floor kitchen",
    "the parking gate opens at seven in the morning",
)

#: 只沾一个关键词的候选句子池。
PARTIAL: Final[tuple[str, ...]] = (
    "we saw {term} behave oddly last night",
    "the dashboard chart for {term} looks flat",
    "ops mentioned {term} in the morning handover",
    "the runbook still has a note about {term}",
)

#: 「该沾词却写成干净」时用的替换句——**必须不在 `CLEAN` 里**：撞上就会先以「候选重复」被
#: 拒掉，测试想验的那条判据反而验不到。
UNRELATED: Final[str] = "an unrelated note about the fire drill schedule"

#: 「该干净却沾上词」时用的替换句——同样必须与已有的句子都不一样。
EXTRA: Final[str] = "a stray mention of {term} in passing"


def draft_data(
    plan: features.RerankPlan,
    *,
    unrealized: bool = False,
    extra_hit: bool = False,
    duplicate: bool = False,
    drop_candidate: bool = False,
    bad_terms: bool = False,
    term_missing_from_query: bool = False,
    fence_line: bool = False,
) -> dict[str, JsonValue]:
    """输入：名次计划、七个「故意写歪」的开关。输出：草稿字典。

    预期行为：默认**完全兑现计划**；开关各自破坏一件事——
    `unrealized` 让一条该沾词的不沾、`extra_hit` 让一条干净候选沾上、`duplicate` 让两条候选
    一模一样、`drop_candidate` 少给一条、`bad_terms` 把关键词换成一对互为子串的短词、
    `term_missing_from_query` 让查询漏掉一个关键词、`fence_line` 让一条候选带围栏。
    `unrealized` 在「一条都不该沾」的计划上没有可破坏的对象，这时退化成 `extra_hit`
    （不然 `relevant=0` 的那几条会照常通过，参数化的反例测试就会漏掉它们）。
    """
    terms = list(TERMS[: plan.terms])
    if bad_terms:
        # 一个短于任何语言下限的单字 + 一串互为子串的词：`check_parts` 必须拦下。
        terms = ["a", "ab", "abc", "abcd"][: plan.terms]
    candidates: list[str] = []
    clean_seen = 0
    partial_seen = 0
    for value in plan.overlaps:
        if value == plan.terms:
            candidates.append("the fix is to tune " + " and ".join(terms) + " carefully")
        elif value > 0:
            candidates.append(PARTIAL[partial_seen % len(PARTIAL)].format(term=terms[0]))
            partial_seen += 1
        else:
            candidates.append(CLEAN[clean_seen % len(CLEAN)])
            clean_seen += 1
    if unrealized:
        target = next((index for index, value in enumerate(plan.overlaps) if value > 0), None)
        if target is None:
            extra_hit = True
        else:
            candidates[target] = UNRELATED
    if extra_hit:
        spare = next((index for index, value in enumerate(plan.overlaps) if value == 0), None)
        if spare is not None:
            candidates[spare] = EXTRA.format(term=terms[0])
    if duplicate and len(candidates) >= 2:
        candidates[1] = candidates[0]
    if drop_candidate:
        candidates = candidates[:-1]
    if fence_line and candidates:
        candidates[0] = "```" + candidates[0]
    query = "how do i fix " + " ".join(terms) + " on the server"
    if term_missing_from_query and terms:
        query = "how do i fix " + " ".join(terms[1:]) + " on the server"
    return {"query": query, "terms": terms, "candidates": candidates}


class FakeBackend:
    """脚本化假后端：草稿由 `task.spec` 推出，不联网。

    字段（都是「故意说反话」的开关，默认全关）：

    - `trace_lies` 反解时把相关条数说成另一个数（验 `trace` 判别器）；
    - `trace_raises` 反解时抛异常（验「端点故障只算没判成」）；
    - 其余见 `draft_data`。
    """

    def __init__(
        self,
        *,
        trace_lies: bool = False,
        trace_raises: Exception | None = None,
        unrealized: bool = False,
        extra_hit: bool = False,
        duplicate: bool = False,
        drop_candidate: bool = False,
        bad_terms: bool = False,
        term_missing_from_query: bool = False,
        fence_line: bool = False,
    ) -> None:
        """输入：各开关。输出：实例。"""
        self.trace_lies = trace_lies
        self.trace_raises = trace_raises
        self.unrealized = unrealized
        self.extra_hit = extra_hit
        self.duplicate = duplicate
        self.drop_candidate = drop_candidate
        self.bad_terms = bad_terms
        self.term_missing_from_query = term_missing_from_query
        self.fence_line = fence_line
        self.tasks: list[str] = []

    def generate(self, task: object) -> Draft:
        """输入：`GenerationTask`。输出：草稿。预期行为：按 `task.spec` 现造，不发网络。"""
        spec = getattr(task, "spec")
        task_id = str(getattr(task, "task_id"))
        self.tasks.append(task_id)
        plan = features.RerankPlan(
            terms=int(spec["case"]["terms"]),
            best_pos=None,
            overlaps=tuple(int(value) for value in spec["overlaps"]),
        )
        if task_id.endswith("-trace"):
            if self.trace_raises is not None:
                raise self.trace_raises
            relevant = plan.relevant
            if self.trace_lies:
                relevant = (relevant + 1) % (features.CANDIDATES + 1)
            return _draft({"relevant": relevant}, f"relevant={relevant}")
        data = draft_data(
            plan,
            unrealized=self.unrealized,
            extra_hit=self.extra_hit,
            duplicate=self.duplicate,
            drop_candidate=self.drop_candidate,
            bad_terms=self.bad_terms,
            term_missing_from_query=self.term_missing_from_query,
            fence_line=self.fence_line,
        )
        return _draft(data, "\n".join(str(item) for item in data["candidates"]))

    def generate_many(self, tasks: list[object]) -> list[Draft]:
        """输入：任务列表。输出：草稿列表。预期行为：逐个走 `generate`。"""
        return [self.generate(task) for task in tasks]


def _draft(data: dict[str, JsonValue], raw: str) -> Draft:
    """输入：解析后的数据、原始文本。输出：`Draft`。预期行为：记账字段填假值，不进产物。"""
    return Draft(
        data=data,
        raw_text=raw,
        backend="fake",
        provider="fake",
        model="fake",
        mode="fake",
    )


@pytest.fixture
def backend() -> FakeBackend:
    """输入：无。输出：一个完全兑现计划的假后端。"""
    return FakeBackend()


@pytest.fixture
def fake_backend() -> type[FakeBackend]:
    """工厂夹具：`fake_backend(trace_lies=True)` 造一个假后端。"""
    return FakeBackend
