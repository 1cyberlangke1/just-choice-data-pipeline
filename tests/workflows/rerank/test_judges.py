"""判别器测试：能抓到什么、端点故障怎么记。"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft
from jc.jsontypes import JsonValue
from workflows.rerank import features, judges, locales, render
from workflows.rerank.spec import RerankSpec

from tests.workflows.rerank.conftest import FakeBackend, draft_data

CASE: Final[features.RerankCase] = features.RerankCase(terms=3, relevant=3, best="first")


def spec_for(case: features.RerankCase, phrasing: int = 0) -> RerankSpec:
    """输入：case、问法编号。输出：真值规格。"""
    return RerankSpec(
        case=case,
        phrasing=phrasing,
        split="train",
        seq=1,
        plan=features.plan_for(case, phrasing),
    )


def state_of(spec: RerankSpec, data: dict[str, JsonValue], lang: str = "zh") -> dict[str, JsonValue]:
    """输入：规格、草稿数据、语言。输出：只带 state 的样本。预期行为：不校验草稿，专供反例。"""
    # 用 `model_construct` 绕开 pydantic 的字段校验：这里要造的正是**不合法**的草稿（少一条候选
    # 之类），走正常构造会被 pydantic 先拦下来，测试就验不到判别器那一层了。
    draft = render.RerankDraft.model_construct(
        query=str(data["query"]),
        terms=[str(item) for item in data["terms"]],
        candidates=[str(item) for item in data["candidates"]],
    )
    return {"state": render.to_state(draft, spec.plan, locales.load(lang))}


def faithful(case: features.RerankCase, lang: str = "zh") -> tuple[RerankSpec, dict[str, JsonValue]]:
    """输入：case、语言。输出：(规格, 样本)。预期行为：草稿完全兑现计划。"""
    spec = spec_for(case)
    data = draft_data(spec.plan)
    render.check_parts(
        str(data["query"]),
        list(data["terms"]),
        list(data["candidates"]),
        spec.plan,
        locales.load(lang),
    )
    return spec, state_of(spec, data, lang)


def test_hygiene_accepts_a_faithful_batch() -> None:
    """输入：兑现计划的样本。输出：无。预期行为：通过。"""
    spec, sample = faithful(CASE)
    verdict = judges.HygieneJudge("zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert "兑现" in verdict.reason


@pytest.mark.parametrize("switch", ["unrealized", "extra_hit", "duplicate", "drop_candidate"])
def test_hygiene_rejects_broken_batches(switch: str) -> None:
    """输入：四种写歪的候选。输出：无。预期行为：一律不通过。"""
    spec = spec_for(CASE)
    data = draft_data(spec.plan, **{switch: True})
    verdict = judges.HygieneJudge("zh").judge(state_of(spec, data), spec)
    assert not verdict.ok and not verdict.skipped


def test_hygiene_rejects_a_state_without_a_query_heading() -> None:
    """输入：一段没有查询抬头的 state。输出：无。预期行为：不通过，理由是认不出查询。"""
    verdict = judges.HygieneJudge("zh").judge({"state": "这里什么抬头都没有"}, spec_for(CASE))
    assert not verdict.ok
    assert "查询" in verdict.reason


def test_frame_accepts_a_faithful_batch() -> None:
    """输入：兑现计划的样本。输出：无。预期行为：通过。"""
    spec, sample = faithful(CASE)
    verdict = judges.FrameJudge("zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert "一致" in verdict.reason


def test_frame_rejects_a_tampered_rule_block() -> None:
    """输入：排序口径被改过的 state。输出：无。预期行为：不通过，点出口径对不上。"""
    spec, sample = faithful(CASE)
    sample["state"] = str(sample["state"]).replace("得分高的排在前面", "得分低的排在前面")
    verdict = judges.FrameJudge("zh").judge(sample, spec)
    assert not verdict.ok
    assert "排序口径" in verdict.reason


def test_frame_rejects_a_wrong_term_count() -> None:
    """输入：关键词个数与注入特征不符的 state。输出：无。预期行为：不通过。"""
    spec, sample = faithful(CASE)
    state = str(sample["state"])
    sample["state"] = state.replace("【关键词】", "【关键词】extra、", 1)
    verdict = judges.FrameJudge("zh").judge(sample, spec)
    assert not verdict.ok
    assert "关键词个数" in verdict.reason


def test_trace_passes_when_the_count_matches() -> None:
    """输入：反解说真话的假后端。输出：无。预期行为：通过，且理由里带条数。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(), "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert str(spec.plan.relevant) in verdict.reason


def test_trace_rejects_a_lying_backend() -> None:
    """输入：反解说反话的假后端。输出：无。预期行为：不通过，理由里两种数都列出来。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(trace_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "计划" in verdict.reason and "反解" in verdict.reason
