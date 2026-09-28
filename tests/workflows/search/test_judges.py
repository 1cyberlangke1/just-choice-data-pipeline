"""检索结果判读的判别器测试：三个判别器都要能放过好数据、拦住坏数据。"""

from __future__ import annotations

from jc.backend.types import TransportError
from jc.jsontypes import JsonValue

from workflows.search import judges, locales, render
from workflows.search.features import TagSet
from workflows.search.spec import SearchSpec

from tests.workflows.search.conftest import TEXTS

TAGS = TagSet(
    query_kind="informational",
    topicality="on_topic",
    spam="clean",
    authority="community",
    freshness="fresh",
)
GOOD = dict(TEXTS[("informational", "on_topic", "clean")])


def sample(state: str) -> dict[str, JsonValue]:
    """输入：state。输出：最小可判样本。"""
    return {"state": state}


def spec() -> SearchSpec:
    """输入：无。输出：与 `TAGS` 对齐的真值规格。"""
    return SearchSpec(tags=TAGS, phrasing=0, split="train", seq=0)


def state_of(data: dict[str, str] | None = None) -> str:
    """输入：可选草稿覆盖。输出：拼好的 state。"""
    loc = locales.load("zh")
    return render.to_state({**GOOD, **(data or {})}, TAGS, loc)


def test_hygiene_passes_a_clean_state() -> None:
    """输入：干净 state。输出：无。预期行为：通过。"""
    verdict = judges.HygieneJudge("zh").judge(sample(state_of()))
    assert verdict.ok and not verdict.skipped


def test_hygiene_catches_leftovers_and_multiline() -> None:
    """输入：带围栏 / 多行 / 空段的 state。输出：无。预期行为：都判不通过。"""
    judge = judges.HygieneJudge("zh")
    assert not judge.judge(sample(state_of().replace("【标题】", "```\n【标题】"))).ok
    assert not judge.judge(sample(state_of({"title": "第一行\n第二行"}))).ok
    assert not judge.judge(sample(state_of({"title": "   "}))).ok


def test_frame_passes_the_injected_row() -> None:
    """输入：与特征一致的 state。输出：无。预期行为：通过。"""
    verdict = judges.FrameJudge("zh").judge(sample(state_of()), spec())
    assert verdict.ok


def test_frame_catches_a_wrong_source_row() -> None:
    """输入：来源行被改成别的取值。输出：无。预期行为：判不通过。"""
    broken = state_of().replace("- community | fresh", "- official | stale")
    assert not judges.FrameJudge("zh").judge(sample(broken), spec()).ok


def test_frame_catches_a_missing_rule_table() -> None:
    """输入：规则表被整段删掉。输出：无。预期行为：判不通过。"""
    loc = locales.load("zh")
    head = state_of().split(locales.header(loc, loc["meta"]["rules_label"]))[0]
    assert not judges.FrameJudge("zh").judge(sample(head), spec()).ok


def test_frame_skips_without_spec() -> None:
    """输入：没有真值规格。输出：无。预期行为：记成跳过，不算不通过。"""
    verdict = judges.FrameJudge("zh").judge(sample(state_of()))
    assert verdict.ok and verdict.skipped


def test_trace_passes_when_the_model_agrees(fake_backend: type) -> None:
    """输入：从不撒谎的假后端。输出：无。预期行为：通过。"""
    verdict = judges.TraceJudge(fake_backend(), "zh").judge(sample(state_of()), spec())
    assert verdict.ok and not verdict.skipped


def test_trace_catches_a_misread_query_kind(fake_backend: type) -> None:
    """输入：反解说反话的假后端。输出：无。预期行为：判不通过。"""
    verdict = judges.TraceJudge(fake_backend(trace_lies_kind=True), "zh").judge(
        sample(state_of()), spec()
    )
    assert not verdict.ok


def test_trace_catches_a_misread_spam_flag(fake_backend: type) -> None:
    """输入：反解把「是不是垃圾」说反。输出：无。预期行为：判不通过。"""
    verdict = judges.TraceJudge(fake_backend(trace_lies_spam=True), "zh").judge(
        sample(state_of()), spec()
    )
    assert not verdict.ok


def test_trace_marks_endpoint_failure_as_skipped(fake_backend: type) -> None:
    """输入：反解时端点故障。输出：无。

    预期行为：记成**跳过**而不是不通过——端点抖动不是数据质量问题（PLAN 3.1.4）。
    """
    backend = fake_backend(trace_raises=TransportError("boom"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())
    assert verdict.ok and verdict.skipped


def test_trace_marks_a_broken_draft_as_skipped(fake_backend: type) -> None:
    """输入：反解草稿缺字段。输出：无。预期行为：记成跳过。"""
    backend = fake_backend(trace_data={"query_kind": "informational"})
    verdict = judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())
    assert verdict.ok and verdict.skipped


def test_option_names_reads_the_product_order() -> None:
    """输入：一道 choice 题。输出：无。预期行为：按产物里的顺序取选项名。"""
    assert judges.option_names({"criteria": {"b": "x", "a": "y"}}) == ["b", "a"]
    assert judges.option_names({}) == []
