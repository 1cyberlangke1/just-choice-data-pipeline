"""工单分诊的判别器测试：三个判别器都要能放过好数据、拦住坏数据。"""

from __future__ import annotations

import pytest

from jc.backend.types import AllProvidersDown, TransportError
from jc.jsontypes import JsonValue

from workflows.routing import features, judges, locales, render
from workflows.routing.features import TagSet
from workflows.routing.spec import RoutingSpec

from tests.workflows.routing.conftest import body_for

TAGS = TagSet(issue="bug", urgency="deadline", tier="pro", compliance="none", area="storage")
GOOD = {"body": body_for(TAGS.as_dict())}


def sample(state: object) -> dict[str, JsonValue]:
    """输入：state。输出：最小可判样本。"""
    return {"state": state}


def spec() -> RoutingSpec:
    """输入：无。输出：与 `TAGS` 对齐的真值规格。"""
    return RoutingSpec(tags=TAGS, phrasing=0, split="train", seq=0)


def state_of(data: dict[str, str] | None = None) -> str:
    """输入：可选草稿覆盖。输出：拼好的 state。"""
    loc = locales.load("zh")
    return render.to_state({**GOOD, **(data or {})}, TAGS, loc)


def test_hygiene_passes_a_clean_state() -> None:
    """输入：干净 state。输出：无。预期行为：通过。"""
    verdict = judges.HygieneJudge("zh").judge(sample(state_of()))
    assert verdict.ok and not verdict.skipped


def test_hygiene_catches_leftovers_multiline_and_empty() -> None:
    """输入：带围栏 / 多行 / 空正文的 state。输出：无。预期行为：都判不通过。"""
    judge = judges.HygieneJudge("zh")
    assert not judge.judge(sample(state_of().replace("【工单正文】", "```\n【工单正文】"))).ok
    assert not judge.judge(sample(state_of({"body": "第一行\n第二行"}))).ok
    assert not judge.judge(sample(state_of({"body": "   "}))).ok


def test_hygiene_catches_html_line_breaks() -> None:
    """输入：正文里混进 `<br>`。输出：无。预期行为：判不通过。"""
    assert not judges.HygieneJudge("zh").judge(sample(state_of({"body": "上一句<br>下一句"}))).ok


def test_hygiene_catches_a_repeated_section_name() -> None:
    """输入：正文里又写了一遍段落名。输出：无。预期行为：判不通过。"""
    broken = state_of({"body": "工单正文：刚才那句话"})
    assert not judges.HygieneJudge("zh").judge(sample(broken)).ok


def test_hygiene_catches_attribute_words_in_the_body() -> None:
    """输入：正文里写了属性词。输出：无。预期行为：判不通过。"""
    assert not judges.HygieneJudge("zh").judge(sample(state_of({"body": "我们是企业版客户"}))).ok


def test_hygiene_lets_a_non_text_state_through() -> None:
    """输入：state 不是文本。输出：无。预期行为：不判、也不炸（当没查到东西）。"""
    verdict = judges.HygieneJudge("zh").judge(sample(123))
    assert verdict.ok


def test_frame_passes_the_injected_row_and_tables() -> None:
    """输入：与特征一致的 state。输出：无。预期行为：通过。"""
    verdict = judges.FrameJudge("zh").judge(sample(state_of()), spec())
    assert verdict.ok


def test_frame_catches_a_wrong_attr_row() -> None:
    """输入：属性行被改成别的取值。输出：无。预期行为：判不通过。"""
    loc = locales.load("zh")
    broken = state_of().replace(render.attr_lines(TAGS, loc)[0], "- storage | free | none")
    assert not judges.FrameJudge("zh").judge(sample(broken), spec()).ok


def test_frame_catches_a_missing_queue_table() -> None:
    """输入：派单表被整段删掉。输出：无。预期行为：判不通过。"""
    loc = locales.load("zh")
    head = state_of().split(locales.header(loc, loc["meta"]["queue_label"]))[0]
    assert not judges.FrameJudge("zh").judge(sample(head), spec()).ok


def test_frame_catches_a_tampered_priority_row() -> None:
    """输入：优先级表某一行的档位被改掉。输出：无。预期行为：判不通过。"""
    loc = locales.load("zh")
    first = render.priority_lines(loc)[0]
    broken = state_of().replace(first, first.replace("| 0 | 0 | 1", "| 3 | 3 | 3"))
    assert not judges.FrameJudge("zh").judge(sample(broken), spec()).ok


def test_frame_catches_a_missing_flag_table() -> None:
    """输入：附加规则被整段删掉。输出：无。预期行为：判不通过。"""
    loc = locales.load("zh")
    head = state_of().split(locales.header(loc, loc["meta"]["flags_label"]))[0]
    assert not judges.FrameJudge("zh").judge(sample(head), spec()).ok


def test_frame_skips_without_spec_or_text() -> None:
    """输入：没有真值规格 / state 不是文本。输出：无。预期行为：都记成跳过。"""
    assert judges.FrameJudge("zh").judge(sample(state_of())).skipped
    assert judges.FrameJudge("zh").judge(sample(123), spec()).skipped


def test_trace_passes_when_the_model_agrees(fake_backend: type) -> None:
    """输入：从不撒谎的假后端。输出：无。预期行为：通过。"""
    verdict = judges.TraceJudge(fake_backend(), "zh").judge(sample(state_of()), spec())
    assert verdict.ok and not verdict.skipped


def test_trace_catches_a_misread_issue(fake_backend: type) -> None:
    """输入：反解把「是什么问题」说反。输出：无。预期行为：判不通过。"""
    backend = fake_backend(trace_lies_issue=True)
    verdict = judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())
    assert not verdict.ok
    assert "issue" in verdict.reason


def test_trace_catches_a_misread_urgency(fake_backend: type) -> None:
    """输入：反解把「有多急」说反。输出：无。预期行为：判不通过。"""
    backend = fake_backend(trace_lies_urgency=True)
    verdict = judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())
    assert not verdict.ok
    assert "urgency" in verdict.reason


def test_trace_marks_endpoint_failure_as_skipped(fake_backend: type) -> None:
    """输入：反解时端点故障。输出：无。

    预期行为：记成**跳过**而不是不通过——端点抖动不是数据质量问题（PLAN 3.1.4）。
    """
    backend = fake_backend(trace_raises=TransportError("boom"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())
    assert verdict.ok and verdict.skipped


def test_trace_reraises_a_fatal_backend_error(fake_backend: type) -> None:
    """输入：全部 provider 都挂了。输出：无。

    预期行为：直接往上抛——那是整批该停下的唯一条件，吞掉它就会产出一整个「全被拒」的
    假数据集（PLAN 3.1.4）。
    """
    backend = fake_backend(trace_raises=AllProvidersDown("全挂了"))
    with pytest.raises(AllProvidersDown):
        judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())


def test_trace_marks_a_broken_draft_as_skipped(fake_backend: type) -> None:
    """输入：反解草稿缺字段。输出：无。预期行为：记成跳过。"""
    backend = fake_backend(trace_data={"issue": "bug"})
    verdict = judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())
    assert verdict.ok and verdict.skipped


def test_trace_skips_without_spec_or_backend(fake_backend: type) -> None:
    """输入：没有真值规格 / 没有后端。输出：无。预期行为：都记成跳过。"""
    assert judges.TraceJudge(fake_backend(), "zh").judge(sample(state_of())).skipped
    assert judges.TraceJudge(None, "zh").judge(sample(state_of()), spec()).skipped


def test_build_trace_prompt_asks_both_families_with_the_shared_options() -> None:
    """输入：一段 state。输出：无。

    预期行为：反解提示词把两族原子题连同选项原样再问一遍——选项与数据集里的一致，
    否则「答对了」与「标签对了」不是一回事。
    """
    loc = locales.load("zh")
    prompt = judges.build_trace_prompt(state_of(), loc)
    assert state_of() in prompt
    assert judges.TRACE_FAMILIES == ("issue", "urgency")
    for family in judges.TRACE_FAMILIES:
        assert str(loc["fam"][family]["q"][0]) in prompt
        for name in loc["fam"][family]["criteria"]:
            assert str(name) in prompt


def test_trace_draft_literals_match_the_feature_tables() -> None:
    """输入：反解草稿的取值域。输出：无。

    预期行为：`TraceDraft` 的两维取值域与 `features.ISSUES` / `features.URGENCIES` 一致——
    对不上就会出现「模型答对了但字段类型不符」这种静默失败。
    """
    assert set(judges.TraceDraft.model_fields["issue"].annotation.__args__) == set(
        features.ISSUES
    )
    assert set(judges.TraceDraft.model_fields["urgency"].annotation.__args__) == set(
        features.URGENCIES
    )


def test_option_names_reads_the_product_order() -> None:
    """输入：一道 choice 题。输出：无。预期行为：按产物里的顺序取选项名。"""
    assert judges.option_names({"criteria": {"b": "x", "a": "y"}}) == ["b", "a"]
    assert judges.option_names({}) == []
