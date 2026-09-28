"""会话上下文压缩的判别器测试：三个判别器都要能放过好数据、拦住坏数据。"""

from __future__ import annotations

import pytest

from jc.backend.types import AllProvidersDown, TransportError
from jc.jsontypes import JsonValue

from workflows.compact import judges, locales, render
from workflows.compact.features import TagSet
from workflows.compact.spec import CompactSpec

from tests.workflows.compact.conftest import BASE

TAGS = TagSet(kind="user", fact="chatter", age="old", pins="unpinned", budget="tight")
GOOD = {"content": BASE["user"]}


def sample(state: object) -> dict[str, JsonValue]:
    """输入：state。输出：最小可判样本。"""
    return {"state": state}


def spec() -> CompactSpec:
    """输入：无。输出：与 `TAGS` 对齐的真值规格。"""
    return CompactSpec(tags=TAGS, phrasing=0, split="train", seq=0)


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
    assert not judge.judge(sample(state_of().replace("【本块内容】", "```\n【本块内容】"))).ok
    assert not judge.judge(sample(state_of({"content": "第一行\n第二行"}))).ok
    assert not judge.judge(sample(state_of({"content": "   "}))).ok


def test_hygiene_catches_html_line_breaks() -> None:
    """输入：正文里混进 `<br>`。输出：无。预期行为：判不通过。"""
    assert not judges.HygieneJudge("zh").judge(sample(state_of({"content": "上一句<br>下一句"}))).ok


def test_hygiene_catches_a_repeated_section_name() -> None:
    """输入：正文里又写了一遍段落名。输出：无。预期行为：判不通过。"""
    broken = state_of({"content": "本块内容：刚才那句话"})
    assert not judges.HygieneJudge("zh").judge(sample(broken)).ok


def test_hygiene_lets_a_non_text_state_through() -> None:
    """输入：state 不是文本。输出：无。预期行为：不判、也不炸（当没查到东西）。"""
    verdict = judges.HygieneJudge("zh").judge(sample(123))
    assert verdict.ok


def test_frame_passes_the_injected_row() -> None:
    """输入：与特征一致的 state。输出：无。预期行为：通过。"""
    verdict = judges.FrameJudge("zh").judge(sample(state_of()), spec())
    assert verdict.ok


def test_frame_catches_a_wrong_where_row() -> None:
    """输入：位置行被改成别的取值。输出：无。预期行为：判不通过。"""
    broken = state_of().replace(render.where_line(TAGS, locales.load("zh")), "- recent | pinned | ample")
    assert not judges.FrameJudge("zh").judge(sample(broken), spec()).ok


def test_frame_catches_a_missing_rule_table() -> None:
    """输入：处置表被整段删掉。输出：无。预期行为：判不通过。"""
    loc = locales.load("zh")
    head = state_of().split(locales.header(loc, loc["meta"]["rules_label"]))[0]
    assert not judges.FrameJudge("zh").judge(sample(head), spec()).ok


def test_frame_catches_a_tampered_verdict() -> None:
    """输入：处置表某一行的处置被改掉。输出：无。预期行为：判不通过。"""
    loc = locales.load("zh")
    first = render.rule_lines(loc)[0]
    broken = state_of().replace(first, first.replace("| keep", "| drop"))
    assert not judges.FrameJudge("zh").judge(sample(broken), spec()).ok


def test_frame_skips_without_spec_or_text() -> None:
    """输入：没有真值规格 / state 不是文本。输出：无。预期行为：都记成跳过。"""
    assert judges.FrameJudge("zh").judge(sample(state_of())).skipped
    assert judges.FrameJudge("zh").judge(sample(123), spec()).skipped


def test_trace_passes_when_the_model_agrees(fake_backend: type) -> None:
    """输入：从不撒谎的假后端。输出：无。预期行为：通过。"""
    verdict = judges.TraceJudge(fake_backend(), "zh").judge(sample(state_of()), spec())
    assert verdict.ok and not verdict.skipped


def test_trace_catches_a_misread_kind(fake_backend: type) -> None:
    """输入：反解把「这一块是什么」说反。输出：无。预期行为：判不通过。"""
    backend = fake_backend(trace_lies_kind=True)
    verdict = judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())
    assert not verdict.ok
    assert "kind" in verdict.reason


def test_trace_catches_a_misread_fact(fake_backend: type) -> None:
    """输入：反解把「有没有承载持久事实」说反。输出：无。预期行为：判不通过。"""
    backend = fake_backend(trace_lies_fact=True)
    verdict = judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())
    assert not verdict.ok
    assert "fact" in verdict.reason


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
    backend = fake_backend(trace_data={"kind": "user"})
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
    assert judges.TRACE_FAMILIES == ("kind", "fact")
    for family in judges.TRACE_FAMILIES:
        assert str(loc["fam"][family]["q"][0]) in prompt
        for name in loc["fam"][family]["criteria"]:
            assert str(name) in prompt


def test_option_names_reads_the_product_order() -> None:
    """输入：一道 choice 题。输出：无。预期行为：按产物里的顺序取选项名。"""
    assert judges.option_names({"criteria": {"b": "x", "a": "y"}}) == ["b", "a"]
    assert judges.option_names({}) == []
