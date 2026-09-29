"""判别器测试：能抓到什么、端点故障怎么记。"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import AllProvidersDown, Draft, TransportError
from jc.jsontypes import JsonValue
from workflows.citation import features, judges, locales
from workflows.citation.spec import CitationSpec

from tests.workflows.citation.conftest import SENTENCES, FakeBackend, faithful, spec_for

CASE: Final[features.CitationCase] = features.CitationCase("partial", "other", "middle", "last")


def _claim_line_index(state: str, loc: dict[str, JsonValue]) -> int:
    """输入：state、语言字典。输出：声称出处那一行的行号。"""
    mark = str(loc["state"]["claim_mark"])
    for index, line in enumerate(state.splitlines()):
        if line.strip().startswith(mark):
            return index
    raise AssertionError("state 里没有声称出处那一行")


def _rewrite_claim(state: str, loc: dict[str, JsonValue], n: int) -> str:
    """输入：state、语言字典、新的节号（1 起）。输出：改过声称出处的 state。"""
    lines = state.splitlines()
    index = _claim_line_index(state, loc)
    lines[index] = f"{loc['state']['claim_mark']}{str(loc['state']['claim_value']).format(n=n)}"
    return "\n".join(lines)


def _rewrite_quote(state: str, loc: dict[str, JsonValue], quote: str) -> str:
    """输入：state、语言字典、新的引文。输出：改过引文的 state。"""
    mark = str(loc["state"]["cite_mark"])
    lines = state.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith(mark):
            lines[index] = f"{mark}{quote}"
            break
    return "\n".join(lines)


def test_hygiene_accepts_a_faithful_sample() -> None:
    """输入：兑现计划的样本。输出：无。预期行为：通过。"""
    spec, sample = faithful(CASE)
    verdict = judges.HygieneJudge("zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert "兑现" in verdict.reason


def test_hygiene_rejects_a_quote_that_does_not_match_the_plan() -> None:
    """输入：引文被换成别的句子的样本。输出：无。预期行为：不通过，理由是引文与计划不符。"""
    spec, sample = faithful(CASE)
    loc = locales.load("zh")
    sample["state"] = _rewrite_quote(str(sample["state"]), loc, SENTENCES[0])
    verdict = judges.HygieneJudge("zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "引文" in verdict.reason


def test_hygiene_rejects_a_duplicated_sentence() -> None:
    """输入：两句一模一样的样本。输出：无。预期行为：不通过（重复句会让「出自哪一句」失去意义）。"""
    spec, sample = faithful(CASE)
    state = str(sample["state"])
    lines = state.splitlines()
    target = next(index for index, line in enumerate(lines) if line.strip() == f"- {SENTENCES[1]}")
    lines[target] = f"- {SENTENCES[0]}"
    sample["state"] = "\n".join(lines)
    verdict = judges.HygieneJudge("zh").judge(sample, spec)
    assert not verdict.ok
    assert "句子重复" in verdict.reason


def test_hygiene_rejects_a_state_without_a_document_heading() -> None:
    """输入：一段没有文档抬头的 state。输出：无。预期行为：不通过，理由是认不出文档。"""
    _, sample = faithful(CASE)
    sample["state"] = "这里什么抬头都没有"
    verdict = judges.HygieneJudge("zh").judge(sample, spec_for(CASE))
    assert not verdict.ok
    assert "文档" in verdict.reason


def test_hygiene_rejects_fence_lines_anywhere() -> None:
    """输入：整段被围栏包起来的 state。输出：无。预期行为：不通过，理由是残留。"""
    spec, sample = faithful(CASE)
    sample["state"] = f"```text\n{sample['state']}\n```"
    verdict = judges.HygieneJudge("zh").judge(sample, spec)
    assert not verdict.ok
    assert "残留" in verdict.reason


def test_hygiene_skips_without_a_spec_or_text_state() -> None:
    """输入：没有 spec、或 state 不是文本。输出：无。预期行为：记成「没判成」。"""
    judge = judges.HygieneJudge("zh")
    assert judge.judge({"state": "x"}, None).skipped
    assert judge.judge({"state": 7}, spec_for(CASE)).skipped


def test_frame_accepts_a_faithful_sample() -> None:
    """输入：兑现计划的样本。输出：无。预期行为：通过。"""
    spec, sample = faithful(CASE)
    verdict = judges.FrameJudge("zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert "一致" in verdict.reason


def test_frame_rejects_a_tampered_claim() -> None:
    """输入：声称出处被改过的样本。输出：无。预期行为：不通过，点出声称出处对不上。"""
    spec, sample = faithful(CASE)
    loc = locales.load("zh")
    other = (spec.plan.claimed + 1) % features.SECTIONS
    sample["state"] = _rewrite_claim(str(sample["state"]), loc, other + 1)
    verdict = judges.FrameJudge("zh").judge(sample, spec)
    assert not verdict.ok
    assert "声称出处" in verdict.reason


def test_frame_rejects_a_missing_section() -> None:
    """输入：少了一节的样本。输出：无。预期行为：不通过，点出小节数对不上。"""
    spec, sample = faithful(CASE)
    loc = locales.load("zh")
    mark = str(loc["state"]["sec_mark"]).format(n=features.SECTIONS)
    lines = [line for line in str(sample["state"]).splitlines() if line.strip() != mark]
    sample["state"] = "\n".join(lines)
    verdict = judges.FrameJudge("zh").judge(sample, spec)
    assert not verdict.ok
    assert "小节数" in verdict.reason


def test_frame_skips_without_a_spec() -> None:
    """输入：没有 spec。输出：无。预期行为：记成「没判成」。"""
    _, sample = faithful(CASE)
    assert judges.FrameJudge("zh").judge(sample, None).skipped


def test_trace_passes_when_the_reader_agrees() -> None:
    """输入：反解说真话的假后端。输出：无。预期行为：通过。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(), "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert "一致" in verdict.reason


def test_trace_rejects_a_lying_backend() -> None:
    """输入：反解说反话的假后端。输出：无。预期行为：不通过，理由里两种说法都列出来。"""
    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(FakeBackend(trace_lies=True), "zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "计划" in verdict.reason and "反解" in verdict.reason


def test_trace_rejects_a_wrong_section_even_when_in_doc_agrees() -> None:
    """输入：只把节号说错的假后端。输出：无。预期行为：不通过（第二问也要对）。"""
    spec, sample = faithful(CASE)

    class HalfLiar(FakeBackend):
        """在文档里但节号说错的假后端。"""

        def generate(self, task: object) -> Draft:
            """输入：任务。输出：草稿。预期行为：in_doc 说真话、section 说反话。"""
            spec_data = getattr(task, "spec")
            return Draft(
                data={
                    "in_doc": True,
                    "section": (int(spec_data["true_section"]) + 1) % features.SECTIONS + 1,
                },
                raw_text="",
                backend="fake",
                provider="fake",
                model="fake",
                mode="fake",
            )

    verdict = judges.TraceJudge(HalfLiar(), "zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped


def test_trace_records_a_transport_failure_as_skipped() -> None:
    """输入：反解时抛 `TransportError`。输出：无。预期行为：记成「没判成」，不记成数据不合格。"""
    spec, sample = faithful(CASE)
    backend = FakeBackend(trace_raises=TransportError("boom"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.skipped and verdict.ok
    assert "端点故障" in verdict.reason


def test_trace_lets_a_fatal_failure_bubble_up() -> None:
    """输入：反解时抛 `AllProvidersDown`。输出：无。预期行为：整批停下，不能吞掉。"""
    spec, sample = faithful(CASE)
    backend = FakeBackend(trace_raises=AllProvidersDown("all down"))
    with pytest.raises(AllProvidersDown):
        judges.TraceJudge(backend, "zh").judge(sample, spec)


def test_trace_skips_without_a_spec() -> None:
    """输入：没有 spec。输出：无。预期行为：记成「没判成」。"""
    _, sample = faithful(CASE)
    assert judges.TraceJudge(FakeBackend(), "zh").judge(sample, None).skipped


def test_trace_skips_when_the_draft_has_no_field() -> None:
    """输入：反解草稿缺字段。输出：无。预期行为：记成「没判成」，不当成数据不合格。"""
    class Mute(FakeBackend):
        """只回一个空字典的假后端。"""

        def generate(self, task: object) -> Draft:
            """输入：任务。输出：空草稿。"""
            return Draft(
                data={},
                raw_text="",
                backend="fake",
                provider="fake",
                model="fake",
                mode="fake",
            )

    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(Mute(), "zh").judge(sample, spec)
    assert verdict.skipped
    assert "in_doc" in verdict.reason


def test_trace_skips_when_the_section_is_not_an_integer() -> None:
    """输入：反解把节号写成字符串。输出：无。预期行为：记成「没判成」。"""
    class Stringy(FakeBackend):
        """节号写成字符串的假后端。"""

        def generate(self, task: object) -> Draft:
            """输入：任务。输出：节号是字符串的草稿。"""
            return Draft(
                data={"in_doc": True, "section": "2"},
                raw_text="",
                backend="fake",
                provider="fake",
                model="fake",
                mode="fake",
            )

    spec, sample = faithful(CASE)
    verdict = judges.TraceJudge(Stringy(), "zh").judge(sample, spec)
    assert verdict.skipped
    assert "section" in verdict.reason


def test_trace_draft_bounds_are_locked() -> None:
    """输入：反解模型的字段约束。输出：无。预期行为：节号只收 1~3。"""
    assert judges.TraceDraft(in_doc=True, section=1).section == 1
    assert judges.TraceDraft(in_doc=False, section=None).section is None
    with pytest.raises(ValueError):
        judges.TraceDraft(in_doc=True, section=0)
    with pytest.raises(ValueError):
        judges.TraceDraft(in_doc=True, section=features.SECTIONS + 1)


def test_trace_prompt_carries_the_state() -> None:
    """输入：一段 state。输出：无。预期行为：反解提示词里原样带着整段材料。"""
    spec, sample = faithful(CASE)
    prompt = judges.build_trace_prompt(str(sample["state"]), locales.load("zh"))
    assert str(sample["state"]) in prompt


def test_option_names_reads_the_written_order() -> None:
    """输入：一道 choice 题。输出：无。预期行为：按产物里的书写顺序给出选项名。"""
    question: dict[str, JsonValue] = {"type": "choice", "criteria": {"b": "x", "a": "y"}}
    assert judges.option_names(question) == ["b", "a"]
    assert judges.option_names({"type": "noul"}) == []


def test_spec_labels_helper_is_the_construction_path() -> None:
    """输入：一份规格。输出：无。预期行为：`spec.labels` 与 `features.label_for` 同源。"""
    spec: CitationSpec = spec_for(CASE)
    assert spec.labels == features.label_for(spec.plan)
