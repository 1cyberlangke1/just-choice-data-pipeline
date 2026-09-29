"""行级规则检查的判别器测试：能抓到什么、端点故障怎么记。"""

from __future__ import annotations

from typing import Final

import pytest

from jc.backend.types import Draft, TransportError
from jc.jsontypes import JsonValue
from workflows.lint import features, judges, locales, render
from workflows.lint.features import LintCase
from workflows.lint.spec import LintSpec

from tests.workflows.lint.conftest import FakeBackend, draft_lines

SINGLE: Final[LintCase] = LintCase(features.RULE_TODO, "code", 1)
MANY: Final[LintCase] = LintCase(features.RULE_PATH, "log", 5)


def spec_for(case: LintCase, phrasing: int = 0) -> LintSpec:
    """输入：case、问法编号。输出：真值规格。"""
    return LintSpec(
        case=case,
        phrasing=phrasing,
        split="train",
        seq=1,
        plan=features.plan_for(case, phrasing),
    )


def state_of(spec: LintSpec, lines: list[str], lang: str = "zh") -> dict[str, JsonValue]:
    """输入：规格、行原文、语言。输出：只带 state 的样本。预期行为：不校验草稿，专供反例。"""
    return {"state": render.to_state(lines, spec.plan, locales.load(lang))}


def faithful(case: LintCase, lang: str = "zh") -> tuple[LintSpec, dict[str, JsonValue]]:
    """输入：case、语言。输出：(规格, 样本)。预期行为：草稿完全兑现计划。"""
    spec = spec_for(case)
    lines = draft_lines(case.rule, list(spec.plan.hit_rows))
    render.check_draft(lines, spec.plan)
    return spec, state_of(spec, lines, lang)


def test_hygiene_accepts_a_faithful_fragment() -> None:
    """输入：兑现计划的样本。输出：无。预期行为：通过。"""
    spec, sample = faithful(MANY)
    verdict = judges.HygieneJudge("zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert "兑现" in verdict.reason


@pytest.mark.parametrize("switch", ["unrealized", "extra_hit", "drop_line", "fence_line"])
def test_hygiene_rejects_broken_fragments(switch: str) -> None:
    """输入：四种写歪的片段。输出：无。预期行为：一律不通过。"""
    spec = spec_for(MANY)
    lines = draft_lines(MANY.rule, list(spec.plan.hit_rows), **{switch: True})
    verdict = judges.HygieneJudge("zh").judge(state_of(spec, lines), spec)
    assert not verdict.ok and not verdict.skipped


def test_hygiene_rejects_an_empty_fragment() -> None:
    """输入：一段没有片段抬头的 state。输出：无。预期行为：不通过，理由是认不出片段。"""
    verdict = judges.HygieneJudge("zh").judge({"state": "这里什么片段都没有"}, spec_for(MANY))
    assert not verdict.ok
    assert "片段" in verdict.reason


def test_hygiene_rejects_fence_lines_anywhere() -> None:
    """输入：片段之外也带围栏的 state。输出：无。预期行为：不通过。"""
    spec, sample = faithful(MANY)
    sample["state"] = f"```text\n{sample['state']}\n```"
    verdict = judges.HygieneJudge("zh").judge(sample, spec)
    assert not verdict.ok
    assert "残留" in verdict.reason


def test_hygiene_skips_without_a_spec_or_text_state() -> None:
    """输入：没有 spec、或 state 不是文本。输出：无。预期行为：记成「没判成」。"""
    judge = judges.HygieneJudge("zh")
    assert judge.judge({"state": "x"}, object()).skipped
    assert judge.judge({"state": 42}, spec_for(MANY)).skipped


def test_frame_accepts_the_injected_features() -> None:
    """输入：兑现计划的样本。输出：无。预期行为：通过。"""
    spec, sample = faithful(SINGLE)
    verdict = judges.FrameJudge("zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped


def test_frame_catches_a_swapped_rule_name() -> None:
    """输入：把规则名换成另一条的 state。输出：无。预期行为：不通过并点名规则名。"""
    spec, sample = faithful(SINGLE)
    loc = locales.load("zh")
    wrong = render.rule_name(loc, features.RULE_DEBUG)
    right = render.rule_name(loc, features.RULE_TODO)
    sample["state"] = str(sample["state"]).replace(right, wrong)
    verdict = judges.FrameJudge("zh").judge(sample, spec)
    assert not verdict.ok
    assert "规则名对不上" in verdict.reason


def test_frame_catches_a_broken_handling_table() -> None:
    """输入：处置办法表被改过的 state。输出：无。预期行为：不通过并点名处置办法表。"""
    spec, sample = faithful(MANY)
    loc = locales.load("zh")
    rows = render.gate_rows(loc)
    sample["state"] = str(sample["state"]).replace(rows[0], "- 命中 0~2 行 → 拦下")
    verdict = judges.FrameJudge("zh").judge(sample, spec)
    assert not verdict.ok
    assert "处置办法表" in verdict.reason


def test_frame_skips_without_a_spec_or_text_state() -> None:
    """输入：没有 spec、或 state 不是文本。输出：无。预期行为：记成「没判成」。"""
    judge = judges.FrameJudge("zh")
    assert judge.judge({"state": "x"}, object()).skipped
    assert judge.judge({"state": None}, spec_for(MANY)).skipped


def test_trace_accepts_a_truthful_reader() -> None:
    """输入：说真话的假后端。输出：无。预期行为：通过，理由里带上反解出来的数。"""
    spec, sample = faithful(MANY)
    verdict = judges.TraceJudge(FakeBackend(), "zh").judge(sample, spec)
    assert verdict.ok and not verdict.skipped
    assert verdict.raw["traced"] == spec.plan.hits


def test_trace_catches_a_reader_that_miscounts() -> None:
    """输入：把命中行数说反的假后端。输出：无。预期行为：不通过，理由里两个数都在。"""
    spec, sample = faithful(MANY)
    verdict = judges.TraceJudge(FakeBackend(trace_lies_hits=True), "zh").judge(sample, spec)
    assert not verdict.ok and not verdict.skipped
    assert "计划里命中" in verdict.reason


def test_trace_records_endpoint_failure_as_skipped() -> None:
    """输入：反解时抛传输错误的假后端。输出：无。预期行为：记成「没判成」，不算数据问题。"""
    spec, sample = faithful(MANY)
    backend = FakeBackend(trace_raises=TransportError("端点炸了"))
    verdict = judges.TraceJudge(backend, "zh").judge(sample, spec)
    assert verdict.ok and verdict.skipped
    assert "端点故障" in verdict.reason


class _BadDraft(FakeBackend):
    """反解草稿里 `hits` 不是整数的假后端（验「判不了」这条路）。"""

    def generate(self, task: object) -> Draft:
        """输入：任务。输出：草稿。预期行为：反解任务回一个字符串。"""
        if str(getattr(task, "task_id")).endswith("-trace"):
            return Draft(
                data={"hits": "many"},
                raw_text="hits=many",
                backend="fake",
                provider="fake",
                model="fake",
                mode="fake",
            )
        return super().generate(task)


def test_trace_skips_when_the_draft_has_no_integer() -> None:
    """输入：`hits` 是字符串的草稿。输出：无。预期行为：记成「没判成」。"""
    spec, sample = faithful(MANY)
    verdict = judges.TraceJudge(_BadDraft(), "zh").judge(sample, spec)
    assert verdict.ok and verdict.skipped
    assert "判不了" in verdict.reason


def test_trace_skips_without_spec_or_backend() -> None:
    """输入：没有 spec、或没有后端。输出：无。预期行为：都记成「没判成」。"""
    spec, sample = faithful(MANY)
    assert judges.TraceJudge(FakeBackend(), "zh").judge(sample, object()).skipped
    assert judges.TraceJudge(None, "zh").judge(sample, spec).skipped


def test_option_names_reads_the_written_order() -> None:
    """输入：一道 choice 题。输出：无。预期行为：按产物里写的顺序给选项名。"""
    question: dict[str, JsonValue] = {"type": "choice", "criteria": {"b": "2", "a": "1"}}
    assert judges.option_names(question) == ["b", "a"]
    assert judges.option_names({"type": "noul"}) == []
