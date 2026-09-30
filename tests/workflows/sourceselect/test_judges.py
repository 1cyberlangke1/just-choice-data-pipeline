"""来源选择的判别器测试：卫生、框架比对与反解，含端点故障的记账。"""

from __future__ import annotations

from jc.jsontypes import JsonValue

from workflows.sourceselect import features, judges, locales, render
from workflows.sourceselect.features import SourceCase
from workflows.sourceselect.spec import SourceSpec

from tests.workflows.sourceselect.conftest import QUERIES, FakeBackend

CASE = SourceCase(need="authority", size=4, shape="tied", near_miss=1)
PLAN = features.plan_for(CASE, 0)
GOOD = {"query": QUERIES["authority"]}


def sample(state: str) -> dict[str, JsonValue]:
    """输入：state。输出：最小可判样本。"""
    return {"state": state}


def spec() -> SourceSpec:
    """输入：无。输出：与 `CASE` 对齐的真值规格。"""
    return SourceSpec(case=CASE, phrasing=0, split="train", seq=0)


def state_of(data: dict[str, str] | None = None) -> str:
    """输入：可选草稿覆盖。输出：拼好的 state。"""
    return render.to_state({**GOOD, **(data or {})}, PLAN, locales.load("zh"))


def test_hygiene_catches_leftovers_and_multiline() -> None:
    """输入：带围栏 / 多行 / 空查询的 state。输出：无。预期行为：都判不通过。"""
    judge = judges.HygieneJudge("zh")
    assert not judge.judge(sample(state_of().replace("【查询】", "```\n【查询】"))).ok
    assert not judge.judge(sample(state_of({"query": "第一行\n第二行"}))).ok
    assert not judge.judge(sample(state_of({"query": "   "}))).ok


def test_hygiene_catches_a_missing_table() -> None:
    """输入：候选表被整段删掉。输出：无。预期行为：判不通过。"""
    loc = locales.load("zh")
    state = state_of()
    for line in render.table_lines(PLAN, loc):
        state = state.replace(f"{line}\n", "")
    assert not judges.HygieneJudge("zh").judge(sample(state)).ok


def test_frame_catches_a_wrong_row() -> None:
    """输入：某一行的取值被改掉。输出：无。预期行为：判不通过。"""
    broken = state_of().replace("official", "community")
    assert not judges.FrameJudge("zh").judge(sample(broken), spec()).ok


def test_trace_catches_a_misread_need(fake_backend: type[FakeBackend]) -> None:
    """输入：反解说反话的假后端。输出：无。预期行为：判不通过。"""
    verdict = judges.TraceJudge(fake_backend(trace_lies_need=True), "zh").judge(
        sample(state_of()), spec()
    )
    assert not verdict.ok


def test_trace_marks_a_broken_draft_as_skipped(fake_backend: type[FakeBackend]) -> None:
    """输入：反解草稿缺字段 / 字段类型不对。输出：无。预期行为：都记成跳过。"""
    for payload in ({"wrong": "authority"}, {"need": 7}):
        backend = fake_backend(trace_data=payload)
        verdict = judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())
        assert verdict.ok and verdict.skipped, payload


def test_trace_skips_without_spec_or_backend(fake_backend: type[FakeBackend]) -> None:
    """输入：缺 spec / 缺后端。输出：无。预期行为：都记成跳过。"""
    assert judges.TraceJudge(fake_backend(), "zh").judge(sample(state_of())).skipped
    assert judges.TraceJudge(None, "zh").judge(sample(state_of()), spec()).skipped


def test_build_trace_prompt_quotes_the_item() -> None:
    """输入：state 与语言字典。输出：无。预期行为：题面里带着 state 与三个选项。"""
    loc = locales.load("zh")
    prompt = judges.build_trace_prompt(state_of(), loc)
    assert QUERIES["authority"] in prompt
    for need in features.NEEDS:
        assert need in prompt
