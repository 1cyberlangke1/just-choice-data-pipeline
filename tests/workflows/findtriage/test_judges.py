"""改动引入性判定的判别器测试：卫生、框架比对与反解，含端点故障的记账。"""

from __future__ import annotations

from jc.jsontypes import JsonValue

from workflows.findtriage import features, judges, locales, questions, render
from workflows.findtriage.features import RecordCase
from workflows.findtriage.spec import FindTriageSpec

from tests.workflows.findtriage.conftest import FakeBackend

CASE = RecordCase(cite="touched", evidence="in_row", fix="concrete", size=4, decoy=1)
PLAN = features.plan_for(CASE, 0)
GOOD = {
    "finding": f"第 {PLAN.cited_line} 行：`{PLAN.evidence_symbol}` 的取值没夹紧。",
    "fix": f"在 `{PLAN.fix_symbol}` 那一处补一个上限校验。",
}


def sample(state: str) -> dict[str, JsonValue]:
    """输入：state。输出：最小可判样本。"""
    return {"state": state}


def spec() -> FindTriageSpec:
    """输入：无。输出：与 `CASE` 对齐的真值规格。"""
    return FindTriageSpec(case=CASE, phrasing=0, split="train", seq=0)


def state_of(data: dict[str, str] | None = None) -> str:
    """输入：可选草稿覆盖。输出：拼好的 state。"""
    return render.to_state({**GOOD, **(data or {})}, PLAN, locales.load("zh"))


def test_hygiene_catches_leftovers_and_multiline() -> None:
    """输入：带围栏 / 多行 / 空修法的 state。输出：无。预期行为：都判不通过。"""
    judge = judges.HygieneJudge("zh")
    assert not judge.judge(sample(state_of().replace("【审查发现】", "```\n【审查发现】"))).ok
    assert not judge.judge(sample(state_of({"finding": "第一行\n第二行"}))).ok
    assert not judge.judge(sample(state_of({"fix": "   "}))).ok


def test_hygiene_catches_a_missing_table() -> None:
    """输入：改动记录被整段删掉。输出：无。预期行为：判不通过。"""
    loc = locales.load("zh")
    state = state_of()
    for line in render.table_lines(PLAN, loc):
        state = state.replace(f"{line}\n", "")
    assert not judges.HygieneJudge("zh").judge(sample(state)).ok


def test_frame_passes_the_injected_table() -> None:
    """输入：与计划一致的 state。输出：无。预期行为：通过。"""
    verdict = judges.FrameJudge("zh").judge(sample(state_of()), spec())
    assert verdict.ok


def test_frame_catches_a_wrong_row() -> None:
    """输入：某一行的改动种类被改掉。输出：无。预期行为：判不通过。"""
    broken = state_of().replace("added", "untouched")
    assert not judges.FrameJudge("zh").judge(sample(broken), spec()).ok


def test_frame_catches_a_missing_rule_table() -> None:
    """输入：口径被整段删掉。输出：无。预期行为：判不通过。"""
    loc = locales.load("zh")
    state = state_of()
    for line in render.rule_lines(loc):
        state = state.replace(f"{line}\n", "")
    assert not judges.FrameJudge("zh").judge(sample(state), spec()).ok


def test_frame_catches_a_missing_owner_table() -> None:
    """输入：归属口径被整段删掉。输出：无。预期行为：判不通过。

    为什么单独钉一条：`owner` 那一族的答案完全由这张表说了算，表没渲染出来这道题就成了
    「没给规则的题」——判别器必须当场拦下，不能让它进产物。
    """
    loc = locales.load("zh")
    state = state_of()
    for line in render.owner_lines(loc):
        state = state.replace(f"{line}\n", "")
    assert not judges.FrameJudge("zh").judge(sample(state), spec()).ok


def test_hygiene_catches_a_missing_owner_table() -> None:
    """输入：归属口径**连抬头一起**删掉。输出：无。预期行为：卫生判别器也判不通过。

    注意删法：只删数据行是不够的——`owner_rules_of` 认「段里有没有能解析的行」，
    而归属表最后一行不带换行，`replace("行\n", "")` 会漏掉它（这次就是这么发现的）。
    要验「整段没了」，得连抬头一起按行过滤。
    """
    loc = locales.load("zh")
    header = f"{locales.header(loc, str(loc['meta']['owner_label']))}"
    owner_rows = set(render.owner_lines(loc))
    state = "\n".join(
        line
        for line in state_of().splitlines()
        if not line.startswith(header) and line not in owner_rows
    )
    assert render.owner_rules_of(state, loc) == ()
    assert not judges.HygieneJudge("zh").judge(sample(state)).ok


def test_frame_skips_without_spec() -> None:
    """输入：没有真值规格。输出：无。预期行为：记成跳过，不算不通过。"""
    verdict = judges.FrameJudge("zh").judge(sample(state_of()))
    assert verdict.ok and verdict.skipped


def test_trace_catches_each_misread(fake_backend: type[FakeBackend]) -> None:
    """输入：三件事实各说反一次。输出：无。预期行为：三次都判不通过。"""
    for switch in ("trace_lies_cite", "trace_lies_symbol", "trace_lies_actionable"):
        judge = judges.TraceJudge(fake_backend(**{switch: True}), "zh")
        assert not judge.judge(sample(state_of()), spec()).ok, switch


def test_trace_marks_a_broken_draft_as_skipped(fake_backend: type[FakeBackend]) -> None:
    """输入：反解草稿缺字段 / 字段类型不对。输出：无。预期行为：都记成跳过。"""
    for payload in (
        {"wrong": "s1"},
        {"cite": 7, "symbol": "x", "actionable": True},
        {"cite": "s1", "symbol": "x", "actionable": "yes"},
    ):
        backend = fake_backend(trace_data=payload)
        verdict = judges.TraceJudge(backend, "zh").judge(sample(state_of()), spec())
        assert verdict.ok and verdict.skipped, payload


def test_trace_skips_without_spec_or_backend(fake_backend: type[FakeBackend]) -> None:
    """输入：没有规格 / 没有后端。输出：无。预期行为：都记成跳过。"""
    assert judges.TraceJudge(fake_backend(), "zh").judge(sample(state_of())).skipped
    assert judges.TraceJudge(None, "zh").judge(sample(state_of()), spec()).skipped


def test_trace_prompt_reuses_the_dataset_options() -> None:
    """输入：一个 state。输出：无。

    预期行为：反解提示词里 `cite` 的选项名与题面那一族**完全一致**——不一致就会出现
    「答对了选项、对不上标签」。
    """
    loc = locales.load("zh")
    prompt = judges.build_trace_prompt(state_of(), PLAN.size, loc)
    options = questions.cite_options(PLAN.size, loc)
    assert judges.render_cite_options(PLAN.size, loc) == options
    for name in options:
        assert name in prompt, name
    assert str(loc["verify"]["symbol_q"]) in prompt
