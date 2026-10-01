"""判别器：重算四个答案、篡改能抓到、缺 spec 跳过、跨语言。"""

from __future__ import annotations

import random
from typing import Final

import pytest

from jc import assemble
from jc.jsontypes import JsonValue
from tests.workflows.planstep.conftest import _make_spec
from workflows.planstep import features as ft
from workflows.planstep import judges, locales, parse, questions, render

LOC: Final[dict[str, object]] = locales.load("zh")
ALL_CASES: Final[tuple[ft.PlanCase, ...]] = tuple(ft.build_cases(512))


def sample_of(case: ft.PlanCase, locale: str = "zh", seed: int = 0) -> dict[str, JsonValue]:
    """输入：case、语言、种子。输出：契约样本。

    预期行为：走真管线那两步（渲染 + 出题 + 组装），不手写样本——判别器要验的正是
    「组装之后还能不能由 state 重算出来」。
    """
    loc = locales.load(locale)
    spec = _make_spec(case.graph, case.done)
    state = render.build(spec, loc)
    labeled = questions.build(spec, loc, spec.phrasing, random.Random(seed))
    return assemble.assemble_sample("s-0000001", state, labeled, random.Random(seed))


def parsed_of(sample: dict[str, JsonValue], locale: str = "zh") -> parse.ParsedPlan:
    """输入：样本、语言。输出：反解结果。预期行为：用该语言的标签表反解。"""
    return parse.parse_state(sample["state"], parse.tables_of(locales.load(locale)))


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_facts_are_recomputed_from_the_state(lang: str) -> None:
    """输入：全空间 case × 两种语言。输出：无。

    预期行为：从 state 反解再重算出来的四个答案，与构造侧标签**逐条一致**——这条是
    「渲染把某一项写歪了」那条路的总闸。
    """
    for case in ALL_CASES:
        facts = judges.facts_of(parsed_of(sample_of(case, lang), lang))
        want = ft.label_for(case)
        assert facts["remaining"] == want.remaining, case.describe()
        assert facts["next"] == want.next_key, case.describe()
        assert facts["rounds"] == want.rounds, case.describe()
        assert facts["unlocks"] == want.unlocks, case.describe()


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_check_sample_passes_for_genuine_samples(lang: str) -> None:
    """输入：真样本。输出：无。预期行为：零问题——判别器不误判。"""
    loc = locales.load(lang)
    for case in ALL_CASES[:12]:
        sample = sample_of(case, lang)
        assert judges.check_sample(sample, judges.facts_of(parsed_of(sample, lang)), loc) == []


def test_check_sample_catches_a_tampered_answer() -> None:
    """输入：把三个答案各改错一处。输出：无。预期行为：每一处都被点名抓到。"""
    sample = sample_of(ft.PlanCase(graph=3, done=1))
    facts = judges.facts_of(parsed_of(sample))
    for qid, wrong in (("remaining", 5), ("rounds", 0), ("unlocks", None)):
        broken = {
            "state": sample["state"],
            "questions": sample["questions"],
            "answers": {key: dict(value) for key, value in sample["answers"].items()},
        }
        if qid == "unlocks":
            broken["answers"][qid]["noul"] = not broken["answers"][qid]["noul"]
        else:
            broken["answers"][qid]["score"] = wrong
        problems = judges.check_sample(broken, facts, LOC)
        assert any(problem.startswith(f"{qid}:") for problem in problems), (qid, problems)


def test_check_sample_catches_a_tampered_choice_index() -> None:
    """输入：把 choice 的编号指到另一个选项。输出：无。预期行为：按描述原文比对，抓到。"""
    sample = sample_of(ft.PlanCase(graph=6, done=0))
    facts = judges.facts_of(parsed_of(sample))
    options = list(sample["questions"]["next"]["criteria"].values())
    index = sample["answers"]["next"]["choice"]
    wrong = (index + 1) % len(options)
    broken = {
        "state": sample["state"],
        "questions": sample["questions"],
        "answers": {key: dict(value) for key, value in sample["answers"].items()},
    }
    broken["answers"]["next"]["choice"] = wrong
    problems = judges.check_sample(broken, facts, LOC)
    assert any(problem.startswith("next:") for problem in problems), problems


def test_check_sample_catches_an_out_of_range_index() -> None:
    """输入：把编号指到选项表外。输出：无。预期行为：报「编号越界」，不去索引列表。"""
    sample = sample_of(ft.PlanCase(graph=0, done=0))
    facts = judges.facts_of(parsed_of(sample))
    broken = {
        "state": sample["state"],
        "questions": sample["questions"],
        "answers": {key: dict(value) for key, value in sample["answers"].items()},
    }
    broken["answers"]["next"]["choice"] = 99
    problems = judges.check_sample(broken, facts, LOC)
    assert any("编号越界" in problem for problem in problems), problems


def test_check_sample_any_locale_accepts_the_other_language() -> None:
    """输入：英文样本、用合并口径比对。输出：无。

    预期行为：审计拿到的语言未必是产物那一种，所以选项描述按任一语言认；但只用中文口径的
    `check_sample` 会把它判错——两种口径的区别就在这里。
    """
    sample = sample_of(ft.PlanCase(graph=4, done=2), "en")
    facts = judges.facts_of(parsed_of(sample, "en"))
    assert judges.check_sample_any_locale(sample, facts) == []
    assert judges.check_sample(sample, facts, LOC) != []


def test_roundtrip_diff_is_empty_for_genuine_renders() -> None:
    """输入：全空间 case × 两种语言。输出：无。预期行为：反解与真值规格逐项逐格一致。"""
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        for case in ALL_CASES:
            spec = _make_spec(case.graph, case.done)
            parsed = parse.parse_state(render.build(spec, loc), parse.tables_of(loc))
            assert judges.roundtrip_diff(spec, parsed, loc) == [], case.describe()


def test_roundtrip_diff_catches_a_rewritten_row() -> None:
    """输入：把某一项的状态与前置都改掉。输出：无。预期行为：两条差异都被点出来。"""
    loc = locales.load("zh")
    case = ft.PlanCase(graph=0, done=2)
    spec = _make_spec(case.graph, case.done)
    text = render.build(spec, loc)
    broken = text.replace("  1. 需求梳理 | 已完成 | 无前置", "  1. 需求梳理 | 待开始 | 前置=2")
    assert broken != text
    parsed = parse.parse_state(broken, parse.tables_of(loc))
    diff = judges.roundtrip_diff(spec, parsed, loc)
    assert any("状态与计划不符" in item for item in diff), diff
    assert any("前置" in item for item in diff), diff


def test_roundtrip_diff_catches_a_broken_request_id() -> None:
    """输入：抬头被改。输出：无。预期行为：请求号对不上就报出来。"""
    loc = locales.load("zh")
    spec = _make_spec(2, 1)
    text = render.build(spec, loc).replace(spec.query_id, "P-000000")
    parsed = parse.parse_state(text, parse.tables_of(loc))
    assert any("请求号" in item for item in judges.roundtrip_diff(spec, parsed, loc))


def test_roundtrip_judge_skips_without_a_spec() -> None:
    """输入：不给真值 spec。输出：无。预期行为：跳过而不是误判。"""
    sample = sample_of(ft.PlanCase(graph=1, done=0))
    verdict = judges.RoundtripJudge(locale="zh").judge(sample)
    assert verdict.ok is True and verdict.raw == {"skipped": True}


def test_roundtrip_judge_passes_and_fails() -> None:
    """输入：真样本与写歪的样本。输出：无。预期行为：真的过、歪的不过，理由是人话。"""
    loc = locales.load("zh")
    case = ft.PlanCase(graph=5, done=1)
    spec = _make_spec(case.graph, case.done)
    sample = sample_of(case)
    assert judges.RoundtripJudge(locale="zh").judge(sample, spec).ok is True
    sample["state"] = str(sample["state"]).replace("待开始", "已完成")
    verdict = judges.RoundtripJudge(locale="zh").judge(sample, spec)
    assert verdict.ok is False and verdict.reason
    assert judges.roundtrip_diff(spec, parse.parse_state(sample["state"], parse.tables_of(loc)), loc)


def test_label_judge_passes_and_fails() -> None:
    """输入：真样本与被改答案的样本。输出：无。预期行为：真的过、歪的不过。"""
    sample = sample_of(ft.PlanCase(graph=8, done=3))
    assert judges.LabelJudge(locale="zh").judge(sample).ok is True
    sample["answers"]["remaining"]["score"] = 5
    verdict = judges.LabelJudge(locale="zh").judge(sample)
    assert verdict.ok is False and "remaining" in verdict.reason


def test_label_judge_reports_an_unparsable_state() -> None:
    """输入：state 被写坏。输出：无。预期行为：判不通过并说明反解失败，不抛异常。"""
    verdict = judges.LabelJudge(locale="zh").judge({"state": "乱写", "questions": {}, "answers": {}})
    assert verdict.ok is False and "重算失败" in verdict.reason


def test_roundtrip_judge_reports_an_unparsable_state() -> None:
    """输入：state 被写坏。输出：无。预期行为：判不通过并说明反解失败，不抛异常。"""
    spec = _make_spec(0, 0)
    verdict = judges.RoundtripJudge(locale="zh").judge({"state": "乱写"}, spec)
    assert verdict.ok is False and "反解失败" in verdict.reason


def test_expected_of_rejects_an_unknown_family() -> None:
    """输入：不存在的题族。输出：无。预期行为：报错——新加了题族却忘了在这儿实现，不能静默放过。"""
    with pytest.raises(ValueError, match="未知题族"):
        judges.expected_of("nope", {"next": "t1"}, LOC)


def test_expected_variants_cover_every_language_for_choice() -> None:
    """输入：一个 choice 事实。输出：无。

    预期行为：choice 的答案按语言各收一份（两种语言文案不同），score / noul 与语言无关
    只收一份。
    """
    facts: dict[str, JsonValue] = {"next": "t3", "remaining": 2, "rounds": 1, "unlocks": False}
    variants = judges.expected_variants("next", facts)
    assert len(variants) == len(locales.available()) == 2
    assert set(variants) == {
        str(locales.load(lang)["task"]["T3"]) for lang in locales.available()
    }
    assert judges.expected_variants("remaining", facts) == (2,)
    assert judges.expected_variants("unlocks", facts) == (False,)


def test_option_text_reports_a_missing_option_key() -> None:
    """输入：重算出来的选项名不在选项表里。输出：无。预期行为：报人话，不静默取空串。"""
    with pytest.raises(ValueError, match="不在选项表里"):
        judges.option_text("next", {"next": "t9"}, LOC)


def test_option_text_reports_a_family_without_options() -> None:
    """输入：score 题族。输出：无。预期行为：报错——带选项的只有 `next` 一族。"""
    with pytest.raises(ValueError, match="带选项的只有"):
        judges.option_text("remaining", {"remaining": 1}, LOC)
