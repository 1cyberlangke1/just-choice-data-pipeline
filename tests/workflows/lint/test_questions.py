"""行级规则检查的题族测试：出哪些题、选项集、金答案。"""

from __future__ import annotations

import pytest

from workflows.lint import features, locales, questions
from workflows.lint.features import LintCase

SINGLE = LintCase(features.RULE_PATH, "log", 1)
ZERO = LintCase(features.RULE_SECRET, "doc", 0)
MANY = LintCase(features.RULE_TODO, "code", 5)


def test_spot_family_only_when_exactly_one_line_hits() -> None:
    """输入：命中 0 / 1 / 5 行的计划。输出：无。预期行为：只有 1 行命中时才出 `spot`。"""
    assert questions.families_for(features.plan_for(SINGLE, 0)) == questions.FAMILIES
    assert questions.families_for(features.plan_for(ZERO, 0)) == questions.ALWAYS_FAMILIES
    assert questions.families_for(features.plan_for(MANY, 0)) == questions.ALWAYS_FAMILIES
    assert "spot" not in questions.ALWAYS_FAMILIES


def test_build_emits_the_declared_families() -> None:
    """输入：三个 case、两个语言。输出：无。

    预期行为：题号是「族名 + 序号」，题族与 `families_for` 一致；`line` 族正好八道，
    其余每族一道。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in ["zh", "en"]:
        for case in [ZERO, SINGLE, MANY]:
            try:
                plan = features.plan_for(case, 0)
                labeled = questions.build(plan, locales.load(lang), 0)
                ids = [item.family for item in labeled]
                assert ids.count("line") == features.SIZE
                assert set(ids) == set(questions.families_for(plan))
                for family in ("count", "gate", "spot"):
                    assert ids.count(family) == (1 if family in questions.families_for(plan) else 0)
            except AssertionError as exc:
                bad.append(f"{(lang, case)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_line_answers_follow_the_plan_in_order() -> None:
    """输入：一个计划。输出：无。预期行为：八道 `line` 题的答案逐行等于计划的命中标记。"""
    plan = features.plan_for(MANY, 0)
    labeled = questions.build(plan, locales.load("zh"), 0)
    lines = [item for item in labeled if item.family == "line"]
    assert [item.answer for item in lines] == list(plan.matched())
    assert [item.question["type"] for item in lines] == ["noul"] * features.SIZE
    for index, item in enumerate(lines, start=1):
        assert str(index) in item.question["instructions"]


def test_count_options_cover_zero_through_eight() -> None:
    """输入：任意 case。输出：无。预期行为：`count` 的选项恒为 c0~c8，答案是 c{hits}。"""
    plan = features.plan_for(MANY, 0)
    item = next(q for q in questions.build(plan, locales.load("en"), 0) if q.family == "count")
    assert set(item.question["criteria"]) == {f"c{value}" for value in range(features.SIZE + 1)}
    assert item.answer == f"c{plan.hits}"


def test_gate_options_are_the_three_handlings() -> None:
    """输入：任意 case。输出：无。预期行为：选项名恒为 pass / review / block。"""
    plan = features.plan_for(MANY, 0)
    item = next(q for q in questions.build(plan, locales.load("en"), 0) if q.family == "gate")
    assert set(item.question["criteria"]) == set(features.GATES)
    assert item.answer == features.gate_of(plan.hits)


def test_spot_options_always_cover_all_eight_rows() -> None:
    """输入：一个「恰好一行命中」的计划。输出：无。

    预期行为：八个选项**全都出**（只出答案那一个会让选项集泄露答案），答案是那一行的行号。
    """
    plan = features.plan_for(SINGLE, 0)
    item = next(q for q in questions.build(plan, locales.load("zh"), 0) if q.family == "spot")
    assert set(item.question["criteria"]) == {f"row{n}" for n in range(1, features.SIZE + 1)}
    assert item.answer == f"row{features.label_for(plan).spot}"


def test_choice_options_allow_shuffling_and_noul_does_not() -> None:
    """输入：一个计划。输出：无。预期行为：choice 允许打乱（防位置捷径），noul 不需要。"""
    labeled = questions.build(features.plan_for(MANY, 0), locales.load("zh"), 0)
    for item in labeled:
        expected = item.question["type"] == "choice"
        assert item.shuffle is expected


def test_answer_must_be_one_of_the_options() -> None:
    """输入：答案不在选项里。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("gate", "题面", {"pass": "放行"}, "block")


@pytest.mark.parametrize("phrasing", [0, 1, 2])
def test_phrasing_index_wraps_around(phrasing: int) -> None:
    """输入：超出候选数的问法编号。输出：无。预期行为：回绕取候选，不报错。"""
    labeled = questions.build(features.plan_for(MANY, phrasing), locales.load("zh"), phrasing)
    assert len(labeled) == features.SIZE + len(questions.ALWAYS_FAMILIES) - 1
