"""内容审核的题族测试：四族的题面、答案与均衡。"""

from __future__ import annotations

import pytest

from jc.workflow import LabeledQuestion

from workflows.guardrail import features, locales, questions
from workflows.guardrail.features import GuardrailCase

#: 全空间 32 组 case。
CASES = features.build_cases(512)

#: 每族在 `build` 里占的位置：前四条逐条款，然后 count / combo / action。
CLAUSE_SPAN = len(features.CLAUSE_IDS)


def _build(
    case: GuardrailCase, lang: str = "zh", phrasing: int = 0
) -> list[LabeledQuestion]:
    """输入：case、语言、问法编号。输出：这一条的题目列表。"""
    return questions.build(features.plan_for(case, phrasing), locales.load(lang), phrasing)


def test_family_order_matches_the_declared_families() -> None:
    """输入：一个 case。输出：无。预期行为：四族顺序与题数固定（逐条款 4 条 + 三条）。"""
    labeled = _build(GuardrailCase("P1+P2", features.DECOY_NONE))
    assert questions.FAMILIES == ("clause", "count", "combo", "action")
    assert [item.family for item in labeled] == ["clause"] * 4 + ["count", "combo", "action"]


def test_clause_family_is_one_noul_per_clause() -> None:
    """输入：一个命中 P2 的 case。输出：无。

    预期行为：逐条款四条，答案就是「这一条在不在命中组合里」；题面点名条款名与判据原文。
    """
    loc = locales.load("zh")
    case = GuardrailCase("P2", features.DECOY_NONE)
    for index, clause in enumerate(features.CLAUSE_IDS):
        item = _build(case)[index]
        assert item.question["type"] == "noul"
        assert item.answer is (clause in case.clauses())
        assert item.shuffle is False
        assert str(loc["clause"][clause]["name"]) in str(item.question["instructions"])
        assert set(item.question["criteria"]) == {"true", "false"}
        # 第二种问法会把判据原文也念一遍（第一种只点条款名——判据本来就在 state 的条款表里）。
        second = _build(case, phrasing=1)[index]
        assert str(loc["clause"][clause]["rule"]) in str(second.question["instructions"])


def test_clause_answers_are_fifty_fifty() -> None:
    """输入：全空间 32 组 case。输出：无。

    预期行为：每条条款在 16 组命中组合里各出现 8 次，32 组 case 里各 16 次，所以逐条款那一族的
    True / False 正好各占一半——这是按构造就成立的均衡，不是调出来的。
    """
    seen = [0] * CLAUSE_SPAN
    for case in CASES:
        for index, item in enumerate(_build(case)[:CLAUSE_SPAN]):
            seen[index] += 1 if item.answer else 0
    assert seen == [len(CASES) // 2] * CLAUSE_SPAN


def test_count_answers_match_the_truth() -> None:
    """输入：全空间 32 组 case。输出：无。预期行为：答案档位 = 命中条数，选项表就是 0~4。"""
    for case in CASES:
        item = _build(case)[CLAUSE_SPAN]
        assert item.question["type"] == "choice"
        assert item.answer == f"c{len(case.clauses())}"
        criteria = item.question["criteria"]
        assert sorted(criteria) == [f"c{value}" for value in range(CLAUSE_SPAN + 1)]
        assert len(set(criteria.values())) == CLAUSE_SPAN + 1


def test_combo_offers_the_truth_plus_three_distractors() -> None:
    """输入：全空间 32 组 case。输出：无。

    预期行为：四个选项 = 真值 + 三个邻近干扰，描述两两不同，真值那一个就是金答案。
    """
    loc = locales.load("zh")
    for case in CASES:
        plan = features.plan_for(case, 0)
        item = _build(case)[CLAUSE_SPAN + 1]
        criteria = item.question["criteria"]
        assert len(criteria) == 4
        assert len(set(criteria.values())) == 4
        assert criteria[item.answer] == questions.combo_text(loc, plan.hits)
        for combo in plan.distractors:
            assert questions.combo_text(loc, combo) in criteria.values()


def test_action_answers_follow_the_thresholds() -> None:
    """输入：全空间 32 组 case。输出：无。预期行为：答案 = 按命中条数落到的那个处置档。"""
    loc = locales.load("zh")
    for case in CASES:
        item = _build(case)[CLAUSE_SPAN + 2]
        count = len(case.clauses())
        assert item.answer == features.action_level(count)
        criteria = item.question["criteria"]
        assert sorted(criteria) == sorted(features.ACTION_LEVELS)
        assert criteria[item.answer] == questions.action_text(loc, count)


def test_choice_options_never_duplicate() -> None:
    """输入：全空间 32 组 case。输出：无。预期行为：任何一道 choice 的选项描述都两两不同。"""
    for case in CASES:
        for item in _build(case):
            criteria = item.question["criteria"]
            assert len(set(criteria.values())) == len(criteria), item.family


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_question_is_self_contained(lang: str) -> None:
    """输入：两种语言 × 32 组 case。输出：无。预期行为：题面与选项都不留花括号。"""
    loc = locales.load(lang)
    for case in CASES:
        for item in questions.build(features.plan_for(case, 1), loc, 1):
            instructions = str(item.question["instructions"])
            assert "{" not in instructions and "}" not in instructions
            for value in item.question["criteria"].values():
                assert "{" not in str(value) and "}" not in str(value)


def test_phrasing_picks_alternate_wording() -> None:
    """输入：同一个 case 的两种问法。输出：无。预期行为：问法池里第 0 / 第 1 条文案不同。"""
    case = GuardrailCase("P1", features.DECOY_NONE)
    first = _build(case, phrasing=0)
    second = _build(case, phrasing=1)
    assert first[0].question["instructions"] != second[0].question["instructions"]
    assert first[0].answer == second[0].answer


