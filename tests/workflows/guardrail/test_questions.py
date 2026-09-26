"""题族：齐全、答案与规格一致、选项合法、问法轮转。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.guardrail import locales, questions
from workflows.guardrail import spec as sp

LANGS = ("zh", "en")


def build(spec: sp.GuardSpec, lang: str, phrasing: int = 0) -> list[Any]:
    """输入：规格、语言、问法编号。输出：题目列表。预期行为：纯函数（不掷骰子）。"""
    return questions.build(spec, locales.load(lang), phrasing, random.Random(0))


def test_families_are_complete_and_in_order(make_spec: Any) -> None:
    """输入：一份规格。输出：无。预期行为：11 个题族各一题，顺序与 `FAMILIES` 一致。"""
    built = build(make_spec(), "zh")
    assert [item.family for item in built] == list(questions.FAMILIES)
    assert len(built) == 11


@pytest.mark.parametrize("lang", LANGS)
def test_answers_match_the_spec(lang: str, make_spec: Any) -> None:
    """输入：一份规格（含前科与未成年人受众）。输出：无。

    预期行为：choice 的答案下标指回规格里的选项名、noul 的布尔与规格一致、
    score 的下标就是严重度。
    """
    spec = make_spec(history="repeat", audience="minors")
    loc = locales.load(lang)
    for item in build(spec, lang):
        if item.question["type"] == "choice":
            options = item.question["criteria"]
            want = spec.action if item.family == "action" else spec.top_clause
            # 组装前的答案存的是**选项名**（组装器才把它换成下标）。
            assert item.answer == want
            assert item.answer in options
            assert options[item.answer] == loc["option"][item.family][want]
        elif item.question["type"] == "noul":
            clause = item.family.removeprefix("violates_")
            want = {
                "minor_exposure": spec.minor_exposure,
                "repeat_offender": spec.repeat_offender,
            }.get(item.family, spec.violated(clause))
            assert item.answer is want
        else:
            assert item.answer == spec.severity


def test_clause_questions_name_their_clause(make_spec: Any) -> None:
    """输入：一份规格。输出：无。预期行为：六条条款的问法与判据里都带着条款名。"""
    loc = locales.load("zh")
    for item in build(make_spec(), "zh"):
        if item.family.startswith("violates_"):
            clause = item.family.removeprefix("violates_")
            assert loc["clause"][clause] in item.question["instructions"]
            assert loc["clause"][clause] in item.question["criteria"]["true"]


def test_choice_is_shufflable_and_options_are_distinct(make_spec: Any) -> None:
    """输入：一份规格。输出：无。

    预期行为：两个 choice 题族允许打乱（选项名没有自然顺序），选项描述两两不同；
    noul 与 score 不打乱。
    """
    for item in build(make_spec(), "zh"):
        if item.question["type"] == "choice":
            assert item.shuffle is True
            options = item.question["criteria"]
            assert len(set(options.values())) == len(options)
        else:
            assert item.shuffle is False


def test_phrasing_pool_rotates(make_spec: Any) -> None:
    """输入：同一份规格、三个问法编号。输出：无。

    预期行为：问法池按编号取，同一题族的三档问法互不相同。
    """
    spec = make_spec()
    texts = {
        index: build(spec, "zh", index)[0].question["instructions"] for index in (0, 1, 2)
    }
    assert len(set(texts.values())) == 3


def test_duplicate_option_descriptions_are_rejected() -> None:
    """输入：两个描述相同的选项。输出：无。预期行为：构造期报错（考生没法分辨）。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("action", "题面", {"a": "一样", "b": "一样"}, "a")


def test_unknown_answer_is_rejected() -> None:
    """输入：答案不在选项里。输出：无。预期行为：构造期报错。"""
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("action", "题面", {"a": "甲", "b": "乙"}, "c")


def test_noul_criteria_are_exactly_true_and_false(make_spec: Any) -> None:
    """输入：一份规格。输出：无。预期行为：noul 的判据恰好 true / false 两行。"""
    for item in build(make_spec(), "zh"):
        if item.question["type"] == "noul":
            assert set(item.question["criteria"]) == {"true", "false"}
