"""来源选择的题族测试：题型、选项、金答案与打乱规则。"""

from __future__ import annotations

from workflows.sourceselect import features, locales, questions
from workflows.sourceselect.features import SourceCase

CASE = SourceCase(need="freshness", size=5, shape="tied", near_miss=0)
PLAN = features.plan_for(CASE, 0)
LABELS = features.label_for(PLAN)


def built() -> dict[str, object]:
    """输入：无。输出：{题族名: 题目}。"""
    return {item.family: item for item in questions.build(PLAN, locales.load("zh"), 0)}


def test_families_are_fixed_and_typed() -> None:
    """输入：一个计划。输出：无。

    预期行为：五个题族恒出；choice / noul / score 三类的划分与题面 `type` 对得上。
    """
    by_family = built()
    assert [item.family for item in questions.build(PLAN, locales.load("zh"), 0)] == list(
        questions.FAMILIES
    )
    for family in questions.CHOICE_FAMILIES:
        assert by_family[family].question["type"] == "choice"
    for family in questions.NOUL_FAMILIES:
        assert by_family[family].question["type"] == "noul"
    for family in questions.SCORE_FAMILIES:
        assert by_family[family].question["type"] == "score"


def test_answers_match_the_labels() -> None:
    """输入：一个计划。输出：无。预期行为：每族金答案等于特征层算出来的那一格。"""
    by_family = built()
    assert by_family["need"].answer == LABELS.need
    assert by_family["pick"].answer == LABELS.pick_key
    assert by_family["enough"].answer is LABELS.enough
    assert by_family["cover"].answer == LABELS.cover
    assert by_family["near_miss"].answer is LABELS.near_miss


def test_pick_options_track_the_table_length() -> None:
    """输入：表长 5 的计划。输出：无。

    预期行为：`pick` 的选项名是 `s1`…`s5` 加一个 `none`，顺序与表序一致——
    答案「第几行」就是在这个顺序里的行号。
    """
    criteria = built()["pick"].question["criteria"]
    assert list(criteria) == ["s1", "s2", "s3", "s4", "s5", features.NO_PICK]
    assert str(criteria["s1"]).strip()
    assert str(criteria[features.NO_PICK]).strip()


def test_pick_options_shrink_with_the_table() -> None:
    """输入：表长 3 的计划。输出：无。预期行为：选项跟着表长走，不会多出第 4 行。"""
    small = features.plan_for(SourceCase(need="coverage", size=3, shape="none", near_miss=0), 0)
    options = questions.pick_options(small.size, locales.load("zh"))
    assert list(options) == ["s1", "s2", "s3", features.NO_PICK]


def test_need_options_are_the_three_requirements() -> None:
    """输入：一个计划。输出：无。预期行为：`need` 的选项名就是三个首要要求。"""
    assert list(built()["need"].question["criteria"]) == list(features.NEEDS)


def test_only_the_score_family_is_unshuffled() -> None:
    """输入：一个计划。输出：无。预期行为：有序阶梯不许打乱，其余 choice 都允许打乱。"""
    by_family = built()
    for family in questions.FAMILIES:
        if family in questions.SCORE_FAMILIES or family in questions.NOUL_FAMILIES:
            assert by_family[family].shuffle is False
        else:
            assert by_family[family].shuffle is True


def test_cover_has_three_levels() -> None:
    """输入：一个计划。输出：无。预期行为：覆盖档正好三档，与 `COVER_LEVELS` 对齐。"""
    assert len(built()["cover"].question["criteria"]) == questions.COVER_LEVELS
