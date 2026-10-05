"""题族构造测试：题号、题型、选项与答案形态。"""

from __future__ import annotations

import pytest

from workflows.modelroute import features, locales, questions
from workflows.modelroute.features import TagSet

LOCALES = locales.available()

#: 一组「四族答案都不退化」的特征：选中 cinder-mini（价位 1 档、窗口富余），
#: 第一行卡在预算上。挑它是因为 `none` / 0 档那类退化答案会掩盖选项映射的错。
CASE = TagSet("code_gen", "small", "batch", "tight", "all_ok")


def _build(case: TagSet, lang: str = "zh") -> list[questions.LabeledQuestion]:
    """输入：一组特征、语言。输出：该特征下的题目列表。"""
    return questions.build(case, locales.load(lang), phrasing=0)


@pytest.mark.parametrize("lang", LOCALES)
def test_question_order_and_families(lang: str) -> None:
    """题号顺序必须按 `FAMILIES` 走，题号就是题族名。"""
    assert [item.family for item in _build(CASE, lang)] == list(questions.FAMILIES)


def test_choice_answers_are_option_names() -> None:
    """choice 的答案必须是选项名本身（编号由组装器最后一步映射）。"""
    for item in _build(CASE):
        if item.question["type"] != "choice":
            continue
        assert item.answer in item.question["criteria"], item.family


def test_choice_options_follow_the_contract_table() -> None:
    """选项表必须与 `features.CHOICE_OPTIONS` 逐项一致（含顺序）。"""
    for item in _build(CASE):
        if item.question["type"] != "choice":
            continue
        assert list(item.question["criteria"]) == list(features.CHOICE_OPTIONS[item.family])


def test_all_choice_families_may_shuffle() -> None:
    """三个 choice 题族都允许打乱：表格按固定顺序列，选项若照抄表序就带上位置捷径。"""
    by_family = {item.family: item for item in _build(CASE)}
    for family in questions.CHOICE_FAMILIES:
        assert by_family[family].shuffle is True, family
    assert not questions.ORDERED


def test_score_answer_is_the_level_index() -> None:
    """价格题的答案是等级下标，等级数与目录里的价格档数一致。"""
    item = {q.family: q for q in _build(CASE)}["cost"]
    assert item.question["type"] == "score"
    assert item.answer == CASE.label().cost
    assert 0 <= item.answer < len(item.question["criteria"])
    assert len(item.question["criteria"]) == 4


def test_noul_answers_are_bools() -> None:
    """noul 的答案只能是 JSON 布尔。"""
    for item in _build(CASE):
        if item.question["type"] == "noul":
            assert isinstance(item.answer, bool)


def test_degenerate_answers_still_map_to_options() -> None:
    """输入：没有任何候选可用的特征。输出：无。

    预期行为：退化到 `none` / 0 档时答案照样落在选项表里——这两种取值最容易在
    选项映射上写歪，单独验一条。
    """
    case = TagSet("math_reason", "huge", "interactive", "tight", "borealis_down")
    by_family = {item.family: item for item in _build(case)}
    assert by_family["route"].answer == features.NO_CANDIDATE
    assert by_family["cost"].answer == 0
    assert by_family["slack"].answer is False
