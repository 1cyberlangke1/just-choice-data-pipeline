"""题族构造测试：题号、题型、选项与答案形态。"""

from __future__ import annotations

import pytest

from workflows.shell import features, locales, questions
from workflows.shell.features import TagSet

LOCALES = locales.available()

#: 一个相容的特征组合：破坏性动作 + 生产位置 + 提权，五族答案都不会退化。
CASE = TagSet("bash", "remove_tree", "production", "elevated")


def _build(case: TagSet, lang: str = "zh"):  # noqa: ANN202 —— 测试内部小工具
    """输入：一组特征、语言。输出：该特征下的题目列表。"""
    return questions.build(case, locales.load(lang), phrasing=0)


@pytest.mark.parametrize("lang", LOCALES)
def test_question_order_and_families(lang: str) -> None:
    """题号顺序必须按 `FAMILIES` 走，题号就是题族名。"""
    assert [item.family for item in _build(CASE, lang)] == list(questions.FAMILIES)


@pytest.mark.parametrize("lang", LOCALES)
def test_every_case_asks_every_family(lang: str) -> None:
    """输入：全部相容组合。输出：无。

    预期行为：五个题族对**任何**相容组合都问得出来——本 workflow 没有条件题族，
    所以每条样本的题族数恒定（有条件的 workflow 会在这一步退化）。
    """
    cases = features.build_cases(32)
    assert cases
    for case in cases:
        assert [item.family for item in _build(case, lang)] == list(questions.FAMILIES)


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


def test_ordered_families_are_not_shuffled() -> None:
    """处置与影响半径是阶梯，必须保持顺序；目标位置允许打乱以打掉位置捷径。"""
    by_family = {item.family: item for item in _build(CASE)}
    assert by_family["verdict"].shuffle is False
    assert by_family["blast_radius"].shuffle is False
    assert by_family["target_owner"].shuffle is True
    assert list(by_family["verdict"].question["criteria"]) == ["allow", "confirm", "deny"]


def test_score_answer_is_the_level_index() -> None:
    """影响半径题的答案是等级下标，等级数与契约上限一致。"""
    item = {q.family: q for q in _build(CASE)}["blast_radius"]
    assert item.question["type"] == "score"
    assert item.answer == CASE.label().blast_radius
    assert 0 <= item.answer < len(item.question["criteria"])
    assert len(item.question["criteria"]) == 5


def test_noul_answers_are_bools() -> None:
    """noul 的答案只能是 JSON 布尔。"""
    for item in _build(CASE):
        if item.question["type"] == "noul":
            assert isinstance(item.answer, bool)


@pytest.mark.parametrize("lang", LOCALES)
def test_phrasing_index_wraps(lang: str) -> None:
    """问法编号越界要回绕，不许报错也不许给出空串。"""
    loc = locales.load(lang)
    count = len(loc["fam"]["verdict"]["q"])
    assert questions.build(CASE, loc, phrasing=count + 1)[0].question["instructions"] == (
        questions.build(CASE, loc, phrasing=1)[0].question["instructions"]
    )
