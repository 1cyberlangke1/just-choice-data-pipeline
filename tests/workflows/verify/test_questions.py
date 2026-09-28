"""完成声明核验的题族：题号、题型、选项与金答案。"""

from __future__ import annotations

from jc.workflow import LabeledQuestion
from workflows.verify import features, locales, questions
from workflows.verify.features import TagSet

CASE = TagSet("bug_fixed", "hedged", "partial_pass", "covered", "uncommitted")


def build(case: TagSet = CASE, lang: str = "zh", phrasing: int = 0) -> list[LabeledQuestion]:
    """输入：一个 case、语言、问法编号。输出：六个题族（题面 + 金答案）。"""
    return questions.build(case, locales.load(lang), phrasing)


def test_families_are_in_a_fixed_order() -> None:
    """输入：一个 case。输出：无。预期行为：题号就是题族名，顺序固定（先读声明再判）。"""
    assert [item.family for item in build()] == list(questions.FAMILIES)


def test_types_match_the_declared_groups() -> None:
    """输入：一个 case。输出：无。

    预期行为：四个 choice、一个 noul、一个 score，分组清单与实际产出的题型一一对应。
    """
    by_family = {item.family: item.question["type"] for item in build()}
    for family in questions.CHOICE_FAMILIES:
        assert by_family[family] == "choice"
    for family in questions.NOUL_FAMILIES:
        assert by_family[family] == "noul"
    for family in questions.SCORE_FAMILIES:
        assert by_family[family] == "score"
    declared = set(questions.CHOICE_FAMILIES) | set(questions.NOUL_FAMILIES) | set(questions.SCORE_FAMILIES)
    assert set(by_family) == declared


def test_choice_answers_are_option_names() -> None:
    """输入：一个 case。输出：无。

    预期行为：choice 的金答案是**选项名**（不是编号），且与判据算出来的那一档一致。
    """
    labels = CASE.label()
    for item in build():
        if item.question["type"] != "choice":
            continue
        assert item.answer in item.question["criteria"]
        assert item.answer == getattr(labels, item.family)


def test_choice_options_are_exactly_the_allowed_ones() -> None:
    """输入：一个 case。输出：无。

    预期行为：题面只收本题允许的选项——多一个选项是送分题，少一个是死选项。
    """
    for item in build():
        if item.question["type"] != "choice":
            continue
        assert set(item.question["criteria"]) == set(features.CHOICE_OPTIONS[item.family])


def test_score_answer_is_a_level_index() -> None:
    """输入：一个 case。输出：无。预期行为：score 的答案是 0~3 的档位下标，且不打乱。"""
    item = next(entry for entry in build() if entry.question["type"] == "score")
    assert item.answer == CASE.label().confidence
    assert len(item.question["criteria"]) == features.GATE_COUNT + 1
    assert isinstance(item.answer, int) and 0 <= item.answer < len(item.question["criteria"])
    assert item.shuffle is False


def test_noul_answer_is_a_boolean() -> None:
    """输入：一个 case。输出：无。预期行为：noul 的答案是布尔，选项恰好 true / false。"""
    item = next(entry for entry in build() if entry.question["type"] == "noul")
    assert item.answer is CASE.label().committed
    assert set(item.question["criteria"]) == {"true", "false"}


def test_choice_families_are_shuffled() -> None:
    """输入：一个 case。输出：无。

    预期行为：四个 choice 题族都允许打乱（没有有自然顺序的集合），防「答案总在第 n 个」。
    """
    assert questions.ORDERED == frozenset()
    for item in build():
        if item.question["type"] == "choice":
            assert item.shuffle is True


def test_phrasing_rotates_the_instructions() -> None:
    """输入：两个问法编号。输出：无。

    预期行为：每个题族至少 3 种问法，所以编号 0 与 1 必得不同的问法（题面不是复制粘贴）。
    """
    first = {item.family: item.question["instructions"] for item in build(phrasing=0)}
    second = {item.family: item.question["instructions"] for item in build(phrasing=1)}
    assert set(first) == set(second)
    for family in first:
        assert first[family] != second[family], family


def test_every_case_answers_every_family() -> None:
    """输入：一批 case。输出：无。

    预期行为：任何特征组合都出满六个题族，且 choice 的答案都落在选项域里。
    """
    for case in features.build_cases(48):
        items = build(case)
        assert [item.family for item in items] == list(questions.FAMILIES)
        for item in items:
            if item.question["type"] == "choice":
                assert item.answer in features.CHOICE_OPTIONS[item.family]