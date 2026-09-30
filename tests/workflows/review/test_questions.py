"""改动审查的题族测试：题号、题型、金答案与选项。"""

from __future__ import annotations

import pytest

from workflows.review import features, locales, questions


def all_tags() -> list[features.TagSet]:
    """输入：无。输出：全空间 108 组相容特征。"""
    return features.build_cases(2048)


def test_families_are_fixed_and_ordered() -> None:
    """输入：任意一组特征。输出：无。预期行为：题族与顺序恒定，且分组成三类题型。"""
    assert questions.FAMILIES == ("secret", "debug", "rule", "severity", "gate")
    assert set(questions.CHOICE_FAMILIES) | set(questions.NOUL_FAMILIES) | set(
        questions.SCORE_FAMILIES
    ) == set(questions.FAMILIES)
    assert questions.ORDERED == frozenset()
    for tags in all_tags():
        assert questions.families_for(tags) == questions.FAMILIES


def test_question_types_follow_the_family_groups() -> None:
    """输入：全空间组合。输出：无。预期行为：每族出对的题型，顺序与 `FAMILIES` 一致。"""
    loc = locales.load("zh")
    for tags in all_tags():
        built = questions.build(tags, loc, 0)
        assert [item.family for item in built] == list(questions.FAMILIES)
        for item in built:
            if item.family in questions.CHOICE_FAMILIES:
                assert item.question["type"] == "choice"
            elif item.family in questions.NOUL_FAMILIES:
                assert item.question["type"] == "noul"
            else:
                assert item.question["type"] == "score"


def test_answers_match_the_labels() -> None:
    """输入：全空间组合。输出：无。

    预期行为：noul 的答案是「该维是不是 yes」、choice 的答案是选项名、score 的答案是档位
    下标——三者的形态在契约里完全不同，混一处组装器就崩。
    """
    loc = locales.load("zh")
    for tags in all_tags():
        labels = tags.label()
        for item in questions.build(tags, loc, 0):
            if item.family in questions.NOUL_FAMILIES:
                assert item.answer == (getattr(labels, item.family) == "yes")
                assert isinstance(item.answer, bool)
            elif item.family in questions.CHOICE_FAMILIES:
                assert item.answer == getattr(labels, item.family)
                assert isinstance(item.answer, str)
            else:
                assert item.answer == labels.severity
                assert isinstance(item.answer, int)


def test_choice_options_are_exactly_the_allowed_set() -> None:
    """输入：全空间组合。输出：无。

    预期行为：choice 的 `criteria` 就是该族的允许选项集合，一个不多一个不少——
    多收一个选项，答案域就与标签域对不上了。
    """
    loc = locales.load("zh")
    for tags in all_tags():
        for item in questions.build(tags, loc, 0):
            if item.family in questions.CHOICE_FAMILIES:
                assert tuple(item.question["criteria"]) == features.CHOICE_OPTIONS[item.family]
            elif item.family in questions.NOUL_FAMILIES:
                assert tuple(item.question["criteria"]) == ("true", "false")
            else:
                assert len(item.question["criteria"]) == features.SEVERITY_LEVELS


def test_only_choice_families_allow_shuffling() -> None:
    """输入：全空间组合。输出：无。

    预期行为：只有 choice 允许打乱——score 是阶梯、noul 的 true/false 也有自然顺序。
    """
    loc = locales.load("zh")
    for tags in all_tags():
        for item in questions.build(tags, loc, 0):
            assert item.shuffle is (item.family in questions.CHOICE_FAMILIES)


def test_phrasing_rotates_the_instruction() -> None:
    """输入：同一条样本的三种问法编号。输出：无。预期行为：问法各不相同、都是非空字符串。"""
    tags = all_tags()[0]
    loc = locales.load("zh")
    seen: set[tuple[str, str]] = set()
    for phrasing in (0, 1, 2):
        for item in questions.build(tags, loc, phrasing):
            seen.add((item.family, str(item.question["instructions"])))
    for family in questions.FAMILIES:
        texts = {text for name, text in seen if name == family}
        assert len(texts) == 3, family
        assert all(texts)


def test_instructions_are_never_empty() -> None:
    """输入：全空间组合 × 两种语言。输出：无。预期行为：题面文案都不为空。"""
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        for tags in all_tags():
            for item in questions.build(tags, loc, 1):
                assert str(item.question["instructions"]).strip()


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_noul_criteria_text_differs_between_true_and_false(lang: str) -> None:
    """输入：一个语言。输出：无。预期行为：true / false 两段文案不同（否则题目没法答）。"""
    loc = locales.load(lang)
    for family in questions.NOUL_FAMILIES:
        criteria = loc["fam"][family]["criteria"]
        assert str(criteria["true"]) != str(criteria["false"])
