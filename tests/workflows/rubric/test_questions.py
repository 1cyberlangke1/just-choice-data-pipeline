"""打分与评分的题族测试：题号、题型、金答案与选项。"""

from __future__ import annotations

import pytest

from workflows.rubric import features, locales, questions


def all_tags() -> list[features.TagSet]:
    """输入：无。输出：全空间 48 组相容特征。"""
    return features.build_cases(2048)


def base_of(qid: str) -> str:
    """输入：题号。输出：题族名（`met` 一族逐条问，题号是 `met_1`…`met_5`）。"""
    return "".join(ch for ch in qid if not ch.isdigit()).rstrip("_")


def test_families_are_fixed_and_ordered() -> None:
    """输入：无。输出：无。预期行为：题族与顺序恒定，且分组成三类题型。"""
    assert questions.FAMILIES == ("met", "band", "gate", "top")
    assert set(questions.CHOICE_FAMILIES) | set(questions.NOUL_FAMILIES) | set(
        questions.SCORE_FAMILIES
    ) == set(questions.FAMILIES)
    assert questions.ORDERED == frozenset({"band"})
    assert questions.ALWAYS_FAMILIES == ("met", "band", "gate")


def test_question_types_follow_the_family_groups() -> None:
    """输入：全空间组合。输出：无。预期行为：每族出对的题型，顺序与 `FAMILIES` 一致。"""
    loc = locales.load("zh")
    for tags in all_tags():
        built = questions.build(tags, loc, 0)
        # `met` 一族展开成五道题，所以按「第一次出现的顺序」取题族名再比
        order = list(dict.fromkeys(base_of(item.family) for item in built))
        assert order == list(questions.families_for(tags))
        for item in built:
            if base_of(item.family) in questions.CHOICE_FAMILIES:
                assert item.question["type"] == "choice"
            elif base_of(item.family) in questions.NOUL_FAMILIES:
                assert item.question["type"] == "noul"
            else:
                assert item.question["type"] == "score"


def test_met_family_asks_every_criterion_in_score_sheet_order() -> None:
    """输入：全空间组合。输出：无。

    预期行为：`met` 一族出五道 noul，题号 `met_1`…`met_5` 与评分单的行序一一对应，
    答案是「这条判据达标没有」。
    """
    loc = locales.load("zh")
    for tags in all_tags():
        labels = tags.label()
        built = [item for item in questions.build(tags, loc, 0) if item.family.startswith("met_")]
        assert [item.family for item in built] == [f"met_{index}" for index in range(1, 6)]
        assert [item.answer for item in built] == list(labels.met)
        for index, item in enumerate(built, start=1):
            assert str(loc["criterion"][features.CRITERIA[index - 1]]) in str(
                item.question["instructions"]
            )


def test_answers_match_the_labels() -> None:
    """输入：全空间组合。输出：无。

    预期行为：noul 的答案是布尔、choice 的答案是选项名、score 的答案是档位下标——
    三者的形态在契约里完全不同，混一处组装器就崩。
    """
    loc = locales.load("zh")
    for tags in all_tags():
        labels = tags.label()
        for item in questions.build(tags, loc, 0):
            family = base_of(item.family)
            if family in questions.NOUL_FAMILIES:
                assert isinstance(item.answer, bool)
            elif family in questions.CHOICE_FAMILIES:
                assert item.answer == (labels.gate if family == "gate" else labels.top)
                assert isinstance(item.answer, str)
            else:
                assert item.answer == features.BANDS.index(labels.band)
                assert isinstance(item.answer, int)


def test_choice_options_are_exactly_the_allowed_set() -> None:
    """输入：全空间组合。输出：无。

    预期行为：choice 的 `criteria` 就是该族的允许选项集合，一个不多一个不少——
    多收一个选项，答案域就与标签域对不上了。
    """
    loc = locales.load("zh")
    for tags in all_tags():
        for item in questions.build(tags, loc, 0):
            family = base_of(item.family)
            if family in questions.CHOICE_FAMILIES:
                assert tuple(item.question["criteria"]) == features.CHOICE_OPTIONS[family]
            elif family in questions.NOUL_FAMILIES:
                assert tuple(item.question["criteria"]) == ("true", "false")
            else:
                assert len(item.question["criteria"]) == len(features.BANDS)


def test_only_choice_families_allow_shuffling() -> None:
    """输入：全空间组合。输出：无。

    预期行为：`band` 是有序阶梯、`met` 的 true/false 也有自然顺序，都不许打乱；
    两个 choice 题族允许打乱（防「答案总在第 n 个」这种位置捷径）。
    """
    loc = locales.load("zh")
    for tags in all_tags():
        for item in questions.build(tags, loc, 0):
            assert item.shuffle is (base_of(item.family) in questions.CHOICE_FAMILIES)
            if base_of(item.family) in questions.ORDERED:
                assert not item.shuffle


def test_top_is_absent_when_there_is_no_unique_answer() -> None:
    """输入：两组没有唯一解的特征。输出：无。

    预期行为：没达标的判据里权重最大的那条并列、或者五条全达标时，`top` 这一族**不出**
    ——硬出一题就是一条错数据。
    """
    loc = locales.load("zh")
    tied = features.TagSet(
        placeholders="yes", citations="no", length="short", form="prose", data="none"
    )
    assert features.top_for(tied) is None
    assert questions.families_for(tied) == questions.ALWAYS_FAMILIES
    assert "top" not in {item.family for item in questions.build(tied, loc, 0)}
    all_met = features.TagSet(
        placeholders="no", citations="yes", length="short", form="bulleted", data="numbers"
    )
    assert features.top_for(all_met) is None
    assert "top" not in {item.family for item in questions.build(all_met, loc, 0)}
    # 有唯一解时它一定出
    unique = features.TagSet(
        placeholders="no", citations="no", length="short", form="prose", data="none"
    )
    assert features.top_for(unique) == "grounded"
    assert "top" in {item.family for item in questions.build(unique, loc, 0)}


def test_phrasing_rotates_the_instruction() -> None:
    """输入：同一条样本的三种问法编号。输出：无。预期行为：问法各不相同、都是非空字符串。"""
    tags = all_tags()[0]
    loc = locales.load("zh")
    seen: dict[str, set[str]] = {}
    for phrasing in (0, 1, 2):
        for item in questions.build(tags, loc, phrasing):
            seen.setdefault(item.family, set()).add(str(item.question["instructions"]))
    for family, texts in seen.items():
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
    """输入：一个语言。输出：无。预期行为：true / false 两段文案不一样。"""
    loc = locales.load(lang)
    for family in questions.NOUL_FAMILIES:
        criteria = loc["fam"][family]["criteria"]
        assert str(criteria["true"]).strip() != str(criteria["false"]).strip()


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_band_levels_state_every_cut_point(lang: str) -> None:
    """输入：一个语言。输出：无。

    预期行为：三档各一条，且把档位表的四个切点（5 / 6 / 7 / 8）都写进文案里——
    题面自己说清「多少分算哪一档」，读者不用回头翻档位表。
    """
    loc = locales.load(lang)
    levels = [str(item) for item in loc["fam"]["band"]["levels"]]
    assert len(levels) == len(features.BANDS)
    text = " ".join(levels)
    for cut in (5, 6, 7, 8):
        assert str(cut) in text
