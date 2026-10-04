"""receiptcat 题族的单元测试：四个题族的选项、答案与问法轮转。

输入：`workflows.receiptcat.questions` 与 `features`。输出：无（断言）。
预期行为：每一族的答案都必须**从注入特征推出来**（不碰文本、不碰模型），题型与选项形态
符合 `jc.contract` 的三原语约定。
"""

from __future__ import annotations

from typing import Final

import pytest

from workflows.receiptcat import features, locales, questions, render
from workflows.receiptcat.features import ReceiptCase

#: 两种语言，参数化用。
LANGS: Final[tuple[str, ...]] = ("zh", "en")

#: 覆盖三档金额、两个诱饵取值的 case 表。
CASES: Final[tuple[ReceiptCase, ...]] = (
    ReceiptCase("C1", "under", "none"),
    ReceiptCase("C2", "mild", "none"),
    ReceiptCase("C3", "heavy", "none"),
    ReceiptCase("C4", "heavy", "near"),
)


def test_every_case_gets_all_four_families() -> None:
    """输入：一个 case、一种语言。输出：无。

    预期行为：四个题族**每一个 case 上都能出题**——出题条件苛刻的题族（比如只在某几档才有
    唯一解的反向问法）在 10 条的小批量里会一条都出不来，审计当场报「题族一条都没出」。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in LANGS:
        for case in CASES:
            try:
                items = questions.build(case, locales.load(lang), 0)
                assert tuple(item.family for item in items) == questions.FAMILIES
            except AssertionError as exc:
                bad.append(f"{(lang, case)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_question_types() -> None:
    """输入：一个 case、一种语言。输出：无。预期行为：cat 是 choice、tier 是 score、其余 noul。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in LANGS:
        for case in CASES:
            try:
                kinds = {item.family: item.question["type"] for item in questions.build(case, locales.load(lang), 0)}
                assert kinds == {"cat": "choice", "over": "noul", "tier": "score", "near": "noul"}
            except AssertionError as exc:
                bad.append(f"{(lang, case)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_cat_options_are_the_four_categories() -> None:
    """输入：一个 case、一种语言。输出：无。

    预期行为：选项就是四个类别名（描述是各自的关键词清单，两两不同），答案是真值那一类——
    选项描述重复时 `_choice` 会当场报错，所以这里顺带把「四个类别的关键词清单互不相同」
    也钉住了。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in LANGS:
        for case in CASES:
            try:
                loc = locales.load(lang)
                item = questions.build(case, loc, 0)[0]
                criteria = item.question["criteria"]
                assert isinstance(criteria, dict)
                assert list(criteria) == [render.category_name(loc, cid) for cid in features.CATEGORY_IDS]
                assert item.answer == render.category_name(loc, case.category)
                assert list(criteria.values()) == [questions.category_text(loc, cid) for cid in features.CATEGORY_IDS]
                assert len(set(criteria.values())) == 4
            except AssertionError as exc:
                bad.append(f"{(lang, case)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_over_and_near_answers() -> None:
    """输入：一个 case、一种语言。输出：无。

    预期行为：`over` 的答案就是「档位不是 under」，`near` 的答案就是「计划里要诱饵」——
    两族共享同一次注入，但问的是两件事（金额超没超 / 有没有近似说法）。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in LANGS:
        for case in CASES:
            try:
                labels = features.label_for(case)
                items = {item.family: item for item in questions.build(case, locales.load(lang), 0)}
                assert items["over"].answer is labels.over
                assert items["near"].answer is labels.near
                for family in ("over", "near"):
                    criteria = items[family].question["criteria"]
                    assert isinstance(criteria, dict)
                    assert sorted(criteria) == ["false", "true"]
                    assert items[family].shuffle is False
            except AssertionError as exc:
                bad.append(f"{(lang, case)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_tier_is_a_score_over_the_tier_table() -> None:
    """输入：一个 case、一种语言。输出：无。

    预期行为：`tier` 的等级数组就是档位表四档的名字（顺序 = `TIERS`），答案是这个数组的
    下标——score 的答案本来就是一个下标，而档位**本来就是有序的**。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in LANGS:
        for case in CASES:
            try:
                loc = locales.load(lang)
                item = {entry.family: entry for entry in questions.build(case, loc, 0)}["tier"]
                criteria = item.question["criteria"]
                assert criteria == questions.tier_levels(loc)
                assert len(criteria) == len(features.TIERS)
                assert item.answer == questions.tier_index(features.label_for(case).tier)
                assert 0 <= int(item.answer) < len(features.TIERS)
                assert item.shuffle is False
            except AssertionError as exc:
                bad.append(f"{(lang, case)!r}: {exc}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("lang", LANGS)
def test_phrasing_rotates_the_wording(lang: str) -> None:
    """输入：两种问法编号。输出：无。

    预期行为：问法编号决定题面用哪一条模板，越界回绕——问法不是 case 维度，所以它不能进
    `stats_dimensions`（否则同一组合会被算成两个组合）。
    """
    loc = locales.load(lang)
    case = ReceiptCase("C1", "under", "none")
    first = questions.build(case, loc, 0)
    second = questions.build(case, loc, 1)
    assert first[0].question["instructions"] != second[0].question["instructions"]
    assert questions.build(case, loc, 2)[0].question["instructions"] == first[0].question["instructions"]
    assert first[1].answer is second[1].answer is False


def test_unknown_family_template_reports_the_family() -> None:
    """输入：一个缺问法模板的语言字典。输出：无。预期行为：报错点名字段，不静默出空题面。"""
    loc = dict(locales.load("zh"))
    loc["fam"] = {}
    with pytest.raises(KeyError):
        questions._instructions(loc, "cat", 0)


def test_choice_rejects_duplicate_option_descriptions() -> None:
    """输入：两个描述相同的选项。输出：无。预期行为：当场报错（考生没法分辨）。"""
    with pytest.raises(ValueError, match="重复"):
        questions._choice("cat", "q", {"a": "同", "b": "同"}, "a")


def test_choice_rejects_an_answer_outside_the_options() -> None:
    """输入：答案不在选项里。输出：无。预期行为：当场报错（组装器会拿不到下标）。"""
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("cat", "q", {"a": "甲", "b": "乙"}, "c")
