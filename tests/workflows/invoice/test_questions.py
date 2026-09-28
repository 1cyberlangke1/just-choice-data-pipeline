"""题族的测试：题型形状、选项、答案语义、题面可反查。"""

from __future__ import annotations

import random


import pytest

from workflows.invoice import locales, questions, render, spec as sp

LOCALES = locales.available()


def test_families_for_drops_tax_consistent_without_a_tax_row(make_spec: object) -> None:
    """输入：有/没有税额行的规格。输出：题族元组。

    预期行为：`tax_consistent` 是条件题族——没有税额行就不问（答案无从谈起）。
    """
    without = make_spec(5, tax_state="none")
    assert questions.families_for(without) == tuple(
        family for family in questions.FAMILIES if family != "tax_consistent"
    )
    assert questions.families_for(make_spec(5, tax_state="ok")) == questions.FAMILIES


@pytest.mark.parametrize("lang", LOCALES)
def test_build_returns_one_question_per_family_in_order(lang: str, make_spec: object) -> None:
    """输入：规格、语言、问法编号。输出：题目列表。

    预期行为：顺序与 `FAMILIES` 一致、每族一题、题型分别是 5 choice + 2 noul + 1 score。
    """
    spec = make_spec(11, tax_state="ok")
    items = questions.build(spec, locales.load(lang), 0, random.Random(1))
    assert [item.family for item in items] == list(questions.FAMILIES)
    assert [item.question["type"] for item in items] == ["choice"] * 5 + ["noul"] * 2 + ["score"]


@pytest.mark.parametrize("lang", LOCALES)
def test_choice_options_are_unique_and_contain_the_answer(lang: str, make_spec: object) -> None:
    """输入：一份 4 行、有税额的单据。输出：题目。

    预期行为：choice 的选项描述两两不同（否则那道题有两个正确答案）、答案在选项里。
    """
    spec = make_spec(21, lines=4, tax_state="ok")
    items = questions.build(spec, locales.load(lang), 1, random.Random(2))
    for item in items:
        if item.question["type"] != "choice":
            continue
        options = item.question["criteria"]
        assert len(options) >= 2, item.family
        assert len(set(options.values())) == len(options), item.family
        assert item.answer in options, item.family


@pytest.mark.parametrize("lang", LOCALES)
def test_choice_answers_point_at_the_rendered_field(lang: str, make_spec: object) -> None:
    """输入：一批规格。输出：题目。

    预期行为：choice 的答案永远指向本单据里真实存在的槽，且那个选项的描述
    就是该槽的渲染文本（选项名 = 槽名，`line_k` 用第 k 行的单价）。
    """
    loc = locales.load(lang)
    for seed in range(8):
        spec = make_spec(100 + seed, lines=4, tax_state="ok")
        for item in questions.build(spec, loc, 0, random.Random(seed)):
            if item.question["type"] != "choice":
                continue
            options = item.question["criteria"]
            if item.family == "line_field":
                index = int(item.answer.split("_")[1]) - 1
                assert options[item.answer] == render.format_money(
                    spec.items[index].unit, spec.case.currency, loc
                )
            else:
                assert options[item.answer] == render.field_text(spec, item.answer, loc)


def test_choice_guard_rejects_duplicate_or_missing_answers() -> None:
    """输入：两个描述相同的选项 / 答案不在选项里。输出：ValueError。

    预期行为：直接炸掉而不是产出一条「有两个正确答案」的坏数据。
    """
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("date_field", "x", {"issue": "同一天", "due": "同一天"}, "issue")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("date_field", "x", {"issue": "a"}, "due")


def test_line_option_numbering_starts_at_one() -> None:
    """输入：明细行下标（从 0 起）。输出：选项名。预期行为：`line_1` 起，与题面行号一致。"""
    assert questions.line_option(0) == "line_1"
    assert questions.line_option(3) == "line_4"


@pytest.mark.parametrize("lang", LOCALES)
def test_every_question_can_be_located_from_its_text(lang: str, make_spec: object) -> None:
    """输入：题目列表。输出：`locate` 的结果。

    预期行为：判别器只靠题面反查「问的是哪个字段」，所以每一道题的题面都必须能查回来，
    且 choice 查回来的目标与金答案一致——这条一破，判别器就是在猜。
    """
    loc = locales.load(lang)
    spec = make_spec(31, lines=3, tax_state="ok")
    for item in questions.build(spec, loc, 2, random.Random(3)):
        found = questions.locate(item.question["instructions"], loc, len(spec.items))
        assert found is not None, item.family
        family, target = found
        assert family == item.family
        if item.question["type"] == "choice":
            assert target == item.answer


def test_locate_returns_none_for_unknown_text() -> None:
    """输入：一句不是本 workflow 的题面。输出：None。预期行为：认不出来就说认不出来。"""
    for lang in LOCALES:
        assert questions.locate("今天天气怎么样？", locales.load(lang), 3) is None


@pytest.mark.parametrize("lang", LOCALES)
def test_phrasing_picks_different_wording(lang: str) -> None:
    """输入：同一个字段的三种问法编号。输出：三句题面。

    预期行为：三种问法互不相同——否则「问法多样性」这条维度就是假的。
    """
    pool = locales.load(lang)["ask"]["due"]["q"]
    assert len({locales.pick(pool, index) for index in (0, 1, 2)}) == 3


def test_build_is_deterministic(make_spec: object) -> None:
    """输入：同一份规格 + 同种子的 RNG。输出：两次题目列表。预期行为：完全相同。"""
    loc = locales.load("zh")
    spec = make_spec(41, lines=4, defects=2)
    assert questions.build(spec, loc, 0, random.Random(7)) == questions.build(
        spec, loc, 0, random.Random(7)
    )


@pytest.mark.parametrize("lang", LOCALES)
def test_noul_and_score_shapes(lang: str, make_spec: object) -> None:
    """输入：一份 3 行、2 处错的单据。输出：题目。

    预期行为：noul 的 criteria 只有 true/false、答案是布尔、不打乱；
    score 的 criteria 是语言文件里的等级表、答案是「几行对不上」、不打乱（有自然顺序）。
    """
    loc = locales.load(lang)
    spec = make_spec(51, lines=3, defects=2, tax_state="ok")
    by_family = {
        item.family: item for item in questions.build(spec, loc, 0, random.Random(4))
    }
    for family in questions.NOUL_FAMILIES:
        item = by_family[family]
        assert set(item.question["criteria"]) == {"true", "false"}
        assert isinstance(item.answer, bool)
        assert item.shuffle is False
    score = by_family["bad_lines"]
    assert score.question["criteria"] == list(loc["fam"]["bad_lines"]["levels"])
    assert score.answer == spec.bad_lines == 2
    assert score.shuffle is False


def test_noul_answers_follow_the_injected_defects(make_spec: object) -> None:
    """输入：小计对/错 × 税额对/错 四种组合。输出：noul 题的答案。

    预期行为：答案与注入的特征一一对应（这是「答案从素材算出来」的核心）。
    """
    loc = locales.load("zh")
    for subtotal_ok in (True, False):
        for tax_state in ("ok", "off"):
            spec = make_spec(61, subtotal_ok=subtotal_ok, tax_state=tax_state)
            answers = {
                item.family: item.answer
                for item in questions.build(spec, loc, 0, random.Random(5))
            }
            assert answers["subtotal_consistent"] is subtotal_ok
            assert answers["tax_consistent"] is (tax_state == "ok")


def test_family_list_shape() -> None:
    """输入：无。输出：题族常量。预期行为：5 choice + 2 noul + 1 score，顺序固定。"""
    assert questions.CHOICE_FAMILIES == (
        "date_field",
        "money_field",
        "party_field",
        "line_field",
        "id_field",
    )
    assert questions.NOUL_FAMILIES == ("subtotal_consistent", "tax_consistent")
    assert questions.SCORE_FAMILIES == ("bad_lines",)
    assert questions.FAMILIES == questions.CHOICE_FAMILIES + questions.NOUL_FAMILIES + questions.SCORE_FAMILIES
    assert sp.MONEY_SLOTS == ("subtotal", "shipping", "tax", "total")
