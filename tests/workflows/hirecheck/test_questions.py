"""hirecheck 题族：四个题族的形态、金答案与平衡性。"""

from __future__ import annotations

import pytest

from workflows.hirecheck import features, locales, questions
from workflows.hirecheck.features import HireCheckCase

LANGS: tuple[str, ...] = ("zh", "en")


def plan_for(value: str, phrasing: int = 0) -> features.HireCheckPlan:
    """输入：命中组合串、问法编号。输出：判定计划。"""
    return features.plan_for(
        HireCheckCase(flags=value, channel=features.CHANNEL_REFERRAL), phrasing
    )


def test_build_returns_the_families_in_order() -> None:
    """输入：一个判定计划。输出：无。

    预期行为：四条逐判据的 noul、一道 score、两道 choice，顺序按 `FAMILIES`——审计按这个顺序读
    `flag` 那一族，顺序错了整批答案会对错位。
    """
    built = questions.build(plan_for("H1+H3"), locales.load("zh"), 0)
    assert [item.family for item in built] == ["flag"] * 4 + ["grade", "combo", "act"]
    assert [item.question["type"] for item in built] == ["noul"] * 4 + [
        "score",
        "choice",
        "choice",
    ]


@pytest.mark.parametrize("lang", LANGS)
def test_flag_family_is_balanced_by_construction(lang: str) -> None:
    """输入：全空间每一条 case。输出：无。

    预期行为：每条判据在 16 组里各命中 8 次，所以这一族的 True / False 天然是 50% / 50%——不需要
    另外去凑平衡，也就不会出现「某一档一条都出不来」。
    """
    loc = locales.load(lang)
    for index, tag in enumerate(features.TAG_IDS):
        answers = [
            questions.build(plan_for(value), loc, 0)[index].answer
            for value in features.HIT_SETS
        ]
        assert answers.count(True) == answers.count(False) == 8, tag


def test_flag_question_names_the_criterion() -> None:
    """输入：一条 case 的逐判据题。输出：无。

    预期行为：题面里点名判据名与判据原文——只读 state 的考生要能自己按字面判，而不是靠猜这一族问的
    是哪条规则。
    """
    loc = locales.load("zh")
    # 第 2 种问法把判据原文也印在题面上（第 1 种只点名判据名），这里要同时验两样。
    question = questions.build(plan_for("H2"), loc, 1)[1]
    entry = loc["fact"]["H2"]
    assert str(entry["name"]) in question.question["instructions"]
    assert str(entry["rule"]) in question.question["instructions"]
    assert question.question["criteria"] == {
        "true": str(loc["fam"]["flag"]["criteria_true"]),
        "false": str(loc["fam"]["flag"]["criteria_false"]),
    }
    assert question.answer is True
    assert question.shuffle is False


def test_grade_family_is_an_ordered_score() -> None:
    """输入：五档等级各一条。输出：无。

    预期行为：选项表就是等级表、**不打乱**，答案就是等级下标（等级 1 → 下标 0）——定级是有序的多档
    判断，打乱会把「3 级和 4 级差多少」这个序丢掉。
    """
    loc = locales.load("zh")
    expected = [questions.level_option(loc, level) for level in features.LEVELS]
    for value in features.HIT_SETS:
        question = questions.build(plan_for(value), loc, 0)[4]
        level = features.level_of(features.parse_hits(value))
        assert question.question["criteria"] == expected
        assert question.answer == level - 1
        assert question.shuffle is False


def test_combo_family_offers_the_truth_plus_three_neighbours() -> None:
    """输入：一条 case 的组合题。输出：无。

    预期行为：四个选项描述两两不同、第一个就是真值，答案是 `k0`——干扰项取「与真值只差一两处」的
    邻近组合，题目才有区分度。
    """
    loc = locales.load("zh")
    plan = plan_for("H1+H3")
    question = questions.build(plan, loc, 0)[5]
    options = question.question["criteria"]
    assert len(options) == 4
    assert len(set(options.values())) == 4
    assert question.answer == "k0"
    assert options["k0"] == questions.hits_text(loc, plan.hits)


def test_combo_none_option_uses_the_local_phrase() -> None:
    """输入：一条判据都不命中的 case。输出：无。预期行为：空组合用该语言那句「一条都没命中」。"""
    loc = locales.load("zh")
    question = questions.build(plan_for(""), loc, 0)[5]
    assert question.question["criteria"]["k0"] == str(loc["fam"]["combo"]["none"])


def test_act_family_offers_every_disposition() -> None:
    """输入：五档等级各一条。输出：无。

    预期行为：四个档位全出（不多不少），答案是按等级查表得到的档位名——`act` 与 `grade` 是同一份
    等级的两个下游，档位表少一档就等于有些等级没有处置。
    """
    loc = locales.load("zh")
    for value in features.HIT_SETS:
        question = questions.build(plan_for(value), loc, 0)[6]
        options = question.question["criteria"]
        assert set(options) == set(features.ACTS)
        level = features.level_of(features.parse_hits(value))
        assert question.answer == features.act_for(level)
        assert options[question.answer] == questions.act_option(loc, features.act_for(level))


def test_hits_text_joins_the_names_it_is_given() -> None:
    """输入：一组判据。输出：无。

    预期行为：按**传进来的顺序**取判据名拼起来（调用方给的都是规范顺序）；空组合给那一句话。
    """
    loc = locales.load("zh")
    assert questions.hits_text(loc, ("H1", "H3")) == (
        f"{loc['fact']['H1']['name']} + {loc['fact']['H3']['name']}"
    )
    assert questions.hits_text(loc, ("H3", "H1")) == (
        f"{loc['fact']['H3']['name']} + {loc['fact']['H1']['name']}"
    )
    assert questions.hits_text(loc, ()) == str(loc["fam"]["combo"]["none"])
    assert " + " in questions.hits_text(loc, ("H1", "H3"))


def test_choice_rejects_duplicate_options_and_unknown_answers() -> None:
    """输入：选项描述重复 / 答案不在选项里。输出：无。预期行为：两种都当场报错（构造 bug 不许静默）。"""
    with pytest.raises(ValueError, match="重复"):
        questions._choice("combo", "题面", {"a": "一样", "b": "一样"}, "a")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("combo", "题面", {"a": "甲", "b": "乙"}, "c")


def test_score_rejects_a_wrong_table_and_an_out_of_range_answer() -> None:
    """输入：档数不对的等级表 / 越界的答案下标。输出：无。预期行为：两种都当场报错。"""
    with pytest.raises(ValueError, match="等级表"):
        questions._score("grade", "题面", ["一档", "二档"], 0)
    with pytest.raises(ValueError, match="越界"):
        questions._score("grade", "题面", [f"L{level}" for level in features.LEVELS], 9)
