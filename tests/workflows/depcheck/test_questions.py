"""题族测试：四族题的形状、答案与自足性（含 `score` 那一族的序）。"""

from __future__ import annotations

import pytest

from workflows.depcheck import features, locales, questions
from workflows.depcheck.features import DepCheckCase

#: 一个普通 case：取到两条、这个包来自 npm。
CASE = DepCheckCase("D1+D3", features.ECO_NPM)


def test_family_order_matches_the_declared_families() -> None:
    """输入：无。输出：无。

    预期行为：`build` 出的题族**首次出现顺序**与 `FAMILIES` 一致（逐标记那一族出四道，所以比的是
    去重之后的顺序）。
    """
    out = questions.build(features.plan_for(CASE, 0), locales.load("zh"), 0)
    assert list(dict.fromkeys(item.family for item in out)) == list(questions.FAMILIES)
    assert questions.FAMILIES == ("flag", "grade", "combo", "act")


def test_flag_answers_are_fifty_fifty() -> None:
    """输入：全空间 48 组 case。输出：无。预期行为：每条判据的 True / False 各占一半。"""
    loc = locales.load("zh")
    totals = dict.fromkeys(features.TAG_IDS, 0)
    for case in features.build_cases(512):
        out = questions.build(features.plan_for(case, 0), loc, 0)
        for tag, item in zip(features.TAG_IDS, out, strict=False):
            if item.answer:
                totals[tag] += 1
    assert len(set(totals.values())) == 1, totals
    assert totals["D1"] == len(features.build_cases(512)) // 2


def test_grade_is_an_ordered_score_with_the_weight_level() -> None:
    """输入：全空间 48 组 case。输出：无。

    预期行为：`grade` 是 score 题型、五档、等级表按 `LEVELS` 的顺序、**不打乱**，答案是等级下标
    （等级 1 → 下标 0），而且选项描述与 state 里那份等级文案是同一个来源。
    """
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in questions.build(plan, loc, 0) if q.family == "grade")
        assert item.question["type"] == "score"
        assert item.shuffle is False
        levels = item.question["criteria"]
        assert isinstance(levels, list)
        assert levels == [questions.level_option(loc, level) for level in features.LEVELS]
        assert item.answer == features.level_of(plan.hits) - 1


def test_combo_offers_the_truth_plus_three_distractors() -> None:
    """输入：全空间 48 组 case。输出：无。

    预期行为：四个选项、答案是 `k0`（真值），另外三个是邻近干扰组合。
    """
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in questions.build(plan, loc, 0) if q.family == "combo")
        criteria = item.question["criteria"]
        assert isinstance(criteria, dict)
        assert len(criteria) == 4
        assert item.answer == "k0"
        assert criteria["k0"] == questions.hits_text(loc, plan.hits)


def test_act_answers_follow_the_level_table() -> None:
    """输入：全空间 48 组 case。输出：无。预期行为：`act` 的答案是等级查档位表算出来的那一档。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in questions.build(plan, loc, 0) if q.family == "act")
        assert item.answer == features.act_for(plan.level)
        criteria = item.question["criteria"]
        assert isinstance(criteria, dict)
        assert criteria[item.answer] == questions.act_option(loc, item.answer)
        assert sorted(criteria) == sorted(features.ACTS)


def test_hits_text_follows_the_canonical_tag_order() -> None:
    """输入：空组合、单条组合、以及全空间计划里的真值与干扰组合。输出：无。

    预期行为：选项描述按**传进来的顺序**取标记名拼，空组合用该语言里那句「一个都没命中」；而
    `plan.hits` 与三个干扰组合本来就按 `TAG_IDS` 的规范顺序排好（`features.plan_for` 保证），
    所以一组标记只有一种写法、四个选项描述两两不同（`_choice` 会当场验这一点）。
    """
    loc = locales.load("zh")
    assert questions.hits_text(loc, ()) == str(loc["fam"]["combo"]["none"])
    assert questions.hits_text(loc, ("D2",)).count(str(loc["fact"]["D2"]["name"])) == 1
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        for combo in (plan.hits, *plan.distractors):
            assert combo == tuple(tag for tag in features.TAG_IDS if tag in combo), combo


def test_choice_options_never_duplicate() -> None:
    """输入：全空间 48 组 case × 两种语言。输出：无。

    预期行为：两族 choice 的选项描述两两不同——两句话一样，考生没法分辨，是构造 bug。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        for case in features.build_cases(512):
            for item in questions.build(features.plan_for(case, 0), loc, 0):
                if item.question["type"] != "choice":
                    continue
                criteria = item.question["criteria"]
                assert isinstance(criteria, dict)
                values = [str(value) for value in criteria.values()]
                assert len(set(values)) == len(values), f"{lang}/{case.describe()}/{item.family}"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_question_is_self_contained(lang: str) -> None:
    """输入：一种语言 × 第二种问法。输出：无。

    预期行为：逐标记那一族的题面里带上要问的那条判据的名字与原文——只读 state 的考生要能自己按
    字面判。（用第二种问法：第一种是短问法，只点名标记，不重述判据原文。）
    """
    loc = locales.load(lang)
    for tag, item in zip(
        features.TAG_IDS, questions.build(features.plan_for(CASE, 0), loc, 1), strict=False
    ):
        entry = loc["fact"][tag]
        instructions = str(item.question["instructions"])
        assert str(entry["name"]) in instructions
        assert str(entry["rule"]) in instructions


def test_bad_answer_names_are_rejected() -> None:
    """输入：不在选项里的答案名。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("act", "题面", {"approve": "直接进依赖树"}, "nope")


def test_duplicate_option_descriptions_are_rejected() -> None:
    """输入：两个描述一样的选项。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice(
            "act", "题面", {"approve": "直接进依赖树", "watch": "直接进依赖树"}, "approve"
        )
