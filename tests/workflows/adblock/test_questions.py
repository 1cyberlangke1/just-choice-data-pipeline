"""题族测试：四族题的形状、答案与自足性。"""

from __future__ import annotations

import pytest

from workflows.adblock import features, locales, questions
from workflows.adblock.features import AdBlockCase

#: 一个普通 case：命中两条，元素长在文章正文页上。
CASE = AdBlockCase("A1+A3", features.SURFACE_ARTICLE)


def test_family_order_matches_the_declared_families() -> None:
    """输入：无。输出：无。

    预期行为：`build` 出的题族**首次出现顺序**与 `FAMILIES` 一致（逐判据那一族出四道，所以比
    的是去重之后的顺序）。
    """
    out = questions.build(features.plan_for(CASE, 0), locales.load("zh"), 0)
    assert list(dict.fromkeys(item.family for item in out)) == list(questions.FAMILIES)
    assert questions.FAMILIES == ("flag", "count", "combo", "act")


def test_flag_family_is_one_noul_per_tag() -> None:
    """输入：无。输出：无。预期行为：四条 noul，一条判据一道，答案与真值一致。"""
    plan = features.plan_for(CASE, 0)
    out = questions.build(plan, locales.load("zh"), 0)
    flags = [item for item in out if item.family == "flag"]
    assert len(flags) == len(features.TAG_IDS)
    for tag, item in zip(features.TAG_IDS, flags, strict=True):
        assert item.question["type"] == "noul"
        assert item.answer is (tag in plan.hits)


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
    assert totals["A1"] == 24


def test_count_answers_match_the_truth() -> None:
    """输入：全空间 48 组 case。输出：无。预期行为：`count` 的答案是命中条数那一档。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in questions.build(plan, loc, 0) if q.family == "count")
        assert item.answer == f"c{len(plan.hits)}"
        assert item.question["criteria"][item.answer] == questions.count_text(loc, len(plan.hits))


def test_combo_offers_the_truth_plus_three_distractors() -> None:
    """输入：全空间 48 组 case。输出：无。

    预期行为：四个选项、答案是 `k0`（真值），另外三个是邻近干扰组合。
    """
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in questions.build(plan, loc, 0) if q.family == "combo")
        assert len(item.question["criteria"]) == 4
        assert item.answer == "k0"
        assert item.question["criteria"]["k0"] == questions.combo_text(loc, plan.hits)


def test_act_answers_follow_the_thresholds() -> None:
    """输入：全空间 48 组 case。输出：无。预期行为：`act` 的答案是阈值算出来的那一档。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in questions.build(plan, loc, 0) if q.family == "act")
        assert item.answer == features.act_level(len(plan.hits))
        assert item.question["criteria"][item.answer] == questions.act_text(loc, len(plan.hits))


def test_choice_options_never_duplicate() -> None:
    """输入：全空间 48 组 case × 两种语言。输出：无。

    预期行为：三族 choice 的选项描述两两不同——两句话一样，考生没法分辨，是构造 bug。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        for case in features.build_cases(512):
            for item in questions.build(features.plan_for(case, 0), loc, 0):
                if item.question["type"] != "choice":
                    continue
                values = [str(value) for value in item.question["criteria"].values()]
                assert len(set(values)) == len(values), f"{lang}/{case.describe()}/{item.family}"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_question_is_self_contained(lang: str) -> None:
    """输入：一种语言 × 第二种问法。输出：无。

    预期行为：题面里带上要问的那条判据的名字与原文——只读 state 的考生要能自己按字面判。
    （用第二种问法：第一种是短问法，只点名判据，不重述判据原文。）
    """
    loc = locales.load(lang)
    for tag, item in zip(
        features.TAG_IDS, questions.build(features.plan_for(CASE, 0), loc, 1), strict=False
    ):
        entry = loc["tag"][tag]
        instructions = str(item.question["instructions"])
        assert str(entry["name"]) in instructions
        assert str(entry["rule"]) in instructions


def test_phrasing_picks_alternate_wording() -> None:
    """输入：同一个 case 的两种问法。输出：无。预期行为：题面文字不同、答案相同。"""
    loc = locales.load("zh")
    plan = features.plan_for(CASE, 0)
    first = questions.build(plan, loc, 0)[0]
    second = questions.build(plan, loc, 1)[0]
    assert first.question["instructions"] != second.question["instructions"]
    assert first.answer == second.answer


def test_bad_answer_names_are_rejected() -> None:
    """输入：不在选项里的答案名。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("count", "题面", {"c0": "0 条"}, "c9")


def test_duplicate_option_descriptions_are_rejected() -> None:
    """输入：两个描述一样的选项。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("count", "题面", {"c0": "0 条", "c1": "0 条"}, "c0")

