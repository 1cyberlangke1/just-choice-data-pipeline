"""题族测试：四族题的形状、答案与自足性。"""

from __future__ import annotations

import pytest

from workflows.adblock import features, locales, questions
from workflows.adblock.features import AdBlockCase

#: 一个普通 case：命中两条，元素长在文章正文页上。
CASE = AdBlockCase("A1+A3", features.SURFACE_ARTICLE)


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


def test_phrasing_picks_alternate_wording() -> None:
    """输入：同一个 case 的两种问法。输出：无。预期行为：题面文字不同、答案相同。"""
    loc = locales.load("zh")
    plan = features.plan_for(CASE, 0)
    first = questions.build(plan, loc, 0)[0]
    second = questions.build(plan, loc, 1)[0]
    assert first.question["instructions"] != second.question["instructions"]
    assert first.answer == second.answer
