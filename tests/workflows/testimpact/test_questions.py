"""testimpact 的题族测试：四族的形状、选项与金答案。"""

from __future__ import annotations

from itertools import combinations

import pytest

from workflows.testimpact import features, locales, questions
from workflows.testimpact.features import ImpactCase

LANGS = ("zh", "en")


def test_family_order_matches_the_declared_families() -> None:
    """输入：一份题目列表。输出：无。预期行为：题族顺序与 `FAMILIES` 一致（审计按顺序比答案）。"""
    plan = features.plan_for(ImpactCase(touch="P1+P2", flavor="fix"), 0)
    built = questions.build(plan, locales.load("zh"), 0)
    assert [item.family for item in built] == [
        "flag",
        "flag",
        "flag",
        "count",
        "which",
        "route",
    ]
    assert tuple(dict.fromkeys(item.family for item in built)) == questions.FAMILIES


def test_flag_questions_follow_the_suites_and_the_labels() -> None:
    """输入：16 组触及组合 × 两种语言。输出：无。

    预期行为：逐套件那一族按 `SUITES` 顺序出题，答案与 `label_for` 算出来的「需不需要跑」逐条
    一致；noul 的判据两句都在、且不许打乱。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in LANGS:
        for touch in features.TOUCH_SETS:
            try:
                loc = locales.load(lang)
                case = ImpactCase(touch=touch, flavor="fix")
                plan = features.plan_for(case, 0)
                labels = features.label_for(plan)
                flags = [item for item in questions.build(plan, loc, 0) if item.family == "flag"]
                assert len(flags) == len(features.SUITES)
                for suite, item in zip(features.SUITES, flags, strict=True):
                    assert item.question["type"] == "noul"
                    assert item.answer is labels.needs(suite)
                    assert set(item.question["criteria"]) == {"true", "false"}
                    assert item.shuffle is False
                    assert loc["suite"][suite]["name"] in str(item.question["instructions"])
            except AssertionError as exc:
                bad.append(f"{(lang, touch)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_count_which_and_route_options() -> None:
    """输入：16 组触及组合 × 两种语言。输出：无。

    预期行为：`count` 四档全出、答案是需要的套数；`which` 是四选一（真值排第一，四个描述两两
    不同）；`route` 四档全出、答案与档位一致。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in LANGS:
        for touch in features.TOUCH_SETS:
            try:
                loc = locales.load(lang)
                plan = features.plan_for(ImpactCase(touch=touch, flavor="fix"), 1)
                labels = features.label_for(plan)
                by_family = {item.family: item for item in questions.build(plan, loc, 1)}
                count = by_family["count"]
                assert list(count.question["criteria"]) == ["c0", "c1", "c2", "c3"]
                assert count.answer == f"c{len(labels.need)}"
                assert questions.count_text(loc, len(labels.need)) in count.question["criteria"].values()
                which = by_family["which"]
                criteria = which.question["criteria"]
                assert list(criteria) == ["k0", "k1", "k2", "k3"]
                assert which.answer == "k0"
                assert criteria["k0"] == questions.need_text(loc, plan.need)
                assert len(set(criteria.values())) == 4
                route = by_family["route"]
                assert list(route.question["criteria"]) == list(features.LEVELS)
                assert route.answer == labels.level
            except AssertionError as exc:
                bad.append(f"{(lang, touch)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_need_text_covers_all_eight_sets_distinctly() -> None:
    """输入：三套测试的全部子集。输出：无。

    预期行为：八个集合的描述两两不同——重复的话 `_choice` 会当场报错，题目也没法答。
    """
    for lang in LANGS:
        loc = locales.load(lang)
        texts = [
            questions.need_text(loc, combo)
            for size in range(len(features.SUITES) + 1)
            for combo in combinations(features.SUITES, size)
        ]
        assert len(texts) == 8
        assert len(set(texts)) == 8


def test_choice_rejects_duplicate_options_and_unknown_answers() -> None:
    """输入：两份坏选项表。输出：无。预期行为：重复描述、答案不在选项里都当场报错。"""
    with pytest.raises(ValueError, match="重复"):
        questions._choice("count", "题面", {"a": "一样", "b": "一样"}, "a")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("count", "题面", {"a": "甲"}, "b")


def test_phrasing_index_wraps_around() -> None:
    """输入：问法编号 0 与 1。输出：无。预期行为：两种问法的题面不同、编号越界会回绕。"""
    loc = locales.load("zh")
    plan = features.plan_for(ImpactCase(touch="P3", flavor="feat"), 0)
    first = questions.build(plan, loc, 0)
    second = questions.build(plan, loc, 1)
    assert first[0].question["instructions"] != second[0].question["instructions"]
    assert locales.pick(["甲", "乙"], 2) == "甲"
