"""题族测试：四族题的形状、答案与自足性。"""

from __future__ import annotations

import pytest

from workflows.segmentation import features, locales, questions
from workflows.segmentation.features import SegmentationCase

#: 一个普通 case：两类、三页（第一类两页），来源是扫描件。
CASE = SegmentationCase("form+receipt", "3", features.SOURCE_SCAN)


def test_family_order_matches_the_declared_families() -> None:
    """输入：无。输出：无。

    预期行为：`build` 出的题族**首次出现顺序**与 `FAMILIES` 一致（逐类那一族出四道，所以比
    的是去重之后的顺序）。
    """
    out = questions.build(features.plan_for(CASE, 0), locales.load("zh"), 0)
    assert list(dict.fromkeys(item.family for item in out)) == list(questions.FAMILIES)
    assert questions.FAMILIES == ("flag", "count", "combo", "route")
    assert len(out) == len(features.CLASS_IDS) + 3


def test_flag_family_is_one_noul_per_class() -> None:
    """输入：无。输出：无。预期行为：四条 noul，一类一道，答案与真值一致，题面带判据原文。"""
    plan = features.plan_for(CASE, 1)
    loc = locales.load("zh")
    out = questions.build(plan, loc, 1)
    flags = [item for item in out if item.family == "flag"]
    assert len(flags) == len(features.CLASS_IDS)
    for cls, item in zip(features.CLASS_IDS, flags, strict=True):
        assert item.question["type"] == "noul"
        assert item.answer is (cls in plan.classes)
        assert str(loc["class"][cls]["rule"]) in item.question["instructions"]
        assert set(item.question["criteria"]) == {"true", "false"}
        assert item.shuffle is False


def test_flag_answers_are_close_to_fifty_fifty() -> None:
    """输入：全空间 126 组 case。输出：无。

    预期行为：每一类的 True 占比落在 30%~70% 之间——`flag` 那一族的答案要是被一边吃掉，
    这一族就退化成常量，均衡闸会当场标红。
    """
    loc = locales.load("zh")
    totals = dict.fromkeys(features.CLASS_IDS, 0)
    cases = features.build_cases(512)
    for case in cases:
        out = questions.build(features.plan_for(case, 0), loc, 0)
        for cls, item in zip(features.CLASS_IDS, out, strict=False):
            if item.answer:
                totals[cls] += 1
    assert len(set(totals.values())) == 1, totals
    share = totals["form"] / len(cases)
    assert 0.3 <= share <= 0.7, share


def test_count_answers_match_the_truth() -> None:
    """输入：全空间 126 组 case。输出：无。预期行为：`count` 的答案是类数那一档。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in questions.build(plan, loc, 0) if q.family == "count")
        assert item.answer == f"n{plan.count}"
        assert item.question["criteria"][item.answer] == questions.count_text(loc, plan.count)
        assert len(item.question["criteria"]) == features.MAX_PACKET_CLASSES


def test_combo_offers_the_truth_plus_three_distractors() -> None:
    """输入：全空间 126 组 case。输出：无。

    预期行为：四个选项、描述两两不同、真值排在第一（编号由组装器最后一步打乱），
    其余三个与真值同大小、差一处。
    """
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in questions.build(plan, loc, 0) if q.family == "combo")
        criteria = item.question["criteria"]
        assert isinstance(criteria, dict)
        assert list(criteria) == ["k0", "k1", "k2", "k3"]
        assert item.answer == "k0"
        assert len(set(criteria.values())) == 4
        assert criteria["k0"] == questions.combo_text(loc, plan.classes)
        for combo in plan.distractors:
            assert questions.combo_text(loc, combo) in criteria.values()


def test_route_answers_match_the_truth() -> None:
    """输入：全空间 126 组 case。输出：无。预期行为：`route` 的答案是份数那一档，三档全出。"""
    loc = locales.load("zh")
    seen: set[str] = set()
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in questions.build(plan, loc, 0) if q.family == "route")
        assert item.answer == plan.route
        assert set(item.question["criteria"]) == set(features.ROUTE_LEVELS)
        assert item.question["criteria"][item.answer] == questions.route_text(loc, plan.docs)
        seen.add(item.answer)
    assert seen == set(features.ROUTE_LEVELS)


def test_every_question_is_self_contained() -> None:
    """输入：一种语言 × 一个计划。输出：无。

    预期行为：题面里没有任何没被填掉的槽（`{...}`），而且 `choice` 的答案一定在选项里——
    考生只读 state 与题面就能答（范式 §9.8）。
    """
    plan = features.plan_for(CASE, 0)
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        for phrasing in (0, 1):
            for item in questions.build(plan, loc, phrasing):
                assert "{" not in item.question["instructions"]
                if item.question["type"] == "choice":
                    assert item.answer in item.question["criteria"]


def test_choice_rejects_duplicate_and_unknown_answers() -> None:
    """输入：坏选项表 / 不在选项里的答案。输出：无。预期行为：都当场抛错（构造 bug）。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("x", "问？", {"a": "一样", "b": "一样"}, "a")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("x", "问？", {"a": "甲", "b": "乙"}, "c")


def test_combo_text_lists_the_classes_in_the_canonical_order() -> None:
    """输入：一组种类。输出：无。预期行为：按 `CLASS_IDS` 顺序拼，顺序不同的同一集合只有一种写法。"""
    loc = locales.load("zh")
    assert questions.combo_text(loc, ("receipt", "form")) == questions.combo_text(
        loc, features.parse_classes("form+receipt")
    )
    assert str(loc["class"]["form"]["name"]) in questions.combo_text(loc, ("form",))
