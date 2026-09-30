"""题族测试：四族题的形状、答案与自足性。"""

from __future__ import annotations

import pytest

from workflows.tracescore import features, locales, questions
from workflows.tracescore.features import TraceScoreCase

#: 一个普通 case：命中两条，入口是命令行代理。
CASE = TraceScoreCase("S1+S3", features.SOURCE_CLI)


def test_family_order_matches_the_declared_families() -> None:
    """输入：无。输出：无。

    预期行为：`build` 出的题族**首次出现顺序**与 `FAMILIES` 一致（逐信号那一族出四道，所以比
    的是去重之后的顺序）。
    """
    out = questions.build(features.plan_for(CASE, 0), locales.load("zh"), 0)
    assert list(dict.fromkeys(item.family for item in out)) == list(questions.FAMILIES)
    assert questions.FAMILIES == ("flag", "count", "combo", "route")


def test_flag_family_is_one_noul_per_tag() -> None:
    """输入：无。输出：无。预期行为：四条 noul，一条信号一道，答案与真值一致。"""
    plan = features.plan_for(CASE, 0)
    out = questions.build(plan, locales.load("zh"), 0)
    flags = [item for item in out if item.family == "flag"]
    assert len(flags) == len(features.TAG_IDS)
    for tag, item in zip(features.TAG_IDS, flags, strict=True):
        assert item.question["type"] == "noul"
        assert item.answer is (tag in plan.hits)


def test_flag_answers_are_fifty_fifty() -> None:
    """输入：全空间 48 组 case。输出：无。预期行为：每条信号的 True / False 各占一半。"""
    loc = locales.load("zh")
    totals = dict.fromkeys(features.TAG_IDS, 0)
    for case in features.build_cases(512):
        out = questions.build(features.plan_for(case, 0), loc, 0)
        for tag, item in zip(features.TAG_IDS, out, strict=False):
            if item.answer:
                totals[tag] += 1
    assert len(set(totals.values())) == 1, totals
    assert totals["S1"] == 24


def test_route_answers_follow_the_score_thresholds() -> None:
    """输入：全空间 48 组 case。输出：无。

    预期行为：`route` 的答案由**推进分**算出来（不是命中条数）——`S1+S3` 是 5 分走继续推进，
    `S2+S4` 是 1 分走停下问人，两条命中数一样。
    """
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in questions.build(plan, loc, 0) if q.family == "route")
        assert item.answer == features.route_for(plan.score)
        assert item.question["criteria"][item.answer] == questions.route_text(loc, plan.score)
    assert questions.route_text(loc, 5) != questions.route_text(loc, 1)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_question_is_self_contained(lang: str) -> None:
    """输入：一种语言 × 第二种问法。输出：无。

    预期行为：题面里带上要问的那条信号的名字与原文——只读 state 的考生要能自己按字面判。
    （用第二种问法：第一种是短问法，只点名信号，不重述判据原文。）
    """
    loc = locales.load(lang)
    for tag, item in zip(
        features.TAG_IDS, questions.build(features.plan_for(CASE, 0), loc, 1), strict=False
    ):
        entry = loc["tag"][tag]
        instructions = str(item.question["instructions"])
        assert str(entry["name"]) in instructions
        assert str(entry["rule"]) in instructions
