"""题族测试：四族题的形状、答案与自足性。"""

from __future__ import annotations


from workflows.memorygate import features, locales, questions
from workflows.memorygate.features import MemoryGateCase

#: 一个普通 case。
CASE = MemoryGateCase("M1+M3", features.FORM_SENTENCE)


def test_family_order_matches_the_declared_families() -> None:
    """输入：无。输出：无。

    预期行为：`build` 出的题族**首次出现顺序**与 `FAMILIES` 一致（逐判据那一族出四道，所以比
    的是去重之后的顺序）。
    """
    out = questions.build(features.plan_for(CASE, 0), locales.load("zh"), 0)
    assert list(dict.fromkeys(item.family for item in out)) == list(questions.FAMILIES)
    assert questions.FAMILIES == ("flag", "count", "combo", "gate")


def test_flag_answers_are_fifty_fifty() -> None:
    """输入：全空间 48 组 case。输出：无。预期行为：每条判据的 True / False 各占一半。"""
    loc = locales.load("zh")
    totals = dict.fromkeys(features.TAG_IDS, 0)
    for case in features.build_cases(512):
        for tag, item in zip(features.TAG_IDS, questions.build(features.plan_for(case, 0), loc, 0), strict=False):
            if item.answer:
                totals[tag] += 1
    assert len(set(totals.values())) == 1, totals
    assert totals["M1"] == 24


def test_gate_answers_follow_the_thresholds() -> None:
    """输入：全空间 48 组 case。输出：无。预期行为：`gate` 的答案是阈值算出来的那一档。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in questions.build(plan, loc, 0) if q.family == "gate")
        assert item.answer == features.handling_level(len(plan.hits))
        assert item.question["criteria"][item.answer] == questions.gate_text(loc, len(plan.hits))
