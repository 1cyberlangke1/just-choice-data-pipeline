"""题族测试：四族题的形状、答案与自足性。"""

from __future__ import annotations


from workflows.subqa import features, locales, questions
from workflows.subqa.features import SubQaCase

#: 一个普通 case：命中两条，字幕对来自电影。
CASE = SubQaCase("S1+S3", features.TRACK_MOVIE)


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
    assert totals["S1"] == 24


def test_combo_text_follows_the_canonical_tag_order() -> None:
    """输入：空组合、单条组合、以及全空间计划里的真值与干扰组合。输出：无。

    预期行为：选项描述按**传进来的顺序**取判据名拼，空组合用该语言里那句「哪条都不算」；而
    `plan.hits` 与三个干扰组合本来就按 `TAG_IDS` 的规范顺序排好（`features.plan_for` 保证），
    所以一组判据只有一种写法、四个选项描述两两不同（`_choice` 会当场验这一点）。
    """
    loc = locales.load("zh")
    assert questions.combo_text(loc, ()) == str(loc["fam"]["combo"]["none"])
    assert questions.combo_text(loc, ("S2",)).count(str(loc["tag"]["S2"]["name"])) == 1
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        for combo in (plan.hits, *plan.distractors):
            assert combo == tuple(tag for tag in features.TAG_IDS if tag in combo), combo
