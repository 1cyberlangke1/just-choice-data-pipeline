"""题族：四道题、题型、答案、判据写进题面、选项表与等级表。"""

from __future__ import annotations

import random
from typing import Final

import pytest

from tests.workflows.sentiment.conftest import _make_spec
from workflows.sentiment import features as ft
from workflows.sentiment import locales, questions, render

LOC: Final[dict[str, object]] = locales.load("zh")
ALL_CASES: Final[tuple[ft.SentimentCase, ...]] = tuple(ft.build_cases(512))


def test_only_choice_options_may_be_shuffled() -> None:
    """输入：一条规格。输出：无。

    预期行为：两个 choice 族允许打乱（选项顺序没有语义），`drop` 的档位有自然顺序（落差从
    0 往上涨），打乱它会把「档位」读成「随便一个等级」。
    """
    for item in questions.build(_make_spec(5, "standup"), LOC, 0, random.Random(0)):
        assert item.shuffle is (item.question["type"] == "choice"), item.family


def test_answers_match_the_spec_for_every_case() -> None:
    """输入：全空间 case。输出：无。

    预期行为：四道题的答案都与真值规格算出来的一致——两个 choice 是选项名（组装器再换成
    编号）、noul 是布尔、score 是档位下标。这里是「生成器把答案算错了」那条路的第一道闸。
    """
    for case in ALL_CASES:
        spec = _make_spec(case.arc, case.topic)
        answers = {item.family: item.answer for item in questions.build(spec, LOC, 0, random.Random(0))}
        assert answers["tone"] == spec.tone, case.describe()
        assert answers["pivot"] == spec.pivot_key, case.describe()
        assert answers["drop"] == spec.drop, case.describe()
        assert answers["rebound"] == spec.rebound, case.describe()


def test_answers_stay_inside_their_criteria() -> None:
    """输入：全空间 case。输出：无。

    预期行为：score 的答案落在等级表下标里、两个 choice 的答案各自落在选项表里、noul 是布尔
    ——越界会在组装器里炸，但那时已经分不清是「标签算错」还是「选项表写歪」。
    """
    for case in ALL_CASES:
        spec = _make_spec(case.arc, case.topic)
        for item in questions.build(spec, LOC, 0, random.Random(0)):
            criteria = item.question["criteria"]
            if item.question["type"] == "score":
                assert isinstance(criteria, list)
                assert isinstance(item.answer, int) and 0 <= item.answer < len(criteria)
            elif item.question["type"] == "choice":
                assert isinstance(criteria, dict) and item.answer in criteria
            else:
                assert isinstance(item.answer, bool)


def test_tone_options_are_the_three_leanings() -> None:
    """输入：语言字典。输出：无。预期行为：三项、键与描述都互不相同。"""
    options = questions.tone_options(locales.load("zh"))
    assert set(options) == {"pos", "neg", "flat"}
    assert len(set(options.values())) == 3


def test_pivot_options_are_the_four_turns_plus_none() -> None:
    """输入：语言字典。输出：无。

    预期行为：五个选项、键与 `PIVOT_KEYS` 完全一致、描述互不相同——少一个键就是某一回合
    永远不是答案。
    """
    options = questions.pivot_options(locales.load("zh"))
    assert tuple(options) == questions.PIVOT_KEYS
    assert len(set(options.values())) == 5


def test_phrasing_picks_a_different_head() -> None:
    """输入：三个问法编号。输出：无。

    预期行为：问法池按编号回绕取——同一道题的措辞会换，判据那一段不变。
    """
    loc = locales.load("zh")
    heads = [
        str(questions.build(_make_spec(1, "support"), loc, phrasing, random.Random(0))[0].question["instructions"]).split("\n")[0]
        for phrasing in (0, 1, 2)
    ]
    assert heads == [str(text) for text in loc["fam"]["tone"]["q"]]


def test_drop_levels_cover_every_answer() -> None:
    """输入：同分布全空间。输出：无。

    预期行为：`drop` 的等级表五档，且每一档都真的当过答案——某一档永远不是答案说明等级表写多了。
    """
    loc = locales.load("zh")
    levels = loc["fam"]["drop"]["levels"]
    assert isinstance(levels, list) and len(levels) == 5
    used = {
        item.answer
        for case in ALL_CASES
        for item in questions.build(_make_spec(case.arc, case.topic), loc, 0, random.Random(0))
        if item.family == "drop"
    }
    assert used == set(range(5))


def test_unknown_phrasing_wraps_instead_of_failing() -> None:
    """输入：越界问法编号。输出：无。预期行为：回绕取模，题面仍然出得来。"""
    assert len(questions.build(_make_spec(0), LOC, 99, random.Random(0))) == len(questions.FAMILIES)


def test_duplicate_tone_options_are_rejected() -> None:
    """输入：三个倾向写成同一个词。输出：无。

    预期行为：组装器拒掉重名选项——考生分辨不出两个写法一样的选项，那是坏数据不是难题。
    """
    loc = dict(LOC)
    fam = dict(loc["fam"])
    fam["tone"] = dict(fam["tone"], criteria={"pos": "同一句", "neg": "同一句", "flat": "同一句"})
    loc["fam"] = fam
    with pytest.raises(ValueError, match="选项描述重复"):
        questions.build(_make_spec(0), loc, 0, random.Random(0))
