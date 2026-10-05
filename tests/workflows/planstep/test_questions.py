"""题族：四道题、题型、答案、判据写进题面、选项表与等级表。"""

from __future__ import annotations

import random
from typing import Final

import pytest

from tests.workflows.planstep.conftest import _make_spec
from workflows.planstep import features as ft
from workflows.planstep import locales, questions, render

LOC: Final[dict[str, object]] = locales.load("zh")
ALL_CASES: Final[tuple[ft.PlanCase, ...]] = tuple(ft.build_cases(512))


def test_families_are_the_four_known_ones() -> None:
    """输入：无。输出：题族清单。

    预期行为：四个题族、顺序稳定（先还剩几项、再下一步、再还要几轮、最后解锁），三张分类表
    正好把四族分完——多一族少一族都会让下面的「答案在选项里」漏查。
    """
    assert questions.FAMILIES == ("remaining", "next", "rounds", "unlocks")
    assert questions.SCORE_FAMILIES == ("remaining", "rounds")
    assert questions.CHOICE_FAMILIES == ("next",)
    assert questions.NOUL_FAMILIES == ("unlocks",)
    assert set(questions.FAMILIES) == (
        set(questions.CHOICE_FAMILIES)
        | set(questions.SCORE_FAMILIES)
        | set(questions.NOUL_FAMILIES)
    )


def test_only_choice_options_may_be_shuffled() -> None:
    """输入：一条规格。输出：无。

    预期行为：只有 choice 允许打乱——两个 score 族的档位有自然顺序（项数从 0 往上涨），
    打乱它会把「档位」读成「随便一个等级」。
    """
    for item in questions.build(_make_spec(5, 1), LOC, 0, random.Random(0)):
        assert item.shuffle is (item.question["type"] == "choice"), item.family


def test_answers_match_the_spec_for_every_case() -> None:
    """输入：全空间 case。输出：无。

    预期行为：四道题的答案都与真值规格算出来的一致——choice 是选项名（组装器再换成编号）、
    noul 是布尔、score 是档位下标。这里是「生成器把答案算错了」那条路的第一道闸。
    """
    for case in ALL_CASES:
        spec = _make_spec(case.graph, case.done)
        answers = {item.family: item.answer for item in questions.build(spec, LOC, 0, random.Random(0))}
        assert answers["remaining"] == spec.remaining, case.describe()
        assert answers["next"] == spec.next_key, case.describe()
        assert answers["rounds"] == spec.rounds, case.describe()
        assert answers["unlocks"] == spec.unlocks, case.describe()


def test_answers_stay_inside_their_criteria() -> None:
    """输入：全空间 case。输出：无。

    预期行为：score 的答案落在等级表下标里、choice 的答案在选项表里、noul 是布尔——越界
    会在组装器里炸，但那时已经分不清是「标签算错」还是「选项表写歪」。
    """
    for case in ALL_CASES:
        spec = _make_spec(case.graph, case.done)
        for item in questions.build(spec, LOC, 0, random.Random(0)):
            criteria = item.question["criteria"]
            if item.question["type"] == "score":
                assert isinstance(criteria, list)
                assert isinstance(item.answer, int) and 0 <= item.answer < len(criteria)
            elif item.question["type"] == "choice":
                assert isinstance(criteria, dict) and item.answer in criteria
            else:
                assert isinstance(item.answer, bool)


def test_next_options_are_the_task_names_plus_none() -> None:
    """输入：语言字典。输出：无。

    预期行为：六个选项、描述互不相同，五项取自 `[task]` 表（与 state 里那几行同一个来源），
    外加一个「没有可开工的项」。
    """
    loc = locales.load("zh")
    options = questions.next_options(loc)
    assert set(options) == {"t1", "t2", "t3", "t4", "t5", "none"}
    assert len(set(options.values())) == 6
    for index, key in enumerate(render.TASK_KEYS, start=1):
        assert options[f"t{index}"] == render.task_name(loc, key)
    assert options["none"] == str(loc["fam"]["next"]["opt"]["none"])


def test_phrasing_picks_a_different_head() -> None:
    """输入：三个问法编号。输出：无。

    预期行为：问法池按编号回绕取——同一道题的措辞会换，判据那一段不变。
    """
    loc = locales.load("zh")
    heads = [
        str(questions.build(_make_spec(1, 0), loc, phrasing, random.Random(0))[0].question["instructions"]).split("\n")[0]
        for phrasing in (0, 1, 2)
    ]
    assert len(set(heads)) == 3, heads
    assert heads == [str(text) for text in loc["fam"]["remaining"]["q"]]


def test_score_levels_cover_every_answer() -> None:
    """输入：同分布全空间。输出：无。

    预期行为：`remaining` 与 `rounds` 两族的等级表长度都是 `TASK_COUNT + 1`，且每一档都
    真的当过答案——某一档永远不是答案说明等级表写多了。
    """
    loc = locales.load("zh")
    for family in questions.SCORE_FAMILIES:
        levels = loc["fam"][family]["levels"]
        assert isinstance(levels, list) and len(levels) == ft.TASK_COUNT + 1
        used = {item.answer for case in ALL_CASES for item in questions.build(_make_spec(case.graph, case.done), loc, 0, random.Random(0)) if item.family == family}
        assert used == set(range(ft.TASK_COUNT + 1)), family


def test_unknown_phrasing_wraps_instead_of_failing() -> None:
    """输入：越界问法编号。输出：无。预期行为：回绕取模，题面仍然出得来。"""
    built = questions.build(_make_spec(0, 0), LOC, 99, random.Random(0))
    assert len(built) == len(questions.FAMILIES)


def test_next_options_rejects_duplicate_descriptions() -> None:
    """输入：五个任务名全写成同一个词。输出：无。

    预期行为：组装器拒掉重名选项——考生分辨不出两个写法一样的选项，那是坏数据不是难题。
    """
    loc = dict(LOC)
    loc["task"] = dict.fromkeys(render.TASK_KEYS, "同名任务")
    with pytest.raises(ValueError, match="选项描述重复"):
        questions.build(_make_spec(0, 0), loc, 0, random.Random(0))
