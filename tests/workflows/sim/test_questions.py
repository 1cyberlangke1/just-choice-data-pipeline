"""题族的测试：出题规则、答案推导、答案分布靠构造、判据写进题面。"""

from __future__ import annotations

import dataclasses
from collections import Counter

import pytest

from tests.workflows.sim.conftest import _make_spec
from workflows.sim import locales, questions, spec as sp

LOC = locales.load("zh")
#: 一屏局面在固定排布与句面下的题量（没结束时多两道 `best` / `cost`）。
BASE_COUNT = 2 * len(sp.ACTIONS) + 1


def _build(**overrides: object) -> tuple[sp.SimSpec, list[object]]:
    """输入：case 覆盖项。输出：(规格, 题目列表)。"""
    spec = _make_spec(**overrides)
    return spec, questions.build(spec.rows, spec, LOC, 0)


def test_families_for_matches_build() -> None:
    """输入：全开网格。输出：无。

    预期行为：`families_for` 点出来的题族与 `build` 真出的题族一一对应——审计靠它算
    「应有题族」，两处不一致就会误报缺题（`best` / `cost` 是最容易写歪的那两个）。
    """
    for case in sp.build_cases(sp.SimConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        built = questions.build(spec.rows, spec, LOC, 0)
        assert set(questions.families_for(spec.rows, spec)) == {item.family for item in built}


def test_can_move_is_asked_for_every_position() -> None:
    """输入：全开网格。输出：无。

    预期行为：`can_move` 一屏六道、位置号与行序一致，答案就是「资源不低于耗、冷却为 0」。
    """
    for case in sp.build_cases(sp.SimConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        items = questions.build(spec.rows, spec, LOC, 0)
        moves = [item for item in items if item.family == "can_move"]
        assert len(moves) == len(sp.ACTIONS)
        for slot, (row, item) in enumerate(zip(spec.rows, moves, strict=True), start=1):
            assert item.answer is sp.legal_of_row(row, spec.resources)
            assert item.question["type"] == "noul"
            assert set(item.question["criteria"]) == {"true", "false"}
            assert item.shuffle is False
            assert f"第 {slot} 个动作" in item.question["instructions"]


def test_finished_is_a_single_question() -> None:
    """输入：全开网格。输出：无。预期行为：`finished` 只有一道，答案是「战利品到 6 或回合到 5」。"""
    for case in sp.build_cases(sp.SimConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        items = [item for item in questions.build(spec.rows, spec, LOC, 0) if item.family == "finished"]
        assert len(items) == 1
        assert items[0].answer is spec.done
        assert items[0].question["type"] == "noul"
        assert "{n}" not in items[0].question["instructions"]


def test_gain_reads_the_position_gain() -> None:
    """输入：全开网格。输出：无。

    预期行为：`gain` 是 score 题、等级数组就是六档，答案下标与那一行的收益档对得上。
    """
    for case in sp.build_cases(sp.SimConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        gains = [item for item in questions.build(spec.rows, spec, LOC, 0) if item.family == "gain"]
        assert len(gains) == len(sp.ACTIONS)
        for slot, (row, item) in enumerate(zip(spec.rows, gains, strict=True), start=1):
            assert item.question["type"] == "score"
            assert len(item.question["criteria"]) == len(sp.GAINS)
            assert item.answer == sp.gain_level_of_row(row)
            assert item.shuffle is False
            assert f"第 {slot} 个动作" in item.question["instructions"]


def test_best_and_cost_only_show_up_while_the_run_is_live() -> None:
    """输入：全开网格。输出：无。

    预期行为：`best` 是 choice 6、答案是位置选项名；`cost` 是 score 4 档、答案是该动作的耗；
    两族**只在还没结束时出**——结束了就没有「这一手该用哪个」。
    """
    for case in sp.build_cases(sp.SimConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        items = questions.build(spec.rows, spec, LOC, 0)
        best = [item for item in items if item.family == "best"]
        cost = [item for item in items if item.family == "cost"]
        if spec.done:
            assert best == [] and cost == []
            assert len(items) == BASE_COUNT
            continue
        assert len(best) == 1 and len(cost) == 1
        assert set(best[0].question["criteria"]) == {f"p{n}" for n in range(1, 7)}
        assert best[0].answer == f"p{spec.best_slot}"
        assert best[0].shuffle is False
        assert len(cost[0].question["criteria"]) == len(sp.COSTS)
        assert cost[0].answer == spec.cost_level
        assert cost[0].shuffle is False


def test_a_board_without_usable_actions_asks_three_families_only() -> None:
    """输入：六行全处在冷却里（一个能用的都没有）。输出：无。

    预期行为：只出 `can_move` × 6 + `finished` + `gain` × 6，`best` / `cost` 一道都不出——
    「该用哪个」没有答案时出这种题就是坏数据。
    """
    spec = _make_spec()
    rows = tuple(sp.SimRow("advance", "c1", "g2", "k1") for _ in range(len(sp.ACTIONS)))
    stuck = dataclasses.replace(spec, rows=rows, best_slot=None, cost_level=None)
    items = questions.build(rows, stuck, LOC, 0)
    assert [item.family for item in items] == ["can_move"] * 6 + ["finished"] + ["gain"] * 6
    assert questions.families_for(rows, stuck) == ("can_move", "finished", "gain")


def test_slot_options_are_pairwise_distinct() -> None:
    """输入：全开网格。输出：无。

    预期行为：位置选项的描述两两不同（位置号保证）——两行四格一样时也不会撞车。
    """
    for case in sp.build_cases(sp.SimConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        options = questions._slot_options(spec.rows, LOC)
        assert len(set(options.values())) == len(sp.ACTIONS)
        assert set(options) == {f"p{n}" for n in range(1, 7)}


def test_answer_distributions_are_built_in() -> None:
    """输入：全开网格（每张局面 × 六种排布，句面固定）。输出：无。

    预期行为：五族的答案分布**构造出来就是均匀的**——能用与不能用各占一半、结束与否各占
    一半、收益六档各占六分之一、该用的位置六种各十二道、该用的耗四档各十八道。
    """
    moves: Counter[bool] = Counter()
    finished: Counter[bool] = Counter()
    gains: Counter[int] = Counter()
    best: Counter[str] = Counter()
    cost: Counter[int] = Counter()
    for case in sp.build_cases(sp.SimConfig(phrasings=(0,))):
        spec = sp.sample_spec(case, "train", 0, 0, None)
        for item in questions.build(spec.rows, spec, LOC, 0):
            if item.family == "can_move":
                moves[item.answer] += 1
            elif item.family == "finished":
                finished[item.answer] += 1
            elif item.family == "gain":
                gains[item.answer] += 1
            elif item.family == "best":
                best[item.answer] += 1
            else:
                cost[item.answer] += 1
    assert moves[True] == moves[False] == 24 * 6 * 3
    assert finished[True] == finished[False] == 24 * 6 // 2
    assert set(gains.values()) == {24 * 6 * 6 // len(sp.GAINS)}
    assert set(best.values()) == {12 * 6 // len(sp.ACTIONS)}
    assert set(cost.values()) == {12 * 6 // len(sp.COSTS)}


def test_phrasing_only_changes_the_wording() -> None:
    """输入：同一屏局面、两套句面。输出：无。

    预期行为：题族、答案、选项**完全一样**，只有问句措辞不同——句面是留出维，但留出它
    不会把任何一族的分布压歪。
    """
    spec = _make_spec(layout="l2")
    first = questions.build(spec.rows, spec, LOC, 0)
    second = questions.build(spec.rows, spec, LOC, 1)
    assert [item.family for item in first] == [item.family for item in second]
    assert [item.answer for item in first] == [item.answer for item in second]
    assert [item.question["criteria"] for item in first] == [
        item.question["criteria"] for item in second
    ]
    assert any(
        left.question["instructions"] != right.question["instructions"]
        for left, right in zip(first, second, strict=True)
    )


def test_fill_reports_leftover_placeholders() -> None:
    """输入：一个没填完的模板。输出：无。预期行为：报人话（题面少填一个槽比报错危险）。"""
    with pytest.raises(ValueError, match="占位符"):
        questions._fill("第 {n} 个动作现在能用吗？", other="x")


def test_choice_rejects_duplicate_descriptions_and_unknown_answers() -> None:
    """输入：两个描述一样的选项 / 答案不在选项里。输出：无。预期行为：都报人话。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("best", "题面", {"a": "同一个", "b": "同一个"}, "a")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("best", "题面", {"a": "甲", "b": "乙"}, "c")


def test_build_rejects_a_wrong_number_of_rows() -> None:
    """输入：五行动作。输出：无。预期行为：报人话（少一行说明构造漏了一行）。"""
    spec = _make_spec()
    with pytest.raises(ValueError, match="局面应该是"):
        questions.build(spec.rows[:5], spec, LOC, 0)
