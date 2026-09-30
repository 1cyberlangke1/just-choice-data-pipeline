"""来源选择的特征层测试：维度表、选源规则与候选表的构造不变量。"""

from __future__ import annotations

from collections import Counter

import pytest

from workflows.sourceselect import features
from workflows.sourceselect.features import SourceCase, SourcePlan, SourceRow


def all_cases() -> list[SourceCase]:
    """输入：无。输出：全空间里所有 case。"""
    return features.build_cases(1024)


def all_plans() -> list[tuple[SourceCase, int, SourcePlan]]:
    """输入：无。输出：全空间 × 三个问法的计划。"""
    return [(case, phrasing, features.plan_for(case, phrasing)) for case in all_cases() for phrasing in (0, 1, 2)]


def literal_winner(rows: tuple[SourceRow, ...], need: str) -> int | None:
    """输入：候选表、首要要求。输出：按**印在 state 里的两条规则**字面筛出来的行。

    预期行为：这里是测试自己的第二套实现——只读那两句口径，不看 `winner_index` 的写法，
    两边对不上就说明代码与题面说的不是一回事。
    """
    axis, good = features.REQUIREMENT[need]
    alive = [index for index, row in enumerate(rows) if row.get(axis) == good]
    if not alive:
        return None
    return max(
        alive,
        key=lambda index: (
            features.AXIS_RANK["type"][rows[index].type],
            features.AXIS_RANK["freshness"][rows[index].freshness],
            features.AXIS_RANK["coverage"][rows[index].coverage],
            -index,
        ),
    )


def test_full_space_is_exactly_the_four_dimensions() -> None:
    """输入：全空间。输出：无。预期行为：3 × 3 × 3 × 2 = 54 组，每一维每个取值都到得了。"""
    cases = all_cases()
    assert len(cases) == 54
    assert len({case.as_dict().__str__() for case in cases}) == 54
    for name, values in features.DIMS.items():
        seen = {str(getattr(case, name)) for case in cases}
        assert seen == {str(value) for value in values}, name


def test_shape_is_realized_by_construction() -> None:
    """输入：全空间的计划。输出：无。

    预期行为：`unique` 恰好一条达标行、`tied` 恰好两条、`none` 一条都没有——
    这一维是注入的，就必须按构造兑现，不能靠采样碰运气。
    """
    want = {"unique": 1, "tied": 2, "none": 0}
    for case, _phrasing, plan in all_plans():
        assert len(features.survivor_indices(plan.rows, case.need)) == want[case.shape]


def test_near_miss_is_realized_by_construction() -> None:
    """输入：全空间的计划。输出：无。

    预期行为：`near_miss=1` 时表里**恰好**一条「只差首要要求那一项」的行，`0` 时一条都没有
    ——其余行刻意避开「另外两项都最高」这一格，所以这一维是精确的。
    """
    for case, _phrasing, plan in all_plans():
        hits = sum(1 for row in plan.rows if features.is_near_miss(row, case.need))
        assert hits == case.near_miss
        assert features.label_for(plan).near_miss is bool(case.near_miss)


def test_tied_winner_is_the_later_survivor() -> None:
    """输入：`tied` 那一档的计划。输出：无。

    预期行为：赢的那条**排在后面**——所以「取第一条达标的」这个捷径必然答错，
    第二条规则（比优劣档）真的在承重。
    """
    checked = 0
    for case, _phrasing, plan in all_plans():
        if case.shape != "tied":
            continue
        survivors = features.survivor_indices(plan.rows, case.need)
        assert features.winner_index(plan.rows, case.need) == survivors[-1]
        checked += 1
    assert checked == 54


def test_ignoring_the_filter_gets_the_wrong_row() -> None:
    """输入：全空间的计划。输出：无。

    预期行为：两条捷径都答错不少——这是「这一族到底在考什么」的判据（范式 §9.76 第一条）：
    - 忽略首要要求、直接挑优劣档最大的行：162 条里 90 条答错；
    - 只在达标行里取**第一条**：54 条（正好是全部 `tied`）答错。
    数字写死会把这条尺子绑在种子身上，所以只钉「错得够多」。
    """
    unfiltered = 0
    first_survivor = 0
    total = 0
    for case, _phrasing, plan in all_plans():
        correct = features.winner_index(plan.rows, case.need)
        naive = max(
            range(len(plan.rows)), key=lambda index: (plan.rows[index].rank_key(), -index)
        )
        if naive != correct:
            unfiltered += 1
        survivors = features.survivor_indices(plan.rows, case.need)
        head = survivors[0] if survivors else None
        if head != correct:
            first_survivor += 1
        total += 1
    assert total == 162
    assert unfiltered >= 40, unfiltered
    assert first_survivor == 54, first_survivor


def test_requirement_table_covers_every_need() -> None:
    """输入：首要要求 → 属性。输出：无。预期行为：三个要求各落在一个属性上、取值合法。"""
    assert set(features.REQUIREMENT) == set(features.NEEDS)
    for need, (axis, good) in features.REQUIREMENT.items():
        assert axis in features.AXIS_ORDER, need
        assert good in features.AXIS_VALUES[axis], need
        assert features.AXIS_RANK[axis][good] == max(features.AXIS_RANK[axis].values()), need


def test_value_at_rank_round_trips() -> None:
    """输入：三个属性的每一档。输出：无。预期行为：档位与取值一一对应。"""
    for axis in features.AXIS_ORDER:
        for rank in range(3):
            value = features.value_at_rank(axis, rank)
            assert features.AXIS_RANK[axis][value] == rank
        with pytest.raises(KeyError):
            features.value_at_rank(axis, 3)


def test_rule_order_is_pinned() -> None:
    """输入：口径表。输出：无。预期行为：顺序就是语义（先过滤、再取最优、都没有就弃权）。"""
    assert features.RULES == ("filter", "rank", "abstain")


def test_case_round_trips_through_dict() -> None:
    """输入：一个 case。输出：无。预期行为：`as_dict` 的键序与 `DIMS` 一致、可重建。"""
    case = SourceCase(need="coverage", size=5, shape="tied", near_miss=1)
    assert list(case.as_dict()) == list(features.DIMS)
    assert SourceCase(**case.as_dict()) == case
    assert "coverage" in case.describe()


def test_plan_rows_use_only_known_tokens() -> None:
    """输入：全空间的计划。输出：无。预期行为：每一格都在取值表里，表长等于注入的表长。"""
    for case, _phrasing, plan in all_plans():
        assert plan.size == case.size
        for row in plan.rows:
            for axis in features.AXIS_ORDER:
                assert row.get(axis) in features.AXIS_VALUES[axis]


def test_plan_is_reproducible_and_phrasing_sensitive() -> None:
    """输入：同一个 case。输出：无。

    预期行为：同一 (case, 问法) 必得同一张表；不同问法**至少有一次**长得不一样——
    问法不进种子的话，这几种问法会渲染出结构完全相同的样本。
    """
    case = all_cases()[0]
    assert features.plan_for(case, 0) == features.plan_for(case, 0)
    tables = {features.plan_for(case, phrasing).rows for phrasing in (0, 1, 2)}
    assert len(tables) > 1


def test_source_row_get_rejects_unknown_axis() -> None:
    """输入：一个不存在的属性名。输出：无。预期行为：报错并列出可用属性。"""
    row = SourceRow(type="official", freshness="fresh", coverage="full")
    assert row.rank_key() == (2, 2, 2)
    with pytest.raises(KeyError, match="未知属性"):
        row.get("size")


def test_labels_cover_every_value() -> None:
    """输入：全空间的计划。输出：无。

    预期行为：五个题族的每一档都到得了——「每一档都到得了」是硬要求，分布偏斜照实标出
    不掩盖（与 `search` 的 `trust` 同一条口径）。
    """
    labels = [features.label_for(plan) for _case, _phrasing, plan in all_plans()]
    assert {item.need for item in labels} == set(features.NEEDS)
    assert {item.pick for item in labels} == {None, 1, 2, 3, 4, 5}
    assert {item.enough for item in labels} == {True, False}
    assert {item.cover for item in labels} == {0, 1, 2}
    assert {item.near_miss for item in labels} == {True, False}
    counts = Counter(item.pick for item in labels)
    assert counts[None] > 0


def test_pick_key_formats() -> None:
    """输入：行号 / None。输出：无。预期行为：行号加 `s` 前缀，弃权是 `none`。"""
    assert features.pick_key(3) == "s3"
    assert features.pick_key(None) == features.NO_PICK
