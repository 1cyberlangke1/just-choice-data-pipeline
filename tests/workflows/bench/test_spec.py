"""真值规格的单元测试：规则函数、三十张成绩表的构造、只吃键的重算、case 校验与集号。"""

from __future__ import annotations

import random
from collections import Counter
from itertools import permutations, product

import pytest

from tests.workflows.bench.conftest import _make_case, TINY_TABLES
from workflows.bench import spec as sp


def _grid(tables: tuple[str, ...]) -> list[tuple[str, str]]:
    """输入：成绩表名单。输出：[(成绩表, 列序)]（六种列序全开，顺序稳定）。"""
    return list(product(tables, sp.LAYOUTS))


def test_layouts_are_the_six_column_orders() -> None:
    """输入：列序表。输出：无。

    预期行为：六种列序就是三个候选的全部排列，而且每个候选在**每个列位上**恰好出现两次——
    这是「逐列分布均匀」与「`best` 三档均匀」的全部理由，破了它两族立刻歪。
    """
    perms = [sp.layout_perm(layout) for layout in sp.LAYOUTS]
    assert len(perms) == 6
    assert sorted(perms) == sorted(permutations(sp.CANDIDATES))
    assert set(sp.LAYOUT_PERMS) == set(sp.LAYOUTS)
    for column in range(len(sp.CANDIDATES)):
        seen = Counter(perm[column] for perm in perms)
        assert set(seen.values()) == {2}, (column, seen)


def test_counts_are_pairwise_distinct_and_in_range() -> None:
    """输入：三十张成绩表。输出：无。

    预期行为：三个候选的答对条数两两不同（模 6 的错位量是 0 / 2 / 4，步长 2 恰好把 6 分成
    三份），而且都落在 0..5 里——「谁最好」因此永远唯一，`gap` 也永远只落在 2 与 4 上。
    """
    for name in sp.TABLES:
        counts = sp.counts_of(sp.table_of(name))
        assert len(counts) == len(sp.CANDIDATES)
        assert len(set(counts)) == len(sp.CANDIDATES)
        assert all(count in sp.COUNTS for count in counts)


def test_correct_sets_match_the_counts() -> None:
    """输入：三十张成绩表。输出：无。

    预期行为：答对集的元素个数就是答对条数，而且三个集合互不相同——条数为 0 与 5 分别是
    空集与全集（同一段循环区间的两端，不是特例）。
    """
    for name in sp.TABLES:
        table = sp.table_of(name)
        sets = sp.correct_sets(table)
        counts = sp.counts_of(table)
        assert list(sets) == list(sp.CANDIDATES)
        for cand, count in zip(sp.CANDIDATES, counts, strict=True):
            assert len(sets[cand]) == count
            assert sets[cand] <= frozenset(range(sp.TOTAL))
        assert len({frozenset(value) for value in sets.values()}) == len(sp.CANDIDATES)
    assert sp.correct_sets(sp.table_of("k0s0"))["m0"] == frozenset()
    assert sp.correct_sets(sp.table_of("k5s0"))["m0"] == frozenset(range(sp.TOTAL))


def test_letters_cycle_and_shift_wraps() -> None:
    """输入：五题 × 四个字母。输出：无。

    预期行为：参考答案字母按题下标循环铺（A、B、C、D、A）；位移在四个字母里回绕。
    """
    table = sp.table_of("k3s0")
    assert [sp.letter_of(table, index) for index in range(sp.TOTAL)] == ["A", "B", "C", "D", "A"]
    assert sp.shift_letter("D", 1) == "A"
    assert sp.shift_letter("A", 3) == "D"
    with pytest.raises(KeyError, match="letter"):
        sp.shift_letter("Z", 1)


def test_answer_letter_is_gold_when_correct_and_shifted_when_wrong() -> None:
    """输入：三十张成绩表 × 六种列序 × 五题。输出：无。

    预期行为：答对就给参考字母、答错就给「参考字母往后位移 1 + 候选序号」——三个候选的错
    答案两两不同，而且都不等于参考（考生没法靠「谁跟别人不一样」猜）。
    """
    for name, layout in _grid(sp.TABLES):
        table = sp.table_of(name)
        sets = sp.correct_sets(table)
        for index in range(sp.TOTAL):
            ref = sp.letter_of(table, index)
            for cand in sp.CANDIDATES:
                letter = sp.answer_letter(table, cand, index)
                assert letter in sp.LETTERS
                if index in sets[cand]:
                    assert letter == ref
                else:
                    assert letter != ref
            wrong = [
                sp.answer_letter(table, cand, index)
                for cand in sp.CANDIDATES
                if index not in sets[cand]
            ]
            assert len(set(wrong)) == len(wrong)
    with pytest.raises(KeyError, match="candidate"):
        sp.answer_letter(sp.table_of("k0s0"), "m9", 0)


def test_rows_of_writes_the_column_order() -> None:
    """输入：三十张成绩表 × 六种列序。输出：无。

    预期行为：行数就是题数、行序就是题号；每一行的列序由排布定，且五行**列序一致**；
    参考答案字母随题号走（与排布无关）。
    """
    for name, layout in _grid(sp.TABLES):
        table = sp.table_of(name)
        rows = sp.rows_of(table, layout)
        assert len(rows) == sp.TOTAL
        want = sp.layout_perm(layout)
        for index, row in enumerate(rows):
            assert tuple(cell.cand for cell in row.cells) == want
            assert row.ref == sp.letter_of(table, index)
            for cell in row.cells:
                assert cell.letter == sp.answer_letter(table, cell.cand, index)


def test_rows_serialize_and_describe_without_changing_anything() -> None:
    """输入：一行记录。输出：无。

    预期行为：`as_dict` 与 `describe` 都是纯序列化——报告与调试靠它们看一行到底长什么样，
    写歪了排查就变成猜。
    """
    row = sp.rows_of(sp.table_of("k3s0"), "l0")[0]
    assert row.as_dict() == {
        "ref": row.ref,
        "cells": [{"candidate": cell.cand, "letter": cell.letter} for cell in row.cells],
    }
    assert row.describe() == "/".join([row.ref, *(cell.describe() for cell in row.cells)])
    assert row.cells[0].as_dict() == {
        "candidate": row.cells[0].cand,
        "letter": row.cells[0].letter,
    }
    assert row.cells[0].describe() == f"{row.cells[0].cand}:{row.cells[0].letter}"


def test_key_only_recompute_matches_rows_of() -> None:
    """输入：三十张成绩表 × 六种列序。输出：无。

    预期行为：只吃键的重算（`correct_of_cell` / `count_of_column` / `pass_of_column` /
    `best_column_of_rows` / `gap_level_of_rows`）与直接按成绩表算出来的一模一样——
    判别器把文本读回来重算时走的就是这条，两条路必须等价。
    """
    for name, layout in _grid(sp.TABLES):
        table = sp.table_of(name)
        rows = sp.rows_of(table, layout)
        counts = sp.counts_of(table)
        for column, cand in enumerate(sp.layout_perm(layout), start=1):
            assert sp.count_of_column(rows, column) == counts[sp.CANDIDATES.index(cand)]
            assert sp.pass_of_column(rows, column, sp.BAR) is (
                counts[sp.CANDIDATES.index(cand)] >= sp.BAR
            )
            for index, row in enumerate(rows):
                want = index in sp.correct_sets(table)[cand]
                assert sp.correct_of_cell(row, column) is want
        best = sp.best_column_of_rows(rows)
        assert counts[sp.CANDIDATES.index(sp.layout_perm(layout)[best - 1])] == max(counts)
        gap = sp.gap_level_of_rows(rows)
        if best == 1:
            assert gap is None
        else:
            assert gap is not None
            assert sp.GAP_VALUES[gap] == max(counts) - counts[sp.CANDIDATES.index(sp.layout_perm(layout)[0])]


def test_recompute_reports_bad_column_and_bad_gap() -> None:
    """输入：越界的列号 / 一张被改坏的成绩表。输出：无。预期行为：都报人话。"""
    rows = sp.rows_of(sp.table_of("k3s0"), "l0")
    with pytest.raises(ValueError, match="列号"):
        sp.correct_of_cell(rows[0], 9)
    with pytest.raises(ValueError, match="列号"):
        sp.count_of_column(rows, 0)
    # 把条数改成 (0, 2, 5)：差 5 不在 GAP_VALUES 里，重算必须当场炸而不是给个错档位。
    broken = (
        sp.BenchRow(ref="A", cells=(sp.BenchCell("m0", "B"), sp.BenchCell("m1", "A"), sp.BenchCell("m2", "A"))),
    ) * 5
    with pytest.raises(ValueError, match="落后幅度"):
        sp.gap_level_of_rows(broken)


def test_best_column_prefers_the_earlier_column_on_a_tie() -> None:
    """输入：两列答对条数一样的五行。输出：无。

    预期行为：并列取列序更靠前的那个。三个候选的条数按构造两两不同，所以这条规则在本数据集
    里从不触发——但它是判据的一部分，得真的实现着。
    """
    row = sp.BenchRow(
        ref="A",
        cells=(sp.BenchCell("m0", "A"), sp.BenchCell("m1", "A"), sp.BenchCell("m2", "B")),
    )
    rows = (row,) * sp.TOTAL
    assert sp.best_column_of_rows(rows) == 1


def test_the_thirty_tables_are_balanced_by_construction() -> None:
    """输入：三十张成绩表 × 六种列序。输出：无。

    预期行为：五族的分布都是**构造出来**的，不靠事后过滤——逐题答对恰好 450（占 900 的
    一半）、答对条数六档各 30、达标 90、最好三档各 60、差距两档各 60。逐题看还各占一半。
    """
    hit: Counter[bool] = Counter()
    per_item: Counter[int] = Counter()
    count: Counter[int] = Counter()
    passes: Counter[bool] = Counter()
    best: Counter[int] = Counter()
    gap: Counter[int | None] = Counter()
    for name, layout in _grid(sp.TABLES):
        rows = sp.rows_of(sp.table_of(name), layout)
        for index, row in enumerate(rows):
            value = sp.correct_of_cell(row, 1)
            hit[value] += 1
            per_item[index] += value
        count[sp.count_of_column(rows, 1)] += 1
        passes[sp.pass_of_column(rows, 1, sp.BAR)] += 1
        best[sp.best_column_of_rows(rows)] += 1
        level = sp.gap_level_of_rows(rows)
        if level is not None:
            gap[level] += 1
    assert len(_grid(sp.TABLES)) == 180
    assert hit[True] == hit[False] == 450
    assert set(per_item.values()) == {90}
    assert set(count.values()) == {30} and set(count) == set(sp.COUNTS)
    assert passes[True] == passes[False] == 90
    assert set(best.values()) == {60} and set(best) == {1, 2, 3}
    assert set(gap.values()) == {60} and set(gap) == set(range(len(sp.GAP_VALUES)))


def test_the_tiny_table_set_is_balanced_too() -> None:
    """输入：六张小规格成绩表 × 六种列序。输出：无。

    预期行为：小规格（`k0s0`..`k5s0`，六个 k 各一张）也是精确均匀的——逐题答对 90（占 180
    的一半）、条数六档各 6、达标 18、最好三档各 12、差距两档各 12。测试与冒烟跑的就是它。
    """
    hit: Counter[bool] = Counter()
    count: Counter[int] = Counter()
    passes: Counter[bool] = Counter()
    best: Counter[int] = Counter()
    gap: Counter[int | None] = Counter()
    for name, layout in _grid(TINY_TABLES):
        rows = sp.rows_of(sp.table_of(name), layout)
        for row in rows:
            hit[sp.correct_of_cell(row, 1)] += 1
        count[sp.count_of_column(rows, 1)] += 1
        passes[sp.pass_of_column(rows, 1, sp.BAR)] += 1
        best[sp.best_column_of_rows(rows)] += 1
        level = sp.gap_level_of_rows(rows)
        if level is not None:
            gap[level] += 1
    assert hit[True] == hit[False] == 90
    assert set(count.values()) == {6}
    assert passes[True] == passes[False] == 18
    assert set(best.values()) == {12}
    assert set(gap.values()) == {12}


def test_unknown_table_and_layout_report_available_names() -> None:
    """输入：两个不存在的名字。输出：无。预期行为：都报 KeyError 并列出可用值。"""
    with pytest.raises(KeyError, match="table"):
        sp.table_of("nope")
    with pytest.raises(KeyError, match="layout"):
        sp.layout_perm("nope")


def test_case_validates_every_dimension() -> None:
    """输入：三个维度各喂一个坏值。输出：无。预期行为：构造期就报人话，不拖到采样。"""
    with pytest.raises(KeyError, match="table"):
        _make_case(table="nope")
    with pytest.raises(KeyError, match="layout"):
        _make_case(layout="nope")
    with pytest.raises(KeyError, match="phrasing"):
        _make_case(phrasing=9)


def test_config_reports_empty_pools() -> None:
    """输入：三张取值表各清空一次。输出：无。预期行为：都报人话并点名字段。"""
    for field in ("tables", "layouts", "phrasings"):
        with pytest.raises(ValueError, match=field):
            sp.BenchConfig(**{field: ()})


def test_request_id_is_deterministic_and_split_free() -> None:
    """输入：同一 case 造两次 / 不同 case。输出：无。

    预期行为：请求号由素材指纹派生——同一 case 必得同一个号；而且它**不泄露切分**，
    所以同一屏记录在 train 与 val 里拿到的是同一个号（不然切分信息就写进 state 了）。
    """
    case = _make_case()
    first = sp.sample_spec(case, "train", 0, 0, None)
    second = sp.sample_spec(case, "val", 7, 1, None)
    assert first.request_id == second.request_id
    assert first.request_id.startswith(sp.REQUEST_PREFIX)
    assert first.request_id != sp.sample_spec(_make_case(layout="l1"), "train", 0, 0, None).request_id
    assert first.bar == sp.BAR


def test_sample_spec_ignores_the_rng() -> None:
    """输入：同一 case、两个不同的 RNG。输出：无。

    预期行为：这个 workflow 的每个自由度都已经在 case 里，`sample_spec` 不掷骰子。
    """
    case = _make_case()
    left = sp.sample_spec(case, "train", 0, 0, random.Random(1))
    right = sp.sample_spec(case, "train", 0, 0, random.Random(2))
    assert left == right


