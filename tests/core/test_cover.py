"""维度组合枚举器测试（通用层，与具体 workflow 无关）。

输入：`jc.cover` 的纯函数。输出：断言结果。
预期行为：小空间全枚举（顺序稳定）；大空间成对覆盖（任意两维的任意取值组合都出现）；
同一输入同一输出；只产出声明过的取值。
"""

from __future__ import annotations

import random

import pytest

from jc import cover


def test_small_space_is_full_product() -> None:
    """组合数不超过阈值时全枚举，顺序与 itertools.product 一致。"""
    dims = {"a": ("1", "2"), "b": ("x", "y", "z")}
    rows = cover.cover_dims(dims)
    assert [tuple(row.values()) for row in rows] == [
        ("1", "x"),
        ("1", "y"),
        ("1", "z"),
        ("2", "x"),
        ("2", "y"),
        ("2", "z"),
    ]


def test_big_space_covers_every_pair() -> None:
    """组合数超过阈值时改用成对覆盖，且行数远小于全枚举。"""
    dims = {f"d{i}": (f"{i}a", f"{i}b", f"{i}c") for i in range(6)}  # 3^6 = 729
    rows = cover.cover_dims(dims, max_full=100)
    assert cover.missing_pairs(dims, rows) == set()
    assert 0 < len(rows) < 3**6


def test_cover_is_deterministic() -> None:
    """同一输入必须给出同一输出（case 库要能逐字节重建）。"""
    dims = {f"d{i}": (f"{i}a", f"{i}b") for i in range(8)}
    assert cover.cover_dims(dims, max_full=1) == cover.cover_dims(dims, max_full=1)


def test_rows_only_use_declared_values() -> None:
    """每行都要覆盖全部维度，且取值必须来自声明。"""
    dims = {"a": ("1", "2"), "b": ("x", "y"), "c": ("m", "n")}
    rows = cover.cover_dims(dims, max_full=1)
    assert rows
    for row in rows:
        assert set(row) == set(dims)
        for name, value in row.items():
            assert value in dims[name]


def test_missing_pairs_reports_uncovered() -> None:
    """未覆盖的取值对要能准确报出来（空集 = 成对覆盖达成）。"""
    dims = {"a": ("1", "2"), "b": ("x", "y")}
    assert cover.missing_pairs(dims, [{"a": "1", "b": "x"}]) == {
        ("a", "1", "b", "y"),
        ("a", "2", "b", "x"),
        ("a", "2", "b", "y"),
    }
    assert cover.missing_pairs(
        dims, [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]
    ) == {("a", "1", "b", "y"), ("a", "2", "b", "x")}
    all_rows = [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}, {"a": "1", "b": "y"}, {"a": "2", "b": "x"}]
    assert cover.missing_pairs(dims, all_rows) == set()


def _not_both_a1_and_bx(dim_a: str, value_a: str, dim_b: str, value_b: str) -> bool:
    """示例取值对约束：`a=1` 与 `b=x` 不能同时出现（顺序无关）。"""
    return frozenset({(dim_a, value_a), (dim_b, value_b)}) != frozenset(
        {("a", "1"), ("b", "x")}
    )


def test_full_enumeration_honours_the_pair_rule() -> None:
    """全枚举路径也要遵守取值对约束：被禁的行直接不进结果，其余顺序不变。"""
    dims = {"a": ("1", "2"), "b": ("x", "y")}
    rows = cover.cover_dims(dims, allow_pair=_not_both_a1_and_bx)
    assert rows == [{"a": "1", "b": "y"}, {"a": "2", "b": "x"}, {"a": "2", "b": "y"}]


def test_pairwise_honours_the_pair_rule() -> None:
    """成对覆盖路径也要遵守约束，并且「被禁的对」不算没覆盖。"""
    dims = {f"d{i}": (f"{i}a", f"{i}b", f"{i}c") for i in range(6)}  # 3^6 = 729

    def rule(dim_a: str, value_a: str, dim_b: str, value_b: str) -> bool:
        """约束：`d0=0a` 与 `d1=1a` 不能同时出现。"""
        pair = frozenset({(dim_a, value_a), (dim_b, value_b)})
        return pair != frozenset({("d0", "0a"), ("d1", "1a")})

    rows = cover.cover_dims(dims, max_full=100, allow_pair=rule)
    assert rows
    assert all(rule("d0", row["d0"], "d1", row["d1"]) for row in rows)
    assert cover.missing_pairs(dims, rows, allow_pair=rule) == set()
    # 不带约束地查，被禁的那一对必须报成「没覆盖」——证明确实被排除了
    assert ("d0", "0a", "d1", "1a") in cover.missing_pairs(dims, rows)


def test_impossible_constraints_are_reported() -> None:
    """约束太紧（某维没有任何可选取值）时报人话，不许静默少给行。"""
    dims = {"a": ("1",), "b": ("x", "y"), "c": ("m",)}

    def rule(dim_a: str, value_a: str, dim_b: str, value_b: str) -> bool:
        """约束：`a` 维与 `b` 维的任何取值都不能共存。"""
        return {dim_a, dim_b} != {"a", "b"}

    with pytest.raises(ValueError, match="没有任何可选取值"):
        cover.cover_dims(dims, max_full=1, allow_pair=rule)


def test_row_allowed_reports_any_banned_pair() -> None:
    """输入：一行组合、取值对约束。输出：断言结果。

    预期行为：逐对比对，**任意一对**被禁就 False，全允许才 True——workflow 的
    `compatible` 就是拿它当「这组特征写得出来吗」。
    """
    names = ["a", "b", "c"]
    row = {"a": "1", "b": "2", "c": "3"}

    def allow_all(dim_a: str, value_a: str, dim_b: str, value_b: str) -> bool:
        """任何取值对都允许。"""
        return True

    def ban_a_with_c(dim_a: str, value_a: str, dim_b: str, value_b: str) -> bool:
        """禁掉 `a` 与 `c` 的任何取值组合。"""
        return {dim_a, dim_b} != {"a", "c"}

    assert cover.row_allowed(names, row, allow_all) is True
    assert cover.row_allowed(names, row, ban_a_with_c) is False


def test_near_miss_combos_are_close_to_truth_and_never_the_truth() -> None:
    """输入：判据编号表、真值组合、固定随机源。输出：断言结果。

    预期行为：给 `count` 个互不相同的组合，都不等于真值、都离真值很近（差一两处），
    且同一入参必得同一结果——干扰项要难分辨，但不能变成答案。
    """
    tag_ids = ("A", "B", "C", "D")
    truth = ("A", "C")
    got = cover.near_miss_combos(tag_ids, truth, random.Random(7), count=3)
    assert len(got) == 3
    assert len(set(got)) == 3
    assert truth not in got
    for combo in got:
        assert set(combo) <= set(tag_ids)
        assert len(set(combo) ^ set(truth)) <= 2
    assert got == cover.near_miss_combos(tag_ids, truth, random.Random(7), count=3)
