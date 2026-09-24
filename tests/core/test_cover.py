"""维度组合枚举器测试（通用层，与具体 workflow 无关）。

输入：`jc.cover` 的纯函数。输出：断言结果。
预期行为：小空间全枚举（顺序稳定）；大空间成对覆盖（任意两维的任意取值组合都出现）；
同一输入同一输出；只产出声明过的取值。
"""

from __future__ import annotations

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
