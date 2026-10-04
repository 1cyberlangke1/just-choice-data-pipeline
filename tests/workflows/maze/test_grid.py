"""迷宫网格测试：生成、搜索、对称变换、砍边。

覆盖：生成结果必是生成树；8 种变换保持树结构且 rot90 四次回到原状；
最短路的步数与朴素 DFS 穷举一致；非法移动必须报错；砍边后出口不可达。
"""

from __future__ import annotations

import random

import pytest

from workflows.maze import grid


def test_generate_is_spanning_tree() -> None:
    for algo in ("dfs", "prim", "wilson"):
        for rows, cols in ((3, 3), (4, 6), (7, 5)):
            rng = random.Random(f"{algo}-{rows}-{cols}")
            mask = grid.generate(rng, rows, cols, algo)
            assert grid.is_spanning_tree(mask, rows, cols)
            assert len(grid.edges(mask, rows, cols)) == rows * cols - 1


def test_generate_is_deterministic() -> None:
    a = grid.generate(random.Random(7), 5, 5, "prim")
    b = grid.generate(random.Random(7), 5, 5, "prim")
    assert a == b


def test_transform_keeps_tree() -> None:
    """本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。"""
    bad: list[str] = []
    for name in grid.TRANSFORMS:
        try:
            mask = grid.generate(random.Random(11), 4, 6, "dfs")
            new, rows, cols = grid.apply_transform(name, mask, 4, 6)
            assert grid.is_spanning_tree(new, rows, cols)
        except AssertionError as exc:
            bad.append(f"{(name,)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_rot90_four_times_is_identity() -> None:
    mask = grid.generate(random.Random(3), 4, 5, "prim")
    cur, rows, cols = mask, 4, 5
    for _ in range(4):
        cur, rows, cols = grid.apply_transform("rot90", cur, rows, cols)
    assert (rows, cols) == (4, 5)
    assert cur == mask


def _brute_force_distance(
    mask: tuple[int, ...], rows: int, cols: int, src: int, dst: int
) -> int:
    """朴素 DFS 穷举最短距离（作为 BFS 的独立对照实现）。"""
    best = None
    stack = [(src, 0, frozenset({src}))]
    while stack:
        cur, steps, seen = stack.pop()
        if cur == dst:
            best = steps if best is None else min(best, steps)
            continue
        for _, nb in grid.open_neighbors(mask, cur, rows, cols):
            if nb not in seen:
                stack.append((nb, steps + 1, seen | {nb}))
    return -1 if best is None else best


def test_bfs_matches_brute_force() -> None:
    mask = grid.generate(random.Random(5), 4, 4, "dfs")
    dist = grid.distances(mask, 4, 4, 0)
    for dst in range(16):
        assert dist[dst] == _brute_force_distance(mask, 4, 4, 0, dst)


def test_shortest_path_is_continuous_and_minimal() -> None:
    mask = grid.generate(random.Random(9), 5, 5, "wilson")
    path = grid.shortest_path(mask, 5, 5, 0, 24)
    assert path is not None
    assert path[0] == 0 and path[-1] == 24
    moves = grid.path_moves(path, 5)
    assert grid.apply_moves(mask, 5, 5, 0, moves) == path
    assert len(path) - 1 == grid.distances(mask, 5, 5, 0)[24]


def test_apply_moves_rejects_illegal_move() -> None:
    mask = grid.generate(random.Random(2), 3, 3, "dfs")
    closed = [d for d in grid.DIRECTIONS if not grid.has_passage(mask, 0, d, 3, 3)]
    if closed:
        with pytest.raises(ValueError):
            grid.apply_moves(mask, 3, 3, 0, [closed[0]])
    else:
        with pytest.raises(ValueError):
            grid.apply_moves(mask, 3, 3, 0, [grid.DIR_UP])  # 越界


def test_remove_edge_cuts_reachability() -> None:
    mask = grid.generate(random.Random(4), 4, 4, "prim")
    path = grid.shortest_path(mask, 4, 4, 0, 15)
    assert path is not None
    cut = grid.remove_edge(mask, path[0], path[1], 4, 4)
    assert grid.distances(cut, 4, 4, 15)[0] < 0
    assert grid.shortest_path(cut, 4, 4, 0, 15) is None


def test_braid_creates_loops() -> None:
    mask = grid.generate(random.Random(6), 5, 5, "dfs")
    braided = grid.braid(random.Random(6), mask, 5, 5, 1.0)
    assert len(grid.edges(braided, 5, 5)) > len(grid.edges(mask, 5, 5))
