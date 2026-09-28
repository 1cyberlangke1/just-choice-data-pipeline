"""迷宫网格：表示、生成、搜索、对称变换。

表示：`mask` 是长度为 rows*cols 的元组，第 i 个元素是格子 i 的「通路位掩码」，
第 d 位为 1 表示从格子 i 沿方向 d 有通路。格子索引 = r*cols + c。

方向约定（屏幕方向，不是东南西北）：0=上、1=右、2=下、3=左。
坐标约定：(行, 列)，0 起，原点在左上，行向下增长。

输入/输出：所有函数都是纯函数（不改入参），返回新的元组或列表。
"""

from __future__ import annotations

import random
from collections import deque
from typing import Final

DIR_UP: Final[int] = 0
DIR_RIGHT: Final[int] = 1
DIR_DOWN: Final[int] = 2
DIR_LEFT: Final[int] = 3
DIRECTIONS: Final[tuple[int, ...]] = (DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT)
DELTAS: Final[tuple[tuple[int, int], ...]] = ((-1, 0), (0, 1), (1, 0), (0, -1))
OPPOSITE: Final[tuple[int, ...]] = (DIR_DOWN, DIR_LEFT, DIR_UP, DIR_RIGHT)

# 8 种对称变换名（抵消生成算法的方向偏好）
TRANSFORMS: Final[tuple[str, ...]] = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "transpose",
    "anti_transpose",
)


def index_of(r: int, c: int, cols: int) -> int:
    """格子坐标 → 线性索引。"""
    return r * cols + c


def rowcol_of(cell: int, cols: int) -> tuple[int, int]:
    """线性索引 → (行, 列)。"""
    return divmod(cell, cols)


def neighbor(cell: int, d: int, rows: int, cols: int) -> int | None:
    """格子沿方向 d 的邻居；越界返回 None。

    输入：格子索引、方向、行列数。
    输出：邻居索引或 None。预期行为：不做通路检查，只看边界。
    """
    r, c = rowcol_of(cell, cols)
    dr, dc = DELTAS[d]
    nr, nc = r + dr, c + dc
    if 0 <= nr < rows and 0 <= nc < cols:
        return index_of(nr, nc, cols)
    return None


def has_passage(mask: tuple[int, ...], cell: int, d: int, rows: int, cols: int) -> bool:
    """格子沿方向 d 是否有通路（越界一律视为无）。"""
    return bool(mask[cell] >> d & 1) and neighbor(cell, d, rows, cols) is not None


def open_neighbors(
    mask: tuple[int, ...], cell: int, rows: int, cols: int
) -> list[tuple[int, int]]:
    """格子的全部可走方向，返回 [(方向, 邻居索引)]，按方向序号升序。"""
    out: list[tuple[int, int]] = []
    for d in DIRECTIONS:
        nxt = neighbor(cell, d, rows, cols)
        if nxt is not None and has_passage(mask, cell, d, rows, cols):
            out.append((d, nxt))
    return out


def _carve(mask: list[int], a: int, d: int, rows: int, cols: int) -> None:
    """打通 a 与 a 沿 d 的邻居（双向置位）。调用方保证邻居存在。"""
    b = neighbor(a, d, rows, cols)
    assert b is not None
    mask[a] |= 1 << d
    mask[b] |= 1 << OPPOSITE[d]


def generate(rng: random.Random, rows: int, cols: int, algo: str = "dfs") -> tuple[int, ...]:
    """生成完美迷宫（生成树：任意两格之间路径唯一）。

    输入：RNG、行列数、算法名（dfs=长走廊 / prim=多分叉短死胡同 / wilson=均匀生成树）。
    输出：`mask` 元组。
    预期行为：同一 RNG 状态必得同一迷宫；边数与连通性满足生成树（边数 = 格子数 - 1）。
    """
    if algo == "dfs":
        mask = _generate_dfs(rng, rows, cols)
    elif algo == "prim":
        mask = _generate_prim(rng, rows, cols)
    elif algo == "wilson":
        mask = _generate_wilson(rng, rows, cols)
    else:
        raise ValueError(f"未知迷宫算法：{algo!r}")
    assert is_spanning_tree(tuple(mask), rows, cols), "生成结果不是完美迷宫"
    return tuple(mask)


def _generate_dfs(rng: random.Random, rows: int, cols: int) -> list[int]:
    """递归回溯（深度优先）：走廊长、分叉少。"""
    total = rows * cols
    mask = [0] * total
    visited = [False] * total
    start = rng.randrange(total)
    visited[start] = True
    stack = [start]
    while stack:
        cur = stack[-1]
        options = [
            d
            for d in DIRECTIONS
            if (nb := neighbor(cur, d, rows, cols)) is not None and not visited[nb]
        ]
        if not options:
            stack.pop()
            continue
        d = rng.choice(options)
        nb = neighbor(cur, d, rows, cols)
        assert nb is not None
        _carve(mask, cur, d, rows, cols)
        visited[nb] = True
        stack.append(nb)
    return mask


def _generate_prim(rng: random.Random, rows: int, cols: int) -> list[int]:
    """随机 Prim：分叉多、死胡同短。"""
    total = rows * cols
    mask = [0] * total
    visited = [False] * total
    start = rng.randrange(total)
    visited[start] = True
    frontier: list[tuple[int, int]] = []

    def push(cell: int) -> None:
        for d in DIRECTIONS:
            nb = neighbor(cell, d, rows, cols)
            if nb is not None and not visited[nb]:
                frontier.append((cell, d))

    push(start)
    while frontier:
        pick = rng.randrange(len(frontier))
        cell, d = frontier[pick]
        frontier[pick] = frontier[-1]
        frontier.pop()
        nb = neighbor(cell, d, rows, cols)
        assert nb is not None
        if visited[nb]:
            continue
        _carve(mask, cell, d, rows, cols)
        visited[nb] = True
        push(nb)
    return mask


def _generate_wilson(rng: random.Random, rows: int, cols: int) -> list[int]:
    """Wilson 算法：均匀生成树（纹理更「自然」，代价是慢一些）。

    输入：RNG、行列数。输出：mask 列表。预期行为：10x10 以内秒级完成。
    """
    total = rows * cols
    mask = [0] * total
    in_tree = [False] * total
    root = rng.randrange(total)
    in_tree[root] = True
    remaining = [i for i in range(total) if i != root]
    rng.shuffle(remaining)
    for seed in remaining:
        if in_tree[seed]:
            continue
        walk = [seed]
        position_of: dict[int, int] = {seed: 0}
        cur = seed
        while not in_tree[cur]:
            options = [
                d for d in DIRECTIONS if neighbor(cur, d, rows, cols) is not None
            ]
            d = rng.choice(options)
            nb = neighbor(cur, d, rows, cols)
            assert nb is not None
            if nb in position_of:
                cut = position_of[nb]
                for gone in walk[cut + 1 :]:
                    position_of.pop(gone, None)
                del walk[cut + 1 :]
            else:
                position_of[nb] = len(walk)
                walk.append(nb)
            cur = nb
        for a, b in zip(walk, walk[1:], strict=False):
            for d in DIRECTIONS:
                if neighbor(a, d, rows, cols) == b:
                    _carve(mask, a, d, rows, cols)
                    break
            in_tree[a] = True
        in_tree[walk[-1]] = True
    return mask


def edges(mask: tuple[int, ...], rows: int, cols: int) -> list[tuple[int, int]]:
    """列出全部通路（每条边只出现一次，a < b）。"""
    out: list[tuple[int, int]] = []
    for cell in range(rows * cols):
        for d in (DIR_RIGHT, DIR_DOWN):
            nb = neighbor(cell, d, rows, cols)
            if nb is not None and has_passage(mask, cell, d, rows, cols):
                out.append((cell, nb))
    return out


def is_spanning_tree(mask: tuple[int, ...], rows: int, cols: int) -> bool:
    """校验 mask 是否是完美迷宫：边数 = 格子数 - 1 且全图连通。

    输入：mask、行列数。输出：bool。预期行为：用于生成后的自检。
    """
    total = rows * cols
    if len(mask) != total:
        return False
    if len(edges(mask, rows, cols)) != total - 1:
        return False
    seen = {0}
    queue = deque([0])
    while queue:
        cur = queue.popleft()
        for _, nb in open_neighbors(mask, cur, rows, cols):
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return len(seen) == total


def distances(mask: tuple[int, ...], rows: int, cols: int, src: int) -> list[int]:
    """BFS 求 src 到各格的最短步数。

    输入：mask、行列数、起点。输出：长度 rows*cols 的列表，不可达为 -1。
    """
    total = rows * cols
    dist = [-1] * total
    dist[src] = 0
    queue = deque([src])
    while queue:
        cur = queue.popleft()
        for _, nb in open_neighbors(mask, cur, rows, cols):
            if dist[nb] < 0:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
    return dist


def shortest_path(
    mask: tuple[int, ...], rows: int, cols: int, src: int, dst: int
) -> list[int] | None:
    """求 src → dst 的一条最短路（BFS 父指针）。

    输入：mask、行列数、起止格。输出：格子序列（含两端），不可达返回 None。
    预期行为：完美迷宫下最短路唯一；有环路时不保证唯一（调用方需自行判断）。
    """
    total = rows * cols
    parent = [-2] * total
    parent[src] = -1
    queue = deque([src])
    while queue:
        cur = queue.popleft()
        if cur == dst:
            break
        for _, nb in open_neighbors(mask, cur, rows, cols):
            if parent[nb] == -2:
                parent[nb] = cur
                queue.append(nb)
    if parent[dst] == -2:
        return None
    path = [dst]
    while path[-1] != src:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def path_moves(path: list[int], cols: int) -> list[int]:
    """把格子序列转成方向序列。

    输入：连续的格子序列、列数。输出：方向列表。预期行为：相邻格必须真的相邻。
    """
    moves: list[int] = []
    for a, b in zip(path, path[1:], strict=False):
        ar, ac = rowcol_of(a, cols)
        br, bc = rowcol_of(b, cols)
        delta = (br - ar, bc - ac)
        if delta not in DELTAS:
            raise ValueError(f"路径不连续：{a} -> {b}")
        moves.append(DELTAS.index(delta))
    return moves


def apply_moves(
    mask: tuple[int, ...], rows: int, cols: int, start: int, moves: list[int] | tuple[int, ...]
) -> list[int]:
    """按方向序列行走，返回途经的格子序列（含起点）。

    输入：mask、行列数、起点、方向序列。
    输出：格子序列。预期行为：撞墙或越界直接抛 ValueError（不静默忽略）。
    """
    cells = [start]
    cur = start
    for d in moves:
        if not has_passage(mask, cur, d, rows, cols):
            raise ValueError(f"非法移动：格子 {cur} 方向 {d} 没有通路")
        nxt = neighbor(cur, d, rows, cols)
        assert nxt is not None
        cells.append(nxt)
        cur = nxt
    return cells


def remove_edge(mask: tuple[int, ...], a: int, b: int, rows: int, cols: int) -> tuple[int, ...]:
    """移除 a 与 b 之间的通路（用于构造「出口不可达」）。

    输入：mask、两端格子、行列数。输出：新的 mask 元组。
    预期行为：a、b 必须相邻；移除后双向置位都清零。
    """
    new = list(mask)
    ar, ac = rowcol_of(a, cols)
    br, bc = rowcol_of(b, cols)
    delta = (br - ar, bc - ac)
    if delta not in DELTAS:
        raise ValueError(f"两端不相邻：{a} 与 {b}")
    d = DELTAS.index(delta)
    new[a] &= ~(1 << d)
    new[b] &= ~(1 << OPPOSITE[d])
    return tuple(new)


def braid(
    rng: random.Random, mask: tuple[int, ...], rows: int, cols: int, prob: float
) -> tuple[int, ...]:
    """以 prob 的概率拆掉死胡同（产生环路）。

    输入：RNG、mask、行列数、概率。
    输出：新的 mask 元组。预期行为：prob<=0 时原样返回；拆墙只针对「只有一个出口」的格子。
    """
    if prob <= 0:
        return mask
    new = list(mask)
    for cell in range(rows * cols):
        if bin(new[cell]).count("1") != 1:
            continue
        if rng.random() >= prob:
            continue
        candidates = [
            d
            for d in DIRECTIONS
            if (nb := neighbor(cell, d, rows, cols)) is not None and not (new[cell] >> d & 1)
        ]
        if candidates:
            _carve(new, cell, rng.choice(candidates), rows, cols)
    return tuple(new)


def transform_dims(name: str, rows: int, cols: int) -> tuple[int, int]:
    """对称变换后的新行列数。

    输入：变换名、原行列数。输出：(新行数, 新列数)。预期行为：旋转/转置会交换行列。
    """
    if name in ("identity", "rot180", "flip_h", "flip_v"):
        return rows, cols
    if name in ("rot90", "rot270", "transpose", "anti_transpose"):
        return cols, rows
    raise ValueError(f"未知变换：{name!r}")


def _map_cell(name: str, r: int, c: int, rows: int, cols: int) -> tuple[int, int]:
    """把 (行, 列) 映射到变换后的坐标。"""
    if name == "identity":
        return r, c
    if name == "rot90":  # 顺时针 90°
        return c, rows - 1 - r
    if name == "rot180":
        return rows - 1 - r, cols - 1 - c
    if name == "rot270":  # 逆时针 90°
        return cols - 1 - c, r
    if name == "flip_h":  # 左右镜像
        return r, cols - 1 - c
    if name == "flip_v":  # 上下镜像
        return rows - 1 - r, c
    if name == "transpose":
        return c, r
    if name == "anti_transpose":
        return cols - 1 - c, rows - 1 - r
    raise ValueError(f"未知变换：{name!r}")


def apply_transform(
    name: str, mask: tuple[int, ...], rows: int, cols: int
) -> tuple[tuple[int, ...], int, int]:
    """对整个迷宫施加对称变换。

    输入：变换名、mask、行列数。
    输出：(新 mask, 新行数, 新列数)。
    预期行为：方向不手工推导，而是由「邻居映射后的坐标差」反查，避免写错 8 套映射。
    """
    new_rows, new_cols = transform_dims(name, rows, cols)
    new = [0] * (new_rows * new_cols)
    for r in range(rows):
        for c in range(cols):
            src = index_of(r, c, cols)
            nr, nc = _map_cell(name, r, c, rows, cols)
            dst = index_of(nr, nc, new_cols)
            for d in DIRECTIONS:
                if not (mask[src] >> d & 1):
                    continue
                dr, dc = DELTAS[d]
                tr, tc = _map_cell(name, r + dr, c + dc, rows, cols)
                nd = DELTAS.index((tr - nr, tc - nc))
                new[dst] |= 1 << nd
    return tuple(new), new_rows, new_cols
