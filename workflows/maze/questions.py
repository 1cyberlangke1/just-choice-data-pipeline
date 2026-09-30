"""迷宫题族：从 spec 纯代码推导题目与金答案（禁止调用任何模型）。

题族一览（局部 = 只看标记格附近；全局 = 需要沿路径走完整个迷宫）：
- next_step     choice 全局：从当前位置出发，下一步朝哪走最快到 E
- off_by_one    choice 全局：当前位置偏离最优路径 1 格，往哪走能回到最优路径
- on_optimal    noul   全局：当前位置是否在 S→E 最优路径上
- deviation     score  全局：当前位置偏离最优路径几步（0/1/2/≥3）
- path_via      choice 全局：S→E 最优路径经过哪个标记格
- on_path       noul   全局：标记格是否在最优路径上
- reachable     noul   全局：S 能否走到 E
- path_length   score  全局：S→E 最少几步（分桶）
- step_result   choice 局部：从标记格某方向走一步会到哪里
- can_move      noul   局部：从标记格某方向能否直接走
- dead_end      noul   局部：标记格是否死胡同
- exits         score  局部：标记格周围有几个方向能走

输入：spec、RNG、语言字典、问法编号。
输出：LabeledQuestion，或 None（该题在当前 spec 下不成立/答案不唯一 → 不出这道题）。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random
from collections.abc import Callable


from jc import contract
from jc.workflow import LabeledQuestion
from . import grid, locales, render
from .render import DIR_KEYS
from .spec import OPTION_CELL, PATH_LENGTH_BUCKETS, MazeSpec, exit_level

# 方向顺序常量（0=上 1=右 2=下 3=左），从网格模块转出，避免这里再写一遍
DIRECTIONS = grid.DIRECTIONS


def _dir_words(maze: MazeSpec, loc: dict[str, JsonValue]) -> dict[int, str]:
    """方向 → 选项文案。

    输入：spec、语言字典。
    输出：{方向: 文案}。预期行为：option_detail=cell 时在方向词后追加「→ 目标方块是什么」。
    """
    out: dict[int, str] = {}
    for d in DIRECTIONS:
        word = loc["moves"][DIR_KEYS[d]]
        if maze.render.option_detail == OPTION_CELL:
            out[d] = f"{word} → {render.cell_kind_label(maze, loc, d)}"
        else:
            out[d] = word
    return out


def _toward_exit(maze: MazeSpec) -> int | None:
    """当前位置朝出口 E 的第一步方向。

    输入：spec。
    输出：方向序号，或 None（已在出口 / 出口不可达 / 答案不唯一）。
    预期行为：完美迷宫下唯一；若出现多个等价方向（有环路）则返回 None，避免出答案不唯一的题。
    """
    cur = maze.current()
    if cur == maze.exit:
        return None
    dist = maze.dist_from_exit()
    if dist[cur] < 0:
        return None
    best: list[int] = []
    for d, nb in grid.open_neighbors(maze.mask, cur, maze.rows, maze.cols):
        if dist[nb] == dist[cur] - 1:
            best.append(d)
    return best[0] if len(best) == 1 else None


def _pick_grouped(rng: random.Random, groups: dict[object, list[JsonValue]]) -> object | None:
    """先均匀挑「目标答案」，再在该组里随机挑一个候选。

    输入：RNG、{答案: [候选…]}（空组应事先剔除）。
    输出：被选中的候选，或 None（没有任何候选）。
    预期行为：这一层是答案均衡的关键——先定答案再挑实现它的输入。
    """
    usable = {k: v for k, v in groups.items() if v}
    if not usable:
        return None
    key = rng.choice(sorted(usable, key=str))
    return rng.choice(usable[key])


def _choice(family: str, question: str, options: list[str], answer: str) -> LabeledQuestion:
    """构造 choice 题：选项描述一律用 null（选项名本身已经说明一切）。"""
    return LabeledQuestion(
        family=family,
        question={"type": contract.TYPE_CHOICE, "instructions": question, "criteria": {o: None for o in options}},
        answer=answer,
        shuffle=True,
    )


def _noul(family: str, question: str, answer: bool) -> LabeledQuestion:
    """构造 noul 题：答案是布尔，不给 criteria（边界已由题面写清）。"""
    return LabeledQuestion(
        family=family,
        question={"type": contract.TYPE_NOUL, "instructions": question},
        answer=answer,
        shuffle=False,
    )


def _score(family: str, question: str, levels: list[str], answer: int) -> LabeledQuestion:
    """构造 score 题：等级描述来自语言文件，顺序即编号（0 起）。"""
    return LabeledQuestion(
        family=family,
        question={"type": contract.TYPE_SCORE, "instructions": question, "criteria": list(levels)},
        answer=answer,
        shuffle=False,
    )


# ---------------------------------------------------------------- 全局题族


def build_next_step(maze: MazeSpec, rng: random.Random, loc: dict[str, JsonValue], idx: int) -> LabeledQuestion | None:
    """从当前位置出发，下一步朝哪走最快到出口（覆盖「往最优道路上走」的情形）。"""
    d = _toward_exit(maze)
    if d is None:
        return None
    words = _dir_words(maze, loc)
    q = locales.phrasing(loc, "next_step", idx)
    return _choice("next_step", q, [words[x] for x in DIRECTIONS], words[d])


def build_off_by_one(maze: MazeSpec, rng: random.Random, loc: dict[str, JsonValue], idx: int) -> LabeledQuestion | None:
    """当前位置偏离最优路径恰好 1 格时，往哪走能回到最优路径。"""
    if maze.agent.deviation != 1:
        return None
    path = maze.path()
    if path is None:
        return None
    d = _toward_exit(maze)
    if d is None:
        return None
    nxt = grid.neighbor(maze.current(), d, maze.rows, maze.cols)
    if nxt is None or nxt not in set(path):
        return None  # 与「偏离 1 格」的定义不符，宁可不问
    words = _dir_words(maze, loc)
    q = locales.phrasing(loc, "off_by_one", idx)
    return _choice("off_by_one", q, [words[x] for x in DIRECTIONS], words[d])


def build_on_optimal(maze: MazeSpec, rng: random.Random, loc: dict[str, JsonValue], idx: int) -> LabeledQuestion | None:
    """当前位置是否在 S→E 最优路径上。"""
    path = maze.path()
    if path is None:
        return None
    q = locales.phrasing(loc, "on_optimal", idx)
    return _noul("on_optimal", q, maze.current() in set(path))


def build_deviation(maze: MazeSpec, rng: random.Random, loc: dict[str, JsonValue], idx: int) -> LabeledQuestion | None:
    """当前位置偏离最优路径几步（0 / 1 / 2 / ≥3）。"""
    if maze.path() is None or maze.agent.deviation < 0:
        return None
    levels = list(loc["fam"]["deviation"]["levels"])
    answer = min(maze.agent.deviation, len(levels) - 1)
    q = locales.phrasing(loc, "deviation", idx)
    return _score("deviation", q, levels, answer)


def build_path_via(maze: MazeSpec, rng: random.Random, loc: dict[str, JsonValue], idx: int) -> LabeledQuestion | None:
    """S→E 最优路径经过哪个标记格（「都不经过」作为正确答案保留）。"""
    path = maze.path()
    if path is None:
        return None
    path_set = set(path)
    on_path = [letter for letter, cell in sorted(maze.marks.items()) if cell in path_set]
    if len(on_path) > 1:
        return None  # 多个标记在路径上 → 答案不唯一，不出这道题
    none_label = loc["fam"]["path_via"]["none"]
    options = [*sorted(maze.marks), none_label]
    answer = on_path[0] if on_path else none_label
    q = locales.phrasing(loc, "path_via", idx)
    return _choice("path_via", q, options, answer)


def build_on_path(maze: MazeSpec, rng: random.Random, loc: dict[str, JsonValue], idx: int) -> LabeledQuestion | None:
    """某个标记格是否在最优路径上（真假尽量 1:1）。"""
    path = maze.path()
    if path is None:
        return None
    path_set = set(path)
    groups: dict[bool, list[str]] = {True: [], False: []}
    for letter, cell in sorted(maze.marks.items()):
        groups[cell in path_set].append(letter)
    picked = _pick_grouped(rng, groups)
    if picked is None:
        return None
    q = locales.phrasing(loc, "on_path", idx, letter=picked)
    return _noul("on_path", q, picked in [l for l, c in sorted(maze.marks.items()) if c in path_set])


def build_reachable(maze: MazeSpec, rng: random.Random, loc: dict[str, JsonValue], idx: int) -> LabeledQuestion | None:
    """S 能否走到 E（被 cut 砍断时为假）。"""
    q = locales.phrasing(loc, "reachable", idx)
    return _noul("reachable", q, maze.reachable())


def build_path_length(maze: MazeSpec, rng: random.Random, loc: dict[str, JsonValue], idx: int) -> LabeledQuestion | None:
    """S→E 最少几步（分桶成 5 档）。"""
    path = maze.path()
    if path is None:
        return None
    levels = list(loc["fam"]["path_length"]["levels"])
    steps = len(path) - 1
    answer = next(
        idx for idx, (low, high) in enumerate(PATH_LENGTH_BUCKETS) if low <= steps <= high
    )
    q = locales.phrasing(loc, "path_length", idx)
    return _score("path_length", q, levels, answer)


# ---------------------------------------------------------------- 局部题族


def _step_result_label(maze: MazeSpec, cell: int, d: int, options: list[str]) -> str | None:
    """从 cell 沿 d 走一步的落点类别。

    输入：spec、起点格、方向、选项表（顺序：被墙挡住/普通通道/起点/出口/迷宫外）。
    输出：选项文案，或 None（落点是标记格 → 与选项表冲突，跳过该组合）。
    """
    blocked, passage, start_l, exit_l, outside = options
    nxt = grid.neighbor(cell, d, maze.rows, maze.cols)
    if nxt is None:
        return outside
    if not grid.has_passage(maze.mask, cell, d, maze.rows, maze.cols):
        return blocked
    if nxt == maze.start:
        return start_l
    if nxt == maze.exit:
        return exit_l
    if nxt in set(maze.marks.values()):
        return None
    return passage


def build_step_result(maze: MazeSpec, rng: random.Random, loc: dict[str, JsonValue], idx: int) -> LabeledQuestion | None:
    """从标记格某方向走一步会到哪里（先抽目标类别，再挑标记与方向）。"""
    options = list(loc["fam"]["step_result"]["options"])
    groups: dict[str, list[tuple[str, int, str]]] = {}
    for letter, cell in sorted(maze.marks.items()):
        for d in DIRECTIONS:
            label = _step_result_label(maze, cell, d, options)
            if label is not None:
                groups.setdefault(label, []).append((letter, d, label))
    picked = _pick_grouped(rng, groups)
    if picked is None:
        return None
    letter, d, label = picked
    q = locales.phrasing(loc, "step_result", idx, letter=letter, dir=loc["moves"][DIR_KEYS[d]])
    return _choice("step_result", q, options, label)


def build_can_move(maze: MazeSpec, rng: random.Random, loc: dict[str, JsonValue], idx: int) -> LabeledQuestion | None:
    """从标记格某方向能否直接走一步（真假 1:1）。"""
    groups: dict[bool, list[tuple[str, int]]] = {True: [], False: []}
    for letter, cell in sorted(maze.marks.items()):
        for d in DIRECTIONS:
            groups[grid.has_passage(maze.mask, cell, d, maze.rows, maze.cols)].append((letter, d))
    picked = _pick_grouped(rng, groups)
    if picked is None:
        return None
    letter, d = picked
    answer = grid.has_passage(maze.mask, maze.marks[letter], d, maze.rows, maze.cols)
    q = locales.phrasing(loc, "can_move", idx, letter=letter, dir=loc["moves"][DIR_KEYS[d]])
    return _noul("can_move", q, answer)


def build_dead_end(maze: MazeSpec, rng: random.Random, loc: dict[str, JsonValue], idx: int) -> LabeledQuestion | None:
    """标记格是否死胡同（只有一个方向能走）。"""
    groups: dict[bool, list[str]] = {True: [], False: []}
    for letter, cell in sorted(maze.marks.items()):
        opened = grid.open_neighbors(maze.mask, cell, maze.rows, maze.cols)
        groups[len(opened) == 1].append(letter)
    picked = _pick_grouped(rng, groups)
    if picked is None:
        return None
    opened = grid.open_neighbors(maze.mask, maze.marks[picked], maze.rows, maze.cols)
    q = locales.phrasing(loc, "dead_end", idx, letter=picked)
    return _noul("dead_end", q, len(opened) == 1)


def _exits_level(maze: MazeSpec, cell: int) -> int:
    """格子可走方向数 → 等级编号（0 死胡同 / 1 拐角 / 2 丁字或十字）。

    只分三档，因为「0 个方向」在迷宫里结构性不可能（每格至少一条通路），
    而「4 个方向」在小迷宫里极罕见；档位必须对应真实可达的情境，否则答案分布必然失衡。
    尺子只有一把（`spec.exit_level`）：挑标记时用它保证三档都有候选，这里用它定答案。
    """
    return exit_level(maze.mask, maze.rows, maze.cols, cell)


def build_exits(maze: MazeSpec, rng: random.Random, loc: dict[str, JsonValue], idx: int) -> LabeledQuestion | None:
    """标记格周围有几个方向能走（各档尽量均匀）。"""
    levels = list(loc["fam"]["exits"]["levels"])
    groups: dict[int, list[str]] = {}
    for letter, cell in sorted(maze.marks.items()):
        groups.setdefault(_exits_level(maze, cell), []).append(letter)
    picked = _pick_grouped(rng, groups)
    if picked is None:
        return None
    answer = _exits_level(maze, maze.marks[picked])
    q = locales.phrasing(loc, "exits", idx, letter=picked)
    return _score("exits", q, levels, answer)


BUILDERS: dict[str, Callable[[MazeSpec, random.Random, dict[str, JsonValue], int], LabeledQuestion | None]] = {
    "next_step": build_next_step,
    "off_by_one": build_off_by_one,
    "on_optimal": build_on_optimal,
    "deviation": build_deviation,
    "path_via": build_path_via,
    "on_path": build_on_path,
    "reachable": build_reachable,
    "path_length": build_path_length,
    "step_result": build_step_result,
    "can_move": build_can_move,
    "dead_end": build_dead_end,
    "exits": build_exits,
}

ALL_FAMILIES: tuple[str, ...] = tuple(BUILDERS)


def build_one(
    family: str, maze: MazeSpec, rng: random.Random, loc: dict[str, JsonValue], idx: int
) -> LabeledQuestion | None:
    """按题族名出题。

    输入：题族名、spec、RNG、语言字典、问法编号。
    输出：LabeledQuestion 或 None。预期行为：未知题族直接报错（不静默跳过）。
    另外：当 state 是数组（block_rows）时，图例与历史无处安放，
    这里统一把它们塞进 instructions 的 object 形态（其余格式返回原样）。
    """
    if family not in BUILDERS:
        raise KeyError(f"未知题族：{family}；可用：{', '.join(ALL_FAMILIES)}")
    item = BUILDERS[family](maze, rng, loc, idx)
    if item is None:
        return None
    extras = render.instructions_extras(maze, loc)
    if not extras:
        return item
    question = dict(item.question)
    question["instructions"] = {"question": question["instructions"], **extras}
    return LabeledQuestion(
        family=item.family, question=question, answer=item.answer, shuffle=item.shuffle
    )
