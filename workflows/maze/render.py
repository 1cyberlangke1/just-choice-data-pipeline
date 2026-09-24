"""迷宫渲染：把 spec 变成契约允许的 state（string / object / array）。

五种格式（覆盖三种 state 形态与两种 instructions 形态）：
- block_text   → string：图例行 + 字符网格（(2r+1) x (2c+1)）
- block_object → object：{"legend": {...}, "grid": [...]}
- block_rows   → array：每个元素一行；图例与历史改放进 instructions 的 object 形态
- wall_ascii   → string：经典 `+--+` 线框画法
- passages     → object：结构化通路列表（自解释，不需要图例）

渲染变体（来自 spec.render）：
- mark_trail：把走过的格子标成 `·`（不覆盖 S/E/字母标记）
- history_len：在 state 里带上最近 n 次决策（0 = 不带）
- option_detail：只影响「方向型选项」的文案（在 questions 里用）

输入：spec、语言字典。输出：state（或渲染片段）。
"""

from __future__ import annotations

from typing import Any, Final

from . import grid, spec as spec_mod

DIR_KEYS: Final[tuple[str, ...]] = ("up", "right", "down", "left")


def cell_overlay(maze: spec_mod.MazeSpec) -> dict[int, str]:
    """算出「格子 → 覆盖字符」。

    输入：spec。
    输出：{格子索引: 字符}。
    预期行为：字母标记与 S/E 优先于走过的路；当前位置 `@` 优先级最高，
    但只有当智能体确实离开起点后才标（否则会盖掉 S）。
    """
    overlay: dict[int, str] = {}
    for letter, cell in sorted(maze.marks.items()):
        overlay[cell] = letter
    overlay[maze.start] = spec_mod.GLYPH_START
    overlay[maze.exit] = spec_mod.GLYPH_EXIT
    if maze.render.mark_trail and maze.agent.moves:
        for cell in maze.trail()[:-1]:
            overlay.setdefault(cell, spec_mod.GLYPH_TRAIL)
    # 当前位置与起点重合时不画 @：起点已由 S 标出，画上去会把 S 盖掉、也无法反解
    if maze.agent.moves and maze.current() != maze.start:
        overlay[maze.current()] = spec_mod.GLYPH_CURRENT
    return overlay


def _symbols(maze: spec_mod.MazeSpec) -> tuple[str, str]:
    """取当前符号集的 (墙字符, 通道字符)。"""
    return spec_mod.SYMBOL_SETS[maze.render.symbols]


def grid_lines(maze: spec_mod.MazeSpec) -> list[str]:
    """字符网格（块状格式）：(2r+1) x (2c+1) 行。

    输入：spec。
    输出：行字符串列表。
    预期行为：格子在奇数行奇数列；格子之间按通路有无画墙或通道；最外圈一律是墙。
    """
    wall_ch, pass_ch = _symbols(maze)
    overlay = cell_overlay(maze)
    rows, cols, mask = maze.rows, maze.cols, maze.mask
    lines: list[str] = []
    for gr in range(2 * rows + 1):
        chars: list[str] = []
        for gc in range(2 * cols + 1):
            if gr % 2 == 1 and gc % 2 == 1:
                cell = (gr // 2) * cols + (gc // 2)
                chars.append(overlay.get(cell, pass_ch))
            elif gr % 2 == 1:
                r, c = gr // 2, gc // 2
                opened = c > 0 and grid.has_passage(
                    mask, grid.index_of(r, c - 1, cols), grid.DIR_RIGHT, rows, cols
                )
                chars.append(pass_ch if opened else wall_ch)
            elif gc % 2 == 1:
                r, c = gr // 2, gc // 2
                opened = r > 0 and grid.has_passage(
                    mask, grid.index_of(r - 1, c, cols), grid.DIR_DOWN, rows, cols
                )
                chars.append(pass_ch if opened else wall_ch)
            else:
                chars.append(wall_ch)
        lines.append("".join(chars))
    return lines


def wall_ascii_lines(maze: spec_mod.MazeSpec) -> list[str]:
    """经典线框画法：`+--+--+` 与 `|` 表示墙，格子内写内容。

    输入：spec。输出：行字符串列表。预期行为：边框一律画墙；格内字符取覆盖表。
    """
    overlay = cell_overlay(maze)
    rows, cols, mask = maze.rows, maze.cols, maze.mask
    lines: list[str] = []
    for gr in range(2 * rows + 1):
        chars: list[str] = []
        for gc in range(2 * cols + 1):
            if gr % 2 == 0 and gc % 2 == 0:
                chars.append("+")
            elif gr % 2 == 0:
                r, c = gr // 2, gc // 2
                opened = r > 0 and grid.has_passage(
                    mask, grid.index_of(r - 1, c, cols), grid.DIR_DOWN, rows, cols
                )
                chars.append(" " if opened else "-")
            elif gc % 2 == 0:
                r, c = gr // 2, gc // 2
                opened = c > 0 and grid.has_passage(
                    mask, grid.index_of(r, c - 1, cols), grid.DIR_RIGHT, rows, cols
                )
                chars.append(" " if opened else "|")
            else:
                cell = (gr // 2) * cols + (gc // 2)
                chars.append(overlay.get(cell, " "))
        lines.append("".join(chars))
    return lines


def legend_map(maze: spec_mod.MazeSpec, loc: dict[str, Any]) -> dict[str, str]:
    """图例：符号 → 含义。

    输入：spec、语言字典。
    输出：{符号: 说明}。预期行为：只列当前真正用到的符号（避免误导）。
    """
    wall_ch, pass_ch = _symbols(maze)
    out: dict[str, str] = {wall_ch: loc["legend"]["wall"], pass_ch: loc["legend"]["passage"]}
    out[spec_mod.GLYPH_START] = loc["legend"]["start"]
    out[spec_mod.GLYPH_EXIT] = loc["legend"]["exit"]
    if maze.agent.moves:
        out[spec_mod.GLYPH_CURRENT] = loc["legend"]["current"]
    if maze.render.mark_trail and maze.agent.moves:
        out[spec_mod.GLYPH_TRAIL] = loc["legend"]["trail"]
    for letter in sorted(maze.marks):
        out[letter] = loc["legend"]["marks"]
    return out


def legend_line(maze: spec_mod.MazeSpec, loc: dict[str, Any]) -> str:
    """图例的单行文本（块状字符串格式用）。

    输入：spec、语言字典。
    输出：形如「#=墙  .=通道  S=起点 …」的一行。
    预期行为：顺序与 legend_map 一致；**条目之间用「, 」分隔而不是空格**，
    因为 `block` 符号集的通道字符本身就是空格，用空格分隔会把图例切坏、无法反解。
    """
    return ", ".join(f"{glyph}={text}" for glyph, text in legend_map(maze, loc).items())


def history_text(maze: spec_mod.MazeSpec, loc: dict[str, Any]) -> str | None:
    """最近 n 次决策的文本；不需要时返回 None。

    输入：spec、语言字典。
    输出：形如「已走过的最近决策：向上 → 向右」的字符串，或 None。
    预期行为：history_len<=0 或无历史时返回 None；只取最后 n 步。
    """
    if maze.render.history_len <= 0 or not maze.agent.moves:
        return None
    moves = maze.agent.moves[-maze.render.history_len :]
    words = [loc["moves"][DIR_KEYS[d]] for d in moves]
    return loc["history"]["prefix"] + loc["history"]["sep"].join(words)


def passages_payload(maze: spec_mod.MazeSpec, loc: dict[str, Any]) -> dict[str, Any]:
    """结构化通路格式的 state。

    输入：spec、语言字典。
    输出：含 size / start / exit / marks / passages 的对象；
    智能体存在时附 `agent.current`，开了 mark_trail 时再附 `agent.trail`。
    """
    payload: dict[str, Any] = {
        "size": [maze.rows, maze.cols],
        "start": list(grid.rowcol_of(maze.start, maze.cols)),
        "exit": list(grid.rowcol_of(maze.exit, maze.cols)),
        "marks": {k: list(grid.rowcol_of(v, maze.cols)) for k, v in sorted(maze.marks.items())},
        "passages": [
            [list(grid.rowcol_of(a, maze.cols)), list(grid.rowcol_of(b, maze.cols))]
            for a, b in grid.edges(maze.mask, maze.rows, maze.cols)
        ],
    }
    if maze.agent.moves:
        agent: dict[str, Any] = {"current": list(grid.rowcol_of(maze.current(), maze.cols))}
        if maze.render.mark_trail:
            # 与字符格式保持一致：起点/出口/字母标记会保留自己的符号，所以不进 trail
            reserved = {maze.start, maze.exit, *maze.marks.values()}
            agent["trail"] = [
                list(grid.rowcol_of(c, maze.cols))
                for c in maze.trail()[:-1]
                if c not in reserved
            ]
        payload["agent"] = agent
    return payload


def render_state(maze: spec_mod.MazeSpec, loc: dict[str, Any]) -> Any:
    """按 spec.render.fmt 渲染 state。

    输入：spec、语言字典。
    输出：string / object / array（契约允许的三种形态之一）。
    预期行为：
    - block_text / wall_ascii：历史行（若有）+ 图例行 + 网格，用 "\\n" 连接；
    - block_object：{"legend", "grid", "history"?}；
    - block_rows：只返回网格行数组（历史与图例交给 instructions 的 object 形态）；
    - passages：结构化对象（历史作为 "history" 字段）。
    """
    fmt = maze.render.fmt
    history = history_text(maze, loc)
    if fmt == spec_mod.FMT_BLOCK_TEXT:
        parts = [p for p in (history, legend_line(maze, loc)) if p]
        return "\n".join([*parts, *grid_lines(maze)])
    if fmt == spec_mod.FMT_WALL_ASCII:
        parts = [p for p in (history, legend_line(maze, loc)) if p]
        return "\n".join([*parts, *wall_ascii_lines(maze)])
    if fmt == spec_mod.FMT_BLOCK_OBJECT:
        payload: dict[str, Any] = {"legend": legend_map(maze, loc), "grid": grid_lines(maze)}
        if history:
            payload["history"] = history
        return payload
    if fmt == spec_mod.FMT_BLOCK_ROWS:
        return grid_lines(maze)
    if fmt == spec_mod.FMT_PASSAGES:
        payload = passages_payload(maze, loc)
        if history:
            payload["history"] = history
        return payload
    raise ValueError(f"未知渲染格式：{fmt!r}")


def instructions_extras(maze: spec_mod.MazeSpec, loc: dict[str, Any]) -> dict[str, Any]:
    """array 型 state（block_rows）无法自带图例与历史，改由 instructions 的 object 形态承载。

    输入：spec、语言字典。
    输出：需要附加到 instructions 的字段（可能为空字典）。
    """
    if maze.render.fmt != spec_mod.FMT_BLOCK_ROWS:
        return {}
    extras: dict[str, Any] = {"legend": legend_map(maze, loc)}
    history = history_text(maze, loc)
    if history:
        extras["history"] = history
    return extras


def cell_kind_label(
    maze: spec_mod.MazeSpec, loc: dict[str, Any], direction: int
) -> str:
    """方向型选项的「目标方块是什么」文案（option_detail=cell 时用）。

    输入：spec、语言字典、方向。
    输出：如「墙壁（走不通）」「普通通道」「出口 E」「标记格 A」「迷宫外」。
    预期行为：目标越界 → 迷宫外；无通路 → 墙壁；否则按目标格身份给出标签。
    """
    cell = maze.current() if maze.agent.moves else maze.start
    nxt = grid.neighbor(cell, direction, maze.rows, maze.cols)
    if nxt is None:
        return loc["cell"]["outside"]
    if not grid.has_passage(maze.mask, cell, direction, maze.rows, maze.cols):
        return loc["cell"]["wall"]
    for letter, mark_cell in sorted(maze.marks.items()):
        if mark_cell == nxt:
            return loc["cell"]["mark"].format(letter=letter)
    if nxt == maze.start:
        return loc["cell"]["start"]
    if nxt == maze.exit:
        return loc["cell"]["exit"]
    if maze.render.mark_trail and nxt in set(maze.trail()[:-1]):
        return loc["cell"]["trail"]
    return loc["cell"]["passage"]
