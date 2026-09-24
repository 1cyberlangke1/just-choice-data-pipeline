"""迷宫产物的审计：直接读 JSONL，核对「要求的 case」是否真的都在数据里。

与「case 矩阵」不同，这里看的是**落地产物**：渲染格式、是否带历史、选项是否写目标方块、
是否标了走过的路、走偏 1 格的样本是否真的偏离 1 格（从 state 反解后重新算距离）。

输入：run 目录、切分列表、语言代码。
输出：(各切分计数, 缺失项列表)；缺失项为空才算通过。
"""

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from typing import Any

from jc import jsonio

from . import grid, locales
from . import spec as maze_spec
from .workflow import ParsedMaze, parse_state


def infer_format(state: Any) -> str:
    """从 state 的形态反推渲染格式（审计时手上没有 spec，只能看产物本身）。

    输入：state。
    输出：格式名。预期行为：数组→block_rows；带 passages→passages；带 grid→block_object；
    字符串里出现以 `+` 开头的行→wall_ascii；其余按 block_text。
    """
    if isinstance(state, list):
        return maze_spec.FMT_BLOCK_ROWS
    if isinstance(state, dict):
        return maze_spec.FMT_PASSAGES if "passages" in state else maze_spec.FMT_BLOCK_OBJECT
    if isinstance(state, str):
        if any(line.startswith("+") for line in state.split("\n")):
            return maze_spec.FMT_WALL_ASCII
        return maze_spec.FMT_BLOCK_TEXT
    return "unknown"


def infer_symbols(
    state: Any, loc: dict[str, Any], instructions: list[Any] | None = None
) -> str | None:
    """从图例反推符号集名字。

    输入：state、语言字典、可选的全部 instructions（block_rows 的图例塞在 instructions 里）。
    输出：符号集名（hash/block/shade/xo）或 None。
    预期行为：图例不一定在第一行（带历史时第一行是历史文本），要找到第一条含 `=` 的行。
    """
    legend: dict[str, str] = {}
    if isinstance(state, dict) and isinstance(state.get("legend"), dict):
        legend = state["legend"]
    elif isinstance(state, str):
        for line in state.split("\n"):
            if "=" not in line:
                continue
            for chunk in line.split(", "):
                if "=" in chunk:
                    glyph, _, text = chunk.partition("=")
                    legend[glyph] = text
            break
    if not legend and instructions:
        for item in instructions:
            if isinstance(item, dict) and isinstance(item.get("legend"), dict):
                legend = item["legend"]
                break
    wall = next((g for g, text in legend.items() if text == loc["legend"]["wall"]), None)
    passage = next((g for g, text in legend.items() if text == loc["legend"]["passage"]), None)
    for name, (wall_ch, pass_ch) in maze_spec.SYMBOL_SETS.items():
        if wall == wall_ch and passage == pass_ch:
            return name
    return None


def infer_case(
    state: Any, loc: dict[str, Any], instructions: list[Any] | None = None
) -> maze_spec.RenderCase | None:
    """从产物反推渲染变体（审计用；格式与符号集都是推断出来的）。"""
    fmt = infer_format(state)
    if fmt == "unknown":
        return None
    if fmt == maze_spec.FMT_PASSAGES:
        symbols = "hash"  # passages 格式不用符号，解析路径不依赖它
    else:
        symbols = infer_symbols(state, loc, instructions)
        if symbols is None:
            return None
    return maze_spec.RenderCase(
        fmt=fmt,
        symbols=symbols,
        mark_trail=maze_spec.GLYPH_TRAIL in str(state),
        history_len=0,
        option_detail=maze_spec.OPTION_CELL,
    )


def dist_to_set(mask: tuple[int, ...], rows: int, cols: int, sources: set[int]) -> dict[int, int]:
    """多源 BFS：每格到 sources 集合的最短距离（独立实现，供审计核对）。"""
    dist = {src: 0 for src in sources}
    queue = deque(sources)
    while queue:
        cur = queue.popleft()
        for _, nb in grid.open_neighbors(mask, cur, rows, cols):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
    return dist


def family_of(qid: str) -> str:
    """题号 → 题族名（去掉 `_2` 之类后缀）。"""
    return "".join(ch for ch in qid if not ch.isdigit()).rstrip("_")


def audit_file(path: Path, loc: dict[str, Any]) -> dict[str, Any]:
    """审计一个 JSONL 文件。

    输入：文件路径、语言字典。
    输出：计数字典（格式分布、各 case 计数、题族计数、走偏 1 格的实测结果）。
    预期行为：解析不出来的样本计入 unparsed；走偏 1 格要重新算距离，不能只数题族。
    """
    counts: dict[str, Any] = {
        "samples": 0,
        "formats": Counter(),
        "families": Counter(),
        "history": 0,
        "option_cell": 0,
        "trail": 0,
        "cut_samples": 0,
        "off_by_one": 0,
        "off_by_one_dev1": 0,
        "next_step_on_path": 0,
        "next_step": 0,
        "unparsed": 0,
        "unparsed_formats": Counter(),
        "unparsed_symbols": Counter(),
        "types": Counter(),
    }
    for _, sample in jsonio.iter_jsonl(path):
        counts["samples"] += 1
        state = sample["state"]
        blob = str(state) + str([q.get("instructions") for q in sample["questions"].values()])
        counts["formats"][infer_format(state)] += 1
        if loc["history"]["prefix"] in blob:
            counts["history"] += 1
        if any("→" in option for q in sample["questions"].values() for option in (q.get("criteria") or {})):
            counts["option_cell"] += 1
        if maze_spec.GLYPH_TRAIL in str(state) or "trail" in str(state):
            counts["trail"] += 1
        instructions = [q.get("instructions") for q in sample["questions"].values()]
        case = infer_case(state, loc, instructions)
        parsed: ParsedMaze | None = parse_state(state, case) if case else None
        if parsed is None:
            counts["unparsed"] += 1
            counts["unparsed_formats"][infer_format(state)] += 1
            counts["unparsed_symbols"][
                str(infer_symbols(state, loc, instructions)) if case is None else "case_ok"
            ] += 1
            continue
        if parsed.current is None:
            counts["cut_samples"] += 1
        path = grid.shortest_path(parsed.mask, parsed.rows, parsed.cols, parsed.start, parsed.exit)
        path_set = set(path or [])
        current = parsed.current if parsed.current is not None else parsed.start
        dist = dist_to_set(parsed.mask, parsed.rows, parsed.cols, path_set) if path_set else {}
        for qid, question in sample["questions"].items():
            family = family_of(qid)
            counts["families"][family] += 1
            counts["types"][question["type"]] += 1
            if family == "off_by_one":
                counts["off_by_one"] += 1
                if dist.get(current) == 1:
                    counts["off_by_one_dev1"] += 1
            if family == "next_step":
                counts["next_step"] += 1
                if current in path_set:
                    counts["next_step_on_path"] += 1
    return counts


def audit_run(run_dir: Path, splits: tuple[str, ...], locale: str) -> tuple[dict[str, Any], list[str]]:
    """审计一个 run 目录的多个切分。

    输入：run 目录、切分列表、语言。
    输出：(各切分计数, 缺失项列表)。预期行为：缺失项为空才算通过。
    """
    loc = locales.load(locale)
    report: dict[str, Any] = {}
    missing: list[str] = []
    for split in splits:
        path = run_dir / f"{split}.jsonl"
        if not path.exists():
            missing.append(f"{split}.jsonl 不存在")
            continue
        counts = audit_file(path, loc)
        report[split] = counts
        if len([f for f in counts["formats"] if f != "unknown"]) < 3:
            missing.append(f"{split}：渲染格式少于 3 种（{dict(counts['formats'])}）")
        if counts["history"] == 0:
            missing.append(f"{split}：没有带历史决策的样本")
        if counts["option_cell"] == 0:
            missing.append(f"{split}：没有「选项写明目标方块」的样本")
        if counts["trail"] == 0:
            missing.append(f"{split}：没有标记走过的路的样本")
        if counts["off_by_one"] == 0 or counts["off_by_one_dev1"] == 0:
            missing.append(f"{split}：缺少「走偏 1 格」的样本")
        if counts["next_step_on_path"] == 0:
            missing.append(f"{split}：缺少「沿最优路径走」的样本")
    return report, missing
