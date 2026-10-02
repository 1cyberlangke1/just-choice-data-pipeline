"""迷宫 workflow：一个**纯构造式** workflow（`needs` 为空，全程不调用任何模型）。

它实现了 `jc.workflow.Workflow` 的全部契约：
- 声明：`kinds`（三题型都产）、`needs`（空 = 不需要后端）、`judges`（规则判别器）；
- 计划：`split_names` / `config_for` / `build_cases` / `case_label` / `stats_dimensions` /
  `dimension_titles` / `plan_sample`；
- 生成：`sample` → `render` → `ask`（纯代码推导答案）→ `reparse`（往返校验）；
- 质量：`make_judges` / `audit`。

导入本模块即自注册（`jc.workflow.load("maze")` 可用）。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path


from jc.backend.types import GenerationBackend
from jc.judge import BARE, Judge, LOCALE, standard_judges
from jc.workflow import LabeledQuestion, SplitSpec, StandardMethods, register

# 说明：audit / judges 都在模块级 import 本模块的解析函数，所以这里**不能在模块级 import 它们**
# （会形成 workflow → audit → workflow 的循环），改为在用到的方法里延迟导入。
from . import grid, locales, questions, render
from . import spec as spec_mod

WORKFLOW_NAME = "maze"
WORKFLOW_VERSION = "0.2.0"

# 题族分组：决定哪些题族能在哪种 case 下出现
LOCAL_FAMILIES: tuple[str, ...] = ("step_result", "can_move", "dead_end", "exits")
PATH_FAMILIES: tuple[str, ...] = (
    "next_step",
    "on_optimal",
    "deviation",
    "path_via",
    "on_path",
    "path_length",
)
CUT_FAMILIES: tuple[str, ...] = ("reachable", *LOCAL_FAMILIES)

#: `deviation` 与 `on_optimal` 的**入选概率**：{题族: (dev=0 的 case 上的概率, dev≠0 上的概率)}。
#: 两族共用「当前位置在不在最优路径上」这一个自由度——dev=0 的 case 里 `deviation` 的答案
#: 恒为 0、`on_optimal` 的答案恒为 True——所以光靠 case 矩阵铺不平：`deviations=(0,0,1,2,3)`
#: 会让 deviation 的 0 档占 0.41（实测 0.164 的偏差）、on_optimal 的 True 占 0.41。
#: 按 case 的 deviation 调入选概率之后，两族的答案各自铺平（推导：设 dev=0 的 case 上
#: `deviation` 入选概率 a、dev≠0 上 b，则 0.4a/(0.4a+0.6b)=1/4 → b=2a；`on_optimal` 同理
#: 0.4c/(0.4c+0.6d)=1/2 → c=1.5d）。
FAMILY_PICK_WEIGHTS: dict[str, tuple[float, float]] = {
    "deviation": (0.5, 1.0),
    "on_optimal": (1.0, 2 / 3),
}

# 两个切分的采样参数：test_ood 用更大的迷宫、更全的格式/符号/问法（留出维度）
TRAIN_CONFIG = spec_mod.MazeConfig(
    size=(3, 7),
    formats=(
        spec_mod.FMT_BLOCK_TEXT,
        spec_mod.FMT_BLOCK_OBJECT,
        spec_mod.FMT_BLOCK_ROWS,
        spec_mod.FMT_WALL_ASCII,
    ),
    symbols=("hash", "block"),
    phrasing_pool=(0, 1),
)
OOD_CONFIG = spec_mod.MazeConfig(
    size=(8, 10),
    formats=spec_mod.FORMATS,
    symbols=tuple(spec_mod.SYMBOL_SETS),
    phrasing_pool=(0, 1, 2),
)

DIMENSION_TITLES: dict[str, str] = {
    "format": "渲染格式",
    "symbols": "符号集",
    "history_len": "历史长度（最近几次决策）",
    "mark_trail": "标记走过的路",
    "option_detail": "选项是否写目标方块",
    "deviation": "目标偏离步数",
    "cut": "是否出口不可达",
}


@dataclass(frozen=True)
class ParsedMaze:
    """从渲染文本反解出来的迷宫事实（用于与 spec 比对）。"""

    rows: int
    cols: int
    mask: tuple[int, ...]
    start: int | None
    exit: int | None
    marks: dict[str, int]
    current: int | None
    trail: list[int]


def _trailing_grid_lines(state: str) -> list[str]:
    """取字符串 state 末尾那段等长行（网格永远渲染在最后）。

    输入：多行字符串。输出：网格行列表。预期行为：末尾连续等长的行就是网格。
    """
    lines = state.split("\n")
    if not lines:
        return []
    width = len(lines[-1])
    out: list[str] = []
    for line in reversed(lines):
        if len(line) != width:
            break
        out.append(line)
    out.reverse()
    return out


def _parse_block(grid_lines: list[str], wall_ch: str, pass_ch: str) -> ParsedMaze:
    """解析块状格式（block_text / block_object）的网格行。"""
    rows = (len(grid_lines) - 1) // 2
    cols = (len(grid_lines[0]) - 1) // 2
    mask = [0] * (rows * cols)
    marks: dict[str, int] = {}
    start: int | None = None
    exit_cell: int | None = None
    current: int | None = None
    trail: list[int] = []
    for r in range(rows):
        for c in range(cols):
            cell = grid.index_of(r, c, cols)
            ch = grid_lines[2 * r + 1][2 * c + 1]
            if ch == spec_mod.GLYPH_START:
                start = cell
            elif ch == spec_mod.GLYPH_EXIT:
                exit_cell = cell
            elif ch == spec_mod.GLYPH_CURRENT:
                current = cell
            elif ch == spec_mod.GLYPH_TRAIL:
                trail.append(cell)
            elif ch in spec_mod.MARK_LETTERS:
                marks[ch] = cell
            if c + 1 < cols and grid_lines[2 * r + 1][2 * c + 2] == pass_ch:
                mask[cell] |= 1 << grid.DIR_RIGHT
                mask[cell + 1] |= 1 << grid.DIR_LEFT
            if r + 1 < rows and grid_lines[2 * r + 2][2 * c + 1] == pass_ch:
                mask[cell] |= 1 << grid.DIR_DOWN
                mask[cell + cols] |= 1 << grid.DIR_UP
    assert wall_ch  # 解析不依赖墙字符本身，仅保留参数以便将来校验边框
    return ParsedMaze(rows, cols, tuple(mask), start, exit_cell, marks, current, trail)


def _parse_wall_ascii(grid_lines: list[str]) -> ParsedMaze:
    """解析线框格式（wall_ascii）：格内字符取内容，格子之间空格表示通路。"""
    rows = (len(grid_lines) - 1) // 2
    cols = (len(grid_lines[0]) - 1) // 2
    mask = [0] * (rows * cols)
    marks: dict[str, int] = {}
    start: int | None = None
    exit_cell: int | None = None
    current: int | None = None
    trail: list[int] = []
    for r in range(rows):
        for c in range(cols):
            cell = grid.index_of(r, c, cols)
            ch = grid_lines[2 * r + 1][2 * c + 1]
            if ch == spec_mod.GLYPH_START:
                start = cell
            elif ch == spec_mod.GLYPH_EXIT:
                exit_cell = cell
            elif ch == spec_mod.GLYPH_CURRENT:
                current = cell
            elif ch == spec_mod.GLYPH_TRAIL:
                trail.append(cell)
            elif ch in spec_mod.MARK_LETTERS:
                marks[ch] = cell
            if c + 1 < cols and grid_lines[2 * r + 1][2 * c + 2] == " ":
                mask[cell] |= 1 << grid.DIR_RIGHT
                mask[cell + 1] |= 1 << grid.DIR_LEFT
            if r + 1 < rows and grid_lines[2 * r + 2][2 * c + 1] == " ":
                mask[cell] |= 1 << grid.DIR_DOWN
                mask[cell + cols] |= 1 << grid.DIR_UP
    return ParsedMaze(rows, cols, tuple(mask), start, exit_cell, marks, current, trail)


def _parse_passages(state: dict[str, JsonValue]) -> ParsedMaze:
    """解析结构化通路格式。"""
    rows, cols = int(state["size"][0]), int(state["size"][1])
    mask = [0] * (rows * cols)
    for a, b in state["passages"]:
        ca = grid.index_of(int(a[0]), int(a[1]), cols)
        cb = grid.index_of(int(b[0]), int(b[1]), cols)
        ar, ac = divmod(ca, cols)
        br, bc = divmod(cb, cols)
        d = grid.DELTAS.index((br - ar, bc - ac))
        mask[ca] |= 1 << d
        mask[cb] |= 1 << grid.OPPOSITE[d]
    marks = {
        letter: grid.index_of(int(pos[0]), int(pos[1]), cols)
        for letter, pos in state.get("marks", {}).items()
    }
    start = grid.index_of(int(state["start"][0]), int(state["start"][1]), cols)
    exit_cell = grid.index_of(int(state["exit"][0]), int(state["exit"][1]), cols)
    agent = state.get("agent") or {}
    current = None
    if "current" in agent:
        current = grid.index_of(int(agent["current"][0]), int(agent["current"][1]), cols)
    trail = [grid.index_of(int(pos[0]), int(pos[1]), cols) for pos in agent.get("trail", [])]
    return ParsedMaze(rows, cols, tuple(mask), start, exit_cell, marks, current, trail)


def parse_state(state: object, case: spec_mod.RenderCase) -> ParsedMaze | None:
    """从渲染出的 state 反解迷宫事实。

    输入：state（string / object / array）与它对应的渲染变体。
    输出：ParsedMaze，或 None（格式不支持反解）。
    预期行为：只按文本/结构本身解析，不复用渲染时的坐标计算，因此能抓到渲染错位。
    """
    fmt = case.fmt
    wall_ch, pass_ch = spec_mod.SYMBOL_SETS[case.symbols]
    if fmt == spec_mod.FMT_BLOCK_ROWS:
        if not isinstance(state, list) or not state:
            return None
        return _parse_block([str(line) for line in state], wall_ch, pass_ch)
    if fmt == spec_mod.FMT_BLOCK_OBJECT:
        if not isinstance(state, dict):
            return None
        return _parse_block([str(line) for line in state["grid"]], wall_ch, pass_ch)
    if fmt == spec_mod.FMT_PASSAGES:
        if not isinstance(state, dict):
            return None
        return _parse_passages(state)
    if not isinstance(state, str):
        return None
    grid_lines = _trailing_grid_lines(state)
    if not grid_lines:
        return None
    if fmt == spec_mod.FMT_WALL_ASCII:
        return _parse_wall_ascii(grid_lines)
    return _parse_block(grid_lines, wall_ch, pass_ch)


def roundtrip_diff(maze: spec_mod.MazeSpec, parsed: ParsedMaze) -> list[str]:
    """比对反解结果与真值 spec，返回差异列表（空 = 一致）。

    输入：spec、反解结果。
    输出：人类可读的差异列表。预期行为：逐项比较尺寸、通路、起止、标记、当前位置与走过的路。
    """
    diff: list[str] = []
    if (parsed.rows, parsed.cols) != (maze.rows, maze.cols):
        diff.append(f"尺寸不一致：渲染 {parsed.rows}x{parsed.cols} vs spec {maze.rows}x{maze.cols}")
    if parsed.mask != maze.mask:
        bad = [i for i in range(min(len(parsed.mask), len(maze.mask))) if parsed.mask[i] != maze.mask[i]]
        diff.append(f"通路不一致：{len(bad)} 个格子不同（前几个：{bad[:5]}）")
    if parsed.start != maze.start:
        diff.append(f"起点不一致：渲染 {parsed.start} vs spec {maze.start}")
    if parsed.exit != maze.exit:
        diff.append(f"出口不一致：渲染 {parsed.exit} vs spec {maze.exit}")
    if parsed.marks != maze.marks:
        diff.append(f"标记不一致：渲染 {parsed.marks} vs spec {maze.marks}")
    # 当前位置与起点重合时渲染不画 @（起点由 S 表示），比对时按 None 处理
    expected_current = maze.current() if maze.agent.moves else None
    if expected_current == maze.start:
        expected_current = None
    if parsed.current != expected_current:
        diff.append(f"当前位置不一致：渲染 {parsed.current} vs spec {expected_current}")
    if maze.render.mark_trail and maze.agent.moves:
        # 起点、出口与字母标记会保留自己的符号，所以「能显示成 · 的格子」要把它们排除
        reserved = {maze.start, maze.exit, *maze.marks.values()}
        expected_trail = [c for c in maze.trail()[:-1] if c not in reserved]
        if sorted(parsed.trail) != sorted(expected_trail):
            diff.append(
                f"走过的路标记不一致：渲染 {sorted(parsed.trail)} vs spec {sorted(expected_trail)}"
            )
    return diff


def _pick_families(
    rng: random.Random, config: spec_mod.MazeConfig, case: spec_mod.MazeCase, force_cut: bool
) -> tuple[tuple[str, ...], int | None]:
    """为一条样本挑题族。

    输入：RNG、配置、case、是否强制出「出口不可达」样本。
    输出：(题族名元组, path_via 的目标标记数)，后者给采样器定答案用（None = 自由）。
    预期行为：
    - 强制 cut 时只用「reachable + 局部题族」（依赖最优路径的题族会被跳过）；
    - deviation=1 时把 off_by_one 加进池子（这是「走偏了 1 格」的专用题族）；
    - `deviation` / `on_optimal` 按 case 的 deviation 调入选概率（见 `FAMILY_PICK_WEIGHTS`）；
    - `reachable` 在两种情况下都必出：cut 样本给 False、非 cut 给 True，真假刚好 1:1；
    - path_via 的目标（路径上有标记 / 全在路径外）先抽，再交给采样器去实现；
      目标是「全在路径外」时把 on_path 从池子里去掉（否则它的答案会全是 False）。
    """
    if force_cut:
        return tuple(CUT_FAMILIES), None
    pool: list[str] = []
    if case.deviation == 1:
        pool.append("off_by_one")
    pool.extend(["next_step", "on_optimal", "deviation"])
    pool.extend(PATH_FAMILIES)
    pool.extend(LOCAL_FAMILIES)
    pool = list(dict.fromkeys(pool))
    level = 0 if case.deviation == 0 else 1
    pool = [
        name
        for name in pool
        if rng.random() < FAMILY_PICK_WEIGHTS.get(name, (1.0, 1.0))[level]
    ]
    marks_on_path: int | None = None
    if "path_via" in pool:
        # 「都不经过」这种 no-match 选项天然占少数，用 1/4 权重去贴近「每个选项约 1/k」
        marks_on_path = 0 if rng.random() < 0.25 else 1
        if marks_on_path == 0:
            pool = [name for name in pool if name != "on_path"]
    low, high = config.questions_range
    count = min(rng.randint(low, high), len(pool))
    chosen = rng.sample(pool, count)
    if "reachable" not in chosen:
        chosen.append("reachable")
    return tuple(chosen), marks_on_path


@register(WORKFLOW_NAME)
class MazeWorkflow(StandardMethods):
    """迷宫 workflow（纯构造式）。"""

    name = WORKFLOW_NAME
    version = WORKFLOW_VERSION
    kinds = frozenset({"choice", "noul", "score"})
    needs: frozenset[str] = frozenset()  # 纯构造式：不需要任何后端
    judges: tuple[str, ...] = ("roundtrip", "label")
    #: 判别器名单表：名字 → （`judges` 里的类名，怎么造）。
    judge_table = {
        "roundtrip": ("RoundtripJudge", BARE),
        "label": ("LabelJudge", LOCALE),
    }
    split_names_tuple = ("train", "val", "test_iid", "test_ood")
    dimension_titles_map = DIMENSION_TITLES
    pkg_locales = locales

    # ---------- 声明与计划 ----------

    def config_for(self, split: str) -> spec_mod.MazeConfig:
        """输入：切分名。输出：该切分的采样参数（test_ood 用更大迷宫与更全格式）。"""
        if split not in self.split_names():
            raise KeyError(f"未知切分：{split!r}；可用：{self.split_names()}")
        return OOD_CONFIG if split == "test_ood" else TRAIN_CONFIG

    def build_cases(self, config: spec_mod.MazeConfig) -> list[spec_mod.MazeCase]:
        """按 7 个维度取全组合：格式 × 符号集 × 历史长度 × 标记走过的路 ×
        选项写目标方块 × 目标偏离步数 × 是否出口不可达。

        输入：采样配置。输出：case 列表（顺序稳定）。
        """
        cases: list[spec_mod.MazeCase] = []
        for fmt, sym, hist, trail, detail, dev, cut in product(
            config.formats,
            config.symbols,
            config.history_lens,
            config.mark_trails,
            config.option_details,
            config.deviations,
            (False, True),
        ):
            cases.append(
                spec_mod.MazeCase(
                    fmt=fmt,
                    symbols=sym,
                    history_len=hist,
                    mark_trail=trail,
                    option_detail=detail,
                    deviation=dev,
                    cut=cut,
                )
            )
        return cases

    def stats_dimensions(self, case: spec_mod.MazeCase) -> dict[str, JsonValue]:
        """把 case 摊平成统计维度：{维度名: 取值}。"""
        return {
            "format": case.fmt,
            "symbols": case.symbols,
            "history_len": case.history_len,
            "mark_trail": case.mark_trail,
            "option_detail": case.option_detail,
            "deviation": case.deviation,
            "cut": case.cut,
        }

    def plan_sample(
        self, rng: random.Random, config: spec_mod.MazeConfig, case: spec_mod.MazeCase, seq: int
    ) -> SplitSpec:
        """决定这一条怎么造：题族、是否强制出口不可达、path_via 的目标。

        输入：RNG、采样参数、case、样本序号（序号未用，保留给将来的分层采样）。
        输出：SplitSpec（params 里放 config / case / families / force_cut / marks_on_path /
        path_bucket）。
        预期行为：`path_length` 的答案桶（下标）在**这里**抽，随计划一起固定下来——
        `sample` 会为同一个计划换迷宫重试多次，桶必须跨重试不变，否则各桶的接受率不等
        会把短桶系统性挤掉（见 `spec.sample_spec` 里的说明）。不出 `path_length` 时为 None。
        """
        families, marks_on_path = _pick_families(rng, config, case, case.cut)
        path_bucket = (
            rng.choice(range(len(spec_mod.PATH_LENGTH_BUCKETS)))
            if "path_length" in families
            else None
        )
        return SplitSpec(
            name="train",  # 切分名只用于记录；真正的分层信息都在 params 里
            params={
                "config": config,
                "case": case,
                "families": families,
                "force_cut": case.cut,
                "marks_on_path": marks_on_path,
                "path_bucket": path_bucket,
            },
        )

    # ---------- 生成三步 ----------

    def sample(self, rng: random.Random, split: SplitSpec) -> spec_mod.MazeSpec:
        """采样一份迷宫 spec。

        输入：RNG、生成计划。
        输出：MazeSpec。预期行为：单个 case 构造失败时换迷宫重试，超过上限直接报错。
        """
        config: spec_mod.MazeConfig = split.params["config"]
        case: spec_mod.MazeCase = split.params["case"]
        families: tuple[str, ...] = split.params["families"]
        path_bucket = split.params.get("path_bucket")
        if not isinstance(path_bucket, int):
            path_bucket = None
        for _ in range(128):
            maze = spec_mod.sample_spec(
                rng,
                config,
                case,
                split.name,
                families=families,
                force_cut=bool(split.params.get("force_cut", False)),
                marks_on_path=split.params.get("marks_on_path"),
                path_bucket=path_bucket,
            )
            if maze is not None:
                return maze
        raise RuntimeError(f"采样失败：{case.describe()} 在 128 次重试内没有构造出合法 spec")

    def render(self, maze: spec_mod.MazeSpec, rng: random.Random, locale: str) -> object:
        """渲染 state（渲染格式与变体都写在 spec 里，rng 未使用）。"""
        return render.render_state(maze, locales.load(locale))

    def ask(
        self, maze: spec_mod.MazeSpec, rng: random.Random, locale: str
    ) -> list[LabeledQuestion]:
        """按 spec.families 出题。

        输入：spec、RNG、语言代码。
        输出：题目列表（不成立的题族会被跳过）。
        预期行为：题族顺序与 spec 一致；每个题族用 spec.phrasing 里定的问法编号。
        """
        loc = locales.load(locale)
        out: list[LabeledQuestion] = []
        for family, idx in zip(maze.families, maze.phrasing, strict=False):
            item = questions.build_one(family, maze, rng, loc, idx)
            if item is not None:
                out.append(item)
        return out

    def reparse(self, state: object, *, case: spec_mod.RenderCase | None = None) -> ParsedMaze | None:
        """反解渲染文本（需要渲染变体 case；缺省返回 None）。"""
        if case is None:
            return None
        return parse_state(state, case)

    # ---------- 质量 ----------

    def make_judges(self, names: tuple[str, ...], locale: str) -> tuple[Judge, ...]:
        """输入：判别器名单、语言。输出：判别器实例。预期行为：未知名字直接报错。

        延迟导入：`judges` 在模块级 import 本模块的解析函数，模块级取类会成环。
        """
        from . import judges as judges_mod
        return standard_judges(names, locale, judges_mod, None, self.judge_table)

    def audit(
        self, run_dir: Path, splits: tuple[str, ...], locale: str
    ) -> tuple[dict[str, JsonValue], list[str]]:
        """审计产物（委托 audit 模块）。"""
        from . import audit as audit_mod  # 延迟导入：避免与 audit 互相 import 成环

        return audit_mod.audit_run(run_dir, splits, locale)

    def bind_backend(self, backend: GenerationBackend) -> None:
        """空操作：迷宫是纯构造式（`needs` 为空），管线不会给后端，也不该给。

        输入：生成后端（实际不会收到）。输出：无。
        预期行为：留着这个方法只为满足 `Workflow` 协议；收到后端说明接线错了，直接报错。
        """
        raise AssertionError("迷宫 workflow 是纯构造式的，不该被绑定生成后端")
