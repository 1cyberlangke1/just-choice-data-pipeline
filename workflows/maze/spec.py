"""迷宫的真值规格（spec）与 case 采样。

设计要点：
- spec 是「真值先行」的核心：从同一份 spec 既能渲染 state，也能纯代码推导所有金答案；
- 渲染变体（格式、符号集、是否标记走过的路、显示最近几次决策、选项是否写明目标方块）
  与构造变体（当前位置偏离最优路径几步）都属于 spec，因为 render 与 ask 都要用到；
- 全部随机性来自传入的 RNG，同一 RNG 状态必得同一 spec。

输入：RNG + MazeConfig + MazeCase + 切分名。
输出：MazeSpec（或 None 表示该 case 在当前迷宫里实现不了，调用方换一个重试）。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Final

from jc.knobs import require_bounds, require_pool

from . import grid

# 渲染格式
FMT_BLOCK_TEXT: Final[str] = "block_text"
FMT_BLOCK_OBJECT: Final[str] = "block_object"
FMT_BLOCK_ROWS: Final[str] = "block_rows"
FMT_WALL_ASCII: Final[str] = "wall_ascii"
FMT_PASSAGES: Final[str] = "passages"
FORMATS: Final[tuple[str, ...]] = (
    FMT_BLOCK_TEXT,
    FMT_BLOCK_OBJECT,
    FMT_BLOCK_ROWS,
    FMT_WALL_ASCII,
    FMT_PASSAGES,
)

# 符号集（墙 / 通道），随机抽取以防模型死记某一套符号
SYMBOL_SETS: Final[dict[str, tuple[str, str]]] = {
    "hash": ("#", "."),
    "block": ("█", " "),
    "shade": ("▓", "░"),
    "xo": ("X", "_"),
}

# 选项是否写明「走向的方块是什么」
OPTION_PLAIN: Final[str] = "plain"
OPTION_CELL: Final[str] = "cell"

# path_length 题族的分桶区间（含端点）；语言文件里的等级文案与这里一一对应。
# 用代码而非文案保存区间，是因为采样时要按目标桶去挑起止点（先定答案再挑输入）。
PATH_LENGTH_BUCKETS: Final[tuple[tuple[int, int], ...]] = (
    (1, 3),
    (4, 7),
    (8, 10**9),
)

#: 要问 `exits` 时至少放几个标记格：角色顺序是「路径上 → 死胡同 → 拐角 → 路口」，
#: 三个档位占后三个角色，少一个就有档位在样本里恒取不到。
MARKS_FOR_EXITS: Final[int] = 4

# 固定标记符（不参与符号集随机，避免歧义）
GLYPH_START: Final[str] = "S"
GLYPH_EXIT: Final[str] = "E"
GLYPH_CURRENT: Final[str] = "@"
GLYPH_TRAIL: Final[str] = "·"
MARK_LETTERS: Final[str] = "ABCD"


@dataclass(frozen=True)
class RenderCase:
    """渲染变体：同一个迷宫可以长成不同样子。"""

    fmt: str
    symbols: str
    mark_trail: bool
    history_len: int
    option_detail: str


@dataclass(frozen=True)
class AgentSpec:
    """智能体的行走状态。

    字段：
    - moves：从起点 S 出发依次走过的方向序列（可能为空 = 还在起点）；
    - deviation：当前位置到「S→E 最优路径」的最短距离（0 表示就在最优路径上）；
      **-1 表示出口不可达、根本没有最优路径**（cut 样本），此时依赖路径的题族一律不出。
    """

    moves: tuple[int, ...]
    deviation: int


@dataclass(frozen=True)
class MazeCase:
    """一个 case 组合（渲染 + 构造变体），供管线按矩阵遍历以保覆盖。

    cut=True 时该样本必定造出「出口不可达」：只出 reachable 与局部题族、不放智能体，
    此时 deviation 不生效。
    """

    fmt: str
    symbols: str
    history_len: int
    mark_trail: bool
    option_detail: str
    deviation: int
    cut: bool = False

    def describe(self) -> str:
        """case 的一行描述，用于统计与文件名。"""
        trail = "trail" if self.mark_trail else "notrail"
        cut = "cut" if self.cut else "nocut"
        return (
            f"{self.fmt}/{self.symbols}/h{self.history_len}/{trail}/"
            f"{self.option_detail}/d{self.deviation}/{cut}"
        )


@dataclass(frozen=True)
class MazeConfig:
    """迷宫的采样参数（领域相关，但全部可配置）。"""

    size: tuple[int, int] = (3, 7)
    algos: tuple[str, ...] = ("dfs", "prim", "wilson")
    formats: tuple[str, ...] = FORMATS
    symbols: tuple[str, ...] = ("hash", "block", "shade")
    history_lens: tuple[int, ...] = (0, 1, 2, 3, 5)
    mark_trails: tuple[bool, ...] = (False, True)
    option_details: tuple[str, ...] = (OPTION_PLAIN, OPTION_CELL)
    # 重复取值即加权：两个 d0 让「智能体在最优路径上」的样本占 40%，
    # 这样 on_optimal（真假）与 deviation（0/1/2/3+）两族的答案分布都不至于太偏。
    # 注意这两族是**耦合**的：P(dev=0) 同时是 on_optimal 的 True 占比，
    # 所以不可能让「deviation 四档各 25%」与「on_optimal 真假 50/50」同时成立。
    deviations: tuple[int, ...] = (0, 0, 1, 2, 3)
    # 标记格数量：默认至少 3 个，好让 exits 的三档（死胡同 / 拐角 / 路口）
    # 以及 path_via 的「有标记在最优路径上」都有候选可用
    marks_range: tuple[int, int] = (3, 4)
    questions_range: tuple[int, int] = (3, 6)
    braid: float = 0.0
    cut_prob: float = 0.15
    phrasing_pool: tuple[int, ...] = (0, 1, 2)  # 每种问法的可选编号（test_ood 用全集）

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在构造期就报人话，不拖到采样阶段。

        以前空表会活到 `rng.choice`、少元素的区间会活到 `rng.randint(*...)`，
        抛的都是裸 Python 异常（校验规则见 `jc.knobs`）。
        """
        require_bounds("size", self.size)
        require_bounds("marks_range", self.marks_range)
        require_bounds("questions_range", self.questions_range)
        for field, values in (
            ("algos", self.algos),
            ("formats", self.formats),
            ("symbols", self.symbols),
            ("history_lens", self.history_lens),
            ("mark_trails", self.mark_trails),
            ("option_details", self.option_details),
            ("deviations", self.deviations),
            ("phrasing_pool", self.phrasing_pool),
        ):
            require_pool(field, values)


@dataclass(frozen=True)
class MazeSpec:
    """一份迷宫真值规格。"""

    rows: int
    cols: int
    algo: str
    mask: tuple[int, ...]
    start: int
    exit: int
    marks: dict[str, int]
    cut: tuple[int, int] | None
    braid: float
    transform: str
    render: RenderCase
    agent: AgentSpec
    families: tuple[str, ...]
    phrasing: tuple[int, ...] = ()  # 与 families 一一对应：每个题族用第几种问法

    # ---------- 派生量（每次现算，避免 spec 里存冗余状态） ----------

    def path(self) -> list[int] | None:
        """S→E 的最短路（完美迷宫下唯一）；出口被砍断时返回 None。"""
        return grid.shortest_path(self.mask, self.rows, self.cols, self.start, self.exit)

    def dist_from_exit(self) -> list[int]:
        """各格到出口 E 的最短步数（-1 = 不可达）。"""
        return grid.distances(self.mask, self.rows, self.cols, self.exit)

    def current(self) -> int:
        """当前位置（从起点按 moves 走完后的格子）。"""
        cells = grid.apply_moves(self.mask, self.rows, self.cols, self.start, self.agent.moves)
        return cells[-1]

    def trail(self) -> list[int]:
        """走过的格子序列（含起点）。"""
        return grid.apply_moves(self.mask, self.rows, self.cols, self.start, self.agent.moves)

    def reachable(self) -> bool:
        """从 S 能否走到 E。"""
        return self.dist_from_exit()[self.start] >= 0


def _distance_to_set(
    mask: tuple[int, ...], rows: int, cols: int, sources: set[int]
) -> dict[int, int]:
    """多源 BFS：每个格子到 sources 集合的最短距离。

    输入：mask、行列数、源点集合。输出：{格子: 距离}（源点自身为 0）。
    """
    dist: dict[int, int] = {src: 0 for src in sources}
    frontier = list(sources)
    step = 0
    while frontier:
        step += 1
        nxt: list[int] = []
        for cell in frontier:
            for _, nb in grid.open_neighbors(mask, cell, rows, cols):
                if nb not in dist:
                    dist[nb] = step
                    nxt.append(nb)
        frontier = nxt
    return dist


def _pick_start_exit(
    rng: random.Random,
    mask: tuple[int, ...],
    rows: int,
    cols: int,
) -> tuple[int, int]:
    """挑起点与出口：出口在「距离 ≥ 2 的格子」里均匀取，保证路径有长度又不过度偏长。

    输入：RNG、mask、行列数。
    输出：(start, exit)。
    预期行为：最短距离 ≥ 2（小迷宫会退化，此时放宽到 ≥1）。这里刻意不做「只取最远格」的偏置：
    否则路径长度会全挤进长桶，`path_length` 题族的答案分布必然失衡。要**按桶定答案**的
    走 `_pick_start_exit_in_bucket`。
    """
    total = rows * cols
    for _ in range(64):
        start = rng.randrange(total)
        dist = grid.distances(mask, rows, cols, start)
        pool = [c for c in range(total) if dist[c] >= 2]
        if not pool:
            pool = [c for c in range(total) if dist[c] >= 1]
        if not pool:
            continue
        return start, rng.choice(pool)
    return 0, total - 1


def _pick_start_exit_in_bucket(
    rng: random.Random, mask: tuple[int, ...], rows: int, cols: int, bucket: tuple[int, int]
) -> tuple[int, int] | None:
    """挑一对起止格，使最短距离落在 `bucket` 里。

    输入：RNG、mask、行列数、目标距离桶 (low, high)。
    输出：(start, exit)，或 None（这个迷宫实现不了该桶，调用方**带着同一个桶**换迷宫重试）。
    预期行为：起点洗牌后逐个试，第一个能走进桶里的起点就定下来。**不退回别的桶**——
    `path_length` 题族是「先定答案（桶）再挑输入」（范式 §5.2），退回就等于让迷宫来决定
    答案分布（实测那样第 3 档只剩 0.25）。
    """
    low, high = bucket
    total = rows * cols
    starts = list(range(total))
    rng.shuffle(starts)
    for start in starts:
        dist = grid.distances(mask, rows, cols, start)
        hit = [c for c in range(total) if low <= dist[c] <= high]
        if hit:
            return start, rng.choice(hit)
    return None


def _pick_marks(
    rng: random.Random, mask: tuple[int, ...], rows: int, cols: int, start: int, exit: int, count: int
    , *, exclude: set[int] | None = None, marks_on_path: int | None = None
) -> dict[str, int]:
    """挑标记格：按「角色优先级」挑，让常见题型的两侧答案都有人能实现。

    输入：RNG、mask、行列数、起止格、标记数量、额外要避开的格子（如智能体当前位置）、
    最优路径上要放几个标记（None = 按角色优先级自由挑；0 = 全部放路径外；1 = 恰好一个在路径上）。
    输出：{"A": cell, ...}。
    预期行为：标记不与起点/出口/当前位置重合（否则会被 S/E/@ 覆盖，往返校验会失败），
    角色优先级为「最优路径上 → 死胡同 → 路径外 → 高连通度（3~4 个方向）」，
    这样 dead_end / on_path / path_via / exits 的真假两侧都不至于全空，答案分布更均衡；
    位次不够时用随机格子补齐；字母按 A、B、C… 顺序分配。
    """
    banned = {start, exit} | set(exclude or ())
    path = grid.shortest_path(mask, rows, cols, start, exit)
    on_path = [c for c in (path or []) if c not in banned]
    others = [c for c in range(rows * cols) if c not in banned and c not in set(on_path)]
    if marks_on_path == 0:
        on_path.clear()  # 全部标记放路径外：清空「路径上」这个候选池（重新绑定会让类型悬空）
    elif marks_on_path == 1 and not on_path:
        marks_on_path = None  # 路径太短、放不下「路径上的标记」→ 退回默认策略
    dead_ends = [
        c
        for c in range(rows * cols)
        if c not in banned and len(grid.open_neighbors(mask, c, rows, cols)) == 1
    ]
    corners = [
        c
        for c in range(rows * cols)
        if c not in banned and len(grid.open_neighbors(mask, c, rows, cols)) == 2
    ]
    high_degree = [
        c
        for c in range(rows * cols)
        if c not in banned and len(grid.open_neighbors(mask, c, rows, cols)) >= 3
    ]
    picked: list[int] = []
    if marks_on_path == 1:
        # 要求「恰好一个标记在最优路径上」，所以死胡同/高连通度这两个角色也要避开路径
        on_path_set = set(on_path)
        dead_ends = [c for c in dead_ends if c not in on_path_set]
        high_degree = [c for c in high_degree if c not in on_path_set]
    # 角色顺序保证 exits 的三档（死胡同 / 拐角 / 路口）都尽量有候选
    roles = (
        (on_path, dead_ends, corners, high_degree, others)
        if marks_on_path != 0
        else (others, dead_ends, corners, high_degree)
    )
    for role in roles:
        if len(picked) >= count:
            break
        fresh = [c for c in role if c not in picked]
        if fresh:
            picked.append(rng.choice(fresh))
    pool = [c for c in range(rows * cols) if c not in banned and c not in picked]
    rng.shuffle(pool)
    while len(picked) < count and pool:
        picked.append(pool.pop())
    rng.shuffle(picked)
    return {MARK_LETTERS[i]: cell for i, cell in enumerate(picked)}


def _pick_agent(
    rng: random.Random,
    mask: tuple[int, ...],
    rows: int,
    cols: int,
    start: int,
    exit: int,
    target_deviation: int,
    max_steps: int,
) -> AgentSpec | None:
    """构造智能体的行走状态，使当前位置的偏离步数等于目标值。

    输入：RNG、mask、行列数、起止格、目标偏离步数、最大步数。
    输出：AgentSpec 或 None（当前迷宫实现不了该目标，调用方换迷宫重试）。
    预期行为：
    - deviation=0：当前位置落在 S→E 最优路径上（含起点，允许 0 步历史）；
    - deviation≥1：当前位置在路径外，且到路径的最短距离恰好等于目标值（3 表示 ≥3）；
    - moves 是从 S 到目标格的唯一走法（完美迷宫里路径唯一），长度 ≤ max_steps。
    """
    path = grid.shortest_path(mask, rows, cols, start, exit)
    if path is None:
        return None
    path_set = set(path)
    dist_to_path = _distance_to_set(mask, rows, cols, path_set)
    total = rows * cols

    if target_deviation == 0:
        candidates = [c for c in path if c != exit]
    elif target_deviation >= 3:
        candidates = [c for c in range(total) if dist_to_path.get(c, -1) >= 3]
    else:
        candidates = [c for c in range(total) if dist_to_path.get(c, -1) == target_deviation]
    if not candidates:
        return None

    rng.shuffle(candidates)
    for target in candidates:
        if target == exit:
            continue
        walk = grid.shortest_path(mask, rows, cols, start, target)
        if walk is None or len(walk) - 1 > max_steps:
            continue
        deviation = dist_to_path.get(target, -1)
        if deviation < 0:
            continue
        return AgentSpec(moves=tuple(grid.path_moves(walk, cols)), deviation=deviation)
    return None


def _random_walk(
    rng: random.Random, mask: tuple[int, ...], rows: int, cols: int, start: int, max_steps: int
) -> tuple[int, ...]:
    """从起点随机走一段（不走回头路优先），用于「出口不可达」样本。

    输入：RNG、mask、行列数、起点、最大步数。
    输出：方向序列（可能为空）。
    预期行为：每步只走有通路的方向；能不回头就不回头，避免原地打转。
    """
    moves: list[int] = []
    cur = start
    prev: int | None = None
    for _ in range(rng.randint(0, max_steps)):
        options = [d for d, _ in grid.open_neighbors(mask, cur, rows, cols)]
        if not options:
            break
        forward = [
            d
            for d in options
            if grid.neighbor(cur, d, rows, cols) != prev
        ]
        if not forward:
            break  # 走到死胡同就停：禁止回溯，保证不会绕回起点（否则会盖掉 S 标记）
        d = rng.choice(forward)
        prev = cur
        cur = grid.neighbor(cur, d, rows, cols)  # type: ignore[assignment]
        moves.append(d)
    return tuple(moves)


def exit_level(mask: tuple[int, ...], rows: int, cols: int, cell: int) -> int:
    """输入：mask、行列数、格子。输出：该格可走方向数的档位（0 死胡同 / 1 拐角 / 2 路口）。

    预期行为：与出题那一侧同一把尺——挑标记时要保证三档都有候选（否则 `exits` 的第 3 档
    在样本里恒取不到），出题时要按同一把尺定答案。只分三档，因为「0 个方向」在迷宫里
    结构性不可能，「4 个方向」在小迷宫里极罕见。
    """
    return min(2, max(0, len(grid.open_neighbors(mask, cell, rows, cols)) - 1))


def sample_spec(
    rng: random.Random,
    config: MazeConfig,
    case: MazeCase,
    split_name: str,
    *,
    families: tuple[str, ...],
    phrasing_pool: tuple[int, ...] | None = None,
    force_cut: bool = False,
    marks_on_path: int | None = None,
    path_bucket: int | None = None,
) -> MazeSpec | None:
    """采样一份迷宫 spec。

    输入：RNG、配置、case 组合、切分名、本题样本要出的题族、问法编号池（None 用配置默认）、
    是否强制砍断（force_cut=True 时必定造出「出口不可达」样本）、
    最优路径上要放几个标记（给 path_via 定答案用）、
    path_length 的答案桶下标（None = 当场抽一个；给值时调用方必须保证跨重试不变）。
    输出：MazeSpec 或 None（该 case 在当前参数下构造不出来，调用方换参数重试）。
    预期行为：
    - 先造迷宫（算法 + 可选 braid），再施加随机对称变换；
    - 按 cut_prob 决定是否砍断 S→E 路径上的一条边（造「出口不可达」样本）；
    - 砍断时不出任何依赖最优路径的题族，也不放智能体；
    - 其余情况按 case.deviation 构造智能体行走状态。
    """
    rows = rng.randint(config.size[0], config.size[1])
    cols = rng.randint(config.size[0], config.size[1])
    algo = rng.choice(config.algos)
    mask = grid.generate(rng, rows, cols, algo)
    mask = grid.braid(rng, mask, rows, cols, config.braid)

    # exits 题族需要「3 个方向以上」的格子，小迷宫偶尔一个都没有 → 换迷宫重试
    if "exits" in families and not any(
        len(grid.open_neighbors(mask, c, rows, cols)) >= 3 for c in range(rows * cols)
    ):
        return None

    transform = rng.choice(grid.TRANSFORMS)
    mask, rows, cols = grid.apply_transform(transform, mask, rows, cols)

    # path_length 题族：**先定答案（桶）再挑输入**，挑不到就换迷宫重试。
    # 桶由计划层给定（`path_bucket`）并跨重试保持不变：桶在重试循环里重抽的话，
    # 「抽到短桶 → 下游约束（智能体 / 标记 / exits 三档）构造失败 → 重试又抽到长桶」
    # 会让各桶的接受率不等（实测 0.67 / 0.94 / 0.97），短桶被系统性挤掉（第 1 档 0.259 vs 1/3）。
    if "path_length" in families:
        bucket = (
            PATH_LENGTH_BUCKETS[path_bucket]
            if path_bucket is not None
            else rng.choice(PATH_LENGTH_BUCKETS)
        )
        pair = _pick_start_exit_in_bucket(rng, mask, rows, cols, bucket)
        if pair is None:
            return None
        start, exit_cell = pair
    else:
        start, exit_cell = _pick_start_exit(rng, mask, rows, cols)
    path = grid.shortest_path(mask, rows, cols, start, exit_cell)
    if path is None:
        return None

    needs_path = bool({"next_step", "off_by_one", "on_optimal", "deviation", "path_via", "on_path", "path_length"} & set(families))
    cut: tuple[int, int] | None = None
    use_cut = (force_cut or rng.random() < config.cut_prob) and "reachable" in families and not needs_path
    if use_cut and len(path) >= 2:
        idx = rng.randrange(len(path) - 1)
        a, b = path[idx], path[idx + 1]
        cut = (a, b)
        mask = grid.remove_edge(mask, a, b, rows, cols)

    if cut is not None:
        # 出口不可达时没有最优路径，但**仍然放一个随机行走的智能体**：
        # 否则「最近的 x 次决策」「标记走过的路」这两类 case 会白白丢掉一半样本
        agent = AgentSpec(
            moves=_random_walk(rng, mask, rows, cols, start, max(2, rows + cols)),
            deviation=-1,
        )
    else:
        max_steps = max(2, rows + cols)
        agent = _pick_agent(rng, mask, rows, cols, start, exit_cell, case.deviation, max_steps)
        if agent is None:
            return None
        if case.history_len > 0 and not agent.moves:
            # 要求显示历史却没有历史 → 换一个迷宫重试
            return None

    # 标记格必须在智能体定下来之后再挑，并避开当前位置（否则 @ 会盖掉字母标记）
    current = grid.apply_moves(mask, rows, cols, start, agent.moves)[-1]
    # exits 的三档（死胡同 / 拐角 / 路口）要**都有标记**，否则那一档在样本里恒取不到
    # （实测第 3 档只剩 0.29）。角色顺序是「路径上 → 死胡同 → 拐角 → 路口」，三档占后三个
    # 角色，所以标记数至少 `MARKS_FOR_EXITS`。
    mark_count = rng.randint(*config.marks_range)
    if "exits" in families:
        mark_count = max(mark_count, MARKS_FOR_EXITS)
    marks = _pick_marks(
        rng,
        mask,
        rows,
        cols,
        start,
        exit_cell,
        mark_count,
        exclude={current},
        marks_on_path=marks_on_path,
    )
    if "exits" in families and len(
        {exit_level(mask, rows, cols, cell) for cell in marks.values()}
    ) < 3:
        return None  # 三档凑不齐（迷宫太小或没有拐角）→ 换迷宫重试

    pool = phrasing_pool if phrasing_pool else config.phrasing_pool
    return MazeSpec(
        rows=rows,
        cols=cols,
        algo=algo,
        mask=mask,
        start=start,
        exit=exit_cell,
        marks=marks,
        cut=cut,
        braid=config.braid,
        transform=transform,
        render=RenderCase(
            fmt=case.fmt,
            symbols=case.symbols,
            mark_trail=case.mark_trail,
            history_len=case.history_len,
            option_detail=case.option_detail,
        ),
        agent=agent,
        families=families,
        phrasing=tuple(rng.choice(pool) for _ in families),
    )
