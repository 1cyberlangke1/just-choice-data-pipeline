"""`path_length` 的答案桶与 `exits` 的档位：三条踩过坑的不变量。

输入：手搭的小迷宫 + 采样配置（不联网、不落盘）。
输出：断言。
预期行为：钉住三件事——
1. `exit_level` 把「可走方向数」映射成三档（1 个方向 = 0、2 个 = 1、>=3 个 = 2）；
2. `path_length` 的答案桶在**计划层**定死、跨「换迷宫重试」不变（桶在重试里重抽会让
   各桶接受率不等，短桶被系统性挤掉：实测第 1 档只剩 0.259 而非 1/3）；
3. 出 `exits` 时至少放 `MARKS_FOR_EXITS` 个标记，三档才都有候选。
"""

from __future__ import annotations

import random


from jc import pipeline, workflow as workflow_mod
from workflows.maze import grid
from workflows.maze import spec as maze_spec
from workflows.maze.workflow import FAMILY_PICK_WEIGHTS, _pick_families

#: 测试用的普通 case：块状文本、无历史、不标记走过的路、选项不写目标方块、不走偏、不砍断。
PLAIN_CASE: maze_spec.MazeCase = maze_spec.MazeCase(
    fmt=maze_spec.FMT_BLOCK_TEXT,
    symbols="hash",
    history_len=0,
    mark_trail=False,
    option_detail=maze_spec.OPTION_PLAIN,
    deviation=0,
)


def mask_from_edges(rows: int, cols: int, edges: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    """输入：行列数、[(格子, 方向), ...]。输出：mask 元组。

    预期行为：逐条打通格子与它的邻居（双向置位）；邻居越界直接报错，
    免得手搭出来的测试迷宫本身就是非法的。
    """
    cells = [0] * (rows * cols)
    for cell, d in edges:
        nxt = grid.neighbor(cell, d, rows, cols)
        assert nxt is not None, f"格子 {cell} 沿方向 {d} 的邻居越界"
        cells[cell] |= 1 << d
        cells[nxt] |= 1 << grid.OPPOSITE[d]
    return tuple(cells)


def fully_open(rows: int, cols: int) -> tuple[int, ...]:
    """输入：行列数。输出：mask。预期行为：相邻格之间全部打通（直径最小的「开广场」）。"""
    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            cell = grid.index_of(r, c, cols)
            if c + 1 < cols:
                edges.append((cell, grid.DIR_RIGHT))
            if r + 1 < rows:
                edges.append((cell, grid.DIR_DOWN))
    return mask_from_edges(rows, cols, tuple(edges))


def test_exit_level_counts_open_directions() -> None:
    """输入：一条 1x3 走廊 + 一个十字路口。输出：档位。

    预期行为：死胡同（1 个方向）= 0、拐角（2 个）= 1、路口（>=3 个）= 2。
    这里钉的是「减一之后封顶」：第一版写成 `min(2, 度数)` 会把拐角也判成 2，
    于是 `exits` 的第 3 档在样本里恒取不到（实测 maze 出现 1200 条拒稿）。
    """
    corridor = mask_from_edges(1, 3, ((0, grid.DIR_RIGHT), (1, grid.DIR_RIGHT)))
    assert maze_spec.exit_level(corridor, 1, 3, 0) == 0
    assert maze_spec.exit_level(corridor, 1, 3, 1) == 1
    assert maze_spec.exit_level(corridor, 1, 3, 2) == 0
    plus = fully_open(3, 3)
    assert maze_spec.exit_level(plus, 3, 3, grid.index_of(1, 1, 3)) == 2


def test_pick_start_exit_in_bucket_never_falls_back_to_another_bucket() -> None:
    """输入：3x3 开广场（直径 4）+ 三个桶。输出：起止格或 None。

    预期行为：命中的起止格最短距离必须落在桶里；桶 1（1~3 步）与桶 2（4~7 步）都能命中；
    桶 3（>=8 步）在直径只有 4 的迷宫上必须返回 None——**不退回别的桶**，否则就是让迷宫
    来决定答案分布（实测那样第 3 档只剩 0.25）。
    """
    mask = fully_open(3, 3)
    rng = random.Random(7)
    for bucket, realizable in (((1, 3), True), ((4, 7), True), ((8, 10**9), False)):
        pair = maze_spec._pick_start_exit_in_bucket(rng, mask, 3, 3, bucket)
        assert (pair is not None) is realizable, f"桶 {bucket} 的实现性判错了"
        if pair is None:
            continue
        start, exit_cell = pair
        low, high = bucket
        assert low <= grid.distances(mask, 3, 3, start)[exit_cell] <= high


def test_sample_spec_honours_the_pinned_bucket() -> None:
    """输入：显式给定的桶下标。输出：采样结果。

    预期行为：给 `path_bucket=k` 时，凡是采样成功的 spec，S->E 步数必须落在第 k 个桶里；
    三个桶都要有成功的样本（桶 3 在 3~5 格的迷宫上更常实现不了，但重试到成功为止）。
    """
    config = maze_spec.MazeConfig(size=(3, 5))
    for idx, (low, high) in enumerate(maze_spec.PATH_LENGTH_BUCKETS):
        ok = 0
        for seed in range(40):
            spec = maze_spec.sample_spec(
                random.Random(seed),
                config,
                PLAIN_CASE,
                "train",
                families=("path_length",),
                path_bucket=idx,
            )
            if spec is None:
                continue
            steps = len(spec.path() or []) - 1
            assert low <= steps <= high, f"桶 {idx} 采到了 {steps} 步"
            ok += 1
        assert ok > 0, f"桶 {idx} 一次都没采样成功"


def test_plan_sample_pins_the_bucket_and_sample_keeps_it() -> None:
    """输入：一个普通 case 的 80 个计划。输出：计划参数与采样结果。

    预期行为：`path_bucket` 只在出 `path_length` 时是非负整数下标，否则是 None；
    `sample` 会为同一个计划反复换迷宫，但**每次都落在同一个桶里**——这正是
    「短桶不被挤掉」的根因修复（桶若在重试里重抽，接受率 0.67 / 0.94 / 0.97 的差别
    会把第 1 档压到 0.259）。
    """
    workflow = workflow_mod.load("maze")
    config = maze_spec.MazeConfig(size=(3, 5))
    pinned = 0
    for seq in range(80):
        rng = pipeline.sample_rng(20260930, workflow.name, "train", seq)
        split = workflow.plan_sample(rng, config, PLAIN_CASE, seq)
        families = split.params["families"]
        bucket_idx = split.params["path_bucket"]
        assert isinstance(families, tuple)
        if "path_length" not in families:
            assert bucket_idx is None, "不出 path_length 却把桶定死了"
            continue
        assert isinstance(bucket_idx, int), "出 path_length 却没把桶定下来"
        assert 0 <= bucket_idx < len(maze_spec.PATH_LENGTH_BUCKETS)
        low, high = maze_spec.PATH_LENGTH_BUCKETS[bucket_idx]
        steps = len(workflow.sample(rng, split).path() or []) - 1
        assert low <= steps <= high, f"计划钉的第 {bucket_idx} 桶，采样却落在 {steps} 步"
        pinned += 1
    assert pinned > 0, "80 个计划里一次都没抽到 path_length"


def test_family_weights_shift_deviation_and_on_optimal() -> None:
    """输入：dev=0 与 dev=2 的 case 各抽 2000 次题族。输出：两族与对照族的入选率之比。

    预期行为：两族共用「当前位置在不在最优路径上」这一个自由度（dev=0 的 case 里
    `deviation` 恒答 0、`on_optimal` 恒答 True），只能按 case 调入选概率来铺平：
    dev=0 上 `deviation` 的入选概率约 0.5、`on_optimal` 约 1.0；dev!=0 上分别约 1.0 与 2/3。

    入选率本身还乘了「池子里再抽 count 个」的概率，所以拿**同一个池子**里权重恒为 1 的
    `step_result` 当对照，只断言比值——这样测的是权重表，不是抽样比例。
    """
    config = maze_spec.MazeConfig(size=(3, 5))
    rng = random.Random(20260930)
    rate: dict[tuple[int, str], float] = {}
    draws = 2000
    for dev in (0, 2):
        case = maze_spec.MazeCase(
            fmt=maze_spec.FMT_BLOCK_TEXT,
            symbols="hash",
            history_len=0,
            mark_trail=False,
            option_detail=maze_spec.OPTION_PLAIN,
            deviation=dev,
        )
        hits = {"deviation": 0, "on_optimal": 0, "step_result": 0}
        for _ in range(draws):
            families, _marks_on_path = _pick_families(rng, config, case, False)
            for name in hits:
                if name in families:
                    hits[name] += 1
        for name, hit in hits.items():
            rate[(dev, name)] = hit / draws
    ratio = {
        (dev, name): rate[(dev, name)] / rate[(dev, "step_result")]
        for dev in (0, 2)
        for name in ("deviation", "on_optimal")
    }
    assert 0.42 < ratio[(0, "deviation")] < 0.58, ratio
    assert 0.85 < ratio[(0, "on_optimal")] < 1.15, ratio
    assert 0.85 < ratio[(2, "deviation")] < 1.15, ratio
    assert 0.58 < ratio[(2, "on_optimal")] < 0.75, ratio
    # 权重表本身也钉一下，免得有人顺手改回去
    assert FAMILY_PICK_WEIGHTS["deviation"] == (0.5, 1.0)
    assert FAMILY_PICK_WEIGHTS["on_optimal"] == (1.0, 2 / 3)


def test_exits_samples_always_cover_all_three_levels() -> None:
    """输入：只出 `exits` 的样本。输出：标记与档位。

    预期行为：出 `exits` 时标记数恰好是 `MARKS_FOR_EXITS`（角色顺序保证三档都有候选），
    且死胡同 / 拐角 / 路口三档都至少有一个标记——凑不齐的迷宫会被 `sample_spec` 拒掉。
    """
    config = maze_spec.MazeConfig(size=(5, 7))
    rng = random.Random(20260930)
    ok = 0
    for _ in range(40):
        spec = maze_spec.sample_spec(rng, config, PLAIN_CASE, "train", families=("exits",))
        if spec is None:
            continue
        assert len(spec.marks) == maze_spec.MARKS_FOR_EXITS
        levels = {
            maze_spec.exit_level(spec.mask, spec.rows, spec.cols, cell)
            for cell in spec.marks.values()
        }
        assert levels == {0, 1, 2}
        ok += 1
    assert ok > 0, "一次都没采到 exits 样本"
