"""金答案的独立复算测试（本仓库最重要的一道闸）。

做法：拿生成好的样本，从**渲染出来的 state 文本**反解迷宫，用**独立实现的 BFS**重算答案，
再与数据里存的金答案逐题比对。这样同时验证了三件事：
1. 渲染没画错（反解结果与真值 spec 一致，由往返校验覆盖）；
2. 答案算对了（独立复算一致）；
3. 编号映射没串位（比对的是「选项文案」，不是下标本身）。

另外显式检查主人要求的两个 case：走偏 1 格（off_by_one）与沿最优路径走（next_step）。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from collections import deque


from jc import assemble, pipeline, workflow as workflow_mod
from workflows.maze import grid, locales
from workflows.maze import questions as maze_questions
from workflows.maze import spec as maze_spec
from workflows.maze.workflow import ParsedMaze, parse_state, roundtrip_diff

LOC = locales.load("zh")
DIR_WORDS = ("向上", "向右", "向下", "向左")


def family_of(qid: str) -> str:
    """题号 → 题族名（去掉 `_2` 这类后缀）。"""
    return "".join(ch for ch in qid if not ch.isdigit()).rstrip("_")


def dist_to_set(mask: tuple[int, ...], rows: int, cols: int, sources: set[int]) -> dict[int, int]:
    """多源 BFS：每格到 sources 集合的最短距离（独立实现）。"""
    dist = {src: 0 for src in sources}
    queue = deque(sources)
    while queue:
        cur = queue.popleft()
        for _, nb in grid.open_neighbors(mask, cur, rows, cols):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                queue.append(nb)
    return dist


def letter_in(text: str) -> str | None:
    """题面里出现的标记字母（A~D）。"""
    for ch in text:
        if ch in "ABCD":
            return ch
    return None


def dir_in(text: str) -> int | None:
    """题面里出现的方向（返回方向序号）。"""
    for idx, word in enumerate(DIR_WORDS):
        if word in text:
            return idx
    return None


def answer_text(question: dict[str, JsonValue], answer: dict[str, JsonValue]) -> str:
    """把 choice 的编号换回选项文案。"""
    return list(question["criteria"])[answer["choice"]]


def question_text(question: dict[str, JsonValue]) -> str:
    """取题面文本：block_rows 的 instructions 是 object 形态（图例与历史塞在里面）。"""
    instructions = question["instructions"]
    return instructions["question"] if isinstance(instructions, dict) else instructions


def expected_exits_level(mask: tuple[int, ...], rows: int, cols: int, cell: int) -> int:
    """独立复算 exits 的等级编号（0 死胡同 / 1 拐角 / 2 丁字或十字）。"""
    count = len(grid.open_neighbors(mask, cell, rows, cols))
    if count <= 1:
        return 0
    if count <= 2:
        return 1
    return 2


def check_sample(sample: dict[str, JsonValue], parsed: ParsedMaze) -> list[str]:
    """逐题独立复算，返回不一致说明列表（空 = 全部一致）。"""
    problems: list[str] = []
    mask, rows, cols = parsed.mask, parsed.rows, parsed.cols
    start, exit_cell = parsed.start, parsed.exit
    assert start is not None and exit_cell is not None
    path = grid.shortest_path(mask, rows, cols, start, exit_cell)
    dist_exit = grid.distances(mask, rows, cols, exit_cell)
    current = parsed.current if parsed.current is not None else start
    path_set = set(path or [])

    for qid, question in sample["questions"].items():
        family = family_of(qid)
        answer = sample["answers"][qid]
        text = question_text(question)

        if family in ("next_step", "off_by_one"):
            if dist_exit[current] < 0:
                problems.append(f"{qid}: 出口不可达却出了导航题")
                continue
            options = list(question["criteria"])
            expected_dirs = [
                d
                for d, nb in grid.open_neighbors(mask, current, rows, cols)
                if dist_exit[nb] == dist_exit[current] - 1
            ]
            if len(expected_dirs) != 1:
                problems.append(f"{qid}: 最优下一步不唯一（{expected_dirs}）")
                continue
            word = DIR_WORDS[expected_dirs[0]]
            matches = [o for o in options if o.startswith(word)]
            if len(matches) != 1:
                problems.append(f"{qid}: 选项里找不到方向 {word} 的唯一项")
                continue
            if answer_text(question, answer) != matches[0]:
                problems.append(
                    f"{qid}: 金答案 {answer_text(question, answer)!r} != 独立复算 {matches[0]!r}"
                )
        elif family == "path_via":
            on_path = [letter for letter, cell in parsed.marks.items() if cell in path_set]
            if len(on_path) > 1:
                problems.append(f"{qid}: 路径上有多个标记，答案不唯一")
                continue
            expected = on_path[0] if on_path else LOC["fam"]["path_via"]["none"]
            if answer_text(question, answer) != expected:
                problems.append(f"{qid}: 金答案 {answer_text(question, answer)!r} != {expected!r}")
        elif family == "step_result":
            letter = letter_in(text)
            d = dir_in(text)
            if letter is None or d is None:
                problems.append(f"{qid}: 题面里读不出标记或方向")
                continue
            cell = parsed.marks[letter]
            vocab = LOC["fam"]["step_result"]["options"]
            nxt = grid.neighbor(cell, d, rows, cols)
            if nxt is None:
                expected = vocab[4]
            elif not grid.has_passage(mask, cell, d, rows, cols):
                expected = vocab[0]
            elif nxt == start:
                expected = vocab[2]
            elif nxt == exit_cell:
                expected = vocab[3]
            else:
                expected = vocab[1]
            if answer_text(question, answer) != expected:
                problems.append(
                    f"{qid}: 金答案 {answer_text(question, answer)!r} != 独立复算 {expected!r}"
                )
        elif family == "can_move":
            letter, d = letter_in(text), dir_in(text)
            if letter is None or d is None:
                problems.append(f"{qid}: 题面里读不出标记或方向")
                continue
            expected = grid.has_passage(mask, parsed.marks[letter], d, rows, cols)
            if answer["noul"] != expected:
                problems.append(f"{qid}: 金答案 {answer['noul']} != 独立复算 {expected}")
        elif family == "dead_end":
            letter = letter_in(text)
            if letter is None:
                problems.append(f"{qid}: 题面里读不出标记")
                continue
            cell = parsed.marks[letter]
            expected = len(grid.open_neighbors(mask, cell, rows, cols)) == 1
            if answer["noul"] != expected:
                problems.append(f"{qid}: 金答案 {answer['noul']} != 独立复算 {expected}")
        elif family == "on_path":
            letter = letter_in(text)
            if letter is None:
                problems.append(f"{qid}: 题面里读不出标记")
                continue
            expected = parsed.marks[letter] in path_set
            if answer["noul"] != expected:
                problems.append(f"{qid}: 金答案 {answer['noul']} != 独立复算 {expected}")
        elif family == "reachable":
            expected = dist_exit[start] >= 0
            if answer["noul"] != expected:
                problems.append(f"{qid}: 金答案 {answer['noul']} != 独立复算 {expected}")
        elif family == "on_optimal":
            expected = current in path_set
            if answer["noul"] != expected:
                problems.append(f"{qid}: 金答案 {answer['noul']} != 独立复算 {expected}")
        elif family == "deviation":
            if not path_set:
                problems.append(f"{qid}: 出口不可达却问了偏离步数")
                continue
            expected = min(dist_to_set(mask, rows, cols, path_set)[current], 3)
            if answer["score"] != expected:
                problems.append(f"{qid}: 金答案 {answer['score']} != 独立复算 {expected}")
        elif family == "path_length":
            # 独立复算：桶区间与 spec.PATH_LENGTH_BUCKETS 一致（3 桶，需与实现同步改）
            steps = len(path) - 1 if path else -1
            if steps <= 3:
                expected = 0
            elif steps <= 7:
                expected = 1
            else:
                expected = 2
            if answer["score"] != expected:
                problems.append(f"{qid}: 金答案 {answer['score']} != 独立复算 {expected}")
        elif family == "exits":
            letter = letter_in(text)
            if letter is None:
                problems.append(f"{qid}: 题面里读不出标记")
                continue
            expected = expected_exits_level(mask, rows, cols, parsed.marks[letter])
            if answer["score"] != expected:
                problems.append(f"{qid}: 金答案 {answer['score']} != 独立复算 {expected}")
        else:
            problems.append(f"{qid}: 未覆盖的题族 {family}")
    return problems


def generate(
    seed: int, config: maze_spec.MazeConfig, cases: list[maze_spec.MazeCase]
) -> list[tuple[dict[str, JsonValue], maze_spec.MazeSpec, ParsedMaze, maze_spec.MazeCase]]:
    """在内存里跑一遍「计划 → 采样 → 渲染 → 出题 → 组装」，产出 (sample, spec, parsed, case)。"""
    workflow = workflow_mod.load("maze")
    out = []
    for seq, case in enumerate(cases):
        rng = pipeline.sample_rng(seed, workflow.name, "train", seq)
        split = workflow.plan_sample(rng, config, case, seq)
        maze = workflow.sample(rng, split)
        state = workflow.render(maze, rng, "zh")
        labeled = workflow.ask(maze, rng, "zh")
        sample = assemble.assemble_sample(f"maze-train-{seq:07d}", state, labeled, rng)
        parsed = parse_state(state, maze.render)
        assert parsed is not None, f"{case.describe()} 无法反解"
        out.append((sample, maze, parsed, case))
    return out


def small_config() -> maze_spec.MazeConfig:
    """小而全的采样配置：覆盖全部七个 case 维度，规模可控。"""
    return maze_spec.MazeConfig(
        size=(3, 5),
        algos=("dfs", "prim", "wilson"),
        formats=maze_spec.FORMATS,
        symbols=("hash", "block", "xo"),
        history_lens=(0, 1, 2, 5),
        mark_trails=(False, True),
        option_details=(maze_spec.OPTION_PLAIN, maze_spec.OPTION_CELL),
        deviations=(0, 1, 2, 3),
    )


def test_all_labels_match_independent_recomputation() -> None:
    workflow = workflow_mod.load("maze")
    config = small_config()
    cases = workflow.build_cases(config)
    problems: list[str] = []
    for sample, maze, parsed, case in generate(20260924, config, cases):
        diff = roundtrip_diff(maze, parsed)
        assert not diff, f"{sample['id']} 往返校验失败：{diff}"
        problems.extend(
            f"{sample['id']} {case.describe()} {p}" for p in check_sample(sample, parsed)
        )
    assert not problems, "金答案与独立复算不一致：\n" + "\n".join(problems[:20])


def test_goal_required_cases_are_present() -> None:
    """主人点名要的 case：走偏 1 格（往最优路径回走）与沿最优路径走。

    不只是数一数，还要从**渲染出来的 state** 反解验证：
    - off_by_one 的样本，当前位置到最优路径的距离必须恰好是 1；
    - next_step 在 deviation=0 的样本里，当前位置必须真的在最优路径上。
    """
    workflow = workflow_mod.load("maze")
    config = small_config()
    cases = workflow.build_cases(config)
    off_by_one = 0
    next_step_on_path = 0
    for sample, maze, parsed, _case in generate(20260924, config, cases):
        path = grid.shortest_path(parsed.mask, parsed.rows, parsed.cols, parsed.start, parsed.exit)
        path_set = set(path or [])
        current = parsed.current if parsed.current is not None else parsed.start
        dist = dist_to_set(parsed.mask, parsed.rows, parsed.cols, path_set)
        for qid in sample["questions"]:
            if family_of(qid) == "off_by_one":
                assert maze.agent.deviation == 1, "off_by_one 只应出现在偏离 1 格的样本里"
                assert dist[current] == 1, "反解后当前位置到最优路径的距离不是 1"
                off_by_one += 1
            if family_of(qid) == "next_step" and maze.agent.deviation == 0:
                assert current in path_set, "deviation=0 的样本里当前位置不在最优路径上"
                next_step_on_path += 1
    assert off_by_one > 0, "没有生成「走偏 1 格」的样本"
    assert next_step_on_path > 0, "没有生成「沿最优路径走」的样本"


def test_case_dimensions_are_all_exercised() -> None:
    """七个 case 维度都要真的被覆盖到（不只是出现在矩阵里）。"""
    workflow = workflow_mod.load("maze")
    config = small_config()
    cases = workflow.build_cases(config)
    seen: dict[str, set] = {
        "format": set(),
        "symbols": set(),
        "history_len": set(),
        "mark_trail": set(),
        "option_detail": set(),
        "deviation": set(),
        "cut": set(),
    }
    for sample, maze, _parsed, case in generate(20260924, config, cases):
        for name, value in workflow.stats_dimensions(case).items():
            seen[name].add(value)
        state = sample["state"]
        if case.history_len > 0 and maze.agent.moves:
            blob = str(state) + str([q.get("instructions") for q in sample["questions"].values()])
            assert LOC["history"]["prefix"] in blob, "历史没写进 state 或 instructions"
        if case.fmt == maze_spec.FMT_BLOCK_ROWS:
            for question in sample["questions"].values():
                assert isinstance(question["instructions"], dict), "block_rows 必须把图例塞进 instructions"
                assert "legend" in question["instructions"]
    assert seen["format"] == set(config.formats)
    assert seen["symbols"] == set(config.symbols)
    assert seen["history_len"] == set(config.history_lens)
    assert seen["mark_trail"] == set(config.mark_trails)
    assert seen["option_detail"] == set(config.option_details)
    assert seen["deviation"] == set(config.deviations)
    assert seen["cut"] == {False, True}


def test_option_detail_cell_mentions_target_cell() -> None:
    """option_detail=cell 时，方向型选项必须写明走向的方块是什么。"""
    workflow = workflow_mod.load("maze")
    config = small_config()
    cases = [
        c
        for c in workflow.build_cases(config)
        if c.option_detail == maze_spec.OPTION_CELL and c.deviation == 0
    ][:6]
    checked = 0
    for sample, _maze, _parsed, _case in generate(7, config, cases):
        for qid, question in sample["questions"].items():
            if family_of(qid) != "next_step":
                continue
            for option in question["criteria"]:
                assert "→" in option, f"选项没写目标方块：{option!r}"
                checked += 1
    assert checked > 0


def test_every_family_builder_is_covered() -> None:
    """所有题族都至少被生成过一次（防止某个 builder 长期无人调用）。"""
    workflow = workflow_mod.load("maze")
    config = small_config()
    cases = workflow.build_cases(config)
    seen = {
        family_of(qid) for sample, *_ in generate(20260924, config, cases) for qid in sample["questions"]
    }
    missing = set(maze_questions.ALL_FAMILIES) - seen
    assert not missing, f"这些题族一次都没出过：{sorted(missing)}"


def test_workflow_declares_itself_as_constructive() -> None:
    """迷宫是一个纯构造式 workflow：不声明任何后端需求，题型三样都产。"""
    workflow = workflow_mod.load("maze")
    assert workflow.needs == frozenset()
    assert workflow.kinds == {"choice", "noul", "score"}
    assert set(workflow.judges) == {"roundtrip", "label"}
    assert workflow.split_names() == ("train", "val", "test_iid", "test_ood")
    assert workflow.config_for("test_ood").size == (8, 10)
