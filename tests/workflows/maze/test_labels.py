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

import re
from collections import deque
from typing import Final

import pytest


from jc import assemble, pipeline, workflow as workflow_mod
from workflows.maze import grid, locales
from workflows.maze import questions as maze_questions
from workflows.maze import spec as maze_spec
from workflows.maze.workflow import ParsedMaze, parse_state, roundtrip_diff

#: 题面里的标记字母：只认**单独一个**大写 A–D。
#: 这里刻意与 `judges.LETTER_PATTERN` 写同一条读法（答案复算部分仍然是各写各的）：
#: 按「第一个落在 ABCD 里的字符」读，英文题面里的 `Can` / `Does` 会撞上 C / D，
#: 真机 en 侧 782 条拒稿就是这么来的。两条读法的一致性由下面那条模板不变量测试保证。
LETTER_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b[A-D]\b")

#: 测试要覆盖的语言：判别器读标记 / 读方向都是语言相关的，只跑一种等于留盲区。
LANGS: Final[tuple[str, ...]] = ("zh", "en")


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
    """题面里出现的标记字母（A~D）。

    预期行为：恰好一个才认；零个（读不出）或两个（题面本身有歧义）都不猜。
    """
    found = LETTER_PATTERN.findall(text)
    return found[0] if len(found) == 1 else None


def dir_in(text: str, loc: dict[str, JsonValue]) -> int | None:
    """题面里出现的方向（返回方向序号）。

    输入：题面文本、语言字典。
    预期行为：方向词表取自 `loc["moves"]`；`loc` 没有默认值——语言必须由调用方显式给，
    给个 zh 的默认值就等于把「另一种语言读不出方向」这个坑重新挖回来。
    """
    for idx, word in enumerate(loc["moves"].values()):
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


def check_sample(
    sample: dict[str, JsonValue], parsed: ParsedMaze, loc: dict[str, JsonValue]
) -> list[str]:
    """逐题独立复算，返回不一致说明列表（空 = 全部一致）。

    输入：样本、反解出来的迷宫、语言字典。
    预期行为：选项文案、方向词表都从 `loc` 取——**语言必须由调用方给**，写死一种语言会让
    另一种语言的样本读不出标记与方向（真机 en 侧就是这么错掉一半样本的）。
    """
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
            word = str(list(loc["moves"].values())[expected_dirs[0]])
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
            expected = on_path[0] if on_path else loc["fam"]["path_via"]["none"]
            if answer_text(question, answer) != expected:
                problems.append(f"{qid}: 金答案 {answer_text(question, answer)!r} != {expected!r}")
        elif family == "step_result":
            letter = letter_in(text)
            d = dir_in(text, loc)
            if letter is None or d is None:
                problems.append(f"{qid}: 题面里读不出标记或方向")
                continue
            cell = parsed.marks[letter]
            vocab = loc["fam"]["step_result"]["options"]
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
            letter, d = letter_in(text), dir_in(text, loc)
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
    seed: int,
    config: maze_spec.MazeConfig,
    cases: list[maze_spec.MazeCase],
    lang: str = "zh",
) -> list[tuple[dict[str, JsonValue], maze_spec.MazeSpec, ParsedMaze, maze_spec.MazeCase]]:
    """在内存里跑一遍「计划 → 采样 → 渲染 → 出题 → 组装」，产出 (sample, spec, parsed, case)。

    输入：种子、采样配置、case 列表、语言。
    预期行为：渲染与出题都用 `lang`——语言必须能换，不然「另一种语言读不出标记」这类错
    永远只会在真机上撞见。
    """
    workflow = workflow_mod.load("maze")
    out: list[tuple[dict[str, JsonValue], maze_spec.MazeSpec, ParsedMaze, maze_spec.MazeCase]] = []
    for seq, case in enumerate(cases):
        rng = pipeline.sample_rng(seed, workflow.name, "train", seq)
        split = workflow.plan_sample(rng, config, case, seq)
        maze = workflow.sample(rng, split)
        state = workflow.render(maze, rng, lang)
        labeled = workflow.ask(maze, rng, lang)
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


def templates_with(slot: str, loc: dict[str, JsonValue]) -> list[tuple[str, str]]:
    """列出所有含某个占位符的问法。

    输入：占位符名（`dir` 或 `letter`）、语言字典。
    输出：[(题族名, 模板文本)]，按 TOML 里的顺序。
    预期行为：不含该占位符的模板不返回。
    """
    found: list[tuple[str, str]] = []
    for family, entry in loc["fam"].items():
        for template in entry["q"]:
            if "{" + slot + "}" in template:
                found.append((family, template))
    return found


@pytest.mark.parametrize("lang", LANGS)
def test_direction_words_are_mutually_unambiguous(lang: str) -> None:
    """方向词表必须两两互不为子串。

    预期行为：`dir_in` 与判别器都是按**子串**读方向的；一旦某个方向词成了另一个
    的子串（比如 en 里同时有 `up` 和 `upward`），子串读法就会把方向读错。
    这条尺子把这个前提钉死——想加新词，先保证词表无包含关系。
    """
    loc = locales.load(lang)
    words = [str(word) for word in loc["moves"].values()]
    for outer_idx, outer in enumerate(words):
        for inner_idx, inner in enumerate(words):
            if outer_idx != inner_idx:
                assert inner not in outer, (
                    f"{lang}: 方向词 {inner!r} 是 {outer!r} 的子串，子串读法会读错方向"
                )


@pytest.mark.parametrize("lang", LANGS)
def test_placeholder_templates_read_back_exactly(lang: str) -> None:
    """模板不变量：四个方向 / 四个字母逐个填进模板，读法必须恰好读回那一个。

    输入：语言代码。
    预期行为：
    - 含 `{dir}` 的模板填第 i 个方向词后，`dir_in` 必须返回 i；
    - 含 `{letter}` 的模板填第 i 个字母后，`letter_in` 必须返回该字母。

    这条测试就是真机 en 侧「782 条拒稿」那个 bug 的机器可查版本：模板里任何
    多余的独立大写字母（`Can` / `Does` 这类）或方向词子串（`end up` 里的 `up`），
    都会在这里被抓出来。中文同理——若哪天模板把 `{letter}` 紧贴在汉字旁，
    `\b[A-D]\b` 的边界就不再成立，这条也会红。
    """
    loc = locales.load(lang)
    words = [str(word) for word in loc["moves"].values()]
    letters = ["A", "B", "C", "D"]
    dir_templates = templates_with("dir", loc)
    letter_templates = templates_with("letter", loc)
    assert dir_templates, f"{lang}: 没有任何含 {{dir}} 的模板，这条尺子失去意义"
    assert letter_templates, f"{lang}: 没有任何含 {{letter}} 的模板，这条尺子失去意义"
    for family, template in dir_templates:
        for idx, word in enumerate(words):
            text = template.replace("{letter}", "A").replace("{dir}", word)
            read = dir_in(text, loc)
            assert read == idx, f"{lang}/{family}: {text!r} 里的方向读成 {read}，应为 {idx}"
    for family, template in letter_templates:
        for letter in letters:
            text = template.replace("{letter}", letter).replace("{dir}", words[0])
            read = letter_in(text)
            assert read == letter, f"{lang}/{family}: {text!r} 里的标记读成 {read!r}，应为 {letter!r}"


@pytest.mark.parametrize("lang", LANGS)
def test_all_labels_match_independent_recomputation(lang: str) -> None:
    """两种语言各跑全 case 矩阵。

    输入：语言代码。
    预期行为：渲染、出题、复算都用同一个 `lang`——判别器读标记 / 读方向的逻辑是
    语言相关的，只跑一种语言会让另一种语言的读法错误永远躲在测试盲区里。
    """
    workflow = workflow_mod.load("maze")
    config = small_config()
    cases = workflow.build_cases(config)
    loc = locales.load(lang)
    problems: list[str] = []
    for sample, maze, parsed, case in generate(20260924, config, cases, lang):
        diff = roundtrip_diff(maze, parsed)
        assert not diff, f"{sample['id']} 往返校验失败：{diff}"
        problems.extend(
            f"{sample['id']} {case.describe()} {p}" for p in check_sample(sample, parsed, loc)
        )
    assert not problems, "金答案与独立复算不一致：\n" + "\n".join(problems[:20])


@pytest.mark.parametrize("lang", LANGS)
def test_goal_required_cases_are_present(lang: str) -> None:
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
    for sample, maze, parsed, _case in generate(20260924, config, cases, lang):
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


@pytest.mark.parametrize("lang", LANGS)
def test_case_dimensions_are_all_exercised(lang: str) -> None:
    """七个 case 维度都要真的被覆盖到（不只是出现在矩阵里），两种语言各验一遍。"""
    workflow = workflow_mod.load("maze")
    config = small_config()
    cases = workflow.build_cases(config)
    loc = locales.load(lang)
    seen: dict[str, set[JsonValue]] = {
        "format": set(),
        "symbols": set(),
        "history_len": set(),
        "mark_trail": set(),
        "option_detail": set(),
        "deviation": set(),
        "cut": set(),
    }
    for sample, maze, _parsed, case in generate(20260924, config, cases, lang):
        for name, value in workflow.stats_dimensions(case).items():
            seen[name].add(value)
        state = sample["state"]
        if case.history_len > 0 and maze.agent.moves:
            blob = str(state) + str([q.get("instructions") for q in sample["questions"].values()])
            assert loc["history"]["prefix"] in blob, "历史没写进 state 或 instructions"
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
