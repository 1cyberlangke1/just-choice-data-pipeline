"""迷宫审计的测试：产物级的闸门（`workflows/maze/audit.py`）。

覆盖：格式 / 符号集 / 变体从产物反推、审计自带的多源 BFS、按题族计数，以及
「跑一遍极小 run 再审计它」的端到端闸门——审计说通过就得真的通过、说缺什么就得真的缺，
否则这道闸门只是装饰（这个模块原来只有 19% 覆盖率，等于没测）。

输入：`workflows.maze.audit` 的纯函数与 `jc.pipeline` 的小规模跑。
输出：断言结果。预期行为：全程零 token、零网络。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from jc import pipeline, workflow as workflow_mod
from workflows.maze import audit, grid, locales, spec as maze_spec


def audit_config() -> maze_spec.MazeConfig:
    """极小配置：**3 种格式**（审计要求至少 3 种）× 1 符号集 × 2 历史 × 2 × 2 × 2 偏离。"""
    return maze_spec.MazeConfig(
        size=(3, 5),
        algos=("dfs",),
        formats=(
            maze_spec.FMT_BLOCK_TEXT,
            maze_spec.FMT_BLOCK_ROWS,
            maze_spec.FMT_BLOCK_OBJECT,
        ),
        symbols=("hash",),
        history_lens=(0, 2),
        mark_trails=(False, True),
        option_details=(maze_spec.OPTION_PLAIN, maze_spec.OPTION_CELL),
        deviations=(0, 1),
    )


def make_config(tmp_path: Path, run_id: str) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用极小迷宫配置试跑的运行配置。"""
    return pipeline.RunConfig(
        workflow="maze",
        split="train",
        locales=("zh",),
        run_seed=7,
        per_case=1,
        concurrency=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=audit_config(),
    )


def test_infer_format_reads_the_product_shape() -> None:
    """格式只能从产物本身的形态反推（审计时手上没有 spec）。

    输入：五种形态各一份 + 一个不是 state 的值。输出：格式名。
    预期行为：数组 → `block_rows`；带 `passages` 的对象 → `passages`；别的对象 →
    `block_object`；有以 `+` 开头那一行的文本 → `wall_ascii`；其余文本 → `block_text`；
    认不出来 → `unknown`（**不能猜**，猜错会让审计把好数据报成坏的）。
    """
    assert audit.infer_format(["#=墙, .=通道", "# #"]) == maze_spec.FMT_BLOCK_ROWS
    assert audit.infer_format({"passages": []}) == maze_spec.FMT_PASSAGES
    assert audit.infer_format({"grid": []}) == maze_spec.FMT_BLOCK_OBJECT
    assert audit.infer_format("+--+\n|  |\n+--+") == maze_spec.FMT_WALL_ASCII
    assert audit.infer_format("S # E") == maze_spec.FMT_BLOCK_TEXT
    assert audit.infer_format(None) == "unknown"


def test_infer_symbols_reads_the_legend() -> None:
    """符号集从图例反推：文本图例、对象图例、以及塞在 `instructions` 里的图例三条路。

    输入：三种带图例的产物 + 一份没有图例的文本。输出：符号集名 / None。
    预期行为：图例不一定在第一行（带历史时第一行是历史文本），所以要扫到第一条含 `=` 的行；
    找不到图例就返回 None（宁可不判，也不猜错符号集）。
    """
    loc = locales.load("zh")
    # 图例是**单独一行**、条目之间用「, 」分隔（`render.legend_line` 的口径）；
    # 带历史时它不在第一行，所以反推要扫到第一条含 `=` 的行
    assert audit.infer_symbols("已走过的最近决策（从起点 S 出发）：\n#=墙, .=通道", loc) == "hash"
    assert audit.infer_symbols({"legend": {"█": "墙", " ": "通道"}}, loc) == "block"
    assert audit.infer_symbols("没有图例", loc) is None
    assert (
        audit.infer_symbols("x", loc, [{"legend": {"▓": "墙", "░": "通道"}}]) == "shade"
    )


def test_infer_case_refuses_what_it_cannot_read() -> None:
    """认不出格式 / 符号集时必须返回 None，而不是硬凑一个变体去解析。"""
    loc = locales.load("zh")
    assert audit.infer_case(123, loc) is None
    assert audit.infer_case("没有图例", loc) is None
    case = audit.infer_case({"passages": []}, loc)
    assert case is not None
    assert case.fmt == maze_spec.FMT_PASSAGES
    # passages 不用符号，解析路径也不依赖它——审计固定按 hash 反解
    assert case.symbols == "hash"


def test_family_of_strips_the_index() -> None:
    """题号 → 题族名：去掉数字后缀与随之空出来的下划线。"""
    assert audit.family_of("off_by_one") == "off_by_one"
    assert audit.family_of("off_by_one_2") == "off_by_one"
    assert audit.family_of("next_step_12") == "next_step"


def test_dist_to_set_matches_the_grid_implementation() -> None:
    """审计自带的多源 BFS 必须与 `grid.distances` 一致（独立实现才能当交叉校验）。

    输入：同一个随机迷宫、同一个源点。输出：两张距离表。
    预期行为：逐格相同——两处实现不一致时，审计报出来的「走偏 1 格」就不可信了。
    """
    mask = grid.generate(random.Random(11), 4, 5, algo="dfs")
    mine = audit.dist_to_set(mask, 4, 5, {0})
    theirs = {cell: d for cell, d in enumerate(grid.distances(mask, 4, 5, 0)) if d >= 0}
    assert mine == theirs


def test_audit_file_counts_an_unparsable_sample() -> None:
    """解析不出来的样本要单独计数，不能静默当成「没问题」。

    输入：一份只有一条坏样本的 JSONL（state 是数字、questions 为空）。
    输出：计数。预期行为：`samples` 记 1、`unparsed` 记 1、格式记成 `unknown`。
    """
    path = Path("unparsable.jsonl")
    path.write_text(json.dumps({"state": 7, "questions": {}}) + "\n", encoding="utf-8")
    try:
        counts = audit.audit_file(path, locales.load("zh"))
    finally:
        path.unlink()
    assert counts["samples"] == 1
    assert counts["unparsed"] == 1
    assert counts["formats"]["unknown"] == 1


def test_audit_run_passes_on_a_fresh_maze_run(tmp_path: Path) -> None:
    """端到端闸门：刚跑出来的产物必须过审，且计数与产物本身对得上。

    输入：极小配置跑一遍（纯构造式、零 token）。输出：审计报告与缺失项。
    预期行为：缺失项为空；格式至少 3 种、有带历史的样本、有写明目标方块的选项、
    有标走过的路的样本、有「走偏 1 格」且实测确实偏 1 格的样本、有沿最优路径走的样本。
    """
    result = pipeline.run(make_config(tmp_path, "audit-ok"))
    report, missing = workflow_mod.load("maze").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    counts = report["train"]
    assert counts["samples"] == result.accepted
    assert len([fmt for fmt in counts["formats"] if fmt != "unknown"]) >= 3
    assert counts["off_by_one"] > 0
    assert counts["off_by_one_dev1"] > 0
    assert counts["next_step_on_path"] > 0
    assert counts["unparsed"] == 0


def _audit_rows(tmp_path: Path, name: str, rows: list[dict[str, object]]) -> list[str]:
    """输入：临时目录、子目录名、样本行。输出：审计的缺失项。

    预期行为：把过滤后的行写成一个只有 `train` 的 run 目录再审计——用来造「缺某个特征」
    的产物，验审计的报警分支（只过滤、不改内容，否则可能变成「解析不出来」而不是「缺特征」）。
    """
    target = tmp_path / name
    target.mkdir(parents=True, exist_ok=True)
    (target / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    _, missing = workflow_mod.load("maze").audit(target, ("train",), "zh")
    return missing


def test_audit_run_flags_every_degraded_feature(tmp_path: Path) -> None:
    """审计的每一条报警都必须真的会响——闸门的分支不测，就等于没有闸门。

    输入：一份刚跑好的产物，按特征分别过滤成六份残缺产物。
    输出：六份的缺失项。预期行为：每份都报出对应的那一条，且报的是**缺什么**而不是别的。
    """
    loc = locales.load("zh")
    result = pipeline.run(make_config(tmp_path, "audit-source"))
    rows = [
        json.loads(line)
        for line in (result.run_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    def blob(row: dict[str, object]) -> str:
        """输入：一条样本。输出：state 与 instructions 拼成的串（审计认历史就靠这个）。"""
        questions = row["questions"]
        return str(row["state"]) + str([q.get("instructions") for q in questions.values()])

    one_format = _audit_rows(
        tmp_path,
        "degraded-format",
        [r for r in rows if audit.infer_format(r["state"]) == maze_spec.FMT_BLOCK_TEXT],
    )
    assert any("渲染格式少于 3 种" in item for item in one_format), one_format

    no_history = _audit_rows(
        tmp_path, "degraded-history", [r for r in rows if loc["history"]["prefix"] not in blob(r)]
    )
    assert any("没有带历史决策的样本" in item for item in no_history), no_history

    no_cell = _audit_rows(
        tmp_path,
        "degraded-cell",
        [
            r
            for r in rows
            if not any(
                "→" in option
                for q in r["questions"].values()
                for option in (q.get("criteria") or {})
            )
        ],
    )
    assert any("没有「选项写明目标方块」的样本" in item for item in no_cell), no_cell

    no_trail = _audit_rows(
        tmp_path,
        "degraded-trail",
        [
            r
            for r in rows
            if maze_spec.GLYPH_TRAIL not in str(r["state"]) and "trail" not in str(r["state"])
        ],
    )
    assert any("没有标记走过的路的样本" in item for item in no_trail), no_trail

    no_off = _audit_rows(
        tmp_path,
        "degraded-offbyone",
        [r for r in rows if not any(audit.family_of(qid) == "off_by_one" for qid in r["questions"])],
    )
    assert any("缺少「走偏 1 格」的样本" in item for item in no_off), no_off

    no_next = _audit_rows(
        tmp_path,
        "degraded-nextstep",
        [r for r in rows if not any(audit.family_of(qid) == "next_step" for qid in r["questions"])],
    )
    assert any("缺少「沿最优路径走」的样本" in item for item in no_next), no_next


def test_audit_run_reports_a_missing_split(tmp_path: Path) -> None:
    """切分文件不在就要点名报出来——审计不能因为「文件没了」就当成通过。"""
    report, missing = workflow_mod.load("maze").audit(tmp_path, ("train",), "zh")
    assert report == {}
    assert missing == ["train.jsonl 不存在"]
