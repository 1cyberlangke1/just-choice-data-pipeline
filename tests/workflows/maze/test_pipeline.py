"""迷宫 workflow 的端到端测试：产物、统计、可复现性、判别器接入。

覆盖：小规模跑一遍全绿、两次同种子生成逐字节一致、stats/manifest 落盘、
case 维度出现在统计里、判别器通过率被记录。
"""

from __future__ import annotations

import json
from pathlib import Path

from jc import pipeline, workflow as workflow_mod
from workflows.maze import spec as maze_spec
from tests.workflows._scaffold import run_config


def tiny_config() -> maze_spec.MazeConfig:
    """极小配置：2 格式 × 1 符号集 × 2 历史 × 2 × 2 × 2 偏离 × 2 cut = 64 个 case。"""
    return maze_spec.MazeConfig(
        size=(3, 5),
        algos=("dfs", "prim"),
        formats=(maze_spec.FMT_BLOCK_TEXT, maze_spec.FMT_BLOCK_ROWS),
        symbols=("hash",),
        history_lens=(0, 2),
        mark_trails=(False, True),
        option_details=(maze_spec.OPTION_PLAIN, maze_spec.OPTION_CELL),
        deviations=(0, 1),
    )


def make_config(tmp_path: Path, run_id: str, *, concurrency: int = 1) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小配置试跑的运行配置。"""
    return run_config(
        tmp_path,
        run_id,
        workflow="maze",
        seed=4242,
        concurrency=concurrency,
        override=tiny_config(),
    )


def test_run_produces_clean_dataset(tmp_path: Path) -> None:
    workflow = workflow_mod.load("maze")
    result = pipeline.run(make_config(tmp_path, "r1"))
    assert result.rejected == []
    assert result.accepted == len(workflow.build_cases(tiny_config()))
    data = (result.run_dir / "train.jsonl").read_text(encoding="utf-8")
    assert data.count("\n") == result.accepted
    assert (result.run_dir / "stats-train.md").exists()
    assert (result.run_dir / "manifest-train.json").exists()
    assert not (result.run_dir / "rejected.jsonl").exists()


def test_run_is_reproducible(tmp_path: Path) -> None:
    first = pipeline.run(make_config(tmp_path, "a"))
    second = pipeline.run(make_config(tmp_path, "b"))
    assert (first.run_dir / "train.jsonl").read_bytes() == (second.run_dir / "train.jsonl").read_bytes()
    assert (first.run_dir / "manifest-train.json").read_text(encoding="utf-8") != (
        second.run_dir / "manifest-train.json"
    ).read_text(encoding="utf-8")  # run_id 不同，manifest 不同


def test_concurrent_run_is_byte_identical(tmp_path: Path) -> None:
    """输入：同一份迷宫配置，一次串行、一次并发 4。

    输出：两次的 `train.jsonl` 与统计。
    预期行为：逐字节一致——每条样本的 RNG 只由序号派生（管线 docstring 的承诺），
    并发只该省时间，不该改内容、顺序或任何计数。
    """
    serial = pipeline.run(make_config(tmp_path, "serial"))
    concurrent = pipeline.run(make_config(tmp_path, "parallel", concurrency=4))
    assert (serial.run_dir / "train.jsonl").read_bytes() == (
        concurrent.run_dir / "train.jsonl"
    ).read_bytes()
    assert serial.stats["accepted"] == concurrent.stats["accepted"]
    assert serial.stats["reject_codes"] == concurrent.stats["reject_codes"]
    assert serial.stats["judges"] == concurrent.stats["judges"]
    assert serial.stats["answers"] == concurrent.stats["answers"]


def test_stats_cover_all_case_dimensions(tmp_path: Path) -> None:
    result = pipeline.run(make_config(tmp_path, "r2"))
    dims = result.stats["dimensions"]
    assert set(dims["format"]) == {maze_spec.FMT_BLOCK_TEXT, maze_spec.FMT_BLOCK_ROWS}
    assert set(dims["symbols"]) == {"hash"}
    assert set(dims["history_len"]) == {"0", "2"}
    assert set(dims["mark_trail"]) == {"False", "True"}
    assert set(dims["option_detail"]) == {maze_spec.OPTION_PLAIN, maze_spec.OPTION_CELL}
    assert set(dims["deviation"]) == {"0", "1"}
    assert set(dims["cut"]) == {"False", "True"}
    assert result.stats["accepted"] > 0


def test_judges_are_recorded(tmp_path: Path) -> None:
    """两个判别器都要真的跑过，且通过率记进统计。"""
    result = pipeline.run(make_config(tmp_path, "r3"))
    assert set(result.stats["judges"]) == {"roundtrip", "label"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0


def test_question_stats_are_per_type(tmp_path: Path) -> None:
    """题数与题型分布必须可查（SFT 导出按题展开，所以这里核对每题统计）。"""
    result = pipeline.run(make_config(tmp_path, "r4"))
    assert sum(result.stats["families"].values()) > result.stats["accepted"]
    assert any(key.startswith("choice:") for key in result.stats["answers"])
    assert any(key.startswith("score:") for key in result.stats["answers"])
    assert set(result.stats["types"]) <= {"choice", "noul", "score"}


def test_manifest_has_no_backend(tmp_path: Path) -> None:
    """纯构造式 run 的 manifest 里 backend 必须是空字典（不编造后端来源）。"""
    result = pipeline.run(make_config(tmp_path, "r5"))
    manifest = json.loads((result.run_dir / "manifest-train.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == {}
    assert manifest["needs"] == []
