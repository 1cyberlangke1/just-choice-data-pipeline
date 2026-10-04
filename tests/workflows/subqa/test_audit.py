"""产物审计测试：纯代码能重算的那几条真的会被抓出来。"""

from __future__ import annotations

from pathlib import Path


from jc import pipeline
from jc.jsontypes import JsonValue

from tests.workflows._scaffold import read_rows as _rows, run_config, write_rows as _write
from tests.workflows.subqa.conftest import FakeBackend
from workflows.subqa import features, render
from workflows.subqa import workflow as subqa_workflow
from workflows.subqa.workflow import SubQaConfig

CASE_LIMIT = 8


def _run(tmp_path: Path, run_id: str = "audit") -> Path:
    """输入：临时目录、run id。输出：run 目录。预期行为：跑一小批，全部合格。"""
    result = pipeline.run(
        run_config(
            tmp_path,
            run_id,
            backend=FakeBackend(),
            workflow="subqa",
            seed=11,
            override=SubQaConfig(split="train", case_limit=CASE_LIMIT),
        )
    )
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    return result.run_dir


def _audit(
    run_dir: Path, splits: tuple[str, ...] = ("train",)
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分。输出：(计数, 缺失项)。"""
    return subqa_workflow.SubQaWorkflow().audit(run_dir, splits, "zh")


def test_a_rewritten_translation_is_caught(tmp_path: Path) -> None:
    """输入：把一条**译文里留着标记**的样本的 `Target:` 那一行换成干净英文。输出：无。

    预期行为：`flag_bad` / `count_bad` / `combo_bad` / `act_bad` 里至少有一个报出来——审计走的是
    「从文本重推标签」这条独立路径，把标记擦掉之后重推出来的命中组合就与题面答案对不上了。
    """
    run_dir = _run(tmp_path, "t-rewrite")
    rows = _rows(run_dir)
    target_row = next(
        row
        for row in rows
        if "S2" in features.hit_set(render.parse_state(row["state"]).text or "")
    )
    lines = [
        "Target: We should move the meeting to Thursday, if that still works."
        if line.strip().startswith("Target:")
        else line
        for line in str(target_row["state"]).splitlines()
    ]
    target_row["state"] = "\n".join(lines)
    _write(run_dir, rows)
    report, _ = _audit(run_dir)
    counts = report["train"]
    assert counts["flag_bad"] + counts["count_bad"] + counts["combo_bad"] + counts["act_bad"] > 0

