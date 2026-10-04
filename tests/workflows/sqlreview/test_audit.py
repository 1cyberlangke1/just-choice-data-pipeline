"""产物审计测试：纯代码能重算的那几条真的会被抓出来。"""

from __future__ import annotations

from pathlib import Path


from jc import pipeline
from jc.jsontypes import JsonValue

from tests.workflows._scaffold import read_rows as _rows, run_config, write_rows as _write
from tests.workflows.sqlreview.conftest import FakeBackend
from workflows.sqlreview import features, render
from workflows.sqlreview import workflow as sqlreview_workflow
from workflows.sqlreview.workflow import SqlReviewConfig

CASE_LIMIT = 8


def _run(tmp_path: Path, run_id: str = "audit") -> Path:
    """输入：临时目录、run id。输出：run 目录。预期行为：跑一小批，全部合格。"""
    result = pipeline.run(
        run_config(
            tmp_path,
            run_id,
            backend=FakeBackend(),
            workflow="sqlreview",
            seed=11,
            override=SqlReviewConfig(split="train", case_limit=CASE_LIMIT),
        )
    )
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    return result.run_dir


def _audit(
    run_dir: Path, splits: tuple[str, ...] = ("train",)
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分。输出：(计数, 缺失项)。"""
    return sqlreview_workflow.SqlReviewWorkflow().audit(run_dir, splits, "zh")


def test_a_rewritten_line_is_caught(tmp_path: Path) -> None:
    """输入：把一条**计划是大表全扫**的样本的 `plan:` 那一行换成走索引的小计划。输出：无。

    预期行为：`flag_bad` / `count_bad` / `combo_bad` / `act_bad` 里至少有一个报出来——审计走的是
    「从文本重推标签」这条独立路径，把形状擦掉之后重推出来的命中组合就与题面答案对不上了。
    """
    run_dir = _run(tmp_path, "t-rewrite")
    rows = _rows(run_dir)
    target_row = next(
        row for row in rows if "Q2" in features.hit_set(render.parse_state(row["state"]).text or "")
    )
    lines = [
        "plan: Index Scan using orders_pkey on orders  (cost=0.43..8.45 rows=1 width=12)"
        if line.strip().startswith("plan:")
        else line
        for line in str(target_row["state"]).splitlines()
    ]
    target_row["state"] = "\n".join(lines)
    _write(run_dir, rows)
    report, _ = _audit(run_dir)
    counts = report["train"]
    assert counts["flag_bad"] + counts["count_bad"] + counts["combo_bad"] + counts["act_bad"] > 0

