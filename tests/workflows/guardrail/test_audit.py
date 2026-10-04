"""产物审计测试：纯代码能重算的那几条真的会被抓出来。"""

from __future__ import annotations

from pathlib import Path

from jc import pipeline
from jc.jsontypes import JsonValue

from tests.workflows._scaffold import flip_choice as _flip_choice, read_rows as _rows, run_config, write_rows as _write
from tests.workflows.guardrail.conftest import FakeBackend
from workflows.guardrail import workflow as guardrail_workflow
from workflows.guardrail.workflow import GuardrailConfig

CASE_LIMIT = 8


#: 计数器 → 缺失项列表里那句话的抬头（报告给人看，所以点名的是人话不是内部键）。
BAD_LABELS = {
    "content_bad": "条款表或正文认不出来",
    "clause_bad": "逐条款答案与正文不自洽",
    "count_bad": "命中条数与正文不自洽",
    "combo_bad": "命中组合与正文不自洽",
    "action_bad": "处置档位与正文不自洽",
}


def _run(tmp_path: Path, run_id: str = "audit") -> Path:
    """输入：临时目录、run id。输出：run 目录。预期行为：跑一小批，全部合格。"""
    result = pipeline.run(
        run_config(
            tmp_path,
            run_id,
            backend=FakeBackend(),
            workflow="guardrail",
            seed=11,
            override=GuardrailConfig(split="train", case_limit=CASE_LIMIT),
        )
    )
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    return result.run_dir


def _audit(
    run_dir: Path, splits: tuple[str, ...] = ("train",)
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分。输出：(计数, 缺失项)。"""
    return guardrail_workflow.GuardrailWorkflow().audit(run_dir, splits, "zh")


def test_clause_answer_mismatch_is_caught(tmp_path: Path) -> None:
    """输入：把一条逐条款 noul 的答案取反。输出：无。预期行为：`clause_bad` 报 1。"""
    run_dir = _run(tmp_path, "t-clause")
    rows = _rows(run_dir)
    answer = rows[0]["answers"]["clause"]
    answer["noul"] = not answer["noul"]
    _write(run_dir, rows)
    report, missing = _audit(run_dir)
    assert report["train"]["clause_bad"] == 1
    assert any(BAD_LABELS["clause_bad"] in item for item in missing)

