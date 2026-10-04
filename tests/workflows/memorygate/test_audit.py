"""产物审计测试：纯代码能重算的那几条真的会被抓出来。"""

from __future__ import annotations

from pathlib import Path

from jc import pipeline
from jc.jsontypes import JsonValue

from tests.workflows._scaffold import flip_choice as _flip_choice, read_rows as _rows, run_config, write_rows as _write
from tests.workflows.memorygate.conftest import FakeBackend
from workflows.memorygate import workflow as memorygate_workflow
from workflows.memorygate.workflow import MemoryGateConfig

CASE_LIMIT = 8


#: 计数器 → 缺失项列表里那句话的抬头（报告给人看，所以点名的是人话不是内部键）。
BAD_LABELS = {
    "content_bad": "判据表或记忆认不出来",
    "flag_bad": "逐判据答案与记忆不自洽",
    "count_bad": "命中条数与记忆不自洽",
    "combo_bad": "命中组合与记忆不自洽",
    "gate_bad": "处置档位与记忆不自洽",
}


def _run(tmp_path: Path, run_id: str = "audit") -> Path:
    """输入：临时目录、run id。输出：run 目录。预期行为：跑一小批，全部合格。"""
    result = pipeline.run(
        run_config(
            tmp_path,
            run_id,
            backend=FakeBackend(),
            workflow="memorygate",
            seed=11,
            override=MemoryGateConfig(split="train", case_limit=CASE_LIMIT),
        )
    )
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    return result.run_dir


def _audit(
    run_dir: Path, splits: tuple[str, ...] = ("train",)
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分。输出：(计数, 缺失项)。"""
    return memorygate_workflow.MemoryGateWorkflow().audit(run_dir, splits, "zh")

