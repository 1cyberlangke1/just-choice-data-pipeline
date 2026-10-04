"""产物审计测试：纯代码能重算的那几条真的会被抓出来。"""

from __future__ import annotations

from pathlib import Path


from jc import pipeline
from jc.jsontypes import JsonValue

from tests.workflows._scaffold import read_rows as _rows, run_config, write_rows as _write
from tests.workflows.debugstep.conftest import FakeBackend
from workflows.debugstep import workflow as debugstep_workflow
from workflows.debugstep.workflow import DebugStepConfig

CASE_LIMIT = 8


def _run(tmp_path: Path, run_id: str = "audit") -> Path:
    """输入：临时目录、run id。输出：run 目录。预期行为：跑一小批，全部合格。"""
    result = pipeline.run(
        run_config(
            tmp_path,
            run_id,
            backend=FakeBackend(),
            workflow="debugstep",
            seed=11,
            override=DebugStepConfig(split="train", case_limit=CASE_LIMIT),
        )
    )
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    return result.run_dir


def _audit(
    run_dir: Path, splits: tuple[str, ...] = ("train",)
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分。输出：(计数, 缺失项)。"""
    return debugstep_workflow.DebugStepWorkflow().audit(run_dir, splits, "zh")


