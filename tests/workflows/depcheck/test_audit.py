"""产物审计测试：纯代码能重算的那几条真的会被抓出来。"""

from __future__ import annotations

from pathlib import Path

from jc import pipeline
from jc.jsontypes import JsonValue

from tests.workflows._scaffold import flip_choice as _flip_choice, read_rows as _rows, run_config, write_rows as _write
from tests.workflows.depcheck.conftest import FakeBackend
from workflows.depcheck import features, render
from workflows.depcheck import workflow as depcheck_workflow
from workflows.depcheck.workflow import DepCheckConfig, DepCheckWorkflow

CASE_LIMIT = 8


#: 计数器 → 缺失项列表里那句话的抬头（报告给人看，所以点名的是人话不是内部键）。
BAD_LABELS = {
    "content_bad": "标记表或依赖记录认不出来",
    "flag_bad": "逐标记答案与依赖记录不自洽",
    "grade_bad": "风险等级与依赖记录不自洽",
    "combo_bad": "命中组合与依赖记录不自洽",
    "act_bad": "处置档位与依赖记录不自洽",
}


def _run(tmp_path: Path, run_id: str = "audit") -> Path:
    """输入：临时目录、run id。输出：run 目录。预期行为：跑一小批，全部合格。"""
    result = pipeline.run(
        run_config(
            tmp_path,
            run_id,
            backend=FakeBackend(),
            workflow="depcheck",
            seed=13,
            override=DepCheckConfig(split="train", case_limit=CASE_LIMIT),
        )
    )
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    return result.run_dir


def _audit(
    run_dir: Path, splits: tuple[str, ...] = ("train",)
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分。输出：(计数, 缺失项)。"""
    return DepCheckWorkflow().audit(run_dir, splits, "zh")


def test_cases_for_is_the_same_list_the_pipeline_sees(tmp_path: Path) -> None:
    """输入：同一条采样参数。输出：无。

    预期行为：`workflow.cases_for` 与管线挑出来的 case 是同一批——审计之外的那几把尺子（题族齐不齐、
    维度跨度）都靠它，两边不一致就会量错东西。
    """
    config = DepCheckConfig(split="train", case_limit=CASE_LIMIT)
    assert len(depcheck_workflow.cases_for(config)) == CASE_LIMIT
