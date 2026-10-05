"""产物审计测试：纯代码能重算的那几条真的会被抓出来。"""

from __future__ import annotations

from pathlib import Path

from jc import pipeline
from jc.jsontypes import JsonValue

from tests.workflows._scaffold import flip_choice as _flip_choice, read_rows as _rows, run_config, write_rows as _write
from tests.workflows.alertrisk.conftest import FakeBackend
from workflows.alertrisk import features, render
from workflows.alertrisk import workflow as alertrisk_workflow
from workflows.alertrisk.workflow import AlertRiskConfig, AlertRiskWorkflow

CASE_LIMIT = 8


#: 计数器 → 缺失项列表里那句话的抬头（报告给人看，所以点名的是人话不是内部键）。
BAD_LABELS = {
    "content_bad": "因子表或警报记录认不出来",
    "fact_bad": "逐因子答案与警报记录不自洽",
    "grade_bad": "风险等级与警报记录不自洽",
    "highs_bad": "命中组合与警报记录不自洽",
    "act_bad": "处置档位与警报记录不自洽",
}


def _run(tmp_path: Path, run_id: str = "audit") -> Path:
    """输入：临时目录、run id。输出：run 目录。预期行为：跑一小批，全部合格。"""
    result = pipeline.run(
        run_config(
            tmp_path,
            run_id,
            backend=FakeBackend(),
            workflow="alertrisk",
            seed=13,
            override=AlertRiskConfig(split="train", case_limit=CASE_LIMIT),
        )
    )
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    return result.run_dir


def _audit(
    run_dir: Path, splits: tuple[str, ...] = ("train",)
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分。输出：(计数, 缺失项)。"""
    return AlertRiskWorkflow().audit(run_dir, splits, "zh")


def test_fact_answer_mismatch_is_caught(tmp_path: Path) -> None:
    """输入：把一条逐因子 noul 的答案取反。输出：无。预期行为：`fact_bad` 报 1。"""
    run_dir = _run(tmp_path, "t-fact")
    rows = _rows(run_dir)
    answer = rows[0]["answers"]["fact"]
    answer["noul"] = not answer["noul"]
    _write(run_dir, rows)
    report, missing = _audit(run_dir)
    assert report["train"]["fact_bad"] == 1
    assert any(BAD_LABELS["fact_bad"] in item for item in missing)


def test_cases_for_is_the_same_list_the_pipeline_sees(tmp_path: Path) -> None:
    """输入：同一条采样参数。输出：无。

    预期行为：`workflow.cases_for` 与管线挑出来的 case 是同一批——审计之外的那几把尺子（题族齐不齐、
    维度跨度）都靠它，两边不一致就会量错东西。
    """
    config = AlertRiskConfig(split="train", case_limit=CASE_LIMIT)
    assert len(alertrisk_workflow.cases_for(config)) == CASE_LIMIT
