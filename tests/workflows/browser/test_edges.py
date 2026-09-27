"""产物审计的每一条分支：缺产物、编号越界、未知题族、读不出来、缺题族、答案对不上。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from jc import assemble, jsonio
from tests.workflows.browser.conftest import _make_spec
from workflows.browser import locales, questions, render
from workflows.browser.workflow import BrowserWorkflow

WF = BrowserWorkflow()
LOC = locales.load("zh")


def _sample_row(**overrides: Any) -> dict[str, Any]:
    """输入：case 覆盖项。输出：一条走生产路径组装出来的合格样本。"""
    spec = _make_spec(**overrides)
    items = questions.build(spec.rows, spec.task, spec.object, LOC, spec.phrasing)
    return assemble.assemble_sample(
        "browser-train-0000000", render.build(spec, LOC), items, random.Random(1)
    )


def test_audit_reports_a_missing_split(tmp_path: Path) -> None:
    """输入：一个不存在的切分。输出：无。预期行为：如实记一笔「产物不存在」。"""
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert any("产物不存在" in item for item in missing)
    assert report["train"]["samples"] == 0


def test_audit_accepts_a_faithful_row(tmp_path: Path) -> None:
    """输入：一条合格产物。输出：无。预期行为：零缺失，并记下完成度与任务的分布。"""
    row = _sample_row()
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == 1
    assert report["train"]["index_bad"] == 0
    assert report["train"]["label_mismatch"] == 0
    assert report["train"]["dones"] == {"False": 1}
    assert report["train"]["tasks"] == {"fill": 1}
    assert set(report["train"]["types"]) == {"choice", "noul"}
    assert set(report["train"]["families"]) == {"done", "target", "action", "risk"}


def test_audit_flags_an_out_of_range_choice_index(tmp_path: Path) -> None:
    """输入：编号越界的产物。输出：无。预期行为：审计记一笔「编号越界」。"""
    row = _sample_row()
    qid = next(key for key in row["questions"] if row["questions"][key]["type"] == "choice")
    row["answers"][qid]["choice"] = 99
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["index_bad"] == 1
    assert any("编号越界" in item for item in missing)


def test_audit_flags_a_non_integer_choice_index(tmp_path: Path) -> None:
    """输入：编号不是整数的产物。输出：无。预期行为：同样记一笔「编号越界」。"""
    row = _sample_row()
    qid = next(key for key in row["questions"] if row["questions"][key]["type"] == "choice")
    row["answers"][qid]["choice"] = True
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, _ = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["index_bad"] == 1


def test_audit_flags_an_unknown_family(tmp_path: Path) -> None:
    """输入：带未知题族的产物。输出：无。预期行为：审计点名那个题族。"""
    row = _sample_row()
    row["questions"]["bogus"] = {"type": "noul", "criteria": {"true": "是", "false": "否"}}
    row["answers"]["bogus"] = {"type": "noul", "noul": True}
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["unknown_families"]["bogus"] == 1
    assert any("未知题族" in item for item in missing)


def test_audit_flags_a_state_it_cannot_read(tmp_path: Path) -> None:
    """输入：state 读不出来的产物。输出：无。

    预期行为：记一条「反解不出来」而不是静默跳过——静默跳过会把一条坏样本算成好样本。
    """
    row = _sample_row()
    row["state"] = "今天天气不错。"
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["label_mismatch"] == 1
    assert any("反解不出来" in item for item in missing)


def test_audit_flags_a_family_that_never_shows_up(tmp_path: Path) -> None:
    """输入：把某一族的题全删掉。输出：无。预期行为：审计记一笔「一条都没出」。"""
    row = _sample_row()
    del row["questions"]["risk"]
    del row["answers"]["risk"]
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["families"]["risk"] == 0
    assert any("risk" in item and "一条都没出" in item for item in missing)


def test_audit_does_not_demand_families_a_finished_task_never_asks(tmp_path: Path) -> None:
    """输入：一条「任务已完成」的产物（只有完成度题）。输出：无。

    预期行为：审计零缺失——定位 / 动作 / 风险三族本来就不该出，按「题族全都要有」点名
    会把它们误报成缺题。
    """
    row = _sample_row(done=True)
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert missing == []
    assert set(report["train"]["families"]) == {"done"}
    assert report["train"]["dones"] == {"True": 1}


def test_audit_flags_an_answer_that_contradicts_the_text(tmp_path: Path) -> None:
    """输入：把风险答案翻过来的产物。输出：无。

    预期行为：记一笔「与文本重算不一致」——审计不依赖真值规格，只靠文本重算。
    """
    row = _sample_row()
    row["answers"]["risk"]["noul"] = not row["answers"]["risk"]["noul"]
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["label_mismatch"] == 1
    assert any("重算不一致" in item for item in missing)


def test_audit_flags_a_question_type_this_workflow_never_emits(tmp_path: Path) -> None:
    """输入：把完成度那道题换成 `score` 题型。输出：无。

    预期行为：记一笔答案对不上——本题型只出 choice / noul，混进来的题不静默放过。
    """
    row = _sample_row()
    row["questions"]["done"] = {"type": "score", "criteria": ["a", "b"]}
    row["answers"]["done"] = {"type": "score", "score": 3}
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["label_mismatch"] == 1
    assert any("重算不一致" in item for item in missing)
