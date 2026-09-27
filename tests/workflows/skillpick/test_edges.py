"""产物审计的每一条分支：缺产物、编号越界、未知题族、读不出来、缺题族、答案对不上。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from jc import assemble, jsonio
from tests.workflows.skillpick.conftest import _make_spec
from workflows.skillpick import locales, questions, render
from workflows.skillpick.workflow import SkillPickWorkflow

WF = SkillPickWorkflow()
LOC = locales.load("zh")


def _sample_row(**overrides: Any) -> dict[str, Any]:
    """输入：case 覆盖项。输出：一条走生产路径组装出来的合格样本。"""
    spec = _make_spec(**overrides)
    items = questions.build(
        spec.case.tag, spec.case.grant, spec.inuse_tag, spec.rows, LOC, spec.phrasing
    )
    return assemble.assemble_sample(
        "skillpick-train-0000000", render.build(spec, LOC), items, random.Random(1)
    )


def test_audit_reports_a_missing_split(tmp_path: Path) -> None:
    """输入：一个不存在的切分。输出：无。预期行为：如实记一笔「产物不存在」。"""
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert any("产物不存在" in item for item in missing)
    assert report["train"]["samples"] == 0


def test_audit_accepts_a_faithful_row(tmp_path: Path) -> None:
    """输入：一条合格产物。输出：无。预期行为：零缺失，并记下题族分布与匹配上的样本数。"""
    row = _sample_row()
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == 1
    assert report["train"]["index_bad"] == 0
    assert report["train"]["label_mismatch"] == 0
    assert report["train"]["fits_true"] == 1
    assert dict(report["train"]["types"]) == {"choice": 2, "noul": 3}
    assert dict(report["train"]["families"]) == {
        "allow": 1,
        "conflict": 1,
        "fits": 1,
        "pick": 1,
        "scope": 1,
    }


def test_audit_does_not_ask_for_four_families_on_a_decoy(tmp_path: Path) -> None:
    """输入：一条干扰标签的产物（只有 `fits` 一族）。输出：无。

    预期行为：零缺失——「应有题族」按产物里真的读出来的标签与技能表点名，
    否则这种样本会被误报成缺题。
    """
    row = _sample_row(tag="tidy_sheet_tpl")
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert missing == []
    assert dict(report["train"]["families"]) == {"fits": 1}
    assert report["train"]["fits_true"] == 0


def test_audit_flags_an_out_of_range_choice_index(tmp_path: Path) -> None:
    """输入：编号越界的产物。输出：无。预期行为：审计记一笔「编号越界」。"""
    row = _sample_row()
    row["answers"]["pick"]["choice"] = 99
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["index_bad"] == 1
    assert any("编号越界" in item for item in missing)


def test_audit_flags_a_non_integer_choice_index(tmp_path: Path) -> None:
    """输入：编号不是整数的产物。输出：无。预期行为：同样记一笔「编号越界」。"""
    row = _sample_row()
    row["answers"]["scope"]["choice"] = True
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
    del row["questions"]["pick"]
    del row["answers"]["pick"]
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["families"]["pick"] == 0
    assert any("pick" in item and "一条都没出" in item for item in missing)
