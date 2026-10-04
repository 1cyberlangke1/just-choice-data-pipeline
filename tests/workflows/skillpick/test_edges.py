"""产物审计的每一条分支：缺产物、编号越界、未知题族、读不出来、缺题族、答案对不上。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random
from pathlib import Path


from jc import assemble, jsonio
from tests.workflows.skillpick.conftest import _make_spec
from workflows.skillpick import locales, questions, render
from workflows.skillpick.workflow import SkillPickWorkflow

WF = SkillPickWorkflow()
LOC = locales.load("zh")


def _sample_row(**overrides: object) -> dict[str, JsonValue]:
    """输入：case 覆盖项。输出：一条走生产路径组装出来的合格样本。"""
    spec = _make_spec(**overrides)
    items = questions.build(
        spec.case.tag, spec.case.grant, spec.inuse_tag, spec.rows, LOC, spec.phrasing
    )
    return assemble.assemble_sample(
        "skillpick-train-0000000", render.build(spec, LOC), items, random.Random(1)
    )


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
