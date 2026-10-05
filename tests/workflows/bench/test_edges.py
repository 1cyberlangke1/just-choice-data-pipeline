"""产物审计的每一条分支：缺产物、编号越界、未知题族、读不出来、缺题族、答案对不上。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random
from pathlib import Path


from jc import assemble, jsonio
from tests.workflows.bench.conftest import _make_spec
from workflows.bench import locales, questions, render
from workflows.bench.workflow import BenchWorkflow

WF = BenchWorkflow()
LOC = locales.load("zh")


def _sample_row(**overrides: object) -> dict[str, JsonValue]:
    """输入：case 覆盖项。输出：一条走产线组装出来的样本。"""
    spec = _make_spec(**overrides)
    items = questions.build(spec.rows, spec, LOC, spec.phrasing)
    return assemble.assemble_sample(
        "bench-train-0000000", render.build(spec, LOC), items, random.Random(1)
    )


def test_audit_passes_a_good_row(tmp_path: Path) -> None:
    """输入：一条合格产物。输出：无。预期行为：零缺失，计数与题族分布对得上。"""
    jsonio.write_jsonl(tmp_path / "train.jsonl", [_sample_row()])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == 1
    assert report["train"]["index_bad"] == 0
    assert report["train"]["label_mismatch"] == 0
    assert report["train"]["hit_true"] == 3
    assert report["train"]["pass_true"] == 1
    assert dict(report["train"]["types"]) == {"choice": 1, "noul": 6, "score": 2}
    assert dict(report["train"]["families"]) == {
        "best": 1,
        "count": 1,
        "gap": 1,
        "hit": 5,
        "pass": 1,
    }


def test_audit_does_not_ask_for_gap_when_the_first_column_leads(tmp_path: Path) -> None:
    """输入：`k4s0`（第 1 列自己就是最好）的产物。输出：无。

    预期行为：零缺失——「应有题族」按产物里真的读出来的那屏记录对照，否则这种样本会被
    误报成缺 `gap`。
    """
    jsonio.write_jsonl(tmp_path / "train.jsonl", [_sample_row(table="k4s0")])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert missing == []
    assert report["train"]["families"]["gap"] == 0
    assert dict(report["train"]["families"]) == {
        "best": 1,
        "count": 1,
        "hit": 5,
        "pass": 1,
    }


def test_audit_flags_an_unknown_family(tmp_path: Path) -> None:
    """输入：带未知题族的产物。输出：无。预期行为：审计点名那个题族。"""
    row = _sample_row()
    row["questions"]["bogus"] = {"type": "noul", "criteria": {"true": "是", "false": "否"}}
    row["answers"]["bogus"] = {"type": "noul", "noul": True}
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["unknown_families"]["bogus"] == 1
    assert any("未知题族" in item for item in missing)


def test_audit_flags_an_answer_that_does_not_match_the_text(tmp_path: Path) -> None:
    """输入：答案被改坏的产物。输出：无。预期行为：记一笔「与文本重算不一致」。"""
    row = _sample_row()
    row["answers"]["pass"] = {"type": "noul", "noul": False}
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["label_mismatch"] == 1
    assert any("与文本重算不一致" in item for item in missing)
