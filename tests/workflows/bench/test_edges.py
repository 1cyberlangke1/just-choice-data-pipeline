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


def test_audit_flags_a_missing_file(tmp_path: Path) -> None:
    """输入：产物不存在。输出：无。预期行为：记一条「产物不存在」，不静默当空跑。"""
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["samples"] == 0
    assert any("产物不存在" in item for item in missing)


def test_audit_flags_an_out_of_range_choice_index(tmp_path: Path) -> None:
    """输入：编号越界的产物。输出：无。预期行为：审计记一笔「编号越界」。"""
    row = _sample_row()
    row["answers"]["best"]["choice"] = 99
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["index_bad"] == 1
    assert any("编号越界" in item for item in missing)


def test_audit_flags_a_non_integer_choice_index(tmp_path: Path) -> None:
    """输入：编号不是整数的产物。输出：无。预期行为：同样记一笔「编号越界」。"""
    row = _sample_row()
    row["answers"]["best"]["choice"] = True
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, _ = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["index_bad"] == 1


def test_audit_flags_an_out_of_range_score_index(tmp_path: Path) -> None:
    """输入：档位下标越界的产物。输出：无。

    预期行为：score 也查下标（条数只有六档、差距只有两档），越界同样记一笔「编号越界」。
    """
    row = _sample_row()
    row["answers"]["count"]["score"] = 9
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["index_bad"] == 1
    assert any("编号越界" in item for item in missing)


def test_audit_flags_malformed_criteria_shapes(tmp_path: Path) -> None:
    """输入：choice 的 criteria 是数组 / score 的 criteria 是对象。输出：无。

    预期行为：形状不对也算「编号越界」——产物坏了要看得见，不能在审计里当成没下标可查。
    """
    row = _sample_row()
    row["questions"]["best"]["criteria"] = ["甲", "乙"]
    row["questions"]["count"]["criteria"] = {"n0": "答对 0 道"}
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, _ = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["index_bad"] == 2


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
    del row["questions"]["best"]
    del row["answers"]["best"]
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["families"]["best"] == 0
    assert any("best" in item and "一条都没出" in item for item in missing)


def test_audit_flags_an_answer_that_does_not_match_the_text(tmp_path: Path) -> None:
    """输入：答案被改坏的产物。输出：无。预期行为：记一笔「与文本重算不一致」。"""
    row = _sample_row()
    row["answers"]["pass"] = {"type": "noul", "noul": False}
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["label_mismatch"] == 1
    assert any("与文本重算不一致" in item for item in missing)
