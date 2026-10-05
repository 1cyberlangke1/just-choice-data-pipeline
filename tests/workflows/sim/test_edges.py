"""产物审计的每一条分支：缺产物、编号越界、未知题族、读不出来、缺题族、答案对不上。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random
from pathlib import Path


from jc import assemble, jsonio
from tests.workflows.sim.conftest import _make_spec
from workflows.sim import locales, questions, render
from workflows.sim.workflow import SimWorkflow

WF = SimWorkflow()
LOC = locales.load("zh")


def _sample_row(**overrides: object) -> dict[str, JsonValue]:
    """输入：case 覆盖项。输出：一条走生产路径组装出来的合格样本。"""
    spec = _make_spec(**overrides)
    items = questions.build(spec.rows, spec, LOC, spec.phrasing)
    return assemble.assemble_sample(
        "sim-train-0000000", render.build(spec, LOC), items, random.Random(1)
    )


def test_audit_accepts_a_faithful_row(tmp_path: Path) -> None:
    """输入：一条合格产物（`b0`：资源 0，只有待机能用）。输出：无。

    预期行为：零缺失，并记下题族分布、能用条数与结束条数。
    """
    row = _sample_row()
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == 1
    assert report["train"]["index_bad"] == 0
    assert report["train"]["label_mismatch"] == 0
    assert report["train"]["legal_true"] == 1
    assert report["train"]["done_true"] == 0
    assert dict(report["train"]["types"]) == {"choice": 1, "noul": 7, "score": 7}
    assert dict(report["train"]["families"]) == {
        "best": 1,
        "can_move": 6,
        "cost": 1,
        "finished": 1,
        "gain": 6,
    }


def test_audit_does_not_ask_for_best_on_a_finished_board(tmp_path: Path) -> None:
    """输入：`b3`（回合到顶）的产物。输出：无。

    预期行为：零缺失——「应有题族」按产物里真的读出来的那屏局面对照，否则这种样本会被
    误报成缺 `best` / `cost`。
    """
    row = _sample_row(board="b3")
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert missing == []
    assert report["train"]["done_true"] == 1
    assert dict(report["train"]["families"]) == {
        "can_move": 6,
        "finished": 1,
        "gain": 6,
    }
