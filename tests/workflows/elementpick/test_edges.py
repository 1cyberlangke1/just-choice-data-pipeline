"""产物审计的每一条分支：缺产物、编号越界、未知题族、读不出来、缺题族、答案对不上。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random
from pathlib import Path


from jc import assemble, jsonio
from tests.workflows.elementpick.conftest import _make_spec
from workflows.elementpick import locales, questions, render
from workflows.elementpick.workflow import ElementPickWorkflow

WF = ElementPickWorkflow()
LOC = locales.load("zh")


def _sample_row(**overrides: object) -> dict[str, JsonValue]:
    """输入：case 覆盖项。输出：一条走生产路径组装出来的合格样本。"""
    spec = _make_spec(**overrides)
    items = questions.build(spec.rows, spec.case.label, LOC, spec.phrasing)
    return assemble.assemble_sample(
        "elementpick-train-0000000", render.build(spec, LOC), items, random.Random(1)
    )


def _sample_row_in(lang: str, **overrides: object) -> dict[str, JsonValue]:
    """输入：语言、case 覆盖项。输出：用那种语言渲染出来的合格样本。"""
    spec = _make_spec(**overrides)
    loc = locales.load(lang)
    items = questions.build(spec.rows, spec.case.label, loc, spec.phrasing)
    return assemble.assemble_sample(
        f"elementpick-train-{lang}-0000", render.build(spec, loc), items, random.Random(1)
    )


def test_audit_accepts_a_faithful_row(tmp_path: Path) -> None:
    """输入：一条有落点的合格产物。输出：无。

    预期行为：零缺失，并记下题族分布、题型分布与有落点的样本数。
    """
    row = _sample_row(extras=2, shadow=True)
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == 1
    assert report["train"]["index_bad"] == 0
    assert report["train"]["label_mismatch"] == 0
    assert report["train"]["found_true"] == 1
    assert dict(report["train"]["types"]) == {"choice": 2, "noul": 2, "score": 1}
    assert dict(report["train"]["families"]) == {
        "decoy": 1,
        "found": 1,
        "role": 1,
        "shadow": 1,
        "target": 1,
    }


def test_audit_does_not_ask_for_three_families_without_a_landing(tmp_path: Path) -> None:
    """输入：一条没有落点的产物（只有 `found` 与 `shadow`）。输出：无。

    预期行为：零缺失——「应有题族」按产物里真的读出来的动作标签与元素表点名，
    否则这种样本会被误报成缺题。
    """
    row = _sample_row(shape="miss")
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert missing == []
    assert dict(report["train"]["families"]) == {"found": 1, "shadow": 1}
    assert report["train"]["found_true"] == 0


def test_audit_flags_an_out_of_range_score(tmp_path: Path) -> None:
    """输入：score 档位越界的产物。输出：无。预期行为：同样记一笔「编号越界」。"""
    row = _sample_row()
    row["answers"]["decoy"]["score"] = 7
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = WF.audit(tmp_path, ("train",), "zh")
    assert report["train"]["index_bad"] == 1
    assert any("编号越界" in item for item in missing)


def test_audit_reads_a_mixed_language_file(tmp_path: Path) -> None:
    """输入：中英两种语言渲染出来的产物拼成一份。输出：无。

    预期行为：审计用**合并标签表**反解，所以调用方给的语言与产物那一种不一致也零缺失
    （产物按 `locales[seq % len]` 轮转渲染，这是常态）。
    """
    mixed = [
        _sample_row_in(lang, **case)
        for case, lang in (
            ({"label": "save"}, "zh"),
            ({"label": "delete", "extras": 2, "shadow": True}, "en"),
        )
    ]
    jsonio.write_jsonl(tmp_path / "train.jsonl", mixed)
    report, missing = WF.audit(tmp_path, ("train",), "en")
    assert missing == []
    assert report["train"]["samples"] == 2
