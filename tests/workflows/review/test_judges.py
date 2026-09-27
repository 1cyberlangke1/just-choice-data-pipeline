"""判别器的测试：反解 → 重算 → 逐格比对，以及每一条「对不上」都要点名。"""

from __future__ import annotations

import random
from typing import Any

from jc import assemble
from tests.workflows.review.conftest import _make_spec
from workflows.review import judges, locales, parse, questions, render, spec as sp


def _sample(**overrides: Any) -> tuple[dict[str, Any], sp.ChangeSpec]:
    """输入：case 覆盖项。输出：(走生产路径组装出来的合格样本, 真值规格)。"""
    spec = _make_spec(**overrides)
    loc = locales.load("zh")
    items = questions.build(spec.rows, loc, spec.phrasing)
    sample = assemble.assemble_sample(
        "review-train-0000000", render.build(spec, loc), items, random.Random(1)
    )
    return sample, spec


def test_answer_value_handles_every_type() -> None:
    """输入：三种题型的若干答案。输出：无。

    预期行为：choice 给选项名（编号越界 / 类型不对给 None），noul 给布尔，score 给下标。
    """
    choice = {"type": "choice", "criteria": {"a": "甲", "b": "乙"}}
    assert judges.answer_value(choice, {"choice": 1}) == "b"
    assert judges.answer_value(choice, {"choice": 2}) is None
    assert judges.answer_value(choice, {"choice": True}) is None
    assert judges.answer_value(choice, {}) is None
    noul = {"type": "noul", "criteria": {"true": "是", "false": "否"}}
    assert judges.answer_value(noul, {"noul": True}) is True
    assert judges.answer_value(noul, {"noul": 1}) is None
    score = {"type": "score", "criteria": ["甲", "乙"]}
    assert judges.answer_value(score, {"score": 1}) == 1
    assert judges.answer_value(score, {}) is None


def test_rows_from_keys_the_parsed_rows() -> None:
    """输入：反解出来的改动单。输出：无。预期行为：抬头三格与规则原样搬回，不重算别的。"""
    spec = _make_spec()
    loc = locales.load("zh")
    change = parse.parse_change(render.build(spec, loc), parse.labels_of(loc))
    rows = judges.rows_from(change)
    assert [(row.rule, row.file, row.change, row.size) for row in rows] == [
        (row.rule, row.file, row.change, row.size) for row in spec.rows
    ]


def test_recompute_matches_the_production_path() -> None:
    """输入：一条合格样本。输出：无。预期行为：按文本重算与产物里的答案零差异。"""
    sample, _ = _sample()
    loc = locales.load("zh")
    change = parse.parse_change(sample["state"], parse.labels_of(loc))
    assert judges.compare(sample, judges.recompute(change, loc)) == []


def test_compare_reports_every_kind_of_mismatch() -> None:
    """输入：四种坏样本。输出：无。

    预期行为：未知题族、多出来的题、答案不对、整族缺席都各记一笔——尤其「缺席」那一趟
    必须遍历**期望表**，遍历产物时整族少掉会一声不响。
    """
    sample, _ = _sample()
    loc = locales.load("zh")
    change = parse.parse_change(sample["state"], parse.labels_of(loc))
    expected = judges.recompute(change, loc)

    unknown = dict(sample["questions"])
    unknown["mystery"] = unknown["flag"]
    assert any(
        "未知题族" in item
        for item in judges.compare({**sample, "questions": unknown}, expected)
    )

    extra = dict(sample["questions"])
    extra["flag_9"] = extra["flag"]
    assert any(
        "多出来了" in item
        for item in judges.compare({**sample, "questions": extra}, expected)
    )

    answers = dict(sample["answers"])
    answers["flag"] = {"type": "noul", "noul": not expected["flag"][0]}
    assert any(
        "按文本重算应为" in item
        for item in judges.compare({**sample, "answers": answers}, expected)
    )

    trimmed = {qid: item for qid, item in sample["questions"].items() if qid != "top"}
    assert any(
        "少了" in item for item in judges.compare({**sample, "questions": trimmed}, expected)
    )


def test_roundtrip_diff_catches_each_cell() -> None:
    """输入：逐格改坏的改动单。输出：无。

    预期行为：改动号 / 处数 / 每一处的抬头三格 / 每一处的正文各改坏一次都能被抓到。
    """
    spec = _make_spec()
    loc = locales.load("zh")
    good = parse.parse_change(render.build(spec, loc), parse.labels_of(loc))
    assert judges.roundtrip_diff(spec, good, loc) == []

    assert judges.roundtrip_diff(
        spec, parse.ParsedChange("R-00000", good.rows), loc
    )
    assert judges.roundtrip_diff(
        spec, parse.ParsedChange(good.change_id, good.rows[:3]), loc
    )

    first = good.rows[0]
    wrong_cell = parse.ParsedRow("doc", first.change, first.size, first.body)
    assert judges.roundtrip_diff(
        spec, parse.ParsedChange(good.change_id, (wrong_cell, *good.rows[1:])), loc
    )
    wrong_body = parse.ParsedRow(first.file, first.change, first.size, "完全另一段正文")
    assert judges.roundtrip_diff(
        spec, parse.ParsedChange(good.change_id, (wrong_body, *good.rows[1:])), loc
    )


def test_roundtrip_judge_skips_without_a_spec() -> None:
    """输入：没有真值规格。输出：无。预期行为：跳过而不是误判。"""
    sample, _ = _sample()
    verdict = judges.RoundtripJudge().judge(sample, None)
    assert verdict.ok and verdict.raw == {"skipped": True}


def test_roundtrip_judge_passes_and_catches_a_wrecked_state() -> None:
    """输入：一条自洽样本 / 一条被改坏的文本。输出：无。预期行为：一个过、一个不过。"""
    sample, spec = _sample()
    judge = judges.RoundtripJudge()
    assert judge.judge(sample, spec).ok
    wrecked = {**sample, "state": sample["state"].replace("源码", "文档")}
    assert not judge.judge(wrecked, spec).ok
    verdict = judge.judge({**sample, "state": "什么都没有"}, spec)
    assert not verdict.ok and "反解失败" in verdict.reason


def test_label_judge_reports_a_wrong_spec_type() -> None:
    """输入：给判别器一个不是规格的东西。输出：无。预期行为：报「类型不对」而不是崩。"""
    sample, _ = _sample()
    verdict = judges.LabelJudge().judge(sample, "not a spec")
    assert not verdict.ok and "类型不对" in verdict.reason


def test_label_judge_checks_ids_and_answers() -> None:
    """输入：一条自洽样本 / 一条改过答案的样本 / 一条改过题号的样本。输出：无。"""
    sample, spec = _sample()
    judge = judges.LabelJudge()
    assert judge.judge(sample, spec).ok

    answers = dict(sample["answers"])
    answers["flag"] = {
        "type": "noul",
        "noul": not judges.answer_value(sample["questions"]["flag"], sample["answers"]["flag"]),
    }
    verdict = judge.judge({**sample, "answers": answers}, spec)
    assert not verdict.ok and "按规格应为" in verdict.reason

    questions_map = dict(sample["questions"])
    questions_map["flag"] = questions_map.pop("flag_2")
    verdict = judge.judge({**sample, "questions": questions_map}, spec)
    assert not verdict.ok and "题号与规格不符" in verdict.reason
