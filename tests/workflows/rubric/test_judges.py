"""判别器的测试：答案取值、重算与产线一致、逐格往返、以及每种坏法都要被抓到。"""

from __future__ import annotations

import random
from typing import Any

from jc import assemble
from tests.workflows.rubric.conftest import _make_spec
from workflows.rubric import judges, locales, parse, questions, render, spec as sp

LOC = locales.load("zh")


def _sample(**overrides: Any) -> tuple[dict[str, Any], sp.RubricSpec]:
    """输入：case 覆盖项。输出：(走产线组装出来的样本, 真值规格)。"""
    spec = _make_spec(**overrides)
    items = questions.build(spec.rows, LOC, spec.phrasing)
    row = assemble.assemble_sample(
        "rubric-train-0000000", render.build(spec, LOC), items, random.Random(1)
    )
    return row, spec


def test_answer_value_handles_every_type() -> None:
    """输入：三种题型的答案 + 各种坏形态。输出：无。

    预期行为：choice 给选项名、score 给等级下标、noul 给布尔；编号越界、类型不对
    （布尔不是编号）、criteria 形状不对都给 None——那是对不上，不是崩。
    """
    choice = {"type": "choice", "criteria": {"a": "甲", "b": "乙"}}
    assert judges.answer_value(choice, {"choice": 1}) == "b"
    assert judges.answer_value(choice, {"choice": 2}) is None
    assert judges.answer_value(choice, {"choice": True}) is None
    assert judges.answer_value(choice, {"choice": "1"}) is None
    assert judges.answer_value({"type": "choice", "criteria": []}, {"choice": 0}) is None

    score = {"type": "score", "criteria": ["低", "中", "高"]}
    assert judges.answer_value(score, {"score": 2}) == 2
    assert judges.answer_value(score, {"score": 3}) is None
    assert judges.answer_value(score, {"score": False}) is None
    assert judges.answer_value({"type": "score", "criteria": {}}, {"score": 0}) is None

    noul = {"type": "noul", "criteria": {"true": "是", "false": "否"}}
    assert judges.answer_value(noul, {"noul": True}) is True
    assert judges.answer_value(noul, {"noul": "true"}) is None
    assert judges.answer_value({"type": "score"}, {}) is None
    assert judges.answer_value({"type": "bogus"}, {}) is None


def test_recompute_matches_the_production_path() -> None:
    """输入：一份规格的文本反解结果。输出：无。

    预期行为：从文本重算出来的答案表与产线出的题目逐族逐条一致——判别器与产线走的是
    同一套规则，所以「渲染写歪」才会变成「答案对不上」。
    """
    spec = _make_spec(sheet="s6", layout="l3", slack="roomy")
    sheet = parse.parse_sheet(render.build(spec, LOC), parse.labels_of(LOC))
    expected = judges.recompute(sheet, LOC)
    produced: dict[str, list[Any]] = {}
    for item in questions.build(spec.rows, LOC, spec.phrasing):
        produced.setdefault(item.family, []).append(item.answer)
    assert expected == produced
    assert set(expected) == {"met", "band", "gate", "top"}


def test_recompute_reads_the_table_not_the_criterion_keys() -> None:
    """输入：把文本里某一行的观测格改小。输出：无。

    预期行为：重算结果跟着**文本**变——只吃评分单里那四格，所以「渲染把某一格写歪」
    会被抓成「答案对不上」，而不是被反查表悄悄修正。
    """
    spec = _make_spec(sheet="s0", layout="l0")
    text = render.build(spec, LOC)
    labels = parse.labels_of(LOC)
    before = judges.recompute(parse.parse_sheet(text, labels), LOC)
    broken = text.replace(str(LOC["count"]["n1"]), str(LOC["count"]["n0"]))
    assert broken != text
    after = judges.recompute(parse.parse_sheet(broken, labels), LOC)
    assert after["met"] != before["met"]


def test_compare_reports_every_kind_of_mismatch() -> None:
    """输入：一条好样本 + 四种改坏法。输出：无。

    预期行为：未知题族、多出来的题、答案不对、少了题各记一笔——不静默取第一个。
    """
    sample, spec = _sample()
    sheet = parse.parse_sheet(sample["state"], parse.labels_of(LOC))
    expected = judges.recompute(sheet, LOC)
    assert judges.compare(sample, expected) == []

    unknown = dict(sample["questions"])
    unknown["mystery"] = unknown["met"]
    assert any(
        "未知题族" in item
        for item in judges.compare({**sample, "questions": unknown}, expected)
    )

    extra = dict(sample["questions"])
    extra["met_9"] = extra["met"]
    assert any(
        "多出来了" in item
        for item in judges.compare({**sample, "questions": extra}, expected)
    )

    answers = dict(sample["answers"])
    answers["met"] = {"type": "noul", "noul": not expected["met"][0]}
    assert any(
        "按文本重算应为" in item
        for item in judges.compare({**sample, "answers": answers}, expected)
    )

    trimmed = {qid: item for qid, item in sample["questions"].items() if qid != "gate"}
    assert any(
        "少了" in item for item in judges.compare({**sample, "questions": trimmed}, expected)
    )


def test_roundtrip_diff_catches_each_cell() -> None:
    """输入：逐格改坏的评分单。输出：无。

    预期行为：单号 / 每一行的四格 / 行数 各改坏一次都能被抓到。
    """
    spec = _make_spec(sheet="s0", layout="l0")
    good = parse.parse_sheet(render.build(spec, LOC), parse.labels_of(LOC))
    assert judges.roundtrip_diff(spec, good, LOC) == []

    assert judges.roundtrip_diff(spec, parse.ParsedSheet("G-00000", good.rows), LOC)
    assert judges.roundtrip_diff(spec, parse.ParsedSheet(good.request_id, good.rows[:3]), LOC)

    first = good.rows[0]
    for wrong in (
        parse.ParsedRow("source", first.weight, first.need, first.count),
        parse.ParsedRow(first.criterion, "w1", first.need, first.count),
        parse.ParsedRow(first.criterion, first.weight, "need3", first.count),
        parse.ParsedRow(first.criterion, first.weight, first.need, "n4"),
    ):
        diff = judges.roundtrip_diff(
            spec, parse.ParsedSheet(good.request_id, (wrong, *good.rows[1:])), LOC
        )
        assert diff, wrong


def test_roundtrip_judge_passes_a_good_sample() -> None:
    """输入：一条合格样本。输出：无。预期行为：结构往返通过。"""
    sample, spec = _sample()
    verdict = judges.RoundtripJudge(locale="zh").judge(sample, spec)
    assert verdict.ok is True
    assert verdict.raw == {"diff": []}


def test_roundtrip_judge_skips_without_a_spec() -> None:
    """输入：没有真值规格 / 规格类型不对。输出：无。

    预期行为：跳过而不是判错——「没判成」与「不通过」是两件事。
    """
    sample, _ = _sample()
    for spec in (None, "not-a-spec"):
        verdict = judges.RoundtripJudge(locale="zh").judge(sample, spec)
        assert verdict.ok is True and verdict.raw == {"skipped": True}


def test_roundtrip_judge_flags_a_broken_state() -> None:
    """输入：state 被改坏。输出：无。预期行为：判不通过并说清是反解失败。"""
    sample, spec = _sample()
    broken = {**sample, "state": "今天天气不错。"}
    verdict = judges.RoundtripJudge(locale="zh").judge(broken, spec)
    assert verdict.ok is False and "反解失败" in verdict.reason


def test_label_judge_passes_a_good_sample() -> None:
    """输入：一条合格样本。输出：无。预期行为：题号与答案与规格逐题一致。"""
    sample, spec = _sample()
    assert judges.LabelJudge(locale="zh").judge(sample, spec).ok is True


def test_label_judge_flags_a_wrong_answer_and_a_wrong_spec() -> None:
    """输入：答案被改坏 / 规格类型不对。输出：无。预期行为：都判不通过并说清原因。"""
    sample, spec = _sample()
    answers = dict(sample["answers"])
    answers["band"] = {"type": "score", "score": (spec.band == "b1") and 2 or 0}
    verdict = judges.LabelJudge(locale="zh").judge({**sample, "answers": answers}, spec)
    assert verdict.ok is False and "band" in verdict.reason

    verdict = judges.LabelJudge(locale="zh").judge(sample, object())
    assert verdict.ok is False and "规格类型不对" in verdict.reason


def test_label_judge_flags_a_missing_question() -> None:
    """输入：把某一族整族删掉。输出：无。预期行为：题号对不上，判不通过。"""
    sample, spec = _sample()
    trimmed = {qid: item for qid, item in sample["questions"].items() if qid != "top"}
    verdict = judges.LabelJudge(locale="zh").judge({**sample, "questions": trimmed}, spec)
    assert verdict.ok is False and "题号与规格不符" in verdict.reason


def test_judges_work_on_a_tied_sheet_without_top() -> None:
    """输入：`s2`（不出 `top`）的样本。输出：无。预期行为：两个判别器都照常通过。"""
    sample, spec = _sample(sheet="s2")
    assert spec.top_slot is None
    assert "top" not in sample["questions"]
    assert judges.RoundtripJudge(locale="zh").judge(sample, spec).ok is True
    assert judges.LabelJudge(locale="zh").judge(sample, spec).ok is True
