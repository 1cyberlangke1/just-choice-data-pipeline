"""判别器的测试：答案取值、重算与产线一致、逐格往返、以及每种坏法都要被抓到。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import dataclasses
import random


import pytest

from jc import assemble
from tests.workflows.bench.conftest import _make_spec
from workflows.bench import judges, locales, parse, questions, render
from workflows.bench import spec as sp

LOC = locales.load("zh")


def _sample(**overrides: object) -> tuple[dict[str, JsonValue], sp.BenchSpec]:
    """输入：case 覆盖项。输出：(走产线组装出来的样本, 真值规格)。"""
    spec = _make_spec(**overrides)
    items = questions.build(spec.rows, spec, LOC, spec.phrasing)
    row = assemble.assemble_sample(
        "bench-train-0000000", render.build(spec, LOC), items, random.Random(1)
    )
    return row, spec


def test_recompute_matches_the_production_path() -> None:
    """输入：一份规格的文本反解结果。输出：无。

    预期行为：从文本重算出来的答案表与产线出的题目逐族逐条一致——判别器与产线走的是
    同一套规则，所以「渲染写歪」才会变成「答案对不上」。
    """
    spec = _make_spec(table="k2s3", layout="l0")
    record = parse.parse_record(render.build(spec, LOC), parse.labels_of(LOC))
    expected = judges.recompute(record, LOC)
    produced: dict[str, list[JsonValue]] = {}
    for item in questions.build(spec.rows, spec, LOC, spec.phrasing):
        produced.setdefault(item.family, []).append(item.answer)
    assert expected == produced
    assert set(expected) == {"hit", "count", "pass", "best", "gap"}


def test_recompute_reads_the_cells_not_the_table_key() -> None:
    """输入：把文本里某一格改掉。输出：无。

    预期行为：重算结果跟着**文本**变——只吃行里的四格与抬头那一格，所以「渲染把某一格
    写歪」会被抓成「答案对不上」，而不是被反查表悄悄修正。

    用的成绩表 `k4s0` 里第 1 列本来就是最好的（没有 `gap`），所以把一格从对改成错不会
    造出「差 1 道」这种表外的差距——那是另一种硬伤，另有断言管。
    """
    spec = _make_spec(table="k4s0", layout="l0")
    text = render.build(spec, LOC)
    labels = parse.labels_of(LOC)
    before = judges.recompute(parse.parse_record(text, labels), LOC)
    assert before["hit"][0] is True
    name = str(LOC["cand"]["m0"])
    broken = text.replace(f"{name} A", f"{name} B")
    assert broken != text
    after = judges.recompute(parse.parse_record(broken, labels), LOC)
    assert after["hit"][0] is False
    assert after["count"] == [before["count"][0] - 1]
    assert "gap" not in before and "gap" not in after


def test_recompute_reports_a_gap_outside_the_band_table() -> None:
    """输入：把一格改坏成「差 1 道」。输出：无。

    预期行为：报人话——表外的差距说明成绩表被改坏了，不能静默当成「没有差距」（那样
    审计就看不见这条坏数据了）。
    """
    spec = _make_spec(table="k3s0", layout="l0")  # 第 1 列落后 2 道
    text = render.build(spec, LOC)
    name = str(LOC["cand"]["m0"])
    broken = text.replace(f"{name} A", f"{name} B")  # 再少对一道 → 差 3 道
    with pytest.raises(ValueError, match="落后幅度"):
        judges.recompute(parse.parse_record(broken, parse.labels_of(LOC)), LOC)


def test_compare_reports_every_kind_of_mismatch() -> None:
    """输入：一条好样本 + 四种改坏法。输出：无。

    预期行为：未知题族、多出来的题、答案不对、少了题各记一笔——不静默取第一个。
    """
    sample, _spec = _sample()
    record = parse.parse_record(sample["state"], parse.labels_of(LOC))
    expected = judges.recompute(record, LOC)
    assert judges.compare(sample, expected) == []

    unknown = dict(sample["questions"])
    unknown["mystery"] = unknown["hit"]
    assert any(
        "未知题族" in item for item in judges.compare({**sample, "questions": unknown}, expected)
    )

    extra = dict(sample["questions"])
    extra["hit_9"] = extra["hit"]
    assert any(
        "多出来了" in item for item in judges.compare({**sample, "questions": extra}, expected)
    )

    answers = dict(sample["answers"])
    answers["hit"] = {"type": "noul", "noul": not expected["hit"][0]}
    assert any(
        "按文本重算应为" in item
        for item in judges.compare({**sample, "answers": answers}, expected)
    )

    trimmed = {qid: item for qid, item in sample["questions"].items() if qid != "best"}
    assert any(
        "少了" in item for item in judges.compare({**sample, "questions": trimmed}, expected)
    )


def test_roundtrip_diff_catches_each_cell() -> None:
    """输入：逐格改坏的记录。输出：无。

    预期行为：集号 / 门槛 / 每一行的参考答案 / 每一行的三格 / 行数 各改坏一次都能被抓到。
    """
    spec = _make_spec(table="k3s0", layout="l0")
    good = parse.parse_record(render.build(spec, LOC), parse.labels_of(LOC))
    assert judges.roundtrip_diff(spec, good, LOC) == []

    assert judges.roundtrip_diff(spec, dataclasses.replace(good, request_id="E-00000"), LOC)
    assert judges.roundtrip_diff(spec, dataclasses.replace(good, bar=4), LOC)
    assert judges.roundtrip_diff(spec, dataclasses.replace(good, rows=good.rows[:3]), LOC)

    first = good.rows[0]
    other_ref = next(letter for letter in sp.LETTERS if letter != first.ref)
    assert judges.roundtrip_diff(
        spec, dataclasses.replace(good, rows=(dataclasses.replace(first, ref=other_ref), *good.rows[1:])), LOC
    )
    for index, cell in enumerate(first.cells):
        flipped = sp.shift_letter(cell.letter, 1)
        broken_cells = list(first.cells)
        broken_cells[index] = parse.ParsedCell(cell.cand, flipped)
        diff = judges.roundtrip_diff(
            spec,
            dataclasses.replace(
                good, rows=(dataclasses.replace(first, cells=tuple(broken_cells)), *good.rows[1:])
            ),
            LOC,
        )
        assert diff, cell


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
    sample, _spec = _sample()
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
    answers["count"] = {"type": "score", "score": 9}
    verdict = judges.LabelJudge(locale="zh").judge({**sample, "answers": answers}, spec)
    assert verdict.ok is False and "count" in verdict.reason

    verdict = judges.LabelJudge(locale="zh").judge(sample, object())
    assert verdict.ok is False and "规格类型不对" in verdict.reason


def test_judges_work_with_and_without_the_gap_family() -> None:
    """输入：第 1 列落后的记录 / 第 1 列自己就是最好的记录。输出：无。

    预期行为：两个判别器在两种记录上都照常通过——第 1 列领先时本来就不出 `gap`，
    这不是缺题。
    """
    behind, behind_spec = _sample(table="k3s0")
    assert "gap" in behind["questions"]
    assert judges.RoundtripJudge(locale="zh").judge(behind, behind_spec).ok is True
    assert judges.LabelJudge(locale="zh").judge(behind, behind_spec).ok is True

    leading, leading_spec = _sample(table="k4s0")
    assert "gap" not in leading["questions"]
    assert judges.RoundtripJudge(locale="zh").judge(leading, leading_spec).ok is True
    assert judges.LabelJudge(locale="zh").judge(leading, leading_spec).ok is True
