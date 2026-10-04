"""判别器的测试：答案取值、重算与产线一致、逐格往返、以及每种坏法都要被抓到。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import dataclasses
import random


from jc import assemble
from tests.workflows.sim.conftest import _make_spec
from workflows.sim import judges, locales, parse, questions, render, spec as sp

LOC = locales.load("zh")


def _sample(**overrides: object) -> tuple[dict[str, JsonValue], sp.SimSpec]:
    """输入：case 覆盖项。输出：(走产线组装出来的样本, 真值规格)。"""
    spec = _make_spec(**overrides)
    items = questions.build(spec.rows, spec, LOC, spec.phrasing)
    row = assemble.assemble_sample(
        "sim-train-0000000", render.build(spec, LOC), items, random.Random(1)
    )
    return row, spec


def test_recompute_matches_the_production_path() -> None:
    """输入：一份规格的文本反解结果。输出：无。

    预期行为：从文本重算出来的答案表与产线出的题目逐族逐条一致——判别器与产线走的是
    同一套规则，所以「渲染写歪」才会变成「答案对不上」。
    """
    spec = _make_spec(board="b6", layout="l3")
    board = parse.parse_board(render.build(spec, LOC), parse.labels_of(LOC))
    expected = judges.recompute(board, LOC)
    produced: dict[str, list[JsonValue]] = {}
    for item in questions.build(spec.rows, spec, LOC, spec.phrasing):
        produced.setdefault(item.family, []).append(item.answer)
    assert expected == produced
    assert set(expected) == {"can_move", "finished", "gain", "best", "cost"}


def test_recompute_reads_the_cells_not_the_action_keys() -> None:
    """输入：把文本里待机那一行的耗格改大。输出：无。

    预期行为：重算结果跟着**文本**变——只吃行里的三格与抬头的资源格，所以「渲染把某一格
    写歪」会被抓成「答案对不上」，而不是被反查表悄悄修正。
    """
    spec = _make_spec(board="b0", layout="l0")  # 资源 0，只有待机（耗 0）能用
    text = render.build(spec, LOC)
    labels = parse.labels_of(LOC)
    before = judges.recompute(parse.parse_board(text, labels), LOC)
    broken = text.replace(str(LOC["cost"]["c0"]), str(LOC["cost"]["c1"]))
    assert broken != text
    after = judges.recompute(parse.parse_board(broken, labels), LOC)
    assert after["can_move"] != before["can_move"]
    assert set(after["can_move"]) == {False}
    assert "best" not in after and "cost" not in after


def test_compare_reports_every_kind_of_mismatch() -> None:
    """输入：一条好样本 + 四种改坏法。输出：无。

    预期行为：未知题族、多出来的题、答案不对、少了题各记一笔——不静默取第一个。
    """
    sample, _spec = _sample()
    board = parse.parse_board(sample["state"], parse.labels_of(LOC))
    expected = judges.recompute(board, LOC)
    assert judges.compare(sample, expected) == []

    unknown = dict(sample["questions"])
    unknown["mystery"] = unknown["can_move"]
    assert any(
        "未知题族" in item for item in judges.compare({**sample, "questions": unknown}, expected)
    )

    extra = dict(sample["questions"])
    extra["can_move_9"] = extra["can_move"]
    assert any(
        "多出来了" in item for item in judges.compare({**sample, "questions": extra}, expected)
    )

    answers = dict(sample["answers"])
    answers["can_move"] = {"type": "noul", "noul": not expected["can_move"][0]}
    assert any(
        "按文本重算应为" in item
        for item in judges.compare({**sample, "answers": answers}, expected)
    )

    trimmed = {qid: item for qid, item in sample["questions"].items() if qid != "best"}
    assert any(
        "少了" in item for item in judges.compare({**sample, "questions": trimmed}, expected)
    )


def test_roundtrip_diff_catches_each_cell() -> None:
    """输入：逐格改坏的局面。输出：无。

    预期行为：单号 / 抬头三格 / 每一行的四格 / 行数 各改坏一次都能被抓到。
    """
    spec = _make_spec(board="b6", layout="l0")
    good = parse.parse_board(render.build(spec, LOC), parse.labels_of(LOC))
    assert judges.roundtrip_diff(spec, good, LOC) == []

    assert judges.roundtrip_diff(spec, dataclasses.replace(good, request_id="S-00000"), LOC)
    other_turn = 5 if spec.turn != 5 else 3
    assert judges.roundtrip_diff(spec, dataclasses.replace(good, turn=other_turn), LOC)
    assert judges.roundtrip_diff(
        spec, dataclasses.replace(good, resources=spec.resources + 1), LOC
    )
    assert judges.roundtrip_diff(spec, dataclasses.replace(good, haul=spec.haul + 1), LOC)
    assert judges.roundtrip_diff(spec, dataclasses.replace(good, rows=good.rows[:3]), LOC)

    first = good.rows[0]
    for wrong in (
        parse.ParsedRow("idle", first.cost, first.gain, first.cooldown),
        parse.ParsedRow(first.action, "c3", first.gain, first.cooldown),
        parse.ParsedRow(first.action, first.cost, "g5", first.cooldown),
        parse.ParsedRow(first.action, first.cost, first.gain, "k2"),
    ):
        diff = judges.roundtrip_diff(
            spec, dataclasses.replace(good, rows=(wrong, *good.rows[1:])), LOC
        )
        assert diff, wrong


def test_label_judge_flags_a_wrong_answer_and_a_wrong_spec() -> None:
    """输入：答案被改坏 / 规格类型不对。输出：无。预期行为：都判不通过并说清原因。"""
    sample, spec = _sample()
    answers = dict(sample["answers"])
    answers["finished"] = {"type": "noul", "noul": not spec.done}
    verdict = judges.LabelJudge(locale="zh").judge({**sample, "answers": answers}, spec)
    assert verdict.ok is False and "finished" in verdict.reason

    verdict = judges.LabelJudge(locale="zh").judge(sample, object())
    assert verdict.ok is False and "规格类型不对" in verdict.reason


def test_judges_work_on_a_finished_board_without_best() -> None:
    """输入：`b3`（回合到顶，结束了）的样本。输出：无。

    预期行为：两个判别器都照常通过——结束了就不出 `best` / `cost`，这不是缺题。
    """
    sample, spec = _sample(board="b3")
    assert spec.done is True
    assert "best" not in sample["questions"] and "cost" not in sample["questions"]
    assert judges.RoundtripJudge(locale="zh").judge(sample, spec).ok is True
    assert judges.LabelJudge(locale="zh").judge(sample, spec).ok is True
