"""判别器的测试：反解 → 重算 → 逐格比对，以及每一条「对不上」都要点名。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random


from jc import assemble
from tests.workflows.elementpick.conftest import _make_spec
from workflows.elementpick import judges, locales, parse, questions, render, spec as sp


def _sample(**overrides: object) -> tuple[dict[str, JsonValue], sp.ElementSpec]:
    """输入：case 覆盖项。输出：(走生产路径组装出来的合格样本, 真值规格)。"""
    spec = _make_spec(**overrides)
    loc = locales.load("zh")
    items = questions.build(spec.rows, spec.case.label, loc, spec.phrasing)
    sample = assemble.assemble_sample(
        "elementpick-train-0000000", render.build(spec, loc), items, random.Random(1)
    )
    return sample, spec


def test_answer_value_handles_every_type() -> None:
    """输入：三种题型的若干答案。输出：无。

    预期行为：choice 给选项名（编号越界 / 类型不对给 None），noul 给布尔，
    score 给下标（越界同样给 None）。
    """
    choice = {"type": "choice", "criteria": {"a": "甲", "b": "乙"}}
    assert judges.answer_value(choice, {"choice": 1}) == "b"
    assert judges.answer_value(choice, {"choice": 2}) is None
    assert judges.answer_value(choice, {"choice": True}) is None
    assert judges.answer_value(choice, {}) is None
    noul = {"type": "noul", "criteria": {"true": "是", "false": "否"}}
    assert judges.answer_value(noul, {"noul": True}) is True
    assert judges.answer_value(noul, {"noul": 1}) is None
    score = {"type": "score", "criteria": ["甲", "乙", "丙"]}
    assert judges.answer_value(score, {"score": 2}) == 2
    assert judges.answer_value(score, {"score": 3}) is None
    assert judges.answer_value(score, {"score": True}) is None
    assert judges.answer_value(score, {}) is None


def test_recompute_matches_the_production_path() -> None:
    """输入：一条合格样本。输出：无。预期行为：按文本重算与产物里的答案零差异。"""
    sample, _ = _sample(extras=2, shadow=True)
    loc = locales.load("zh")
    screen = parse.parse_screen(sample["state"], parse.labels_of(loc))
    assert judges.compare(sample, judges.recompute(screen, loc)) == []


def test_recompute_reads_the_table_not_the_case() -> None:
    """输入：一条把可见性格改歪的清单。输出：无。

    预期行为：重算只吃文本里读回来的四格，所以「渲染把某一格写成另一档」会直接变成
    「答案对不上」——这是判别器真的在独立重算的证据。
    """
    sample, _ = _sample()
    loc = locales.load("zh")
    screen = parse.parse_screen(sample["state"], parse.labels_of(loc))
    rows = list(screen.rows)
    first = rows[0]
    rows[0] = parse.ParsedRow(first.role, "hidden", first.clickable, first.text)
    broken = parse.ParsedScreen(screen.screen_id, screen.action, tuple(rows))
    assert judges.compare(sample, judges.recompute(broken, loc)) != []


def test_compare_reports_every_kind_of_mismatch() -> None:
    """输入：四种坏样本。输出：无。

    预期行为：未知题族、多出来的题、答案不对、整族缺席都各记一笔——尤其「缺席」那一趟
    必须遍历**期望表**，遍历产物时整族少掉会一声不响。
    """
    sample, _ = _sample()
    loc = locales.load("zh")
    screen = parse.parse_screen(sample["state"], parse.labels_of(loc))
    expected = judges.recompute(screen, loc)

    unknown = dict(sample["questions"])
    unknown["mystery"] = unknown["found"]
    assert any(
        "未知题族" in item
        for item in judges.compare({**sample, "questions": unknown}, expected)
    )

    extra = dict(sample["questions"])
    extra["target_9"] = extra["target"]
    assert any(
        "多出来了" in item
        for item in judges.compare({**sample, "questions": extra}, expected)
    )

    answers = dict(sample["answers"])
    answers["found"] = {"type": "noul", "noul": not expected["found"][0]}
    assert any(
        "按文本重算应为" in item
        for item in judges.compare({**sample, "answers": answers}, expected)
    )

    trimmed = {qid: item for qid, item in sample["questions"].items() if qid != "decoy"}
    assert any(
        "少了" in item for item in judges.compare({**sample, "questions": trimmed}, expected)
    )


def test_roundtrip_diff_catches_each_cell() -> None:
    """输入：逐格改坏的清单。输出：无。

    预期行为：屏号 / 动作 / 每一行的四格 / 行数 各改坏一次都能被抓到。
    """
    spec = _make_spec(extras=1, shadow=True)
    loc = locales.load("zh")
    good = parse.parse_screen(render.build(spec, loc), parse.labels_of(loc))
    assert judges.roundtrip_diff(spec, good, loc) == []

    assert judges.roundtrip_diff(
        spec, parse.ParsedScreen("S-00000", good.action, good.rows), loc
    )
    assert judges.roundtrip_diff(
        spec, parse.ParsedScreen(good.screen_id, "cancel", good.rows), loc
    )
    assert judges.roundtrip_diff(
        spec, parse.ParsedScreen(good.screen_id, good.action, good.rows[:3]), loc
    )

    first = good.rows[0]
    wrong_cell = parse.ParsedRow(first.role, first.visibility, "no", first.text)
    assert judges.roundtrip_diff(
        spec,
        parse.ParsedScreen(good.screen_id, good.action, (wrong_cell, *good.rows[1:])),
        loc,
    )


def test_roundtrip_judge_skips_without_a_spec() -> None:
    """输入：没有真值规格。输出：无。

    预期行为：跳过而不是判错——「没判成」与「不通过」是两件事。
    """
    sample, _ = _sample()
    verdict = judges.RoundtripJudge(locale="zh").judge(sample, None)
    assert verdict.ok is True and verdict.raw == {"skipped": True}


def test_roundtrip_judge_flags_a_broken_state() -> None:
    """输入：state 被改坏 / 规格类型不对。输出：无。

    预期行为：state 改坏判不通过；规格类型不对走「没有 spec」那条路（跳过），
    与 `LabelJudge` 的严格检查相反——结构往返没有规格也能只看文本，所以跳过是对的。
    """
    sample, spec = _sample()
    broken = {**sample, "state": "今天天气不错。"}
    verdict = judges.RoundtripJudge(locale="zh").judge(broken, spec)
    assert verdict.ok is False and "反解失败" in verdict.reason
    verdict = judges.RoundtripJudge(locale="zh").judge(sample, "not-a-spec")
    assert verdict.ok is True and verdict.raw == {"skipped": True}


def test_label_judge_passes_a_good_sample() -> None:
    """输入：一条合格样本。输出：无。预期行为：题号与答案与规格逐题一致。"""
    sample, spec = _sample(shape="miss")
    verdict = judges.LabelJudge(locale="zh").judge(sample, spec)
    assert verdict.ok is True
