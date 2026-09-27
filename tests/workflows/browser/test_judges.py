"""判别器的单元测试：往返、重算、比对，以及两条路各自要能抓到的错。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from jc import assemble
from tests.workflows.browser.conftest import _make_case, _make_spec
from workflows.browser import judges, locales, parse, questions, render, spec as sp


def _sample(**overrides: Any) -> tuple[dict[str, Any], sp.ScreenSpec]:
    """输入：case 覆盖项。输出：(样本, 真值规格)。预期行为：选项打乱用固定种子。"""
    spec = _make_spec(**overrides)
    loc = locales.load("zh")
    labeled = questions.build(spec.rows, spec.task, spec.object, loc, spec.phrasing)
    sample = assemble.assemble_sample(
        "browser-train-0", render.build(spec, loc), labeled, random.Random(11)
    )
    return sample, spec


def test_answer_value_reads_both_question_types() -> None:
    """输入：四种答案形态。输出：无。预期行为：choice 给选项名、noul 给布尔、越界给 None。"""
    choice = {"type": "choice", "criteria": {"a": "甲", "b": "乙"}}
    assert judges.answer_value(choice, {"choice": 1}) == "b"
    assert judges.answer_value(choice, {"choice": 5}) is None
    assert judges.answer_value(choice, {"choice": True}) is None
    assert judges.answer_value(choice, {}) is None
    assert judges.answer_value({"type": "noul"}, {"noul": False}) is False
    assert judges.answer_value({"type": "noul"}, {"noul": "no"}) is None
    assert judges.answer_value({"type": "score"}, {"score": 2}) == 2


def test_recompute_matches_the_builders() -> None:
    """输入：全部 480 个 case 的反解结果。输出：无。

    预期行为：重算出来的期望答案与真值逐题一致——两条路共用同一套出题逻辑。
    """
    loc = locales.load("zh")
    labels = parse.labels_of(loc)
    for case in sp.build_cases(sp.ScreenConfig()):
        spec = sp.sample_spec(case, "train", 0, 0, None)
        screen = parse.parse_screen(render.build(spec, loc), labels)
        expected = judges.recompute(screen, loc)
        assert expected["done"][0] is spec.done
        if spec.target_position is None:
            assert set(expected) == {"done"}
        else:
            assert expected["target"][0] == f"p{spec.target_position}"
            assert expected["action"][0] == sp.action_of(case.task)
            assert expected["risk"][0] is sp.is_sensitive(spec.object)


def test_compare_accepts_a_faithful_sample() -> None:
    """输入：一条自洽样本。输出：无。预期行为：差异为空。"""
    sample, _ = _sample()
    screen = parse.parse_screen(sample["state"], parse.labels_of(locales.load("zh")))
    assert judges.compare(sample, judges.recompute(screen, locales.load("zh"))) == []


def test_compare_reports_every_kind_of_mismatch() -> None:
    """输入：四种坏样本。输出：无。

    预期行为：未知题族、多出来的题、答案不对、整族缺席都各记一笔——尤其「缺席」那一趟
    必须遍历**期望表**，遍历产物时整族少掉会一声不响。
    """
    sample, spec = _sample()
    screen = parse.parse_screen(sample["state"], parse.labels_of(locales.load("zh")))
    expected = judges.recompute(screen, locales.load("zh"))

    unknown = dict(sample["questions"])
    unknown["mystery"] = unknown["done"]
    assert any(
        "未知题族" in item
        for item in judges.compare({**sample, "questions": unknown}, expected)
    )

    extra = dict(sample["questions"])
    extra["done_2"] = extra["done"]
    assert any(
        "多出来了" in item
        for item in judges.compare({**sample, "questions": extra}, expected)
    )

    answers = dict(sample["answers"])
    answers["done"] = {"type": "noul", "noul": not spec.done}
    assert any(
        "按文本重算应为" in item
        for item in judges.compare({**sample, "answers": answers}, expected)
    )

    trimmed = {qid: item for qid, item in sample["questions"].items() if qid != "risk"}
    assert any(
        "少了" in item
        for item in judges.compare({**sample, "questions": trimmed}, expected)
    )


def test_roundtrip_diff_catches_each_cell() -> None:
    """输入：逐格改坏的界面。输出：无。

    预期行为：窗口号 / 任务意图 / 对象 / 行数 / 每一行的三格各改坏一次都能被抓到。
    """
    spec = _make_spec()
    loc = locales.load("zh")
    good = parse.parse_screen(render.build(spec, loc), parse.labels_of(loc))
    assert judges.roundtrip_diff(spec, good, loc) == []

    def diff(**changes: Any) -> list[str]:
        fields = {**vars(good), **changes}
        return judges.roundtrip_diff(spec, parse.ParsedScreen(**fields), loc)

    assert diff(screen_id="W-00000")
    assert diff(task="open")
    assert diff(object="draft")
    assert judges.roundtrip_diff(
        spec, parse.ParsedScreen(good.screen_id, good.task, good.object, good.rows[:3]), loc
    )
    bad_row = parse.ParsedElement("button", "refresh", "ready")
    assert judges.roundtrip_diff(
        spec,
        parse.ParsedScreen(good.screen_id, good.task, good.object, (bad_row, *good.rows[1:])),
        loc,
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
    wrecked = {**sample, "state": sample["state"].replace("可用", "已是当前")}
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
    answers["done"] = {"type": "noul", "noul": not spec.done}
    verdict = judge.judge({**sample, "answers": answers}, spec)
    assert not verdict.ok and "按规格应为" in verdict.reason

    questions_map = dict(sample["questions"])
    questions_map["done"] = questions_map.pop("target")
    verdict = judge.judge({**sample, "questions": questions_map}, spec)
    assert not verdict.ok and "题号与规格不符" in verdict.reason


def test_rows_from_keys_the_parsed_rows() -> None:
    """输入：反解出来的界面。输出：无。预期行为：三格原样搬进 `Element`，不重算判断。"""
    case = _make_case()
    rows = sp.rows_of(case)
    parsed = parse.ParsedScreen(
        "W-1", "fill", "note", tuple(parse.ParsedElement(r.role, r.object, r.state) for r in rows)
    )
    assert judges.rows_from(parsed) == rows
