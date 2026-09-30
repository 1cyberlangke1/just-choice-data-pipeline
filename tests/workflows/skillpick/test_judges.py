"""判别器的测试：反解 → 重算 → 逐格比对，以及每一条「对不上」都要点名。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random


from jc import assemble
from tests.workflows.skillpick.conftest import _make_spec
from workflows.skillpick import judges, locales, parse, questions, render, spec as sp


def _sample(**overrides: object) -> tuple[dict[str, JsonValue], sp.SkillSpec]:
    """输入：case 覆盖项。输出：(走生产路径组装出来的合格样本, 真值规格)。"""
    spec = _make_spec(**overrides)
    loc = locales.load("zh")
    items = questions.build(
        spec.case.tag, spec.case.grant, spec.inuse_tag, spec.rows, loc, spec.phrasing
    )
    sample = assemble.assemble_sample(
        "skillpick-train-0000000", render.build(spec, loc), items, random.Random(1)
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


def test_recompute_matches_the_production_path() -> None:
    """输入：一条合格样本。输出：无。预期行为：按文本重算与产物里的答案零差异。"""
    sample, _ = _sample()
    loc = locales.load("zh")
    request = parse.parse_request(sample["state"], parse.labels_of(loc))
    assert judges.compare(sample, judges.recompute(request, loc)) == []


def test_recompute_reads_the_table_not_the_entry_keys() -> None:
    """输入：一条把权限格改歪的选择单。输出：无。

    预期行为：重算只吃文本里读回来的三格，所以「渲染把权限格写成另一档」会直接变成
    「答案对不上」——这是判别器真的在独立重算的证据。
    """
    sample, _ = _sample()
    loc = locales.load("zh")
    labels = parse.labels_of(loc)
    request = parse.parse_request(sample["state"], labels)
    rows = list(request.rows)
    first = rows[0]
    rows[0] = parse.ParsedRow(first.tag, "admin", first.resource)
    broken = parse.ParsedRequest(request.request_id, request.tag, request.grant, request.inuse, tuple(rows))
    assert judges.compare(sample, judges.recompute(broken, loc)) != []


def test_compare_reports_every_kind_of_mismatch() -> None:
    """输入：四种坏样本。输出：无。

    预期行为：未知题族、多出来的题、答案不对、整族缺席都各记一笔——尤其「缺席」那一趟
    必须遍历**期望表**，遍历产物时整族少掉会一声不响。
    """
    sample, _ = _sample()
    loc = locales.load("zh")
    request = parse.parse_request(sample["state"], parse.labels_of(loc))
    expected = judges.recompute(request, loc)

    unknown = dict(sample["questions"])
    unknown["mystery"] = unknown["fits"]
    assert any(
        "未知题族" in item
        for item in judges.compare({**sample, "questions": unknown}, expected)
    )

    extra = dict(sample["questions"])
    extra["pick_9"] = extra["pick"]
    assert any(
        "多出来了" in item
        for item in judges.compare({**sample, "questions": extra}, expected)
    )

    answers = dict(sample["answers"])
    answers["fits"] = {"type": "noul", "noul": not expected["fits"][0]}
    assert any(
        "按文本重算应为" in item
        for item in judges.compare({**sample, "answers": answers}, expected)
    )

    trimmed = {qid: item for qid, item in sample["questions"].items() if qid != "conflict"}
    assert any(
        "少了" in item for item in judges.compare({**sample, "questions": trimmed}, expected)
    )


def test_roundtrip_diff_catches_each_cell() -> None:
    """输入：逐格改坏的选择单。输出：无。

    预期行为：请求号 / 标签 / 授权 / 已占用 / 每一行的三格 / 行数 各改坏一次都能被抓到。
    """
    spec = _make_spec()
    loc = locales.load("zh")
    good = parse.parse_request(render.build(spec, loc), parse.labels_of(loc))
    assert judges.roundtrip_diff(spec, good, loc) == []

    assert judges.roundtrip_diff(spec, parse.ParsedRequest("Q-00000", good.tag, good.grant, good.inuse, good.rows), loc)
    assert judges.roundtrip_diff(spec, parse.ParsedRequest(good.request_id, "edit_code", good.grant, good.inuse, good.rows), loc)
    assert judges.roundtrip_diff(spec, parse.ParsedRequest(good.request_id, good.tag, "write", good.inuse, good.rows), loc)
    assert judges.roundtrip_diff(spec, parse.ParsedRequest(good.request_id, good.tag, good.grant, "edit_code", good.rows), loc)
    assert judges.roundtrip_diff(spec, parse.ParsedRequest(good.request_id, good.tag, good.grant, good.inuse, good.rows[:3]), loc)

    first = good.rows[0]
    wrong_cell = parse.ParsedRow(first.tag, "admin", first.resource)
    assert judges.roundtrip_diff(
        spec, parse.ParsedRequest(good.request_id, good.tag, good.grant, good.inuse, (wrong_cell, *good.rows[1:])), loc
    )


def test_label_judge_passes_a_good_sample() -> None:
    """输入：一条合格样本。输出：无。预期行为：题号与答案与规格逐题一致。"""
    sample, spec = _sample()
    verdict = judges.LabelJudge(locale="zh").judge(sample, spec)
    assert verdict.ok is True


def test_label_judge_flags_a_wrong_answer_and_a_wrong_spec() -> None:
    """输入：答案被改坏 / 规格类型不对。输出：无。预期行为：都判不通过并说清原因。"""
    sample, spec = _sample()
    answers = dict(sample["answers"])
    answers["fits"] = {"type": "noul", "noul": not sample["answers"]["fits"]["noul"]}
    verdict = judges.LabelJudge(locale="zh").judge({**sample, "answers": answers}, spec)
    assert verdict.ok is False and "fits" in verdict.reason

    verdict = judges.LabelJudge(locale="zh").judge(sample, object())
    assert verdict.ok is False and "规格类型不对" in verdict.reason


def test_label_judge_flags_a_missing_question() -> None:
    """输入：把某一族整族删掉。输出：无。预期行为：题号对不上，判不通过。"""
    sample, spec = _sample()
    trimmed = {qid: item for qid, item in sample["questions"].items() if qid != "scope"}
    verdict = judges.LabelJudge(locale="zh").judge({**sample, "questions": trimmed}, spec)
    assert verdict.ok is False and "题号与规格不符" in verdict.reason
