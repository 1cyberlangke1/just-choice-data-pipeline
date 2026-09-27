"""判别器：走得通、抓得住、坏输入不崩。"""

from __future__ import annotations

import copy
import random
from typing import Any

import pytest

from jc import assemble
from tests.workflows.autoformat.conftest import _make_spec
from workflows.autoformat import judges, locales, parse, questions, render, spec as sp

LANGS = ("zh", "en")


def _sample(spec: sp.DocSpec, locale: str = "zh", seed: int = 1) -> dict[str, Any]:
    """输入：规格、语言、种子。输出：契约形态的样本。预期行为：走组装器（编号只由它产生）。"""
    loc = locales.load(locale)
    labeled = questions.build(spec, loc, spec.phrasing, random.Random(0))
    return assemble.assemble_sample(
        "autoformat-train-0000000", render.render_state(spec, loc), labeled, random.Random(seed)
    )


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"shape": "list", "variant": "ordered"},
        {"shape": "code", "code_family": "container"},
        {"shape": "callout", "variant": "tip"},
        {"shape": "mixed", "variant": "warning", "code_family": "vcs"},
    ],
)
def test_roundtrip_judge_passes_on_real_samples(lang: str, overrides: dict) -> None:
    """输入：五种形状 × 两种语言。输出：无。

    预期行为：从文本反解后重算四个题族，逐题都对得上。
    """
    spec = _make_spec(**overrides)
    verdict = judges.RoundtripJudge(locale=lang).judge(_sample(spec, lang), spec)
    assert verdict.ok, verdict.reason


@pytest.mark.parametrize("lang", LANGS)
def test_label_judge_passes_on_real_samples(lang: str) -> None:
    """输入：两种语言。输出：无。预期行为：按规格重出一次题，题号与答案完全一致。"""
    spec = _make_spec(shape="mixed", variant="note", code_family="build")
    verdict = judges.LabelJudge(locale=lang).judge(_sample(spec, lang), spec)
    assert verdict.ok, verdict.reason


def test_roundtrip_judge_catches_a_flipped_noul() -> None:
    """输入：把一道 noul 的答案翻过来。输出：无。预期行为：判不通过并点名题号。"""
    spec = _make_spec(shape="list", variant="ordered")
    sample = _sample(spec)
    qid = next(qid for qid, q in sample["questions"].items() if q["type"] == "noul")
    sample["answers"][qid]["noul"] = not sample["answers"][qid]["noul"]
    verdict = judges.RoundtripJudge(locale="zh").judge(sample, spec)
    assert not verdict.ok
    assert qid in verdict.reason


def test_roundtrip_judge_catches_a_shifted_choice() -> None:
    """输入：把一道 choice 的编号挪一格。输出：无。预期行为：判不通过。"""
    spec = _make_spec(shape="mixed", variant="warning", code_family="build")
    sample = _sample(spec)
    qid = next(qid for qid, q in sample["questions"].items() if q["type"] == "choice")
    count = len(sample["questions"][qid]["criteria"])
    sample["answers"][qid]["choice"] = (sample["answers"][qid]["choice"] + 1) % count
    verdict = judges.RoundtripJudge(locale="zh").judge(sample, spec)
    assert not verdict.ok


def test_roundtrip_judge_catches_a_broken_state() -> None:
    """输入：把 state 里一行的行号前缀删掉。输出：无。预期行为：判不通过并说明反解失败。"""
    spec = _make_spec()
    sample = _sample(spec)
    sample["state"] = sample["state"].replace("L002| ", "第二行 ", 1)
    verdict = judges.RoundtripJudge(locale="zh").judge(sample, spec)
    assert not verdict.ok
    assert "反解失败" in verdict.reason


def test_label_judge_rejects_a_foreign_spec() -> None:
    """输入：把规格换成别的东西。输出：无。预期行为：判不通过并说明类型不对。"""
    spec = _make_spec()
    verdict = judges.LabelJudge(locale="zh").judge(_sample(spec), {"not": "a spec"})
    assert not verdict.ok
    assert "类型不对" in verdict.reason


def test_label_judge_catches_a_missing_question() -> None:
    """输入：从样本里删掉一道题。输出：无。预期行为：判不通过（题号对不上）。"""
    spec = _make_spec(shape="list", variant="ordered")
    sample = _sample(spec)
    qid = next(iter(sample["questions"]))
    sample["questions"].pop(qid)
    sample["answers"].pop(qid)
    verdict = judges.LabelJudge(locale="zh").judge(sample, spec)
    assert not verdict.ok
    assert "题号与规格不符" in verdict.reason


def test_compare_reports_extra_and_missing_questions() -> None:
    """输入：期望表里多一条、少一条。输出：无。预期行为：两种偏差各记一笔。"""
    spec = _make_spec(shape="callout", variant="note")
    sample = _sample(spec)
    expected = judges.recompute(
        parse.parse_state(sample["state"], locales.load("zh")), locales.load("zh")
    )
    assert judges.compare(sample, expected) == []
    fewer = {key: list(value) for key, value in expected.items()}
    fewer["block_type"] = fewer["block_type"][:-1]
    problems = judges.compare(sample, fewer)
    assert any("多出来了" in item for item in problems)
    more = {key: list(value) for key, value in expected.items()}
    more["join"] = [*more["join"], True]
    problems = judges.compare(sample, more)
    assert any("少了" in item for item in problems)


def test_compare_reports_an_unknown_family() -> None:
    """输入：样本里混进一个没见过的题族。输出：无。预期行为：记一笔未知题族。"""
    spec = _make_spec()
    sample = _sample(spec)
    sample["questions"]["mystery"] = {"type": "noul", "instructions": "?"}
    sample["answers"]["mystery"] = {"type": "noul", "noul": True}
    problems = judges.compare(sample, {"join": [], "block_type": [], "callout_kind": [], "list_ordered": []})
    assert any("未知题族" in item for item in problems)


def test_answer_value_guards_bad_shapes() -> None:
    """输入：越界编号、非布尔、非整数。输出：无。预期行为：一律给 None（对不上，不崩）。"""
    choice = {"type": "choice", "instructions": "x", "criteria": {"a": "甲", "b": "乙"}}
    assert judges.answer_value(choice, {"choice": 1}) == "b"
    assert judges.answer_value(choice, {"choice": 5}) is None
    assert judges.answer_value(choice, {"choice": True}) is None
    assert judges.answer_value(choice, {"choice": "1"}) is None
    noul = {"type": "noul", "instructions": "x"}
    assert judges.answer_value(noul, {"noul": True}) is True
    assert judges.answer_value(noul, {"noul": 1}) is None
    score = {"type": "score", "instructions": "x", "criteria": ["低", "高"]}
    assert judges.answer_value(score, {"score": 1}) == 1


def test_judges_do_not_modify_the_sample() -> None:
    """输入：一份样本。输出：无。预期行为：判别器只读，不修改样本。"""
    spec = _make_spec(shape="mixed", variant="note", code_family="package")
    sample = _sample(spec)
    before = copy.deepcopy(sample)
    judges.RoundtripJudge(locale="zh").judge(sample, spec)
    judges.LabelJudge(locale="zh").judge(sample, spec)
    assert sample == before
