"""判别器：从文本重算的往返、按规格重算的题号比对，以及差异报告的每一种记账。"""

from __future__ import annotations

import random
from typing import Any

from jc import assemble
from jc.workflow import family_of

from workflows.lint import judges, locales, questions, render
from workflows.lint.spec import SIZE, LintCase, sample_spec


def make_sample(
    rule: str = "secret",
    genre: str = "code",
    hits: int = 0,
    phrasing: int = 0,
    lang: str = "zh",
    seed: int = 0,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """输入：四个维度 + 语言 + 打乱种子。输出：(样本, 真值规格, 语言字典)。

    预期行为：走生产路径（sample → render → ask → assemble），判别器看到的就是产物。
    """
    loc = locales.load(lang)
    spec = sample_spec(LintCase(rule, genre, hits, phrasing), "train", 0)
    lines = render.render_lines(spec.plan, loc)
    labeled = questions.build(spec.plan.rule, lines, spec.plan.matched(), loc, phrasing)
    sample = assemble.assemble_sample(
        "lint-train-0000000", render.render_state(spec, loc), labeled, random.Random(seed)
    )
    return sample, spec, loc


def test_roundtrip_passes_on_a_fresh_sample() -> None:
    """输入：一条刚生成的样本。输出：无。预期行为：通过，理由是人话。"""
    sample, _, _ = make_sample(hits=3)
    verdict = judges.RoundtripJudge("zh").judge(sample)
    assert verdict.ok and "逐题都对上" in verdict.reason


def test_label_passes_on_a_fresh_sample() -> None:
    """输入：一条刚生成的样本 + 真值规格。输出：无。预期行为：通过。"""
    sample, spec, _ = make_sample(hits=1)
    verdict = judges.LabelJudge("zh").judge(sample, spec)
    assert verdict.ok and "逐题一致" in verdict.reason


def test_roundtrip_catches_a_tampered_line() -> None:
    """输入：把一行背景行改成命中行的 state。输出：无。

    预期行为：不通过——这一条正是「渲染把某一行写串味了」的等价情形：文本变了，
    而答案还是老的。
    """
    sample, _, _ = make_sample(hits=0)
    rows = sample["state"].split("\n")
    rows[2] = '1. API_KEY = "sk-live-9f2b2c"'
    broken = dict(sample, state="\n".join(rows))
    verdict = judges.RoundtripJudge("zh").judge(broken)
    assert not verdict.ok and "重算应为" in verdict.reason


def test_roundtrip_catches_an_unreadable_state() -> None:
    """输入：不含规则短语的 state。输出：无。预期行为：不通过，理由是「读不出来」。"""
    sample, _, _ = make_sample()
    broken = dict(sample, state="随便一段文本\n抬头\n1. x")
    verdict = judges.RoundtripJudge("zh").judge(broken)
    assert not verdict.ok and "读不出来" in verdict.reason


def test_roundtrip_catches_a_wrong_answer() -> None:
    """输入：把某道逐行题的答案翻过来。输出：无。预期行为：不通过。"""
    sample, _, _ = make_sample(hits=0)
    qid = next(iter(sample["questions"]))
    flipped = dict(sample["answers"][qid], noul=True)
    broken = dict(sample, answers={**sample["answers"], qid: flipped})
    verdict = judges.RoundtripJudge("zh").judge(broken)
    assert not verdict.ok


def test_label_rejects_a_wrong_spec_type() -> None:
    """输入：给 `label` 传一个不是规格的对象。输出：无。预期行为：不通过（不静默放行）。"""
    sample, _, _ = make_sample()
    verdict = judges.LabelJudge("zh").judge(sample, {"not": "a spec"})
    assert not verdict.ok and "类型不对" in verdict.reason


def test_label_catches_a_missing_question() -> None:
    """输入：从产物里删掉一道题。输出：无。预期行为：不通过，理由是题号对不上。"""
    sample, spec, _ = make_sample(hits=1)
    qid = next(iter(sample["questions"]))
    broken = dict(
        sample,
        questions={key: value for key, value in sample["questions"].items() if key != qid},
    )
    verdict = judges.LabelJudge("zh").judge(broken, spec)
    assert not verdict.ok and "题号与规格不符" in verdict.reason


def test_label_catches_a_wrong_answer() -> None:
    """输入：把 count 题的答案改掉。输出：无。预期行为：不通过并点名是哪道题。"""
    sample, spec, _ = make_sample(hits=2)
    broken = dict(
        sample,
        answers={**sample["answers"], "count": {"type": "choice", "choice": 0}},
    )
    verdict = judges.LabelJudge("zh").judge(broken, spec)
    assert not verdict.ok and verdict.reason.startswith("count")


def test_recompute_groups_answers_in_question_order() -> None:
    """输入：一条命中 1 行的样本。输出：无。

    预期行为：`line` 那一族的期望答案按行号顺序排列，且 `spot` 只在恰好一行命中时出现。
    """
    sample, spec, loc = make_sample(hits=1)
    from workflows.lint import parse

    reading = parse.read_state(sample["state"], loc)
    expected = judges.recompute(reading, loc)
    assert expected["line"] == list(reading.matched)
    assert len(expected["line"]) == SIZE
    # choice 族的期望答案是**选项名**（编号只由组装器产生），所以这里要写成 `c1`。
    assert expected["count"] == ["c1"]
    assert expected["gate"] == ["pass"]
    assert len(expected["spot"]) == 1


def test_compare_reports_an_extra_question() -> None:
    """输入：期望表里没有、产物里却有的题。输出：无。预期行为：记一笔「多出来了」。"""
    sample, _, _ = make_sample(hits=0)
    sample = dict(sample, questions={**sample["questions"], "line_99": {"type": "noul"}})
    problems = judges.compare(sample, {"line": [False]})
    assert any("多出来了" in item for item in problems)


def test_compare_reports_an_unknown_family() -> None:
    """输入：产物里出现不认识的题族。输出：无。预期行为：记一笔「未知题族」。"""
    sample, _, _ = make_sample(hits=0)
    sample = dict(sample, questions={"nope": {"type": "noul"}})
    problems = judges.compare(sample, {})
    assert any("未知题族" in item for item in problems)


def test_compare_reports_a_whole_family_missing() -> None:
    """输入：期望表里有、产物里整族缺席。输出：无。

    预期行为：记一笔「少了 N 道题」——「少题」那一趟必须遍历**期望表**：遍历产物时整族
    缺席会一声不响（漏报比误报危险）。
    """
    sample, _, _ = make_sample(hits=0)
    trimmed = dict(
        sample,
        questions={
            key: value
            for key, value in sample["questions"].items()
            if family_of(key) != "gate"
        },
    )
    problems = judges.compare(trimmed, {"gate": ["pass"], "line": [False] * SIZE})
    assert any("gate 少了 1 道题" in item for item in problems)


def test_compare_reports_a_wrong_index_as_a_mismatch() -> None:
    """输入：choice 的编号越界。输出：无。预期行为：记一笔答案对不上，而不是崩。"""
    sample, _, _ = make_sample(hits=0)
    qid = "count"
    broken = dict(sample, answers={**sample["answers"], qid: {"type": "choice", "choice": 99}})
    problems = judges.compare(broken, {"count": ["c0"]})
    assert any("count：" in item for item in problems)


def test_answer_value_tolerates_junk() -> None:
    """输入：编号是布尔 / 字符串的答案。输出：无。预期行为：给 None（对不上，不是崩）。"""
    question = {"type": "choice", "criteria": {"a": "甲", "b": "乙"}}
    assert judges.answer_value(question, {"choice": True}) is None
    assert judges.answer_value(question, {"choice": "0"}) is None
    assert judges.answer_value({"type": "noul"}, {"noul": "yes"}) is None
    assert judges.answer_value({"type": "score"}, {"score": 2}) == 2
