"""判别器：结构往返、答案重算，以及比对函数的每一条分支。"""

from __future__ import annotations

import random
from typing import Any

from jc import assemble
from workflows.context import judges, locales, parse, questions, render
from workflows.context import spec as sp

LOC = locales.load("zh")


def _spec(**overrides: object) -> sp.SessionSpec:
    """输入：case 覆盖项。输出：一条真值规格（固定种子）。"""
    fields = {
        "keep_form": "err_small",
        "drop_form": "off_small",
        "kind_set": "read_run",
        "extra": "keep",
        "order": "p0",
        "phrasing": 0,
    }
    fields.update(overrides)
    return sp.sample_spec(sp.SessionCase(**fields), "train", 0, 0, random.Random(3))  # type: ignore[arg-type]


def _sample(**overrides: object) -> tuple[dict[str, Any], sp.SessionSpec]:
    """输入：case 覆盖项。输出：(契约样本, 真值规格)。预期行为：走生产路径组装。"""
    spec = _spec(**overrides)
    items = questions.build(spec.blocks, spec.budget, LOC, spec.phrasing)
    sample = assemble.assemble_sample(
        "context-train-0000000", render.build(spec, LOC), items, random.Random(1)
    )
    return sample, spec


def test_recompute_matches_the_spec_answers() -> None:
    """输入：一条从文本反解出来的会话。输出：无。

    预期行为：重算出来的期望答案表与规格出题的结果逐条一致——判别器与构造侧共用同一套
    规则函数，两边不该有第二份真值。
    """
    spec = _spec(order="p4")
    session = parse.parse_session(render.build(spec, LOC), parse.labels_of(LOC))
    expected = judges.recompute(session, LOC)
    assert expected == {
        "action": [block.action() for block in spec.blocks],
        "compact": [spec.compact],
        "spot": [f"p{spec.drop_position}"],
        "top": [f"p{spec.top_position}"],
    }


def test_blocks_from_reads_back_the_keys() -> None:
    """输入：一条反解出来的会话。输出：无。预期行为：四格原文翻回键，与真值块一致。"""
    spec = _spec(order="p2")
    session = parse.parse_session(render.build(spec, LOC), parse.labels_of(LOC))
    assert judges.blocks_from(session) == spec.blocks


def test_roundtrip_judge_passes_on_a_faithful_sample() -> None:
    """输入：一条合格样本。输出：无。预期行为：通过。"""
    sample, spec = _sample()
    verdict = judges.RoundtripJudge(locale="zh").judge(sample, spec)
    assert verdict.ok and verdict.raw["diff"] == []


def test_roundtrip_judge_catches_a_tampered_cell() -> None:
    """输入：把某一块的「大小」格改掉。输出：无。

    预期行为：报出那一格不一致——这条路只看文本，不看答案，所以能抓到渲染串味。
    """
    sample, spec = _sample()
    sample["state"] = sample["state"].replace("小 · ", "大 · ", 1)
    verdict = judges.RoundtripJudge(locale="zh").judge(sample, spec)
    assert not verdict.ok and "第 1 块抬头" in verdict.reason


def test_roundtrip_judge_catches_a_tampered_body_line() -> None:
    """输入：把某一块的正文改掉。输出：无。预期行为：报出那一块正文不一致。"""
    sample, spec = _sample()
    lines = sample["state"].splitlines()
    lines[6] = "    这里是另一段完全不同的内容。"
    sample["state"] = "\n".join(lines)
    verdict = judges.RoundtripJudge(locale="zh").judge(sample, spec)
    assert not verdict.ok and "第 1 块正文" in verdict.reason


def test_roundtrip_judge_catches_a_missing_block() -> None:
    """输入：删掉最后一块（抬头 + 正文）。输出：无。预期行为：报出块数不一致。"""
    sample, spec = _sample()
    sample["state"] = "\n".join(sample["state"].splitlines()[:-2])
    verdict = judges.RoundtripJudge(locale="zh").judge(sample, spec)
    assert not verdict.ok and "块数" in verdict.reason


def test_roundtrip_judge_reports_an_unreadable_state() -> None:
    """输入：读不出来的 state。输出：无。预期行为：不通过并说明反解失败。"""
    sample, spec = _sample()
    sample["state"] = "今天天气不错。"
    verdict = judges.RoundtripJudge(locale="zh").judge(sample, spec)
    assert not verdict.ok and "反解失败" in verdict.reason


def test_roundtrip_judge_skips_without_a_spec() -> None:
    """输入：没有真值规格。输出：无。预期行为：跳过（不误判成不通过）。"""
    sample, _ = _sample()
    verdict = judges.RoundtripJudge(locale="zh").judge(sample, None)
    assert verdict.ok and verdict.raw == {"skipped": True}


def test_label_judge_passes_on_a_faithful_sample() -> None:
    """输入：一条合格样本。输出：无。预期行为：题号与答案逐题一致。"""
    sample, spec = _sample(order="p3")
    assert judges.LabelJudge(locale="zh").judge(sample, spec).ok


def test_label_judge_catches_a_tampered_answer() -> None:
    """输入：把某个 noul 答案翻过来。输出：无。预期行为：报出那道题对不上。"""
    sample, spec = _sample()
    sample["answers"]["compact"]["noul"] = not sample["answers"]["compact"]["noul"]
    verdict = judges.LabelJudge(locale="zh").judge(sample, spec)
    assert not verdict.ok and "compact" in verdict.reason


def test_label_judge_catches_a_missing_question() -> None:
    """输入：删掉一道题。输出：无。预期行为：题号对不上，直接不通过。"""
    sample, spec = _sample()
    qid = next(iter(sample["questions"]))
    del sample["questions"][qid]
    del sample["answers"][qid]
    verdict = judges.LabelJudge(locale="zh").judge(sample, spec)
    assert not verdict.ok and "题号与规格不符" in verdict.reason


def test_label_judge_rejects_a_foreign_spec() -> None:
    """输入：拿错的规格对象。输出：无。预期行为：报类型不对，不静默通过。"""
    sample, _ = _sample()
    verdict = judges.LabelJudge(locale="zh").judge(sample, {"not": "a spec"})
    assert not verdict.ok and "规格类型不对" in verdict.reason


def test_compare_reports_unknown_extra_missing_and_wrong() -> None:
    """输入：四种坏产物。输出：无。

    预期行为：未知题族、多出来的题、少掉的题、答案不对各记一笔——「少题」那一趟遍历
    期望表，整族缺席也报得出来。
    """
    sample, spec = _sample()
    expected = judges.recompute(parse.parse_session(sample["state"], parse.labels_of(LOC)), LOC)
    assert judges.compare(sample, expected) == []
    sample["questions"]["bogus"] = {"type": "noul", "criteria": {"true": "是", "false": "否"}}
    sample["answers"]["bogus"] = {"type": "noul", "noul": True}
    problems = judges.compare(sample, expected)
    assert any("未知题族" in item for item in problems)
    sample["questions"]["action_9"] = {"type": "noul", "criteria": {"true": "是", "false": "否"}}
    sample["answers"]["action_9"] = {"type": "noul", "noul": True}
    problems = judges.compare(sample, expected)
    assert any("多出来了" in item for item in problems)
    del sample["questions"]["action_9"]
    del sample["answers"]["action_9"]
    del sample["questions"]["action_5"]
    del sample["answers"]["action_5"]
    problems = judges.compare(sample, expected)
    assert any("少了 1 道题" in item for item in problems)
    sample["answers"]["compact"]["noul"] = not sample["answers"]["compact"]["noul"]
    problems = judges.compare(sample, expected)
    assert any("按文本重算应为" in item for item in problems)


def test_answer_value_handles_every_shape() -> None:
    """输入：各种题与答案。输出：无。

    预期行为：choice 给选项名、noul 给布尔、score 给下标；编号越界 / 类型不对给 None
    （那是对不上，不是崩）。
    """
    choice = {"type": "choice", "criteria": {"a": "甲", "b": "乙"}}
    assert judges.answer_value(choice, {"choice": 1}) == "b"
    assert judges.answer_value(choice, {"choice": 9}) is None
    assert judges.answer_value(choice, {"choice": True}) is None
    assert judges.answer_value(choice, {}) is None
    assert judges.answer_value({"type": "noul"}, {"noul": True}) is True
    assert judges.answer_value({"type": "noul"}, {"noul": "yes"}) is None
    assert judges.answer_value({"type": "score"}, {"score": 2}) == 2
