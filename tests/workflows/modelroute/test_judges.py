"""判别器：结构往返、答案重算、以及差异报告的每一种记账。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from jc import assemble

from workflows.modelroute import judges, locales, parse, questions, render
from workflows.modelroute.spec import RouteCase, sample_spec


def make_sample(
    task: str = "codefix",
    phrasing: int = 0,
    lang: str = "zh",
    seed: int = 0,
    **overrides: Any,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """输入：任务、问法、语言、打乱种子与维度覆盖。输出：(样本, 真值规格, 语言字典)。

    预期行为：走生产路径（sample → render → ask → assemble），判别器看到的就是产物。
    """
    fields: dict[str, Any] = {
        "task": task,
        "complexity": "mid",
        "latency": "normal",
        "cost": "normal",
        "retry": "first",
        "data": "public",
        "phrasing": phrasing,
    }
    fields.update(overrides)
    loc = locales.load(lang)
    spec = sample_spec(RouteCase(**fields), "train", 0, phrasing)
    items = questions.build(spec, loc, phrasing, random.Random(seed))
    sample = assemble.assemble_sample(
        "modelroute-train-0000000", render.build(spec, loc), items, random.Random(seed)
    )
    return sample, spec, loc


# ---------- 小工具 ----------


def test_reverse_maps_text_back_to_keys() -> None:
    """输入：一张词表。输出：无。预期行为：原文 → 键；原文重复时报错。"""
    assert judges.reverse({"a": "甲", "b": "乙"}) == {"甲": "a", "乙": "b"}
    with pytest.raises(ValueError, match="文案重复"):
        judges.reverse({"a": "甲", "b": "甲"})


def test_task_of_requires_exactly_one_hit() -> None:
    """输入：认不出 / 认出两个任务短语的正文。输出：无。预期行为：两种都报错，不猜一个。"""
    loc = locales.load("zh")
    assert judges.task_of(loc["body"]["codefix"], loc) == "codefix"
    with pytest.raises(ValueError, match="认不出唯一的任务类型"):
        judges.task_of("今天天气不错。", loc)
    with pytest.raises(ValueError, match="认不出唯一的任务类型"):
        judges.task_of("归类 和 抽取 都做一下。", loc)


# ---------- 重算 ----------


def test_recompute_derives_every_answer_from_the_text() -> None:
    """输入：一条渲染出来的画像。输出：无。预期行为：六个答案与真值规格一致。"""
    sample, spec, loc = make_sample(task="extract", complexity="high", latency="relaxed", data="private")
    request = parse.parse_request(sample["state"], parse.label_index(loc))
    expected = judges.recompute(request, loc)
    assert expected == {
        "tier": spec.tier,
        "next": spec.next_tier,
        "effort": spec.effort,
        "is_first": spec.is_first,
        "degrade": spec.degrade,
        "cache": spec.cache,
    }


def test_recompute_omits_next_when_there_is_no_chain() -> None:
    """输入：`reason` 的画像。输出：无。

    预期行为：结果里**没有** `next`——`max` 之上没有档。`LabelJudge` 两边都查：
    这里少一格，产物里多一道 `next` 题也算错。
    """
    sample, spec, loc = make_sample(task="reason")
    request = parse.parse_request(sample["state"], parse.label_index(loc))
    expected = judges.recompute(request, loc)
    assert spec.next_tier is None and "next" not in expected
    assert expected["tier"] == "max"


# ---------- 结构往返 ----------


def test_roundtrip_diff_is_empty_on_a_fresh_sample() -> None:
    """输入：一条刚生成的样本。输出：无。预期行为：逐格一致。"""
    sample, spec, loc = make_sample()
    request = parse.parse_request(sample["state"], parse.label_index(loc))
    assert judges.roundtrip_diff(spec, request, loc) == []


def test_roundtrip_diff_names_the_broken_grid() -> None:
    """输入：把正文换成别的任务的正文。输出：无。预期行为：点名「正文」不一致。"""
    sample, spec, loc = make_sample(task="codefix")
    state = sample["state"].replace(loc["body"]["codefix"], loc["body"]["reason"])
    request = parse.parse_request(state, parse.label_index(loc))
    diff = judges.roundtrip_diff(spec, request, loc)
    assert len(diff) == 1 and diff[0].startswith("正文")


def test_roundtrip_judge_passes_and_reports_reason() -> None:
    """输入：一条刚生成的样本。输出：无。预期行为：通过，理由是人话。"""
    sample, spec, _ = make_sample()
    verdict = judges.RoundtripJudge(locale="zh").judge(sample, spec)
    assert verdict.ok and not verdict.raw.get("diff")


def test_roundtrip_judge_skips_without_a_spec() -> None:
    """输入：不给真值规格。输出：无。预期行为：跳过（不误判成不合格）。"""
    sample, _, _ = make_sample()
    verdict = judges.RoundtripJudge(locale="zh").judge(sample, None)
    assert verdict.ok and verdict.raw == {"skipped": True}


def test_roundtrip_judge_catches_an_unreadable_state() -> None:
    """输入：state 被换成一段没有标签的文本。输出：无。预期行为：不通过，理由是反解失败。"""
    sample, spec, _ = make_sample()
    broken = dict(sample, state="今天天气不错。")
    verdict = judges.RoundtripJudge(locale="zh").judge(broken, spec)
    assert not verdict.ok and "反解失败" in verdict.reason


# ---------- 答案比对 ----------


def test_label_judge_passes_on_a_fresh_sample() -> None:
    """输入：一条刚生成的样本。输出：无。预期行为：通过，没有任何问题项。"""
    sample, _, _ = make_sample(task="extract")
    verdict = judges.LabelJudge(locale="zh").judge(sample)
    assert verdict.ok and verdict.raw["problems"] == []


def test_label_judge_catches_a_wrong_choice_answer() -> None:
    """输入：把 `tier` 的答案改到别的选项。输出：无。预期行为：不通过并点名那道题。"""
    sample, _, _ = make_sample()
    right = sample["answers"]["tier"]["choice"]
    wrong = (right + 1) % len(sample["questions"]["tier"]["criteria"])
    broken = dict(sample, answers={**sample["answers"], "tier": {"type": "choice", "choice": wrong}})
    verdict = judges.LabelJudge(locale="zh").judge(broken)
    assert not verdict.ok and verdict.reason.startswith("tier")


def test_label_judge_catches_a_wrong_noul_answer() -> None:
    """输入：把 `cache` 的答案翻过来。输出：无。预期行为：不通过。"""
    sample, _, _ = make_sample()
    broken = dict(sample, answers={**sample["answers"], "cache": {"type": "noul", "noul": False}})
    verdict = judges.LabelJudge(locale="zh").judge(broken)
    assert not verdict.ok and "cache" in verdict.reason


def test_label_judge_catches_a_missing_question() -> None:
    """输入：从产物里删掉一道题。输出：无。

    预期行为：不通过，理由是「少了」——这一趟遍历**期望表**，遍历产物会一声不响。
    """
    sample, _, _ = make_sample()
    broken = dict(
        sample,
        questions={key: value for key, value in sample["questions"].items() if key != "effort"},
    )
    verdict = judges.LabelJudge(locale="zh").judge(broken)
    assert not verdict.ok and "effort 少了 1 道题" in verdict.reason


def test_label_judge_catches_a_question_that_should_not_exist() -> None:
    """输入：给 `reason` 的样本硬塞一道 `next` 题。输出：无。

    预期行为：不通过——`max` 之上没有档，这道题本来就不该出（条件题族两边都要对得上）。
    """
    sample, _, _ = make_sample(task="reason")
    broken = dict(
        sample,
        questions={**sample["questions"], "next": {"type": "choice", "criteria": {"max": "max"}}},
        answers={**sample["answers"], "next": {"type": "choice", "choice": 0}},
    )
    verdict = judges.LabelJudge(locale="zh").judge(broken)
    assert not verdict.ok and "不该出这道题" in verdict.reason


def test_label_judge_catches_an_unknown_family() -> None:
    """输入：产物里出现不认识的题族。输出：无。预期行为：不通过并点名。"""
    sample, _, _ = make_sample()
    broken = dict(
        sample,
        questions={**sample["questions"], "bogus": {"type": "noul", "criteria": {"true": "是", "false": "否"}}},
        answers={**sample["answers"], "bogus": {"type": "noul", "noul": True}},
    )
    verdict = judges.LabelJudge(locale="zh").judge(broken)
    assert not verdict.ok and "bogus" in verdict.reason


def test_label_judge_catches_a_question_type_this_workflow_never_emits() -> None:
    """输入：把 `cache` 那道题换成 `score` 题型。输出：无。

    预期行为：不通过——本题型只出 choice / noul，混进来的 score 题按答案对不上记账
    （不静默放过，也不崩）。
    """
    sample, _, _ = make_sample()
    broken = dict(
        sample,
        questions={**sample["questions"], "cache": {"type": "score", "criteria": ["a", "b"]}},
        answers={**sample["answers"], "cache": {"type": "score", "score": 3}},
    )
    verdict = judges.LabelJudge(locale="zh").judge(broken)
    assert not verdict.ok and "cache" in verdict.reason


def test_label_judge_catches_an_out_of_range_index() -> None:
    """输入：choice 的编号越界。输出：无。预期行为：不通过，而不是崩。"""
    sample, _, _ = make_sample()
    broken = dict(sample, answers={**sample["answers"], "tier": {"type": "choice", "choice": 99}})
    verdict = judges.LabelJudge(locale="zh").judge(broken)
    assert not verdict.ok and "编号越界" in verdict.reason


def test_label_judge_reports_a_state_it_cannot_recompute() -> None:
    """输入：state 读不出来。输出：无。预期行为：不通过，理由是重算失败。"""
    sample, _, _ = make_sample()
    broken = dict(sample, state="随便一段文本")
    verdict = judges.LabelJudge(locale="zh").judge(broken)
    assert not verdict.ok and "重算失败" in verdict.reason


def test_label_judge_handles_the_english_locale() -> None:
    """输入：英文样本。输出：无。预期行为：英文路径也能重算并通过。"""
    sample, _, _ = make_sample(lang="en", task="classify")
    verdict = judges.LabelJudge(locale="en").judge(sample)
    assert verdict.ok


def test_label_judge_trims_the_reason_list() -> None:
    """输入：三道题同时答错。输出：无。预期行为：理由最多列 `max_problems` 条。"""
    sample, _, _ = make_sample()
    answers = dict(sample["answers"])
    for qid, question in sample["questions"].items():
        if question["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": not answers[qid]["noul"]}
    verdict = judges.LabelJudge(locale="zh", max_problems=1).judge(dict(sample, answers=answers))
    assert not verdict.ok and verdict.reason.count(";") == 0
