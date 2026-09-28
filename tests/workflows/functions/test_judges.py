"""判别器：两条独立的路（从文本重算 / 拿规格重算）都要抓得住产物里的错。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random


import pytest

from jc import assemble
from workflows.functions import judges, locales, parse, questions, render, spec as sp
from workflows.functions.workflow import FunctionsWorkflow


def _built(case: sp.CallCase, lang: str = "zh", seed: int = 1) -> tuple[sp.CallSpec, dict[str, JsonValue]]:
    """输入：case、语言、打乱种子。输出：(规格, 契约样本)。"""
    loc = locales.load(lang)
    spec = sp.sample_spec(case, "train", 0)
    state = render.render_state(spec, loc)
    items = questions.build(spec.plan, loc, spec.plan.phrasing, random.Random(0))
    return spec, assemble.assemble_sample("s1", state, items, random.Random(seed))


def _case(tool: str = "plot_price", voice: str = "chatty") -> sp.CallCase:
    """输入：工具与语气。输出：一个 case。"""
    return sp.CallCase(tool, voice, "1w", "ticker", 0)


# ---------- answer_value ----------


def test_answer_value_reads_choice_by_option_name() -> None:
    """输入：一道 choice 题与编号。输出：无。

    预期行为：给**选项名**（不是编号）——编号只由组装器产生，判别器不该依赖它。
    """
    question = {"type": "choice", "criteria": {"a": "甲", "b": "乙"}}
    assert judges.answer_value(question, {"choice": 0}) == "a"
    assert judges.answer_value(question, {"choice": 1}) == "b"


@pytest.mark.parametrize("bad", [2, -1, "0", None, True, 1.5])
def test_answer_value_rejects_a_bad_choice_index(bad: object) -> None:
    """输入：越界 / 类型不对的编号。输出：无。预期行为：给 None（对不上，不是崩）。"""
    question = {"type": "choice", "criteria": {"a": "甲", "b": "乙"}}
    assert judges.answer_value(question, {"choice": bad}) is None


def test_answer_value_reads_noul_and_rejects_non_bools() -> None:
    """输入：noul 答案。输出：无。预期行为：只认真布尔，其余给 None。"""
    question = {"type": "noul"}
    assert judges.answer_value(question, {"noul": True}) is True
    assert judges.answer_value(question, {"noul": 1}) is None
    assert judges.answer_value(question, {}) is None


def test_answer_value_passes_score_through() -> None:
    """输入：score 题（本 workflow 不出，但通用形状要认得）。输出：无。"""
    assert judges.answer_value({"type": "score"}, {"score": 3}) == 3


# ---------- recompute / compare ----------


@pytest.mark.parametrize("tool", sorted(sp.TOOLS))
def test_compare_is_clean_on_a_generated_sample(tool: str) -> None:
    """输入：四个工具各一条真样本。输出：无。预期行为：差异列表为空。"""
    for voice in sp.VOICES:
        case = _case(tool, voice)
        loc = locales.load("zh")
        _, sample = _built(case)
        plan = parse.read_request(sample["state"], loc)
        assert judges.compare(sample, judges.recompute(plan, loc)) == []


def test_compare_reports_an_extra_question() -> None:
    """输入：产物里多出一道题。输出：无。预期行为：记一笔「多出来了」。"""
    loc = locales.load("zh")
    _, sample = _built(_case())
    plan = parse.read_request(sample["state"], loc)
    sample["questions"]["arg_window_2"] = sample["questions"]["arg_window"]
    sample["answers"]["arg_window_2"] = dict(sample["answers"]["arg_window"])
    problems = judges.compare(sample, judges.recompute(plan, loc))
    assert any("多出来了" in item for item in problems)


def test_compare_reports_a_missing_question() -> None:
    """输入：产物里少了一道题。输出：无。预期行为：记一笔「少了」。"""
    loc = locales.load("zh")
    _, sample = _built(_case())
    plan = parse.read_request(sample["state"], loc)
    del sample["questions"]["stated_style"]
    del sample["answers"]["stated_style"]
    problems = judges.compare(sample, judges.recompute(plan, loc))
    assert any("少了 1 道题" in item for item in problems)


def test_compare_reports_a_wrong_answer() -> None:
    """输入：把一道题的答案改错。输出：无。预期行为：点名题号与两边取值。"""
    loc = locales.load("zh")
    _, sample = _built(_case())
    plan = parse.read_request(sample["state"], loc)
    qid = "stated_include_volume"
    sample["answers"][qid]["noul"] = not sample["answers"][qid]["noul"]
    problems = judges.compare(sample, judges.recompute(plan, loc))
    assert any(qid in item for item in problems)


def test_compare_reports_an_unknown_family() -> None:
    """输入：产物里出现不认识的题族。输出：无。预期行为：点名「未知题族」。"""
    loc = locales.load("zh")
    _, sample = _built(_case())
    plan = parse.read_request(sample["state"], loc)
    sample["questions"]["bogus"] = {"type": "noul", "criteria": {"true": "是", "false": "否"}}
    sample["answers"]["bogus"] = {"type": "noul", "noul": True}
    problems = judges.compare(sample, judges.recompute(plan, loc))
    assert any("未知题族" in item for item in problems)


def test_recompute_groups_answers_by_family() -> None:
    """输入：一份计划。输出：无。预期行为：{题族: [答案]}，且每个题族恰好一条答案。"""
    plan = sp.build_plan(_case(tool="compare_returns"))
    stated = frozenset(binding.name for binding in plan.args if binding.stated)
    table = judges.recompute(plan, locales.load("zh"))
    assert set(table) == set(questions.families_for("compare_returns", stated))
    assert all(len(answers) == 1 for answers in table.values())


# ---------- RoundtripJudge ----------


def test_roundtrip_judge_accepts_a_faithful_sample() -> None:
    """输入：真样本。输出：无。预期行为：通过，理由是「逐题都对上」。"""
    _, sample = _built(_case())
    verdict = judges.RoundtripJudge("zh").judge(sample)
    assert verdict.ok is True
    assert "对上" in verdict.reason


def test_roundtrip_judge_reports_an_unreadable_state() -> None:
    """输入：认不出意图的 state。输出：无。预期行为：不通过，理由是读不出来。"""
    _, sample = _built(_case())
    sample["state"] = "今天天气不错。"
    verdict = judges.RoundtripJudge("zh").judge(sample)
    assert verdict.ok is False
    assert "读不出来" in verdict.reason


def test_roundtrip_judge_reports_mismatched_answers() -> None:
    """输入：把答案改错的样本。输出：无。预期行为：不通过，且原始结果里带上全部差异。"""
    _, sample = _built(_case())
    qid = "arg_window"
    sample["answers"][qid]["choice"] = (sample["answers"][qid]["choice"] + 1) % 2
    verdict = judges.RoundtripJudge("zh").judge(sample)
    assert verdict.ok is False
    assert verdict.raw["problems"]


def test_roundtrip_judge_ignores_the_spec_argument() -> None:
    """输入：多余传进来的 spec。输出：无。预期行为：这条路只读文本，规格传什么都无所谓。"""
    _, sample = _built(_case())
    assert judges.RoundtripJudge("zh").judge(sample, object()).ok is True


# ---------- LabelJudge ----------


def test_label_judge_accepts_a_faithful_sample() -> None:
    """输入：真样本 + 真规格。输出：无。预期行为：通过，题号与答案逐题一致。"""
    spec, sample = _built(_case())
    verdict = judges.LabelJudge("zh").judge(sample, spec)
    assert verdict.ok is True


def test_label_judge_rejects_a_wrong_spec_type() -> None:
    """输入：不是 `CallSpec` 的规格。输出：无。预期行为：不通过，理由是类型不对。"""
    _, sample = _built(_case())
    verdict = judges.LabelJudge("zh").judge(sample, {"nope": 1})
    assert verdict.ok is False
    assert "类型不对" in verdict.reason


def test_label_judge_reports_a_question_id_mismatch() -> None:
    """输入：产物里多出一道题。输出：无。预期行为：不通过，理由是题号与规格不符。"""
    spec, sample = _built(_case())
    sample["questions"]["arg_style_2"] = sample["questions"]["arg_style"]
    verdict = judges.LabelJudge("zh").judge(sample, spec)
    assert verdict.ok is False
    assert "题号与规格不符" in verdict.reason


def test_label_judge_reports_a_wrong_answer() -> None:
    """输入：把一道题的答案改错。输出：无。预期行为：不通过，点名题号与两边取值。"""
    spec, sample = _built(_case())
    qid = "stated_style"
    sample["answers"][qid]["noul"] = not sample["answers"][qid]["noul"]
    verdict = judges.LabelJudge("zh").judge(sample, spec)
    assert verdict.ok is False
    assert qid in verdict.reason


def test_label_judge_is_stable_across_option_order() -> None:
    """输入：两种打乱种子。输出：无。预期行为：选项打乱不影响答案对错（答案按选项名比）。"""
    spec, first = _built(_case(), seed=1)
    _, second = _built(_case(), seed=9)
    assert list(first["questions"]["arg_window"]["criteria"]) != list(
        second["questions"]["arg_window"]["criteria"]
    )
    assert judges.LabelJudge("zh").judge(first, spec).ok is True
    assert judges.LabelJudge("zh").judge(second, spec).ok is True


# ---------- 接线 ----------


def test_make_judges_builds_the_declared_pair() -> None:
    """输入：workflow 声明的判别器名单。输出：无。预期行为：名字与实例一一对应。"""
    workflow = FunctionsWorkflow()
    built = workflow.make_judges(workflow.judges, "zh")
    assert [judge.name for judge in built] == ["roundtrip", "label"]


def test_make_judges_rejects_an_unknown_name() -> None:
    """输入：不存在的判别器名。输出：无。预期行为：KeyError。"""
    with pytest.raises(KeyError):
        FunctionsWorkflow().make_judges(("nope",), "zh")
