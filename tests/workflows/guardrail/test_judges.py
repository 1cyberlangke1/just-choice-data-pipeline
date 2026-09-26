"""判别器：结构往返与答案重算（只看文本）。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.guardrail import judges, locales, parse, questions, render
from workflows.guardrail import spec as sp


def sample_of(spec: sp.GuardSpec, lang: str = "zh") -> dict[str, Any]:
    """输入：规格、语言。输出：一条样本（题号 = 题族名，答案按契约编码）。"""
    loc = locales.load(lang)
    built_questions: dict[str, Any] = {}
    built_answers: dict[str, Any] = {}
    for item in questions.build(spec, loc, spec.phrasing, random.Random(0)):
        built_questions[item.family] = item.question
        if item.question["type"] == "choice":
            options = list(item.question["criteria"])
            built_answers[item.family] = {"type": "choice", "choice": options.index(item.answer)}
        elif item.question["type"] == "noul":
            built_answers[item.family] = {"type": "noul", "noul": item.answer}
        else:
            built_answers[item.family] = {"type": "score", "score": item.answer}
    return {
        "state": render.build(spec, loc),
        "questions": built_questions,
        "answers": built_answers,
    }


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_roundtrip_judge_passes(lang: str, make_spec: Any) -> None:
    """输入：一条真样本（带擦边句）。输出：无。预期行为：结构判别通过。"""
    spec = make_spec(mentions=("meds", "privacy"))
    verdict = judges.RoundtripJudge(locale=lang).judge(sample_of(spec, lang), spec)
    assert verdict.ok, verdict.reason


def test_roundtrip_judge_catches_a_tampered_field(make_spec: Any) -> None:
    """输入：把传播面那一格改掉。输出：无。预期行为：判不通过并点名那一格。"""
    spec = make_spec()
    sample = sample_of(spec)
    loc = locales.load("zh")
    sample["state"] = sample["state"].replace(loc["reach"]["public"], loc["reach"]["group"])
    verdict = judges.RoundtripJudge().judge(sample, spec)
    assert not verdict.ok
    assert verdict.raw["diff"]
    assert any("传播面" in item for item in verdict.raw["diff"])


def test_roundtrip_judge_skips_without_spec(make_spec: Any) -> None:
    """输入：不给真值规格。输出：无。预期行为：跳过而不是误判。"""
    verdict = judges.RoundtripJudge().judge(sample_of(make_spec()), None)
    assert verdict.ok and verdict.raw.get("skipped") is True


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_label_judge_passes(lang: str, make_spec: Any) -> None:
    """输入：一条真样本。输出：无。预期行为：重算出来的答案与样本一致。"""
    verdict = judges.LabelJudge(locale=lang).judge(sample_of(make_spec(), lang))
    assert verdict.ok, verdict.reason


def test_label_judge_catches_a_flipped_answer(make_spec: Any) -> None:
    """输入：把一条 noul 答案翻过来。输出：无。预期行为：判不通过并点名题号。"""
    sample = sample_of(make_spec())
    sample["answers"]["violates_spam"]["noul"] = True
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok
    assert any(problem.startswith("violates_spam:") for problem in verdict.raw["problems"])


def test_label_judge_catches_an_unparseable_state(make_spec: Any) -> None:
    """输入：state 只剩抬头一行。输出：无。预期行为：判不通过并说明反解失败。"""
    sample = sample_of(make_spec())
    sample["state"] = sample["state"].splitlines()[0]
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok
    assert "重算失败" in verdict.reason


def test_recompute_reproduces_the_spec(make_spec: Any) -> None:
    """输入：一条真值规格（前科 + 未成年人受众）。输出：无。预期行为：重算逐族等于规格。"""
    spec = make_spec(history="repeat", audience="minors")
    loc = locales.load("zh")
    message = parse.parse_message(render.build(spec, loc), parse.label_index(loc))
    expected = judges.recompute(message, loc)
    assert expected["severity"] == spec.severity
    assert expected["action"] == spec.action
    assert expected["top_clause"] == spec.top_clause
    assert expected["minor_exposure"] is spec.minor_exposure
    assert expected["repeat_offender"] is spec.repeat_offender
    for clause in sp.CLAUSES:
        assert expected[f"violates_{clause}"] is spec.violated(clause)


@pytest.mark.parametrize(
    "overrides",
    ({"mentions": ("harass", "scam", "cheat")}, {"wording": "softened"}),
)
def test_distractors_do_not_change_the_answers(overrides: dict[str, Any], make_spec: Any) -> None:
    """输入：只改擦边句 / 措辞的两条规格。输出：无。

    预期行为：重算出来的答案完全一样——这两维只改文本、不改判据。判别器若被它们带偏
    （把擦边句当违规、把委婉语气当免责），或者句序指纹里混进了这两维（于是 `top_clause`
    跟着换），这条都会红。
    """
    loc = locales.load("zh")

    def answers(spec: sp.GuardSpec) -> dict[str, Any]:
        message = parse.parse_message(render.build(spec, loc), parse.label_index(loc))
        return judges.recompute(message, loc)

    assert answers(make_spec()) == answers(make_spec(**overrides))


def test_mentions_are_not_counted_as_violations(make_spec: Any) -> None:
    """输入：只有擦边句、没有违规句的正文。输出：无。预期行为：违规条款为空。"""
    loc = locales.load("zh")
    spec = make_spec(violations=(), mentions=("harass", "scam", "cheat"))
    violations, mentions = judges.read_body(render.body_lines(spec, loc), loc)
    assert violations == ()
    assert sorted(mentions) == sorted(("harass", "scam", "cheat"))


def test_read_body_rejects_an_unknown_sentence(make_spec: Any) -> None:
    """输入：正文里塞一句谁也不认识的句子。输出：无。

    预期行为：报错——那说明渲染漏了模板或语言文件缺句，是硬伤，不能当填充句蒙过去。
    """
    loc = locales.load("zh")
    spec = make_spec()
    body = render.body_lines(spec, loc) + ("这句谁也没写过。",)
    with pytest.raises(ValueError, match="认不出来"):
        judges.read_body(body, loc)
