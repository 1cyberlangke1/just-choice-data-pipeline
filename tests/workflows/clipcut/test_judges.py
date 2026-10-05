"""判别器：主题名归一化、重算四个答案、篡改能抓到、缺 spec 跳过、跨语言。"""

from __future__ import annotations

import random

import pytest

from jc import assemble
from jc.jsontypes import JsonValue
from tests.workflows.clipcut.conftest import _make_spec
from workflows.clipcut import features as ft
from workflows.clipcut import judges, locales, parse, questions, render
from workflows.clipcut import spec as sp

LOC = locales.load("zh")


def sample_of(spec: sp.ClipSpec, locale: str = "zh", seed: int = 0) -> dict[str, JsonValue]:
    """输入：规格、语言、种子。输出：契约样本。

    预期行为：走真管线那两步（渲染 + 出题 + 组装），不手写样本——判别器要验的正是
    「组装之后还能不能由 state 重算出来」。
    """
    loc = locales.load(locale)
    state = render.build(spec, loc)
    labeled = questions.build(spec, loc, spec.phrasing, random.Random(seed))
    return assemble.assemble_sample("s-0000001", state, labeled, random.Random(seed))


def parsed_of(sample: dict[str, JsonValue], locale: str = "zh") -> parse.ParsedClip:
    """输入：样本、语言。输出：反解结果。预期行为：用该语言的标签表反解。"""
    return parse.parse_state(sample["state"], parse.tables_of(locales.load(locale)))


def facts_of(sample: dict[str, JsonValue], locale: str = "zh") -> dict[str, JsonValue]:
    """输入：样本、语言。输出：从 state 重算出来的四个答案。"""
    return judges.facts_of(parsed_of(sample, locale), locales.words())


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_to_keys_normalizes_both_languages(lang: str) -> None:
    """输入：两种语言渲染的样本。输出：无。

    预期行为：主题名与高亮词归一化之后与构造侧的序列**逐段**一致——只认一种语言的名字会把
    另一种语言的产物整片读成「表外名字」。
    """
    case = ft.ClipCase(topics=1, flags=5)
    spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
    keys = judges.to_keys(parsed_of(sample_of(spec, lang), lang).segments, locales.words())
    assert tuple(topic for topic, _ in keys) == case.topic_seq
    assert tuple(flag for _, flag in keys) == case.flag_seq


def test_to_keys_reports_names_outside_the_table() -> None:
    """输入：表外的主题名 / 高亮词。输出：无。

    预期行为：报人话——渲染写出了表外的名字不能静默当成「没高亮」，那会把渲染 bug 伪装成
    一道难题。
    """
    with pytest.raises(ValueError, match="主题不在名字表里"):
        judges.to_keys((("乱写", "高亮"),), locales.words())
    with pytest.raises(ValueError, match="高亮词不在表里"):
        judges.to_keys((("开场白", "乱写"),), locales.words())


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_facts_of_recomputes_every_answer_for_every_case(lang: str) -> None:
    """输入：全空间 case × 两种语言。输出：无。

    预期行为：只看 state 重算出来的答案与真值规格完全一致——「生成器把答案算错了」与
    「渲染把某一段写歪了」是两条独立的路，这里是审计侧那条。
    """
    for case in ft.build_cases(512):
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        facts = facts_of(sample_of(spec, lang), lang)
        assert facts["keep_count"] == spec.keep_count, case.describe()
        assert facts["first_kept"] == spec.first_kept, case.describe()
        assert facts["blocked_any"] is spec.blocked_any, case.describe()
        assert facts["dup_any"] is spec.dup_any, case.describe()
        assert facts["topics"] == list(case.topic_seq), case.describe()
        assert facts["flags"] == list(case.flag_seq), case.describe()


def test_check_sample_accepts_a_fresh_sample() -> None:
    """输入：刚组装的样本。输出：无。预期行为：四道题全部对得上（零问题）。"""
    sample = sample_of(_make_spec(2))
    assert judges.check_sample(sample, facts_of(sample), LOC) == []


def test_check_sample_catches_a_tampered_answer() -> None:
    """输入：把某道题的答案改错。输出：无。预期行为：那一题被报出来，别的题不受影响。"""
    sample = sample_of(_make_spec(3))
    facts = facts_of(sample)
    qid = next(name for name in sample["questions"] if name.startswith("keep_count"))
    sample["answers"][qid]["score"] = (int(sample["answers"][qid]["score"]) + 1) % 5
    problems = judges.check_sample(sample, facts, LOC)
    assert len(problems) == 1
    assert problems[0].startswith(qid)


def test_any_locale_check_accepts_the_other_languages_option_text() -> None:
    """输入：英文渲染的样本、用中文口径判。输出：无。

    预期行为：判别器按「任一支持语言」认选项描述，所以英文产物不会被中文口径误判；
    反过来，只认一种语言的 `check_sample` 会当场报出来——审计必须用宽的那把尺。
    """
    spec = _make_spec(5, topics=0, flags=0)
    sample = sample_of(spec, "en")
    facts = facts_of(sample, "en")
    assert judges.check_sample_any_locale(sample, facts) == []
    problems = judges.check_sample(sample, facts, LOC)
    assert problems and "first_kept" in problems[0]


def test_roundtrip_diff_reports_a_tampered_segment() -> None:
    """输入：把某一段的高亮词改掉。输出：无。预期行为：报出那一段，不静默放过。"""
    spec = _make_spec(6, topics=2, flags=0)
    loc = locales.load("zh")
    state = render.build(spec, loc).replace("（高亮）", "（无高亮）", 1)
    parsed = parse.parse_state(state, parse.tables_of(loc))
    diff = judges.roundtrip_diff(spec, parsed, loc)
    assert diff and "第 1 段" in diff[0]


def test_roundtrip_diff_reports_a_wrong_segment_count() -> None:
    """输入：少了一段的状态。输出：无。预期行为：报「片段数」不对。"""
    spec = _make_spec(7)
    loc = locales.load("zh")
    lines = render.build(spec, loc).splitlines()
    parsed = parse.parse_state("\n".join(lines[:-1]) + "\n", parse.tables_of(loc))
    diff = judges.roundtrip_diff(spec, parsed, loc)
    assert diff and "片段数" in diff[0]


def test_roundtrip_judge_skips_without_a_spec() -> None:
    """输入：没有真值规格。输出：Verdict。预期行为：通过但标注跳过——不误判。"""
    sample = sample_of(_make_spec(8))
    verdict = judges.RoundtripJudge().judge(sample, None)
    assert verdict.ok is True
    assert verdict.raw["skipped"] is True


def test_roundtrip_judge_catches_a_tampered_state() -> None:
    """输入：被改过的 state + 真值规格。输出：Verdict。预期行为：判不通过并给出理由。"""
    spec = _make_spec(9, topics=3, flags=0)
    loc = locales.load("zh")
    sample = sample_of(spec)
    sample["state"] = str(sample["state"]).replace("（高亮）", "（无高亮）", 1)
    verdict = judges.RoundtripJudge(locale="zh").judge(sample, spec)
    assert verdict.ok is False
    assert "第 1 段" in verdict.reason
    assert loc["topic"][ft.TOPIC_IDS[0]] in render.build(spec, loc)


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_label_judge_passes_a_fresh_sample(lang: str) -> None:
    """输入：刚组装的样本（两种语言）。输出：Verdict。预期行为：判通过，零问题。"""
    sample = sample_of(_make_spec(11, topics=1, flags=2), lang)
    verdict = judges.LabelJudge(locale=lang).judge(sample)
    assert verdict.ok is True, verdict.reason
    assert verdict.raw["problems"] == []
