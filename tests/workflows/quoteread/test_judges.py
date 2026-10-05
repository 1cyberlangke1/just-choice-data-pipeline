"""判别器：标的名归一化、读数换算、重算四个答案、篡改能抓到、缺 spec 跳过、跨语言。"""

from __future__ import annotations

import random

import pytest

from jc import assemble
from jc.jsontypes import JsonValue
from tests.workflows.quoteread.conftest import _make_spec
from workflows.quoteread import features as ft
from workflows.quoteread import judges, locales, parse, questions, render
from workflows.quoteread import spec as sp

LOC = locales.load("zh")


def sample_of(spec: sp.QuoteSpec, locale: str = "zh", seed: int = 0) -> dict[str, JsonValue]:
    """输入：规格、语言、种子。输出：契约样本。

    预期行为：走真管线那两步（渲染 + 出题 + 组装），不手写样本——判别器要验的正是
    「组装之后还能不能由 state 重算出来」。
    """
    loc = locales.load(locale)
    state = render.build(spec, loc)
    labeled = questions.build(spec, loc, spec.phrasing, random.Random(seed))
    return assemble.assemble_sample("s-0000001", state, labeled, random.Random(seed))


def parsed_of(sample: dict[str, JsonValue], locale: str = "zh") -> parse.ParsedQuote:
    """输入：样本、语言。输出：反解结果。预期行为：用该语言的标签表反解。"""
    return parse.parse_state(sample["state"], parse.tables_of(locales.load(locale)))


def facts_of(sample: dict[str, JsonValue], locale: str = "zh") -> dict[str, JsonValue]:
    """输入：样本、语言。输出：从 state 重算出来的四个答案。"""
    return judges.facts_of(parsed_of(sample, locale), locales.words())


def test_parse_bp_converts_percent_into_basis_points() -> None:
    """输入：几种读数写法。输出：无。

    预期行为：百分比按 `1% = 100 基准点` 展开（两位小数）、基准点直接读整数；写法不合规
    当场报人话，不静默取近似值。
    """
    assert judges.parse_bp("-0.45", "pct") == -45
    assert judges.parse_bp("+0.08", "pct") == 8
    assert judges.parse_bp("1.50", "pct") == 150
    assert judges.parse_bp("0.00", "pct") == 0
    assert judges.parse_bp("-150", "bp") == -150
    assert judges.parse_bp("+200", "bp") == 200
    with pytest.raises(ValueError, match="百分比读数"):
        judges.parse_bp("1.5", "pct")
    with pytest.raises(ValueError, match="基准点读数"):
        judges.parse_bp("1.5", "bp")
    with pytest.raises(ValueError, match="未知单位键"):
        judges.parse_bp("10", "permille")


def test_split_value_prefers_the_longest_unit_suffix() -> None:
    """输入：一个单位写法互为前缀的假表。输出：无。

    预期行为：最长后缀优先——单位写法之间有前缀关系时才不会把 `bp` 认成 `b`；
    只有单位、没有数字的串不算读数。
    """
    units: dict[str, JsonValue] = {"b": "short", "bp": "bp"}
    assert judges.split_value("10bp", units) == ("10", "bp")
    assert judges.split_value("-1.50%", {"%": "pct"}) == ("-1.50", "pct")
    with pytest.raises(ValueError, match="认不出单位"):
        judges.split_value("bp", units)


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_to_readings_normalizes_both_languages(lang: str) -> None:
    """输入：两种语言渲染的样本。输出：无。

    预期行为：标的名与读数归一化之后与构造侧的序列**逐行**一致——只认一种语言的写法会把
    另一种语言的产物整片读成「表外写法」。
    """
    case = ft.QuoteCase(values=4, units=2, names=0)
    spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
    readings = judges.to_readings(parsed_of(sample_of(spec, lang), lang).rows, locales.words())
    assert tuple(name for name, _ in readings) == case.name_ids
    assert tuple(bp for _, bp in readings) == case.bp_values


def test_to_readings_reports_names_and_units_outside_the_table() -> None:
    """输入：表外的标的名 / 读数写法。输出：无。

    预期行为：报人话——渲染写出了表外的写法不能静默当成「读数为零」，那会把渲染 bug
    伪装成一道难题。
    """
    with pytest.raises(ValueError, match="标的名不在名字表里"):
        judges.to_readings((("乱写", "-1.50%"),), locales.words())
    with pytest.raises(ValueError, match="认不出单位"):
        judges.to_readings((("大盘指数", "乱写"),), locales.words())


def test_to_readings_reports_a_broken_lookup_table() -> None:
    """输入：形状不对的反查表。输出：无。预期行为：报形状，不抛 AttributeError。"""
    with pytest.raises(ValueError, match="反查表形状不对"):
        judges.to_readings((("大盘指数", "-1.50%"),), {"name": [], "unit": {}})


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_facts_of_recomputes_every_answer_for_every_case(lang: str) -> None:
    """输入：全空间 case × 两种语言。输出：无。

    预期行为：只看 state 重算出来的答案与真值规格完全一致——「生成器把答案算错了」与
    「渲染把某一行写歪了」是两条独立的路，这里是审计侧那条。
    """
    for case in ft.build_cases(512):
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        facts = facts_of(sample_of(spec, lang), lang)
        assert facts["tier"] == spec.tier, case.describe()
        assert facts["leader"] == spec.leader, case.describe()
        assert facts["alert"] is spec.alert, case.describe()
        assert facts["divergent"] is spec.divergent, case.describe()
        assert facts["names"] == list(case.name_ids), case.describe()
        assert facts["values"] == list(case.bp_values), case.describe()


def test_check_sample_catches_a_tampered_answer() -> None:
    """输入：把某道题的答案改错。输出：无。预期行为：那一题被报出来，别的题不受影响。"""
    sample = sample_of(_make_spec(3))
    facts = facts_of(sample)
    qid = next(name for name in sample["questions"] if name.startswith("tier"))
    sample["answers"][qid]["score"] = (int(sample["answers"][qid]["score"]) + 1) % len(
        ft.TIER_FLOORS
    )
    problems = judges.check_sample(sample, facts, LOC)
    assert len(problems) == 1
    assert problems[0].startswith(qid)


def test_any_locale_check_accepts_the_other_languages_option_text() -> None:
    """输入：英文渲染的样本、用中文口径判。输出：无。

    预期行为：判别器按「任一支持语言」认选项描述，所以英文产物不会被中文口径误判；
    反过来，只认一种语言的 `check_sample` 会当场报出来——审计必须用宽的那把尺。
    """
    sample = sample_of(_make_spec(5, values=1), "en")
    facts = facts_of(sample, "en")
    assert judges.check_sample_any_locale(sample, facts) == []
    problems = judges.check_sample(sample, facts, LOC)
    assert problems and "leader" in problems[0]


def test_roundtrip_diff_reports_a_tampered_row() -> None:
    """输入：把某一行的读数改掉。输出：无。预期行为：报出那一行，不静默放过。"""
    spec = _make_spec(6, values=0, units=0)
    loc = locales.load("zh")
    state = render.build(spec, loc).replace("-1.50%", "-1.51%", 1)
    parsed = parse.parse_state(state, parse.tables_of(loc))
    diff = judges.roundtrip_diff(spec, parsed, loc)
    assert diff and "第 1 行" in diff[0]


def test_roundtrip_diff_reports_a_wrong_row_count() -> None:
    """输入：少了一行的状态。输出：无。预期行为：报「读数行数」不对。"""
    spec = _make_spec(7)
    loc = locales.load("zh")
    lines = render.build(spec, loc).splitlines()
    parsed = parse.parse_state("\n".join(lines[:-1]) + "\n", parse.tables_of(loc))
    diff = judges.roundtrip_diff(spec, parsed, loc)
    assert diff and "读数行数" in diff[0]


def test_roundtrip_judge_catches_a_tampered_state() -> None:
    """输入：被改过的 state + 真值规格。输出：Verdict。预期行为：判不通过并给出理由。"""
    spec = _make_spec(9, values=0, units=0)
    sample = sample_of(spec)
    sample["state"] = str(sample["state"]).replace("+0.25%", "+0.26%", 1)
    verdict = judges.RoundtripJudge(locale="zh").judge(sample, spec)
    assert verdict.ok is False
    assert "第 3 行" in verdict.reason


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_label_judge_passes_a_fresh_sample(lang: str) -> None:
    """输入：刚组装的样本（两种语言）。输出：Verdict。预期行为：判通过，零问题。"""
    sample = sample_of(_make_spec(11, values=4, units=2), lang)
    verdict = judges.LabelJudge(locale=lang).judge(sample)
    assert verdict.ok is True, verdict.reason
    assert verdict.raw["problems"] == []
