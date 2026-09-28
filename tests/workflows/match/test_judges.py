"""判别器：归一化、三种关系、重算、篡改能抓到、缺 spec 跳过。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random


import pytest

from jc import assemble
from tests.workflows.match.conftest import _make_spec
from workflows.match import judges, locales, parse, questions, render
from workflows.match import spec as sp

LOC = locales.load("zh")


def sample_of(spec: sp.MatchSpec, locale: str = "zh", seed: int = 0) -> dict[str, JsonValue]:
    """输入：规格、语言、种子。输出：契约样本。

    预期行为：走真管线那两步（渲染 + 出题 + 组装），不手写样本——判别器要验的正是
    「组装之后还能不能由 state 重算出来」。
    """
    loc = locales.load(locale)
    state = render.build(spec, loc)
    labeled = questions.build(spec, loc, spec.phrasing, random.Random(seed))
    return assemble.assemble_sample("m-0000001", state, labeled, random.Random(seed))


def facts_of(sample: dict[str, JsonValue], locale: str = "zh") -> judges.Facts:
    """输入：样本、语言。输出：从 state 重算出来的事实。"""
    loc = locales.load(locale)
    return judges.recompute(parse.parse_state(sample["state"], parse.tables_of(loc)), loc)


def test_normalize_name_ignores_case_and_repeated_spaces() -> None:
    """输入：大小写 / 空格不同的名称。输出：无。预期行为：归一化后相等。"""
    assert judges.normalize("name", "River Lager", LOC) == judges.normalize("name", "river  lager", LOC)
    assert judges.normalize("brand", "Riverside  Brewery", LOC) == "riverside brewery"


def test_normalize_size_converts_to_ml() -> None:
    """输入：三种写法。输出：无。预期行为：都换算成 ml 的数值（1 L = 1000 ml）。"""
    assert judges.normalize("size", "750 ml", LOC) == judges.normalize("size", "0.75 L", LOC)
    assert judges.normalize("size", "0.75 L", LOC) == judges.normalize("size", "750ml", LOC)
    assert judges.normalize("size", "1 L", LOC) == judges.normalize("size", "1000 ml", LOC)
    assert judges.normalize("size", "1 L", LOC) != judges.normalize("size", "750 ml", LOC)


def test_normalize_abv_keeps_the_number() -> None:
    """输入：三种写法。输出：无。预期行为：`度` 与 `% vol` 同义，只比数值。"""
    assert judges.normalize("abv", "5.0% vol", LOC) == judges.normalize("abv", "5.0 度", LOC)
    assert judges.normalize("abv", "5.0 度", LOC) == judges.normalize("abv", "5%vol", LOC)
    assert judges.normalize("abv", "0.0% vol", LOC) != judges.normalize("abv", "4.5% vol", LOC)


def test_normalize_pack_uses_the_synonym_table() -> None:
    """输入：同义词表里的三种写法。输出：无。预期行为：都归到同一个包装键。"""
    assert judges.normalize("pack", "玻璃瓶", LOC) == "glass"
    assert judges.normalize("pack", "Glass", LOC) == "glass"
    assert judges.normalize("pack", "罐", LOC) == "can"


def test_normalize_reports_unreadable_text() -> None:
    """输入：表外的写法。输出：无。预期行为：报人话（渲染写出了表外的东西）。"""
    with pytest.raises(ValueError, match="规格读不出"):
        judges.normalize("size", "一打", LOC)
    with pytest.raises(ValueError, match="度数读不出"):
        judges.normalize("abv", "很烈", LOC)
    with pytest.raises(ValueError, match="包装不在同义词表"):
        judges.normalize("pack", "布袋", LOC)


def test_kinds_of_classifies_the_three_relations() -> None:
    """输入：三对文本。输出：无。

    预期行为：逐字相同 → same；写法不同但归一化相同 → variant；归一化也不同 → diff。
    判定顺序反了会把 same 读成 variant，所以这一条同时钉住了顺序。
    """
    left = ("River Lager", "Riverside Brewery", "750 ml", "5.0% vol", "玻璃瓶")
    right = ("River Lager", "Riverside  Brewery", "0.75 L", "5.0 度", "Can")
    assert judges.kinds_of(left, right, LOC) == (sp.SAME, sp.VARIANT, sp.VARIANT, sp.VARIANT, sp.DIFF)


def test_recompute_matches_the_spec_for_every_case() -> None:
    """输入：全部 1728 个 case × 两种语言。输出：无。

    预期行为：只看 state 重算出来的关系与四个派生量，与真值规格完全一致——「生成器把答案
    算错了」与「渲染把某一栏写歪了」这两类错会分别被抓到。
    """
    for lang in ("zh", "en"):
        loc = locales.load(lang)
        for case in sp.build_cases(sp.MatchConfig()):
            spec = sp.sample_spec(None, case, "train", 0, 0)
            parsed = parse.parse_state(render.build(spec, loc), parse.tables_of(loc))
            facts = judges.recompute(parsed, loc)
            assert facts.kinds == case.pattern, case.describe()
            assert facts.agree == spec.agree
            assert facts.band == spec.band
            assert facts.verdict == spec.verdict
            assert facts.variant_count == spec.variant_count


def test_facts_same_reports_each_field() -> None:
    """输入：一份事实。输出：无。预期行为：`same(栏)` 与那一栏的关系一致。"""
    spec = _make_spec(3, pattern=sp.PATTERNS[8])
    facts = facts_of(sample_of(spec))
    for field in sp.FIELDS:
        assert facts.same(field) == (facts.kinds[sp.FIELDS.index(field)] != sp.DIFF)


def test_label_judge_passes_on_fresh_samples() -> None:
    """输入：新样本。输出：无。预期行为：八个答案全对得上。"""
    verdict = judges.LabelJudge().judge(sample_of(_make_spec(4)))
    assert verdict.ok, verdict.reason


def test_label_judge_catches_a_tampered_choice_answer() -> None:
    """输入：把某个 choice 的编号改掉。输出：无。预期行为：判不通过，并指出是哪一题。"""
    sample = sample_of(_make_spec(5))
    answer = sample["answers"]["verdict"]
    answer["choice"] = (answer["choice"] + 1) % 2
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok and "verdict" in verdict.reason


def test_label_judge_catches_a_tampered_noul_answer() -> None:
    """输入：把某个 noul 的布尔翻掉。输出：无。预期行为：判不通过。"""
    sample = sample_of(_make_spec(6))
    answer = sample["answers"]["same_name"]
    answer["noul"] = not answer["noul"]
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok and "same_name" in verdict.reason


def test_label_judge_catches_a_tampered_score() -> None:
    """输入：把档位改掉。输出：无。预期行为：判不通过。"""
    sample = sample_of(_make_spec(7))
    sample["answers"]["band"]["score"] = (sample["answers"]["band"]["score"] + 1) % 3
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok and "band" in verdict.reason


def test_label_judge_catches_an_out_of_range_index() -> None:
    """输入：把编号改成越界值。输出：无。预期行为：判不通过并点名越界。"""
    sample = sample_of(_make_spec(8))
    sample["answers"]["variant_count"]["choice"] = 99
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok and "越界" in verdict.reason


def test_label_judge_catches_an_unknown_question_type() -> None:
    """输入：把某题的题型改掉。输出：无。预期行为：判不通过。"""
    sample = sample_of(_make_spec(9))
    sample["questions"]["band"]["type"] = "multi"
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok and "未知题型" in verdict.reason


def test_label_judge_reports_a_broken_state() -> None:
    """输入：state 里出现表外的写法。输出：无。预期行为：判不通过，理由是重算失败。"""
    sample = sample_of(_make_spec(10))
    sample["state"] = sample["state"].replace("750 ml", "一打")
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok and "重算失败" in verdict.reason


def test_label_judge_catches_an_unparsable_state() -> None:
    """输入：state 缺了抬头。输出：无。预期行为：判不通过（反解失败）。"""
    sample = sample_of(_make_spec(11))
    sample["state"] = "\n".join(sample["state"].splitlines()[1:])
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok and "重算失败" in verdict.reason


def test_roundtrip_judge_passes_and_skips_without_spec() -> None:
    """输入：新样本（带 / 不带真值规格）。输出：无。预期行为：带规格时通过，不带时跳过。"""
    spec = _make_spec(12, extra="origin")
    sample = sample_of(spec)
    assert judges.RoundtripJudge().judge(sample, spec).ok
    skipped = judges.RoundtripJudge().judge(sample)
    assert skipped.ok and skipped.raw["skipped"] is True


def test_roundtrip_judge_catches_a_tampered_cell() -> None:
    """输入：把某一栏的原文改掉。输出：无。预期行为：判不通过并点名是哪一格。"""
    spec = _make_spec(13)
    sample = sample_of(spec)
    sample["state"] = sample["state"].replace("River Lager", "River  Lager", 1)
    verdict = judges.RoundtripJudge().judge(sample, spec)
    assert not verdict.ok and "A/name" in verdict.reason


def test_roundtrip_judge_catches_a_missing_extra_line() -> None:
    """输入：删掉额外栏那一行。输出：无。预期行为：判不通过（该写的一栏没写）。"""
    spec = _make_spec(14, extra="origin")
    sample = sample_of(spec)
    sample["state"] = "\n".join(
        line for line in sample["state"].splitlines() if LOC["extra"]["origin_label"] not in line
    )
    verdict = judges.RoundtripJudge().judge(sample, spec)
    assert not verdict.ok and "产地" in verdict.reason


def test_roundtrip_judge_catches_an_unexpected_extra_line() -> None:
    """输入：给不该有额外栏的样本塞一行产地。输出：无。预期行为：判不通过。"""
    spec = _make_spec(15)
    sample = sample_of(spec)
    extra_line = f"  {LOC['extra']['origin_label']}{LOC['meta']['sep']}{LOC['extra']['origin_left']}"
    sample["state"] = sample["state"] + extra_line + "\n"
    verdict = judges.RoundtripJudge().judge(sample, spec)
    assert not verdict.ok and "不该有额外栏" in verdict.reason


def test_roundtrip_judge_catches_an_unparsable_state() -> None:
    """输入：state 认不出。输出：无。预期行为：判不通过（反解失败）。"""
    sample = sample_of(_make_spec(16))
    sample["state"] = "乱七八糟\n"
    verdict = judges.RoundtripJudge().judge(sample, _make_spec(16))
    assert not verdict.ok and "反解失败" in verdict.reason


def test_roundtrip_judge_catches_a_wrong_request_id() -> None:
    """输入：把请求号改掉。输出：无。预期行为：判不通过。"""
    spec = _make_spec(17)
    sample = sample_of(spec)
    sample["state"] = sample["state"].replace(spec.query_id, "M-00000")
    verdict = judges.RoundtripJudge().judge(sample, spec)
    assert not verdict.ok and "请求号" in verdict.reason


def test_english_samples_pass_both_judges() -> None:
    """输入：英文渲染的样本。输出：无。预期行为：两个判别器都通过（语言不参与判据）。"""
    spec = _make_spec(18, extra="origin")
    sample = sample_of(spec, locale="en")
    assert judges.RoundtripJudge(locale="en").judge(sample, spec).ok
    assert judges.LabelJudge(locale="en").judge(sample).ok


def test_expected_of_rejects_unknown_family() -> None:
    """输入：不存在的题族。输出：无。预期行为：报错（新加题族忘了实现不能静默放过）。"""
    with pytest.raises(ValueError, match="未知题族"):
        judges.expected_of("nope", facts_of(sample_of(_make_spec(19))), LOC)


def test_option_text_uses_the_locale_templates() -> None:
    """输入：两个 choice 题族。输出：无。预期行为：金答案的描述原文来自语言文件。"""
    facts = facts_of(sample_of(_make_spec(20, pattern=sp.PATTERNS[8])))
    assert judges.option_text("verdict", facts, LOC) == LOC["fam"]["verdict"]["criteria"][facts.verdict]
    wanted = LOC["fam"]["variant_count"]["criteria"][f"v{facts.variant_count}"]
    assert judges.option_text("variant_count", facts, LOC) == wanted


def test_check_sample_reports_every_problem() -> None:
    """输入：八个答案全改坏。输出：无。预期行为：八条问题全报出来（不是只报第一条）。"""
    sample = sample_of(_make_spec(21))
    for answer in sample["answers"].values():
        if answer["type"] == "choice":
            answer["choice"] = (answer["choice"] + 1) % 2
        elif answer["type"] == "noul":
            answer["noul"] = not answer["noul"]
        else:
            answer["score"] = (answer["score"] + 1) % 3
    problems = judges.check_sample(sample, facts_of(sample), LOC)
    assert len(problems) == len(questions.FAMILIES)


def test_check_sample_is_silent_on_a_good_sample() -> None:
    """输入：新样本。输出：无。预期行为：零问题（审计与判别器共用这一把尺）。"""
    sample = sample_of(_make_spec(22))
    assert judges.check_sample(sample, facts_of(sample), LOC) == []
