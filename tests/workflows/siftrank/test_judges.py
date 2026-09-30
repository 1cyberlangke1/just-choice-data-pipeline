"""判别器：条目名归一化、重算四个答案、篡改能抓到、缺 spec 跳过、跨语言。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random

import pytest

from jc import assemble
from tests.workflows.siftrank.conftest import _make_spec
from workflows.siftrank import judges, locales, parse, questions, render
from workflows.siftrank import spec as sp

LOC = locales.load("zh")


def sample_of(spec: sp.RankSpec, locale: str = "zh", seed: int = 0) -> dict[str, JsonValue]:
    """输入：规格、语言、种子。输出：契约样本。

    预期行为：走真管线那两步（渲染 + 出题 + 组装），不手写样本——判别器要验的正是
    「组装之后还能不能由 state 重算出来」。
    """
    loc = locales.load(locale)
    state = render.build(spec, loc)
    labeled = questions.build(spec, loc, spec.phrasing, random.Random(seed))
    return assemble.assemble_sample("s-0000001", state, labeled, random.Random(seed))


def parsed_of(sample: dict[str, JsonValue], locale: str = "zh") -> parse.ParsedRank:
    """输入：样本、语言。输出：反解结果。预期行为：用该语言的标签表反解。"""
    return parse.parse_state(sample["state"], parse.tables_of(locales.load(locale)))


def facts_of(
    sample: dict[str, JsonValue], locale: str = "zh"
) -> tuple[dict[str, JsonValue], tuple[str, str]]:
    """输入：样本、语言。输出：(从 state 重算出来的事实, 被比较的两个条目键)。"""
    words = locales.words()
    pair = judges.pair_of(sample, words)
    return judges.facts_of(parsed_of(sample, locale), words, pair), pair


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_to_keys_normalizes_both_languages(lang: str) -> None:
    """输入：两种语言渲染的样本。输出：无。

    预期行为：条目名归一化之后与构造侧的批次**逐批逐条**一致——只认一种语言的名字会把
    另一种语言的产物整片读成「表外名字」。
    """
    case = sp.RankCase("swirl", "rotate2", "launch")
    spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
    parsed = parsed_of(sample_of(spec, lang), lang)
    batches = tuple(judges.to_keys(batch, locales.words()) for batch in parsed.batches)
    assert batches == case.batches()


def test_to_keys_reports_names_outside_the_table() -> None:
    """输入：表外的条目名。输出：无。

    预期行为：报人话——渲染写出了表外的名字不能静默当成「另一个条目」，那会把渲染 bug
    伪装成一道难题。
    """
    with pytest.raises(ValueError, match="条目名不在写法表里"):
        judges.to_keys(("乱写",), locales.words())


def test_to_keys_reports_a_broken_lookup_table() -> None:
    """输入：形状不对的反查表。输出：无。预期行为：报形状，不抛 AttributeError。"""
    with pytest.raises(ValueError, match="反查表形状不对"):
        judges.to_keys(("概览说明",), {"item": []})


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_pair_of_reads_the_two_items_from_the_options(lang: str) -> None:
    """输入：两种语言的样本。输出：无。

    预期行为：从 `a_first` / `b_first` 两条描述里认出被比较的两个条目，且与真值规格一致。
    """
    spec = _make_spec(1, shape="swirl", perm="reverse")
    sample = sample_of(spec, lang)
    assert judges.pair_of(sample, locales.words()) == spec.pair


def test_pair_of_rejects_a_broken_option_set() -> None:
    """输入：选项描述里认不出条目名的样本。输出：无。预期行为：报人话，不猜。"""
    spec = _make_spec(2)
    sample = sample_of(spec)
    for qid, question in sample["questions"].items():
        if qid.startswith("pair_order"):
            question["criteria"] = {"a_first": "甲在前", "b_first": "乙在前", "tie": "并列"}
    with pytest.raises(ValueError, match="认出 0 个条目名"):
        judges.pair_of(sample, locales.words())


def test_pair_of_reports_a_sample_without_that_family() -> None:
    """输入：没有 `pair_order` 那一族的样本。输出：无。预期行为：报「认不出」，不返回空。"""
    spec = _make_spec(3)
    sample = sample_of(spec)
    sample["questions"] = {"agg_top": sample["questions"]["agg_top"]}
    sample["answers"] = {"agg_top": sample["answers"]["agg_top"]}
    with pytest.raises(ValueError, match="没有 pair_order"):
        judges.pair_of(sample, locales.words())


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_facts_of_recomputes_every_answer_for_every_case(lang: str) -> None:
    """输入：全空间 case × 两种语言。输出：无。

    预期行为：只看 state 重算出来的答案与真值规格完全一致——「生成器把答案算错了」与
    「渲染把某个批次写歪了」这两类错会分别被抓到。
    """
    for case in sp.build_cases(sp.RankConfig(split="train")):
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        facts, pair = facts_of(sample_of(spec, lang), lang)
        assert pair == spec.pair, case.describe()
        assert facts["cut"] == case.cut(), case.describe()
        assert facts["stable"] == case.stable(), case.describe()
        assert facts["top"] == case.top(), case.describe()
        assert facts["relation"] == case.relation(), case.describe()
        assert facts["totals"] == case.totals(), case.describe()


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_check_sample_accepts_a_clean_sample(lang: str) -> None:
    """输入：一条干净样本。输出：无。预期行为：两种口径（本语言 / 任一语言）都零问题。"""
    spec = _make_spec(4, shape="chain")
    sample = sample_of(spec, lang)
    facts, pair = facts_of(sample, lang)
    assert judges.check_sample(sample, facts, locales.load(lang), pair) == []
    assert judges.check_sample_any_locale(sample, facts, pair) == []


def test_check_sample_catches_a_wrong_answer() -> None:
    """输入：被改坏的答案。输出：无。预期行为：逐题报出「答案指向 …，按 state 重算是 …」。"""
    spec = _make_spec(5, shape="twin")
    sample = sample_of(spec)
    sample["answers"]["cut_size"]["score"] = (sample["answers"]["cut_size"]["score"] + 1) % len(
        sp.CUTS
    )
    facts, pair = facts_of(sample)
    problems = judges.check_sample(sample, facts, LOC, pair)
    assert problems and "cut_size" in problems[0]


def test_check_sample_reports_an_out_of_range_index() -> None:
    """输入：编号越界的 choice 答案。输出：无。预期行为：报「答案编号越界」。"""
    spec = _make_spec(6)
    sample = sample_of(spec)
    sample["answers"]["agg_top"]["choice"] = 99
    facts, pair = facts_of(sample)
    assert any("编号越界" in item for item in judges.check_sample(sample, facts, LOC, pair))


def test_option_text_rejects_an_unknown_family() -> None:
    """输入：不存在的题族。输出：无。预期行为：报错（不然会静默返回另一族的描述）。"""
    spec = _make_spec(7)
    facts, pair = facts_of(sample_of(spec))
    with pytest.raises(ValueError, match="未知题族"):
        judges.option_text("mystery", facts, LOC, pair)
    with pytest.raises(ValueError, match="未知题族"):
        judges.expected_of("mystery", facts, LOC, pair)


def test_roundtrip_judge_passes_and_catches_a_tampered_batch() -> None:
    """输入：干净样本、把某个批次换掉的样本。输出：无。

    预期行为：干净样本通过；批次被动过之后结构判别器必须判不通过（哪怕答案仍然自洽——
    结构判别器比的是「渲染有没有兑现计划」，不是「答案对不对」）。
    """
    spec = _make_spec(8, shape="stair")
    judge = judges.RoundtripJudge(locale="zh")
    sample = sample_of(spec)
    assert judge.judge(sample, spec).ok
    lines = str(sample["state"]).splitlines()
    lines[-1] = "批次 4：概览说明 > 操作手册 > 变更记录 > 监控指标"
    sample["state"] = "\n".join(lines) + "\n"
    verdict = judge.judge(sample, spec)
    assert not verdict.ok and "批次" in verdict.reason


def test_roundtrip_judge_skips_without_spec() -> None:
    """输入：没有 spec 的样本。输出：无。预期行为：跳过而不是误判（`skipped` 记一笔）。"""
    sample = sample_of(_make_spec(9))
    verdict = judges.RoundtripJudge().judge(sample, None)
    assert verdict.ok and verdict.raw.get("skipped") is True


def test_label_judge_passes_and_catches_a_tampered_answer() -> None:
    """输入：干净样本、被改坏的样本。输出：无。预期行为：答案判别器分别通过 / 判不通过。"""
    spec = _make_spec(10, shape="peak")
    judge = judges.LabelJudge(locale="zh")
    sample = sample_of(spec)
    assert judge.judge(sample, spec).ok
    sample["answers"]["stable"]["noul"] = not sample["answers"]["stable"]["noul"]
    verdict = judge.judge(sample, spec)
    assert not verdict.ok and "stable" in verdict.reason


def test_label_judge_reports_an_unparsable_state() -> None:
    """输入：state 认不出来的样本。输出：无。预期行为：报「重算失败」，不抛异常。"""
    sample = sample_of(_make_spec(11))
    sample["state"] = "乱七八糟\n"
    verdict = judges.LabelJudge().judge(sample, None)
    assert not verdict.ok and "重算失败" in verdict.reason
