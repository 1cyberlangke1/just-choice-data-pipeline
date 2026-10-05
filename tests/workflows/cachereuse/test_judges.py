"""判别器：写法归一化、重算四个答案、篡改能抓到、缺 spec 跳过、跨语言。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random

import pytest

from jc import assemble
from tests.workflows.cachereuse.conftest import _make_spec
from workflows.cachereuse import judges, locales, parse, questions, render
from workflows.cachereuse import spec as sp

LOC = locales.load("zh")


def sample_of(spec: sp.ReuseSpec, locale: str = "zh", seed: int = 0) -> dict[str, JsonValue]:
    """输入：规格、语言、种子。输出：契约样本。

    预期行为：走真管线那两步（渲染 + 出题 + 组装），不手写样本——判别器要验的正是
    「组装之后还能不能由 state 重算出来」。
    """
    loc = locales.load(locale)
    state = render.build(spec, loc)
    labeled = questions.build(spec, loc, spec.phrasing, random.Random(seed))
    return assemble.assemble_sample("c-0000001", state, labeled, random.Random(seed))


def parsed_of(sample: dict[str, JsonValue], locale: str = "zh") -> parse.ParsedReuse:
    """输入：样本、语言。输出：反解结果。预期行为：用该语言的标签表反解。"""
    return parse.parse_state(sample["state"], parse.tables_of(locales.load(locale)))


def facts_of(sample: dict[str, JsonValue], locale: str = "zh") -> dict[str, JsonValue]:
    """输入：样本、语言。输出：从 state 重算出来的事实（含四个答案）。"""
    return judges.facts_of(parsed_of(sample, locale), locales.words())


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_to_keys_normalizes_both_spellings_to_the_same_key(lang: str) -> None:
    """输入：同义写法渲染的样本。输出：无。

    预期行为：两条轨迹归一化之后**逐步**与构造侧的步骤键一致——等价步两侧写法不同但键相同，
    这正是「语义可复用」那一层；只比文本的人会在这里判错。
    """
    case = sp.ReuseCase(6, "half", "action", sp.SYNONYM, "web")
    spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
    parsed = parsed_of(sample_of(spec, lang), lang)
    words = locales.words()
    assert judges.to_keys(parsed.cached, words) == sp.traces_of(case)[0]
    assert judges.to_keys(parsed.current, words) == sp.traces_of(case)[1]
    for index in range(case.length):
        equivalent = index < case.prefix or case.prefix < index <= case.prefix + case.tail
        if equivalent:
            assert parsed.cached[index] != parsed.current[index], index


def test_to_keys_reports_words_outside_the_table() -> None:
    """输入：表外的动作 / 元素。输出：无。

    预期行为：报人话——渲染写出了表外的写法不能静默当成「不同」，那会把渲染 bug 伪装成难题。
    """
    words = locales.words()
    with pytest.raises(ValueError, match="动作不在写法表里"):
        judges.to_keys((("乱写", "搜索框"),), words)
    with pytest.raises(ValueError, match="元素不在写法表里"):
        judges.to_keys((("打开", "乱写"),), words)


def test_flavor_key_translates_and_reports_unknown() -> None:
    """输入：来源原文。输出：无。预期行为：中英两种写法都归到同一个键，表外的报人话。"""
    words = locales.words()
    assert judges.flavor_key(LOC["flavor"]["web"], words) == "web"
    assert judges.flavor_key(locales.load("en")["flavor"]["kiosk"], words) == "kiosk"
    with pytest.raises(ValueError, match="来源不在写法表里"):
        judges.flavor_key("平行宇宙版", words)


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_facts_of_recomputes_every_answer_for_every_case(lang: str) -> None:
    """输入：全空间 case × 两种语言。输出：无。

    预期行为：只看 state 重算出来的四个答案与真值规格完全一致——「生成器把答案算错了」与
    「渲染把某一步写歪了」这两类错会分别被抓到。
    """
    loc = locales.load(lang)
    for case in sp.build_cases(sp.ReuseConfig(split="train")):
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        sample = sample_of(spec, lang)
        facts = judges.facts_of(parse.parse_state(sample["state"], parse.tables_of(loc)), locales.words())
        assert facts["band"] == case.band, case.describe()
        assert facts["diverge"] == case.diverge_band(), case.describe()
        assert facts["diff_kind"] == case.diff_kind, case.describe()
        assert facts["reuse_half"] == case.reuse_half, case.describe()
        assert facts["prefix"] == case.prefix, case.describe()
        assert facts["equivalent_total"] == case.equivalent_total, case.describe()
        assert facts["flavor"] == case.flavor, case.describe()


def test_facts_of_reports_an_unreadable_step() -> None:
    """输入：把某一栏改成表外写法。输出：无。预期行为：报「不在写法表里」。"""
    sample = sample_of(_make_spec(1))
    noun = parsed_of(sample).cached[0][1]
    sample["state"] = sample["state"].replace(noun, "异次元框")
    with pytest.raises(ValueError, match="不在写法表里"):
        facts_of(sample)


def test_both_judges_pass_a_good_sample() -> None:
    """输入：新样本。输出：无。预期行为：结构判别器与答案判别器都通过。"""
    spec = _make_spec(2)
    sample = sample_of(spec)
    assert judges.RoundtripJudge().judge(sample, spec).ok
    assert judges.LabelJudge().judge(sample).ok


def test_roundtrip_judge_skips_without_a_spec() -> None:
    """输入：样本（不带真值规格）。输出：无。预期行为：跳过而不是误判。"""
    verdict = judges.RoundtripJudge().judge(sample_of(_make_spec(3)))
    assert verdict.ok and verdict.raw["skipped"] is True


def test_roundtrip_judge_catches_a_tampered_step() -> None:
    """输入：把缓存轨迹第 1 步的动作换成另一个动作。输出：无。预期行为：判不通过并点名第几步。"""
    spec = _make_spec(4)
    sample = sample_of(spec)
    cached, _ = sp.traces_of(spec.case)
    action = next(item for item in sp.ACTIONS if item != cached[0][0])
    sample["state"] = sample["state"].replace(
        render.step_text(LOC, cached[0][0], cached[0][1], 0),
        render.step_text(LOC, action, cached[0][1], 0),
        1,
    )
    verdict = judges.RoundtripJudge().judge(sample, spec)
    assert not verdict.ok and "缓存第 1 步" in verdict.reason


def test_roundtrip_judge_catches_a_wrong_flavor_line() -> None:
    """输入：把来源那一行改成另一档。输出：无。预期行为：判不通过并点名来源。"""
    spec = _make_spec(5)
    sample = sample_of(spec)
    sample["state"] = sample["state"].replace(LOC["flavor"]["web"], LOC["flavor"]["desktop"], 1)
    verdict = judges.RoundtripJudge().judge(sample, spec)
    assert not verdict.ok and "来源" in verdict.reason


def test_roundtrip_judge_catches_a_wrong_request_id() -> None:
    """输入：把请求号改掉。输出：无。预期行为：判不通过。"""
    spec = _make_spec(6)
    sample = sample_of(spec)
    sample["state"] = sample["state"].replace(spec.query_id, "C-000000", 1)
    verdict = judges.RoundtripJudge().judge(sample, spec)
    assert not verdict.ok and "请求号" in verdict.reason


def test_roundtrip_judge_catches_an_unparsable_state() -> None:
    """输入：state 认不出。输出：无。预期行为：判不通过（反解失败）。"""
    spec = _make_spec(7)
    sample = sample_of(spec)
    sample["state"] = "乱七八糟\n"
    verdict = judges.RoundtripJudge().judge(sample, spec)
    assert not verdict.ok and "反解失败" in verdict.reason


def test_roundtrip_judge_catches_a_truncated_trace() -> None:
    """输入：删掉当前轨迹最后一步。输出：无。预期行为：判不通过（两条步数不同）。"""
    spec = _make_spec(8)
    sample = sample_of(spec)
    sample["state"] = "\n".join(sample["state"].splitlines()[:-1]) + "\n"
    verdict = judges.RoundtripJudge().judge(sample, spec)
    assert not verdict.ok and "步数不同" in verdict.reason


def test_label_judge_catches_a_wrong_choice_index() -> None:
    """输入：把位置档的答案编号挪一格。输出：无。预期行为：判不通过。"""
    sample = sample_of(_make_spec(9))
    answer = sample["answers"]["diverge_at"]
    answer["choice"] = (answer["choice"] + 1) % len(sample["questions"]["diverge_at"]["criteria"])
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok and "diverge_at" in verdict.reason


def test_label_judge_catches_a_flipped_noul() -> None:
    """输入：把布尔答案取反。输出：无。预期行为：判不通过。"""
    sample = sample_of(_make_spec(10))
    sample["answers"]["reuse_half"]["noul"] = not sample["answers"]["reuse_half"]["noul"]
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok and "reuse_half" in verdict.reason


def test_label_judge_catches_a_wrong_score() -> None:
    """输入：把档位下标挪一格。输出：无。预期行为：判不通过。"""
    sample = sample_of(_make_spec(11))
    sample["answers"]["reuse_band"]["score"] = (sample["answers"]["reuse_band"]["score"] + 1) % 4
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok and "reuse_band" in verdict.reason


def test_check_sample_reports_every_problem() -> None:
    """输入：四个答案全改坏。输出：无。预期行为：四条问题全报出来（不是只报第一条）。"""
    sample = sample_of(_make_spec(14))
    sample["answers"]["diverge_at"]["choice"] = (
        sample["answers"]["diverge_at"]["choice"] + 1
    ) % len(sample["questions"]["diverge_at"]["criteria"])
    sample["answers"]["diff_kind"]["choice"] = (
        sample["answers"]["diff_kind"]["choice"] + 1
    ) % len(sample["questions"]["diff_kind"]["criteria"])
    sample["answers"]["reuse_half"]["noul"] = not sample["answers"]["reuse_half"]["noul"]
    sample["answers"]["reuse_band"]["score"] = (sample["answers"]["reuse_band"]["score"] + 1) % 4
    problems = judges.check_sample(sample, facts_of(sample), LOC)
    assert len(problems) == len(questions.FAMILIES)


def test_check_sample_any_locale_accepts_a_rotated_language() -> None:
    """输入：英文渲染的样本，分别按单一语言与「任一语言」两把尺比。输出：无。

    预期行为：产物按 `locales[seq % len]` 轮转渲染，审计拿到的语言未必是产物那一种；
    只认一种语言的文案会把另一种语言的产物整片误判成「答案指向 …」。
    """
    sample = sample_of(_make_spec(16), locale="en")
    parsed = parse.parse_state(sample["state"], parse.merged_tables())
    facts = judges.facts_of(parsed, locales.words())
    assert judges.check_sample_any_locale(sample, facts) == []
    assert judges.check_sample(sample, facts, LOC), "按 zh 的文案比应当报出来（英文选项对不上）"


def test_option_text_uses_the_locale_templates() -> None:
    """输入：两个 choice 题族。输出：无。预期行为：金答案的描述原文来自语言文件。"""
    facts = facts_of(sample_of(_make_spec(18)))
    assert judges.option_text("diverge_at", facts, LOC) == (
        LOC["fam"]["diverge_at"]["criteria"][str(facts["diverge"])]
    )
    assert judges.option_text("diff_kind", facts, LOC) == (
        LOC["fam"]["diff_kind"]["criteria"][str(facts["diff_kind"])]
    )


def test_expected_variants_cover_each_language_for_choices() -> None:
    """输入：一条事实。输出：无。

    预期行为：choice 两族的可接受答案是**各语言各写一份**的选项描述，score / noul 与语言无关
    只取一份；未知题族直接报错（与 `expected_of` 同一口径）。
    """
    facts = facts_of(sample_of(_make_spec(19)))
    for family in questions.CHOICE_FAMILIES:
        variants = judges.expected_variants(family, facts)
        assert len(variants) == len(locales.available())
        assert len(set(variants)) == len(variants)
    assert judges.expected_variants("reuse_half", facts) == (facts["reuse_half"],)
    assert judges.expected_variants("reuse_band", facts) == (
        sp.BANDS.index(str(facts["band"])),
    )
    with pytest.raises(ValueError, match="未知题族"):
        judges.expected_variants("nope", facts)


def test_english_samples_pass_both_judges() -> None:
    """输入：英文渲染的样本。输出：无。预期行为：两个判别器都通过（语言不参与判据）。"""
    spec = _make_spec(20, spelling=sp.SYNONYM, flavor="desktop")
    sample = sample_of(spec, locale="en")
    assert judges.RoundtripJudge(locale="en").judge(sample, spec).ok
    assert judges.LabelJudge(locale="en").judge(sample).ok
