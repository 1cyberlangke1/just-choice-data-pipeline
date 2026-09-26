"""判别器：结构往返与答案重算（只看 state 与题面）。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from jc import assemble
from workflows.rerank import judges, locales, parse, questions, render
from workflows.rerank import spec as sp


def sample_of(spec: sp.RerankSpec, lang: str = "zh") -> dict[str, Any]:
    """输入：规格、语言。输出：一条样本（选项顺序与答案编号由真实组装器定）。

    预期行为：必须走 `assemble.assemble_sample`——判别器读的是**组装之后**的样本，
    那时 `criteria` 已经是「选项名 → 描述」的对象、`choice` 是描述在其中的书写顺序；
    手工拼样本会把这个形状拼错，判别器也就测不到真实路径。
    """
    loc = locales.load(lang)
    labeled = questions.build(spec, loc, spec.phrasing, random.Random(0))
    return assemble.assemble_sample(
        "rerank-x", render.build(spec, loc), labeled, random.Random(0)
    )


def spec_of(shape: sp.Shape, **overrides: Any) -> sp.RerankSpec:
    """输入：画像与 case 覆盖项。输出：一条真值规格。"""
    fields: dict[str, Any] = {
        "topic": "backup",
        "aspect": "steps",
        "shape": shape,
        "traps": (),
        "wording": "plain",
    }
    fields.update(overrides)
    return sp.sample_spec(random.Random(0), sp.RerankCase(**fields), "train", 0, 0)


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_roundtrip_judge_passes(lang: str, make_spec: Any) -> None:
    """输入：一条真样本（带全部噪声句与标题式措辞）。输出：无。预期行为：结构判别通过。"""
    spec = make_spec(traps=("forwarded", "saved", "edited"), wording="titled")
    verdict = judges.RoundtripJudge(locale=lang).judge(sample_of(spec, lang), spec)
    assert verdict.ok, verdict.reason


def test_roundtrip_judge_catches_a_tampered_day(make_spec: Any) -> None:
    """输入：把某条候选的更新日期改掉。输出：无。预期行为：判不通过并点名那一格。"""
    spec = make_spec()
    sample = sample_of(spec)
    day = spec.case.shape.cands[1].day
    sample["state"] = sample["state"].replace(day, "2020-01-01", 1)
    verdict = judges.RoundtripJudge().judge(sample, spec)
    assert not verdict.ok
    assert any("更新" in item for item in verdict.raw["diff"])


def test_roundtrip_judge_catches_an_extra_body_line(make_spec: Any) -> None:
    """输入：往 C1 正文里插一行谁也不认识的句子。输出：无。预期行为：点名 C1 正文。"""
    spec = make_spec()
    sample = sample_of(spec)
    lines = sample["state"].splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("C1"))
    lines.insert(start + 1, "   （多出来的一行正文）")
    sample["state"] = "\n".join(lines)
    verdict = judges.RoundtripJudge().judge(sample, spec)
    assert not verdict.ok
    assert any("C1 正文" in item for item in verdict.raw["diff"])


def test_roundtrip_judge_skips_without_spec(make_spec: Any) -> None:
    """输入：不给真值规格。输出：无。预期行为：跳过而不是误判。"""
    verdict = judges.RoundtripJudge().judge(sample_of(make_spec()), None)
    assert verdict.ok and verdict.raw.get("skipped") is True


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_label_judge_passes(lang: str, make_spec: Any) -> None:
    """输入：一条真样本。输出：无。预期行为：重算出来的答案与样本一致。"""
    verdict = judges.LabelJudge(locale=lang).judge(sample_of(make_spec(), lang))
    assert verdict.ok, verdict.reason


def test_label_judge_passes_for_every_shape() -> None:
    """输入：十个画像各一条样本。输出：无。

    预期行为：全部通过——`choice` 的答案编号指的是**选项描述的书写顺序**，
    判别器若拿 `criteria` 的 key 当描述比（原始实现就是这样）会十条全红。
    """
    for shape in sp.SHAPES:
        verdict = judges.LabelJudge().judge(sample_of(spec_of(shape)))
        assert verdict.ok, f"{shape.key()}: {verdict.reason}"


def test_label_judge_catches_a_flipped_answer(make_spec: Any) -> None:
    """输入：把一条 noul 答案翻过来。输出：无。预期行为：判不通过并点名题号。"""
    sample = sample_of(make_spec())
    sample["answers"]["ahead"]["noul"] = not sample["answers"]["ahead"]["noul"]
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok
    assert any(problem.startswith("ahead:") for problem in verdict.raw["problems"])


def test_label_judge_catches_a_wrong_choice_index(make_spec: Any) -> None:
    """输入：把 `first` 的答案编号挪一位。输出：无。预期行为：判不通过并点名题号。"""
    sample = sample_of(make_spec())
    answer = sample["answers"]["first"]
    size = len(sample["questions"]["first"]["criteria"])
    answer["choice"] = (answer["choice"] + 1) % size
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok
    assert any(problem.startswith("first:") for problem in verdict.raw["problems"])


def test_label_judge_catches_a_wrong_score(make_spec: Any) -> None:
    """输入：把 `relevance` 的档位改错。输出：无。预期行为：判不通过并点名题号。"""
    sample = sample_of(make_spec())
    sample["answers"]["relevance"]["score"] = (
        sample["answers"]["relevance"]["score"] + 1
    ) % 5
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok
    assert any(problem.startswith("relevance:") for problem in verdict.raw["problems"])


def test_label_judge_catches_an_unparseable_state(make_spec: Any) -> None:
    """输入：state 只剩抬头一行。输出：无。预期行为：判不通过并说明重算失败。"""
    sample = sample_of(make_spec())
    sample["state"] = sample["state"].splitlines()[0]
    verdict = judges.LabelJudge().judge(sample)
    assert not verdict.ok
    assert "重算失败" in verdict.reason


def test_recompute_reproduces_the_spec(make_spec: Any) -> None:
    """输入：一条真值规格。输出：无。预期行为：重算出来的名次与相关度逐格等于规格。"""
    spec = make_spec(topic="cache", aspect="config", traps=("saved",), wording="terse")
    loc = locales.load("zh")
    parsed = parse.parse_list(render.build(spec, loc), parse.label_index(loc))
    facts = judges.recompute(parsed, loc)
    assert facts.cids == render.CAND_IDS
    assert facts.order == spec.order
    assert facts.relevance == spec.relevance


@pytest.mark.parametrize("wording", sp.WORDINGS)
def test_distractors_do_not_change_the_answers(wording: str, make_spec: Any) -> None:
    """输入：只改噪声句与措辞的规格。输出：无。

    预期行为：重算出来的名次与相关度与默认规格完全一样——这两维只改文本、不改判据。
    判别器若被它们带偏（把噪声句当判据），这一条会红。
    """
    loc = locales.load("zh")
    base = judges.recompute(
        parse.parse_list(render.build(make_spec(), loc), parse.label_index(loc)), loc
    )
    other = judges.recompute(
        parse.parse_list(
            render.build(make_spec(traps=("forwarded", "saved", "edited"), wording=wording), loc),
            parse.label_index(loc),
        ),
        loc,
    )
    assert other.order == base.order
    assert other.relevance == base.relevance


def test_body_regex_handles_the_terse_template() -> None:
    """输入：`terse` 那种「主题 / 方面」写法。输出：无。

    预期行为：前一个占位符懒惰、后一个贪婪——两个都懒惰时后一个只吃一个字符，
    两个都贪婪时前一个会把分隔符一起吃进去。
    """
    loc = locales.load("zh")
    template = loc["sentence"]["body"]["terse"]
    text = template.format(topic="缓存策略", aspect="常见故障")
    found = judges.body_regex(template).fullmatch(text)
    assert found is not None
    assert found.group("topic") == "缓存策略"
    assert found.group("aspect") == "常见故障"


def test_read_cand_rejects_an_unknown_line(make_spec: Any) -> None:
    """输入：正文里塞一句谁也不认识的句子。输出：无。

    预期行为：报错——那说明渲染漏了模板或语言文件缺句，是硬伤，不能当噪声句蒙过去。
    """
    loc = locales.load("zh")
    spec = make_spec()
    body = render.cand_lines(spec.case, 0, loc) + ("这句谁也没写过。",)
    with pytest.raises(ValueError, match="认不出的行"):
        judges.read_cand(body, loc)


def test_read_cand_needs_an_example_line(make_spec: Any) -> None:
    """输入：正文里删掉示例句。输出：无。预期行为：报错点名读不出「有没有示例」。"""
    loc = locales.load("zh")
    spec = make_spec()
    examples = set(loc["sentence"]["example"]["plain"].values())
    body = tuple(line for line in render.cand_lines(spec.case, 0, loc) if line not in examples)
    with pytest.raises(ValueError, match="有没有示例"):
        judges.read_cand(body, loc)


def test_read_cand_rejects_a_body_without_a_topic_line() -> None:
    """输入：正文里只有示例句。输出：无。预期行为：报错点名读不出主题与方面。"""
    loc = locales.load("zh")
    body = (loc["sentence"]["example"]["plain"]["yes"],)
    with pytest.raises(ValueError, match="主题与方面"):
        judges.read_cand(body, loc)


def test_named_cands_reads_the_ids_in_order() -> None:
    """输入：题面。输出：无。预期行为：按出现顺序读出编号，不把 `C1` 认成 `C10` 的一部分。"""
    assert judges.named_cands("候选 C3 是否排在候选 C1 前面？") == ["C3", "C1"]
    assert judges.named_cands("C10 与 C1") == ["C1"]


def test_expected_of_reports_an_unreadable_position(make_spec: Any) -> None:
    """输入：读不出位次的题面。输出：无。预期行为：报人话而不是猜一个位次。"""
    loc = locales.load("zh")
    spec = make_spec()
    parsed = parse.parse_list(render.build(spec, loc), parse.label_index(loc))
    facts = judges.recompute(parsed, loc)
    with pytest.raises(ValueError, match="读不出位次"):
        judges.expected_of("nth", "这条题面里没有位次。", facts, loc)


def test_expected_of_reports_an_unknown_family(make_spec: Any) -> None:
    """输入：不存在的题族。输出：无。预期行为：报人话点名题族。"""
    loc = locales.load("zh")
    spec = make_spec()
    parsed = parse.parse_list(render.build(spec, loc), parse.label_index(loc))
    facts = judges.recompute(parsed, loc)
    with pytest.raises(ValueError, match="未知题族"):
        judges.expected_of("nope", "题面", facts, loc)
