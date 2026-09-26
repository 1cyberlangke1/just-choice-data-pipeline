"""判别器：结构往返 + 答案重算。两者都只看文本，不看 spec 里的答案。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from jc import assemble
from workflows.citation import judges, locales, parse, questions, render
from workflows.citation import spec as sp


def _spec(**overrides: object) -> sp.CitationSpec:
    """输入：维度覆盖项。输出：真值规格（默认「鉴权 / 通过 / 照抄 / 三节」）。"""
    fields: dict[str, object] = {
        "topic": "auth",
        "status": "verified",
        "quote_form": "verbatim",
        "sections": 3,
        "noise": "plain",
        "claim_style": 1,
        "where": "middle",
        "doc_style": "bracketed",
    }
    fields.update(overrides)
    return sp.sample_spec(random.Random(3), sp.CitationCase(**fields), "train", 0, 0)  # type: ignore[arg-type]


def _row(spec: sp.CitationSpec, lang: str = "zh") -> dict[str, Any]:
    """输入：规格、语言。输出：契约样本行。预期行为：走真的题族与组装器。"""
    loc = locales.load(lang)
    rng = random.Random(11)
    labeled = questions.build(spec, loc, spec.phrasing, rng)
    return assemble.assemble_sample(
        "citation-train-0000000", render.build(spec, loc), labeled, rng
    )


# ---------- 句子表与主张反查 ----------


def test_sentence_tables_cover_every_text_of_every_language() -> None:
    """输入：无。输出：两张表。预期行为：两种语言的句子与转述都进表，且映射唯一。"""
    sentences, claims = judges.tables()
    expected = len(sp.TOPIC_NAMES) * 6 * 2 * len(locales.available())
    assert len(sentences) == expected
    assert len(claims) == expected
    assert sentences["访问令牌必须定期轮换"] == sp.Fact("auth", "token_rotation", False)
    assert sentences["访问令牌无需定期轮换"] == sp.Fact("auth", "token_rotation", True)
    assert claims["访问令牌应当定期更换"] == sp.Fact("auth", "token_rotation", False)


def test_put_refuses_one_text_for_two_facts() -> None:
    """输入：同一段文本映射到两个不同事实。输出：异常。

    预期行为：当场报错——反解会随机挑一个，等于把好数据判成坏的。
    """
    table: dict[str, sp.Fact] = {}
    judges._put(table, "同一句", sp.Fact("auth", "token_rotation"), "zh")
    judges._put(table, "同一句", sp.Fact("auth", "token_rotation"), "zh")  # 同一个事实，允许
    with pytest.raises(ValueError, match="两个不同的事实"):
        judges._put(table, "同一句", sp.Fact("auth", "session_scope"), "zh")


def test_claim_fact_requires_exactly_one_paraphrase() -> None:
    """输入：主张那一行的值。输出：事实。预期行为：认出恰好一条；认出零条或两条都报错。"""
    assert judges.claim_fact("文档里写着：访问令牌应当定期更换") == sp.Fact(
        "auth", "token_rotation", False
    )
    with pytest.raises(ValueError, match="不是恰好一条"):
        judges.claim_fact("文档里写着：这句话不在表里")
    two = "访问令牌应当定期更换，并且会话的权限应当收窄到最小"
    with pytest.raises(ValueError, match="不是恰好一条"):
        judges.claim_fact(two)


# ---------- 重算 ----------


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_recompute_agrees_with_the_spec_for_every_case(lang: str) -> None:
    """输入：全部 case 渲染出来的文本（两种语言）。输出：无。

    预期行为：只吃文本重算出来的七个答案，与构造侧从 spec 结构算出来的一模一样——
    这是「判别器独立重算」这句话的全部证据。
    """
    loc = locales.load(lang)
    labels = parse.label_index(loc)
    cases = sp.build_cases(
        sp.CitationConfig(
            topics=sp.TOPIC_NAMES,
            statuses=sp.STATUSES,
            quote_forms=sp.QUOTE_FORMS,
            section_counts=(2, 3),
            noises=("plain",),
            claim_styles=(0, 1),
            wheres=sp.WHERES,
            doc_styles=sp.DOC_STYLES,
        )
    )
    for index, case in enumerate(cases):
        spec = sp.sample_spec(random.Random(index), case, "train", index, 0)
        parsed = parse.parse_citation(render.build(spec, loc), labels)
        assert judges.recompute(parsed) == sp.answers_of(*spec.truth()), case.describe()


def test_recompute_reports_a_sentence_that_is_not_in_the_table() -> None:
    """输入：一节里塞了一句表外的句子。输出：异常。预期行为：报错点名那句，不猜。"""
    loc = locales.load("zh")
    spec = _spec()
    text = render.build(spec, loc)
    broken = text.replace(render.fact_text(spec.sections[0].facts[0], loc), "这是一句表外的话")
    with pytest.raises(ValueError, match="不在句子表里"):
        judges.recompute(parse.parse_citation(broken, parse.label_index(loc)))


# ---------- 结构判别器 ----------


def test_roundtrip_judge_passes_a_good_sample() -> None:
    """输入：一条正常样本。输出：Verdict。预期行为：通过、理由为空、无差异。"""
    spec = _spec()
    verdict = judges.RoundtripJudge().judge(_row(spec), spec)
    assert verdict.ok is True
    assert verdict.raw["diff"] == []


def test_roundtrip_judge_catches_a_dropped_sentence() -> None:
    """输入：句子被删掉一行的样本。输出：Verdict。预期行为：不通过，并指出是哪个小节。"""
    loc = locales.load("zh")
    spec = _spec()
    row = _row(spec)
    gone = render.fact_text(spec.sections[1].facts[1], loc)
    row["state"] = row["state"].replace(f"  {gone}\n", "")
    verdict = judges.RoundtripJudge().judge(row, spec)
    assert verdict.ok is False
    assert "句子" in verdict.reason


def test_roundtrip_judge_catches_a_swapped_quote() -> None:
    """输入：引文被换成别的小节里的一句。输出：Verdict。预期行为：不通过。"""
    loc = locales.load("zh")
    spec = _spec()
    row = _row(spec)
    other = render.fact_text(spec.sections[2].facts[0], loc)
    row["state"] = row["state"].replace(render.quote_text(spec, loc), other)
    verdict = judges.RoundtripJudge().judge(row, spec)
    assert verdict.ok is False
    assert "引文" in verdict.reason


def test_roundtrip_judge_skips_without_a_spec() -> None:
    """输入：没有真值规格。输出：Verdict。预期行为：跳过（不误判），理由里说明原因。"""
    verdict = judges.RoundtripJudge().judge(_row(_spec()), None)
    assert verdict.ok is True
    assert verdict.raw == {"skipped": True}


def test_roundtrip_judge_reports_unparseable_text() -> None:
    """输入：不是这份契约的文本。输出：Verdict。预期行为：不通过，理由是反解失败。"""
    verdict = judges.RoundtripJudge().judge({"state": "随便一段话"}, _spec())
    assert verdict.ok is False
    assert "反解失败" in verdict.reason


# ---------- 答案判别器 ----------


def test_label_judge_passes_a_good_sample() -> None:
    """输入：一条正常样本。输出：Verdict。预期行为：通过、无问题项。"""
    verdict = judges.LabelJudge().judge(_row(_spec()), None)
    assert verdict.ok is True
    assert verdict.raw["problems"] == []


def test_label_judge_does_not_read_the_spec() -> None:
    """输入：一条正常样本 + 一份**对不上**的规格。输出：Verdict。

    预期行为：照样通过——答案判别器只看文本与题面。它要是拿 spec 里的答案去比，
    这条就会「不通过」，那也说明它在拿答案验证答案。
    """
    row = _row(_spec(status="contradicted"))
    stranger = _spec(status="fabricated", quote_form="reworded")
    assert judges.LabelJudge().judge(row, stranger).ok is True


def test_label_judge_catches_a_tampered_answer() -> None:
    """输入：把结论题的答案编号改掉。输出：Verdict。预期行为：不通过并点名题号。"""
    row = _row(_spec())
    question = row["questions"]["verdict"]
    options = list(question["criteria"])
    row["answers"]["verdict"]["choice"] = (row["answers"]["verdict"]["choice"] + 1) % len(options)
    verdict = judges.LabelJudge().judge(row, None)
    assert verdict.ok is False
    assert any(item.startswith("verdict:") for item in verdict.raw["problems"])


def test_label_judge_catches_a_tampered_noul_and_score() -> None:
    """输入：把 noul 与 score 的答案改掉。输出：Verdict。预期行为：两条都报出来。"""
    row = _row(_spec())
    row["answers"]["found"]["noul"] = not row["answers"]["found"]["noul"]
    row["answers"]["evidence_strength"]["score"] = 3 - row["answers"]["evidence_strength"]["score"]
    verdict = judges.LabelJudge().judge(row, None)
    assert verdict.ok is False
    problems = " ".join(verdict.raw["problems"])
    assert "found" in problems and "evidence_strength" in problems


def test_label_judge_reports_an_unknown_family_and_an_out_of_range_index() -> None:
    """输入：题号不认识的题 + 越界的选项编号。输出：Verdict。预期行为：两条都报。"""
    row = _row(_spec())
    row["questions"]["made_up"] = {
        "type": "choice",
        "instructions": "x",
        "criteria": {"a": "1", "b": "2"},
    }
    row["answers"]["made_up"] = {"type": "choice", "choice": 0}
    row["answers"]["verdict"]["choice"] = 99
    verdict = judges.LabelJudge().judge(row, None)
    assert verdict.ok is False
    problems = " ".join(verdict.raw["problems"])
    assert "未知题族" in problems and "越界" in problems


def test_label_judge_reports_unparseable_text() -> None:
    """输入：反解不了的文本。输出：Verdict。预期行为：不通过，理由是反解失败。"""
    verdict = judges.LabelJudge().judge({"state": "随便一段话", "questions": {}, "answers": {}}, None)
    assert verdict.ok is False
    assert "反解失败" in verdict.reason


def test_judges_declare_their_names_and_default_locale() -> None:
    """输入：无。输出：无。预期行为：判别器名与 workflow 声明一致，默认语言是中文。"""
    assert judges.RoundtripJudge().name == "roundtrip"
    assert judges.LabelJudge().name == "label"
    assert judges.RoundtripJudge().locale == "zh" and judges.LabelJudge().locale == "zh"
