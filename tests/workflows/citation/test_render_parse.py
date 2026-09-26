"""渲染与反解：版式骨架、截断规则、两种编号写法、往返一致。"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from tests.workflows.citation.conftest import tiny_config
from workflows.citation import judges, locales, parse, render
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


# ---------- 渲染 ----------


def test_build_writes_the_documented_skeleton() -> None:
    """输入：一份规格。输出：整段文本。预期行为：抬头 → 来源文档 → 每节（小节行 + 两句）
    → 主张 / 引文 / 声称出处，且小节行是 `[n] 标题`、句子行缩进两格。
    """
    loc = locales.load("zh")
    spec = _spec()
    lines = render.build(spec, loc).splitlines()
    meta = loc["meta"]
    assert lines[0] == meta["title"]
    assert lines[1].startswith(meta["doc_label"] + meta["colon"])
    assert lines[1].endswith(meta["doc_suffix"].format(n=len(spec.sections)))
    body = lines[2:]
    heads = [line for line in body if line.startswith("[")]
    assert len(heads) == len(spec.sections)
    assert heads[0].startswith("[1] ")
    sentences = [line for line in body if line.startswith("  ")]
    assert len(sentences) == 2 * len(spec.sections)
    assert body[-3].startswith(meta["claim_label"] + meta["colon"])
    assert body[-2].startswith(meta["quote_label"] + meta["colon"])
    assert body[-1].startswith(meta["source_label"] + meta["colon"])
    assert body[-1].endswith(f"{meta['section_word']} {spec.claimed}")


def test_dotted_style_uses_the_other_marker() -> None:
    """输入：编号写法 `dotted` 的规格。输出：无。预期行为：小节行写成 `1. 标题`。"""
    loc = locales.load("zh")
    lines = render.build(_spec(doc_style="dotted"), loc).splitlines()
    heads = [line for line in lines[2:] if line.startswith("1.")]
    assert heads and heads[0].startswith("1. ")
    assert not any(line.startswith("[") for line in lines[2:])


def test_render_is_a_pure_function() -> None:
    """输入：同一份规格渲染两次。输出：两段文本。预期行为：逐字节一致。"""
    loc = locales.load("zh")
    spec = _spec(status="mislocated", noise="off_topic")
    assert render.build(spec, loc) == render.build(spec, loc)


def test_fact_text_refuses_an_empty_sentence() -> None:
    """输入：空句子。输出：异常。预期行为：报错——静默渲染成空串会产出一条空引文。"""
    with pytest.raises(ValueError, match="空的"):
        render.fact_text(None, locales.load("zh"))


def test_truncate_returns_a_strict_prefix_at_a_word_boundary() -> None:
    """输入：两种语言的每一句原文。输出：截断结果。预期行为：非空、是真前缀、比原文短；
    有空格的语言在词边界切（切出来不会拖半个词）。
    """
    loc = locales.load("en")
    for topic in sp.TOPIC_NAMES:
        for sentence in sp.SENTENCE_IDS[topic]:
            text = loc["topic"][topic]["sentence"][sentence]["doc"]
            cut = render.truncate(text)
            assert 0 < len(cut) < len(text)
            assert text.startswith(cut)
            if " " in text:
                assert text[len(cut)] == " ", (text, cut)


def test_quote_text_follows_the_case() -> None:
    """输入：三种引文写法的规格。输出：引文文本。

    预期行为：`verbatim` 照抄来源句；`truncated` 是它的前半句；`reworded` 写的是主张的转述。
    后两种都**不等于**文档里的任何整句——这正是「编造」这一档成立的原因。
    """
    loc = locales.load("zh")
    verbatim = _spec()
    assert render.quote_text(verbatim, loc) == render.fact_text(verbatim.quote_source, loc)
    truncated = _spec(status="fabricated", quote_form="truncated")
    assert render.quote_text(truncated, loc) == render.truncate(
        render.fact_text(truncated.quote_source, loc)
    )
    reworded = _spec(status="fabricated", quote_form="reworded")
    assert render.quote_text(reworded, loc) == render.claim_text(reworded, loc)


def test_claim_line_wraps_the_paraphrase_in_a_frame() -> None:
    """输入：同一份规格、两种主张措辞。输出：主张那一行的值。

    预期行为：两种框架不同、都裹着同一句转述。**同一份规格**这点很重要——整页文本由
    case 的八个维度一起决定，换 `claim_style` 会连带换掉句子。
    """
    loc = locales.load("zh")
    spec = _spec()
    line = render.claim_line(spec, loc)
    assert render.claim_text(spec, loc) in line
    other = replace(spec, case=replace(spec.case, claim_style=0))
    assert render.claim_line(other, loc) != line
    assert render.claim_text(other, loc) in render.claim_line(other, loc)


# ---------- 反解 ----------


def test_section_head_recognises_both_styles() -> None:
    """输入：两种编号写法的行。输出：无。预期行为：都读得出编号与标题；别的行给 None。"""
    assert parse.section_head("[2] 退款与冲正") == (2, "退款与冲正")
    assert parse.section_head("3. 出账周期") == (3, "出账周期")
    assert parse.section_head("  这是一句正文") is None
    assert parse.section_head("主张：文档里写着：x") is None


def test_label_matching_takes_the_longest_label() -> None:
    """输入：一个标签是另一个前缀的标签表。输出：无。预期行为：取最长匹配，冒号被去掉。"""
    labels = {"来源": "short", "来源文档": "doc_label"}
    assert parse._match_label("来源文档：服务鉴权规范", labels) == ("doc_label", "服务鉴权规范")
    assert parse._match_label("来源：x", labels) == ("short", "x")
    assert parse._match_label("无关的一行", labels) is None


def test_parse_reads_back_the_whole_page() -> None:
    """输入：渲染出来的文本。输出：反解结果。预期行为：来源文档、小节、主张、引文、声称出处逐字对得上。"""
    loc = locales.load("zh")
    spec = _spec(status="mislocated", doc_style="dotted", noise="decoy")
    parsed = parse.parse_citation(render.build(spec, loc), parse.label_index(loc))
    assert parsed.doc == render.doc_line(spec, loc)
    assert len(parsed.sections) == len(spec.sections)
    for want in spec.sections:
        got = parsed.section(want.index)
        assert got.heading == render.heading_text(want, loc)
        assert got.sentences == tuple(render.fact_text(fact, loc) for fact in want.facts)
    assert parsed.claim == render.claim_line(spec, loc)
    assert parsed.quote == render.quote_text(spec, loc)
    assert parsed.source == render.source_line(spec, loc)
    assert parse.claimed_index(parsed.source) == spec.claimed


def test_parse_ignores_the_header_lines() -> None:
    """输入：抬头那两行。输出：无。预期行为：不进任何字段——抬头不是字段行。"""
    loc = locales.load("zh")
    spec = _spec()
    parsed = parse.parse_citation(render.build(spec, loc), parse.label_index(loc))
    assert loc["meta"]["title"] not in parsed.claim
    assert loc["meta"]["title"] not in parsed.doc


def test_parse_rejects_text_without_sections() -> None:
    """输入：一段没有小节行的文本。输出：异常。预期行为：报错说「一个小节都没有」。"""
    with pytest.raises(ValueError, match="一个小节都没有"):
        parse.parse_citation("这只是一段普通文本", parse.merged_labels())


def test_claimed_index_requires_exactly_one_number() -> None:
    """输入：声称出处那一行的值。输出：无。预期行为：恰好一个数字才认；零个或两个都报错。"""
    assert parse.claimed_index("小节 2") == 2
    with pytest.raises(ValueError, match="不是恰好一个"):
        parse.claimed_index("小节")
    with pytest.raises(ValueError, match="不是恰好一个"):
        parse.claimed_index("小节 2 第 3 段")


def test_parsed_section_lookup_reports_missing_indices() -> None:
    """输入：不存在的编号。输出：异常。预期行为：报人话，不返回空小节。"""
    loc = locales.load("zh")
    parsed = parse.parse_citation(render.build(_spec(), loc), parse.label_index(loc))
    with pytest.raises(ValueError, match="没有编号"):
        parsed.section(99)


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_roundtrip_holds_for_every_case(lang: str) -> None:
    """输入：全部 case（两种语言）。输出：无。

    预期行为：渲染出来的文本反解回去与真值规格**逐字**一致——这是整条链的底线，
    也是判别器能独立重算答案的前提。
    """
    loc = locales.load(lang)
    labels = parse.label_index(loc)
    for index, case in enumerate(sp.build_cases(tiny_config())):
        spec = sp.sample_spec(random.Random(index), case, "train", index, 0)
        parsed = parse.parse_citation(render.build(spec, loc), labels)
        assert judges.roundtrip_diff(spec, parsed, loc) == [], case.describe()
