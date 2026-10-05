"""改动引入性判定的文本外壳测试：提示词、草稿校验、state 拼装与反解。"""

from __future__ import annotations

import pytest

from workflows.findtriage import features, locales, render
from workflows.findtriage.features import RecordCase

CASE = RecordCase(cite="touched", evidence="in_row", fix="concrete", size=4, decoy=1)
PLAN = features.plan_for(CASE, 0)
GOOD = {
    "finding": f"第 {PLAN.cited_line} 行：`{PLAN.evidence_symbol}` 的取值没夹紧。",
    "fix": f"在 `{PLAN.fix_symbol}` 那一处补一个上限校验。",
}


def test_prompt_lists_only_the_text_dimensions_and_the_scene_facts() -> None:
    """输入：一个计划、语言字典。输出：无。

    预期行为：提示词只提文本承载的两维（指向何处、修法点不点名）与三行场景事实
    （指向第几行、引用哪个符号、修法点名哪个符号）；表长 / 证据落点 / 诱饵行**一个字都不提**
    ——提了模型就会把它们写进发现，与代码渲染的那张表抢同一件事。
    """
    loc = locales.load("zh")
    prompt = render.build_prompt(PLAN, loc, 0)
    feature_block = prompt.split("\n\n")[1]
    assert "发现指向的那一行" in feature_block
    assert "建议修法" in feature_block
    assert "改动记录的条数" not in feature_block
    assert "证据符号落在哪里" not in feature_block
    assert "诱饵" not in feature_block
    assert "{" not in prompt
    assert str(PLAN.cited_line) in prompt
    assert f"`{PLAN.evidence_symbol}`" in prompt


def test_prompt_rotates_rules() -> None:
    """输入：三个问法编号。输出：无。预期行为：规则顺序轮换，但条数不变。"""
    loc = locales.load("zh")
    first = render.build_prompt(PLAN, loc, 0).splitlines()
    second = render.build_prompt(PLAN, loc, 1).splitlines()
    assert first != second
    assert len(first) == len(second)


def test_prompt_asks_for_a_vague_fix_when_the_case_says_so() -> None:
    """输入：`fix=vague` 的计划。输出：无。

    预期行为：提示词换成「只写泛泛的话、别点名」，而且**不再给**修法要点的那个符号——
    给了符号又说别点名，模型只会两头不是人。
    """
    vague = features.plan_for(
        RecordCase(cite="touched", evidence="in_row", fix="vague", size=4, decoy=0), 0
    )
    prompt = render.build_prompt(vague, locales.load("zh"), 0)
    assert str(locales.load("zh")["prompt"]["fix_vague_line"]) in prompt
    assert str(locales.load("zh")["prompt"]["fix_symbol_line"]).split("{")[0] not in prompt


def test_check_draft_accepts_a_well_formed_draft() -> None:
    """输入：合规草稿。输出：无。预期行为：不抛异常。"""
    render.check_draft(GOOD, PLAN, locales.load("zh"))


@pytest.mark.parametrize("field", ["finding", "fix"])
@pytest.mark.parametrize("bad", ["   ", "第一行\n第二行", "```坏```"])
def test_check_draft_rejects_blank_multiline_and_fences(field: str, bad: str) -> None:
    """输入：空 / 多行 / 带围栏的那两行。输出：无。预期行为：都抛 ValueError。"""
    draft = dict(GOOD)
    draft[field] = bad
    with pytest.raises(ValueError):
        render.check_draft(draft, PLAN, locales.load("zh"))


def test_check_draft_rejects_a_wrong_line_number() -> None:
    """输入：发现里写的是别的行号。输出：无。预期行为：抛 ValueError。"""
    draft = dict(GOOD)
    draft["finding"] = f"第 {PLAN.cited_line + 100} 行：`{PLAN.evidence_symbol}` 有问题。"
    with pytest.raises(ValueError):
        render.check_draft(draft, PLAN, locales.load("zh"))


def test_check_draft_rejects_a_missing_line_number() -> None:
    """输入：发现里压根没写行号。输出：无。预期行为：抛 ValueError。"""
    draft = dict(GOOD)
    draft["finding"] = f"`{PLAN.evidence_symbol}` 的取值没夹紧。"
    with pytest.raises(ValueError):
        render.check_draft(draft, PLAN, locales.load("zh"))


def test_check_draft_rejects_an_unquoted_evidence_symbol() -> None:
    """输入：发现里没把证据符号括起来。输出：无。预期行为：抛 ValueError。"""
    draft = dict(GOOD)
    draft["finding"] = f"第 {PLAN.cited_line} 行：{PLAN.evidence_symbol} 的取值没夹紧。"
    with pytest.raises(ValueError):
        render.check_draft(draft, PLAN, locales.load("zh"))


def test_check_draft_enforces_the_backtick_rule_on_the_fix() -> None:
    """输入：两种写歪的修法。输出：无。

    预期行为：`concrete` 的修法必须点名符号，`vague` 的修法一个反引号都不许有——
    这条是审计能重推「落不落地」的根基。
    """
    loc = locales.load("zh")
    plain = dict(GOOD)
    plain["fix"] = "在那一处补一个上限校验。"
    with pytest.raises(ValueError):
        render.check_draft(plain, PLAN, loc)
    vague_plan = features.plan_for(
        RecordCase(cite="touched", evidence="in_row", fix="vague", size=4, decoy=0), 0
    )
    draft = dict(GOOD)
    draft["finding"] = f"第 {vague_plan.cited_line} 行：`{vague_plan.evidence_symbol}` 有问题。"
    draft["fix"] = f"需要进一步排查 `{vague_plan.evidence_symbol}`。"
    with pytest.raises(ValueError):
        render.check_draft(draft, vague_plan, loc)


def test_state_round_trips() -> None:
    """输入：一份合规草稿。输出：无。预期行为：拼出来的 state 能逐字段反解回原文。"""
    loc = locales.load("zh")
    state = render.to_state(GOOD, PLAN, loc)
    assert render.finding_of(state, loc) == GOOD["finding"]
    assert render.fix_of(state, loc) == GOOD["fix"]
    assert render.rows_of(state, loc) == PLAN.rows
    assert render.rules_of(state, loc) == features.RULES
    assert render.section_lines(state, loc, "rules_label") == render.rule_lines(loc)
    assert render.owner_rules_of(state, loc) == features.OWNER_RULES
    assert render.section_lines(state, loc, "owner_label") == render.owner_lines(loc)
    parsed = render.parse_state(state)
    assert parsed.finding == GOOD["finding"]
    assert parsed.fix == GOOD["fix"]
    assert parsed.rows == PLAN.rows
    assert not parsed.is_empty
    assert render.cite_line_of(GOOD["finding"], loc) == PLAN.cited_line
    assert render.cite_line_any(GOOD["finding"]) == PLAN.cited_line


def test_state_carries_the_record_not_the_finding() -> None:
    """输入：一份合规草稿。输出：无。

    预期行为：改动记录的取值词元只出现在表与口径里，那两行模型写的文本里一个都没有
    ——这是「两边清单不重叠」在产物上的体现。
    """
    loc = locales.load("zh")
    state = render.to_state(GOOD, PLAN, loc)
    for token in ("added", "modified", "untouched"):
        assert token not in GOOD["finding"]
        assert token not in GOOD["fix"]
    assert "added" in state or "modified" in state


def test_parse_state_tolerates_junk() -> None:
    """输入：非文本 / 空文本。输出：无。预期行为：不抛异常，返回空事实。"""
    assert render.parse_state(None).rows == ()
    assert render.parse_state("").is_empty
    assert render.rows_of(None, locales.load("zh")) == ()
    assert render.rules_of(None, locales.load("zh")) == ()
    assert render.section_lines(None, locales.load("zh"), "finding_label") == []
    assert render.cite_line_any("") is None


def test_rule_lines_follow_the_code_order() -> None:
    """输入：两语言口径。输出：无。预期行为：行数与顺序严格跟着 `features.RULES`。"""
    for lang in locales.available():
        lines = render.rule_lines(locales.load(lang))
        assert len(lines) == len(features.RULES)
        assert render.canonical_rules(locales.load(lang)) == tuple(lines)
        for name, line in zip(features.RULES, lines):
            assert name in line


def test_owner_lines_follow_the_code_order() -> None:
    """输入：两语言归属口径。输出：无。预期行为：行数与顺序严格跟着 `features.OWNER_RULES`。"""
    for lang in locales.available():
        loc = locales.load(lang)
        lines = render.owner_lines(loc)
        assert len(lines) == len(features.OWNER_RULES)
        assert render.canonical_owners(loc) == tuple(lines)
        for name, line in zip(features.OWNER_RULES, lines):
            assert name in line


def test_owner_table_is_independent_of_the_triage_table() -> None:
    """输入：两语言的归属段与判定段。输出：无。

    预期行为：两段抬头**互不为前缀**——`_sections` 是按行首切段的，归属段的抬头要是判定段
    抬头的前缀，整段 state 会当场错位。

    两张表的**规则名是可以重名的**（`baseless` / `else` 两张表都有，语义也确实相同），
    所以这里再钉一条：哪怕重名，两个解析器也只读自己那一段——读串了归属就会被判定口径算错。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        rules_header = str(loc["meta"]["rules_label"])
        owner_header = str(loc["meta"]["owner_label"])
        assert rules_header != owner_header
        assert not owner_header.startswith(rules_header)
        assert not rules_header.startswith(owner_header)
        state = render.to_state(GOOD, PLAN, loc)
        assert render.rules_of(state, loc) == features.RULES
        assert render.owner_rules_of(state, loc) == features.OWNER_RULES
        # 把归属段整段删掉：判定口径照旧读得出来，归属读不出来——两边不互相顶替。
        without_owners = "\n".join(
            line
            for line in state.splitlines()
            if not line.startswith(f"{locales.header(loc, owner_header)}")
            and line not in render.owner_lines(loc)
        )
        assert render.rules_of(without_owners, loc) == features.RULES
        assert render.owner_rules_of(without_owners, loc) == ()


def test_backticked_reads_only_quoted_marks() -> None:
    """输入：几段文本。输出：无。预期行为：只认反引号包起来的记号。"""
    assert render.backticked("看 `parse_timeout` 这里") == ["parse_timeout"]
    assert render.backticked("没有记号") == []
    assert render.backticked("`a` 与 `b`") == ["a", "b"]
