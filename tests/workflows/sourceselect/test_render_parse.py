"""来源选择的文本外壳测试：提示词、草稿校验、state 拼装与反解。"""

from __future__ import annotations

import pytest

from workflows.sourceselect import features, locales, render
from workflows.sourceselect.features import SourceCase

from tests.workflows.sourceselect.conftest import QUERIES

CASE = SourceCase(need="authority", size=4, shape="unique", near_miss=1)
PLAN = features.plan_for(CASE, 0)
GOOD = {"query": QUERIES["authority"]}


def test_prompt_lists_only_the_text_dimension() -> None:
    """输入：一个 case、语言字典。输出：无。

    预期行为：提示词只提文本承载的那一维（首要要求），绝不提候选表的结构——
    提了模型就会把表写进查询，与代码渲染的那张表抢同一件事。
    """
    loc = locales.load("zh")
    prompt = render.build_prompt(CASE, loc, 0)
    feature_block = prompt.split("\n\n")[1]
    assert "首要要求" in feature_block
    assert "候选表条数" not in feature_block
    assert "表的形状" not in feature_block
    assert "诱饵" not in feature_block
    assert "{" not in prompt


def test_prompt_rotates_rules() -> None:
    """输入：三个问法编号。输出：无。预期行为：规则顺序轮换，但条数不变。"""
    loc = locales.load("zh")
    first = render.build_prompt(CASE, loc, 0).splitlines()
    second = render.build_prompt(CASE, loc, 1).splitlines()
    assert first != second
    assert len(first) == len(second)


@pytest.mark.parametrize("bad", ["   ", "第一行\n第二行", "```查询```"])
def test_check_draft_rejects_blank_multiline_and_fences(bad: str) -> None:
    """输入：空 / 多行 / 带围栏的查询。输出：无。预期行为：都抛 ValueError。"""
    with pytest.raises(ValueError):
        render.check_draft({"query": bad}, locales.load("zh"))


def test_check_draft_rejects_a_repeated_section_name() -> None:
    """输入：查询开头又写一遍「查询：」。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError):
        render.check_draft({"query": f"查询：{GOOD['query']}"}, locales.load("zh"))


def test_check_draft_rejects_table_talk() -> None:
    """输入：查询里谈候选表。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError):
        render.check_draft({"query": "按候选来源表的序号挑一行"}, locales.load("zh"))


def test_check_draft_allows_the_words_that_express_the_need() -> None:
    """输入：用「官方 / 最新 / 最全」说要求的查询。输出：无。

    预期行为：这几个词正是「首要要求」在自然语言里的正常说法，**不许**被禁用语拦下——
    拦下就等于把这道题唯一要读的那句话堵死。
    """
    for text in ("找官方文档", "要最新的版本说明", "给我覆盖最全的清单"):
        render.check_draft({"query": text}, locales.load("zh"))


def test_state_round_trips() -> None:
    """输入：一份合规草稿。输出：无。预期行为：拼出来的 state 能逐字段反解回原文。"""
    loc = locales.load("zh")
    state = render.to_state(GOOD, PLAN, loc)
    assert render.query_of(state, loc) == GOOD["query"]
    assert render.rows_of(state, loc) == PLAN.rows
    assert render.rules_of(state, loc) == features.RULES
    assert render.section_lines(state, loc, "rules_label") == render.rule_lines(loc)
    parsed = render.parse_state(state)
    assert parsed.query == GOOD["query"]
    assert parsed.rows == PLAN.rows
    assert not parsed.is_empty


def test_state_carries_the_table_not_the_query() -> None:
    """输入：一份合规草稿。输出：无。

    预期行为：候选表的取值词元只出现在表与口径里；那句查询里一个都没有——这是「两边清单
    不重叠」在产物上的体现。
    """
    loc = locales.load("zh")
    state = render.to_state(GOOD, PLAN, loc)
    query = render.query_of(state, loc)
    for token in ("official", "community", "unknown", "fresh", "recent", "stale", "full", "partial"):
        assert token not in query, token
    assert "official" in state


def test_rows_of_rejects_a_broken_table() -> None:
    """输入：序号断档 / 词元不认识的表。输出：无。预期行为：整张表判为解不出来。"""
    loc = locales.load("zh")
    state = render.to_state(GOOD, PLAN, loc)
    assert render.rows_of(state.replace("- 2 | ", "- 7 | "), loc) == ()
    assert render.rows_of(state.replace("official", "vip"), loc) == ()


def test_parse_state_tolerates_junk() -> None:
    """输入：非文本 / 空文本。输出：无。预期行为：不抛异常，返回空框架。"""
    assert render.parse_state(None).rows == ()
    assert render.parse_state("").is_empty
    assert render.rows_of(None, locales.load("zh")) == ()
    assert render.rules_of(None, locales.load("zh")) == ()
    assert render.section_lines(None, locales.load("zh"), "query_label") == []


def test_banned_echo_is_case_insensitive() -> None:
    """输入：大小写不同的禁用语。输出：无。预期行为：都认出来。"""
    loc = locales.load("en")
    assert render.banned_echo("Candidate Source list", loc)
    assert render.banned_echo("a | b", loc)
    assert render.banned_echo("official docs", loc) == []
