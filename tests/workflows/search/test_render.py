"""检索结果判读的文本外壳测试：提示词、草稿校验、state 拼装与反解。"""

from __future__ import annotations

import pytest

from workflows.search import locales, render
from workflows.search.features import TagSet

from tests.workflows.search.conftest import TEXTS

TAGS = TagSet(
    query_kind="informational",
    topicality="on_topic",
    spam="clean",
    authority="community",
    freshness="fresh",
)
GOOD = dict(TEXTS[("informational", "on_topic", "clean")])


def test_prompt_lists_only_the_text_dimensions() -> None:
    """输入：一组特征、语言字典。输出：无。

    预期行为：提示词只提文本承载的三维，绝不提来源类型与时效——提了模型就会写进文本，
    与「来源与时效」那一行抢同一件事（范式 §9.13）。
    """
    loc = locales.load("zh")
    prompt = render.build_prompt(TAGS, loc, 0)
    feature_block = prompt.split("\n\n")[1]
    assert "查询意图" in feature_block
    assert "是否切题" in feature_block
    assert "内容性质" in feature_block
    assert "来源类型" not in feature_block
    assert "时效" not in feature_block
    assert "{" not in prompt


@pytest.mark.parametrize(
    "field",
    ["query", "title", "snippet"],
)
def test_check_draft_rejects_blank_and_multiline(field: str) -> None:
    """输入：某个字段被写空 / 写成两行。输出：无。预期行为：两种都抛 ValueError。"""
    loc = locales.load("zh")
    for bad in ("   ", f"{GOOD[field]}\n第二行", f"```{GOOD[field]}```"):
        data = dict(GOOD)
        data[field] = bad
        with pytest.raises(ValueError):
            render.check_draft(data, loc)


def test_check_draft_rejects_a_repeated_section_name() -> None:
    """输入：标题开头又写一遍「标题：」。输出：无。预期行为：抛 ValueError。"""
    loc = locales.load("zh")
    data = dict(GOOD, title="标题：蓝鲸笔记桌面版下载页")
    with pytest.raises(ValueError):
        render.check_draft(data, loc)


def test_check_draft_rejects_source_words_in_title_and_snippet() -> None:
    """输入：标题 / 摘要里写了来源词。输出：无。预期行为：都抛 ValueError。"""
    loc = locales.load("zh")
    for field in ("title", "snippet"):
        data = dict(GOOD)
        data[field] = f"官方网站 {GOOD[field]}"
        with pytest.raises(ValueError):
            render.check_draft(data, loc)


def test_check_draft_does_not_police_the_query() -> None:
    """输入：查询里出现来源词。输出：无。

    预期行为：查询本身不描述「来源」，所以不查它——照抄来源词的是标题与摘要。
    """
    loc = locales.load("zh")
    render.check_draft(dict(GOOD, query="小蓝鲸 官方 登录"), loc)


def test_state_round_trips() -> None:
    """输入：一份合规草稿。输出：无。预期行为：拼出来的 state 能逐字段反解回原文。"""
    loc = locales.load("zh")
    state = render.to_state(GOOD, TAGS, loc)
    assert render.query_of(state, loc) == GOOD["query"]
    assert render.title_of(state, loc) == GOOD["title"]
    assert render.snippet_of(state, loc) == GOOD["snippet"]
    assert render.source_of(state, loc) == render.SourceState("community", "fresh")
    assert render.rules_of(state, loc) == render.canonical_rules(loc)
    parsed = render.parse_state(state)
    assert parsed.query == GOOD["query"]
    assert parsed.source == render.SourceState("community", "fresh")
    assert not parsed.is_empty


def test_state_carries_the_source_row_not_the_text() -> None:
    """输入：一份合规草稿。输出：无。

    预期行为：来源类型与时效只出现在那一行里；文本里一个字都没有——这是「两边清单
    不重叠」在产物上的体现。
    """
    loc = locales.load("zh")
    state = render.to_state(GOOD, TAGS, loc)
    head, _, _tail = state.partition("【标题】")
    assert "community" in head and "fresh" in head
    text = render.title_of(state, loc) + render.snippet_of(state, loc)
    assert "community" not in text and "fresh" not in text


def test_parse_state_tolerates_junk() -> None:
    """输入：非文本 / 空文本。输出：无。预期行为：不抛异常，返回空框架。"""
    assert render.parse_state(None).source is None
    assert render.parse_state("").is_empty
    assert render.source_of(None, locales.load("zh")) is None


def test_banned_echo_is_case_insensitive() -> None:
    """输入：大小写不同的来源词。输出：无。预期行为：都能认出来。"""
    loc = locales.load("en")
    assert render.banned_echo("Official docs", loc)
    assert render.banned_echo("official docs", loc)
