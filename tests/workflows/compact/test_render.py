"""会话上下文压缩的文本外壳测试：提示词、草稿校验、state 拼装与反解。"""

from __future__ import annotations

import pytest

from workflows.compact import locales, render
from workflows.compact.features import TagSet

from tests.workflows.compact.conftest import BASE

TAGS = TagSet(kind="user", fact="chatter", age="old", pins="unpinned", budget="tight")
GOOD = {"content": BASE["user"]}


def test_prompt_lists_only_the_text_dimensions() -> None:
    """输入：一组特征、语言字典。输出：无。

    预期行为：提示词只提由文本承载的两维，绝不提轮次位置 / 是否钉住 / 剩余预算——
    提了模型就会把它们写进正文，与「本块位置」那一行抢同一件事（范式 §9.13）。
    """
    loc = locales.load("zh")
    prompt = render.build_prompt(TAGS, loc, 0)
    feature_block = prompt.split("\n\n")[1]
    assert "这一块是什么" in feature_block
    assert "有没有承载持久事实" in feature_block
    assert "轮次位置" not in feature_block
    assert "是否钉住" not in feature_block
    assert "剩余预算" not in feature_block
    assert "{" not in prompt


def test_build_task_carries_the_response_model_and_tags() -> None:
    """输入：一组特征、语言字典、任务 id。输出：无。

    预期行为：任务带 pydantic 草稿模型（结构化输出要用）、带 tags（假后端与事后切片要用）、
    而且提示词里没有答案。
    """
    loc = locales.load("zh")
    task = render.build_task(TAGS, loc, "compact-train-0000000-render")
    assert task.response_model is render.CompactDraft
    assert task.spec["tags"] == TAGS.as_dict()
    assert task.task_id == "compact-train-0000000-render"


def test_check_draft_rejects_blank_multiline_and_fences() -> None:
    """输入：三种坏草稿。输出：无。预期行为：都抛 ValueError。"""
    loc = locales.load("zh")
    for bad in ("   ", f"{GOOD['content']}\n第二行", f"```{GOOD['content']}```"):
        with pytest.raises(ValueError):
            render.check_draft({"content": bad}, loc)


def test_check_draft_rejects_a_repeated_section_name() -> None:
    """输入：正文开头又写一遍「本块内容：」。输出：无。预期行为：抛 ValueError。"""
    loc = locales.load("zh")
    with pytest.raises(ValueError):
        render.check_draft({"content": f"本块内容：{GOOD['content']}"}, loc)


def test_check_draft_rejects_where_words_in_the_body() -> None:
    """输入：正文里写了位置词 / 钉住词。输出：无。

    预期行为：两种语言都抛 ValueError——那三件事归「本块位置」那一行管，
    正文里再写一遍就可能与那一行打架。
    """
    for lang, bad in (("zh", "这一块已经被钉住了"), ("en", "this block is pinned")):
        with pytest.raises(ValueError):
            render.check_draft({"content": bad}, locales.load(lang))


def test_check_draft_passes_a_clean_body() -> None:
    """输入：一份合规草稿。输出：无。预期行为：不抛异常。"""
    render.check_draft(GOOD, locales.load("zh"))
    render.check_draft(GOOD, locales.load("en"))


def test_label_leak_only_fires_at_the_start() -> None:
    """输入：正文中间提到段落名。输出：无。

    预期行为：只有「段落名 + 冒号」开头才算残留；正文里正常提到这些词不算。
    """
    loc = locales.load("zh")
    assert render.label_leak(f"本块内容：{GOOD['content']}", loc) == "本块内容"
    assert render.label_leak(f"他说的「本块内容」不是这个意思", loc) == ""
    assert render.label_leak(GOOD["content"], loc) == ""


def test_kind_label_leak_catches_a_self_labelled_body() -> None:
    """输入：正文开头自己写上「这一块是什么」那一维的档名。输出：无。

    预期行为：也算残留——`fam.kind` 问的就是「这一块是哪一档」，正文写「助手：…」等于把答案
    印在题面里（真机 zh 侧 28 条里 8 条这么写，见范式 §9.101）。只认开头 + 冒号，
    正文里正常提到这些词不算。
    """
    loc = locales.load("zh")
    for label in ("助手", "助手的回复", "工具输出", "系统提示词", "随手备注"):
        assert render.label_leak(f"{label}：{GOOD['content']}", loc) == label
    assert render.label_leak("把那段助手回复整理一下", loc) == ""


def test_check_draft_rejects_a_self_labelled_body() -> None:
    """输入：正文开头写了档名的草稿。输出：无。预期行为：抛 ValueError，重渲染。"""
    loc = locales.load("zh")
    with pytest.raises(ValueError):
        render.check_draft({"content": f"助手：{GOOD['content']}"}, loc)


def test_state_round_trips() -> None:
    """输入：一份合规草稿。输出：无。预期行为：拼出来的 state 能逐段反解回原文。"""
    loc = locales.load("zh")
    state = render.to_state(GOOD, TAGS, loc)
    assert render.content_of(state, loc) == GOOD["content"]
    assert render.where_of(state, loc) == render.WhereState("old", "unpinned", "tight")
    assert render.rules_of(state, loc) == render.canonical_rules(loc)
    parsed = render.parse_state(state)
    assert parsed.content == GOOD["content"]
    assert parsed.where == render.WhereState("old", "unpinned", "tight")
    assert parsed.rules == render.canonical_rules(loc)
    assert not parsed.is_empty


def test_state_carries_the_where_row_not_the_text() -> None:
    """输入：一份合规草稿。输出：无。

    预期行为：位置三格只出现在那一行里，正文里一个词元都没有——这是「两边清单不重叠」
    在产物上的体现。
    """
    loc = locales.load("zh")
    state = render.to_state(GOOD, TAGS, loc)
    head, _, _tail = state.partition(render.where_line(TAGS, loc))
    assert render.where_line(TAGS, loc) in state
    assert head
    content = render.content_of(state, loc)
    for token in ("old", "pinned", "tight"):
        assert token not in content


def test_rule_lines_follow_the_code_order() -> None:
    """输入：两语言处置表。输出：无。

    预期行为：渲染出来的行序 == `features.RULES`——处置表是「从上往下第一条命中」。
    """
    from workflows.compact import features

    for lang in locales.available():
        loc = locales.load(lang)
        prefix = str(loc["meta"]["row_prefix"])
        lines = render.rule_lines(loc)
        assert len(lines) == len(features.RULES)
        names = [line.split(render.CELL_SEP)[0].removeprefix(prefix).strip() for line in lines]
        assert names == list(features.RULES)


def test_parse_state_tolerates_junk() -> None:
    """输入：非文本 / 空文本 / 只有段落头。输出：无。预期行为：不抛异常，返回空框架。"""
    assert render.parse_state(None).where is None
    assert render.parse_state("").is_empty
    assert render.content_of(None, locales.load("zh")) == ""
    assert render.where_of(None, locales.load("zh")) is None
    assert render.parse_state("【本块位置】").where is None


def test_where_of_rejects_an_unknown_token() -> None:
    """输入：位置行里写了个不在取值表里的词。输出：无。

    预期行为：解不出来（None）——判别器会把它当成「位置行对不上」，
    而不是把串味的词元当成合法取值放过去。
    """
    loc = locales.load("zh")
    state = render.to_state(GOOD, TAGS, loc).replace(
        render.where_line(TAGS, loc), "- yesterday | unpinned | tight"
    )
    assert render.where_of(state, loc) is None


def test_banned_echo_is_case_insensitive() -> None:
    """输入：大小写不同的钉住词。输出：无。预期行为：都能认出来。"""
    loc = locales.load("en")
    assert render.banned_echo("This block is Pinned", loc)
    assert render.banned_echo("this block is pinned", loc)
    assert render.banned_echo("a plain sentence", loc) == []
