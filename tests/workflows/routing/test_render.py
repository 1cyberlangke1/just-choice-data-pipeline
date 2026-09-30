"""工单分诊的文本外壳测试：提示词、草稿校验、state 拼装与反解。"""

from __future__ import annotations

import pytest

from workflows.routing import features, locales, render
from workflows.routing.features import TagSet

from tests.workflows.routing.conftest import body_for

TAGS = TagSet(issue="bug", urgency="deadline", tier="pro", compliance="none", area="storage")
GOOD = {"body": body_for(TAGS.as_dict())}


def test_prompt_lists_only_the_text_dimensions() -> None:
    """输入：一组特征、语言字典。输出：无。

    预期行为：提示词只提由正文承载的两维，绝不提客户等级 / 合规标记 / 产品域——
    提了模型就会把它们写进正文，与「工单属性」那一行抢同一件事（范式 §9.18）。
    """
    loc = locales.load("zh")
    prompt = render.build_prompt(TAGS, loc, 0)
    feature_block = prompt.split("\n\n")[1]
    assert "这条工单是什么问题" in feature_block
    assert "它有多急" in feature_block
    assert "客户等级" not in feature_block
    assert "合规标记" not in feature_block
    assert "产品域" not in feature_block
    assert "{" not in prompt


def test_build_task_carries_the_response_model_and_tags() -> None:
    """输入：一组特征、语言字典、任务 id。输出：无。

    预期行为：任务带 pydantic 草稿模型（结构化输出要用）、带 tags（假后端与事后切片要用）、
    而且提示词里没有答案。
    """
    loc = locales.load("zh")
    task = render.build_task(TAGS, loc, "routing-train-0000000-render")
    assert task.response_model is render.RoutingDraft
    assert task.spec["tags"] == TAGS.as_dict()
    assert task.task_id == "routing-train-0000000-render"


def test_check_draft_rejects_blank_multiline_and_fences() -> None:
    """输入：三种坏草稿。输出：无。预期行为：都抛 ValueError。"""
    loc = locales.load("zh")
    for bad in ("   ", f"{GOOD['body']}\n第二行", f"```{GOOD['body']}```"):
        with pytest.raises(ValueError):
            render.check_draft({"body": bad}, loc)


def test_check_draft_rejects_a_repeated_section_name() -> None:
    """输入：正文开头又写一遍「工单正文：」。输出：无。预期行为：抛 ValueError。"""
    loc = locales.load("zh")
    with pytest.raises(ValueError):
        render.check_draft({"body": f"工单正文：{GOOD['body']}"}, loc)


def test_check_draft_rejects_attribute_words_in_the_body() -> None:
    """输入：正文里写了属性词 / 表名。输出：无。

    预期行为：两种语言都抛 ValueError——那几件事归「工单属性」那一行与三张表管，
    正文里再写一遍就可能与它们打架。
    """
    for lang, bad in (("zh", "我们是企业版客户"), ("en", "we are on the enterprise plan")):
        with pytest.raises(ValueError):
            render.check_draft({"body": bad}, locales.load(lang))


def test_label_leak_only_fires_at_the_start() -> None:
    """输入：正文中间提到段落名。输出：无。

    预期行为：只有「段落名 + 冒号」开头才算残留；正文里正常提到这些词不算。
    """
    loc = locales.load("zh")
    assert render.label_leak(f"工单正文：{GOOD['body']}", loc) == "工单正文"
    assert render.label_leak(f"他说的「工单正文」不是这个意思", loc) == ""
    assert render.label_leak(GOOD["body"], loc) == ""


def test_state_round_trips() -> None:
    """输入：一份合规草稿。输出：无。预期行为：拼出来的 state 能逐段反解回原文。"""
    loc = locales.load("zh")
    state = render.to_state(GOOD, TAGS, loc)
    assert render.body_of(state, loc) == GOOD["body"]
    assert render.attrs_of(state, loc) == render.AttrsState("storage", "pro", "none")
    assert render.queue_rules_of(state, loc) == render.queue_rule_rows(loc)
    assert render.priority_of(state, loc) == render.priority_rows(loc)
    assert render.flags_of(state, loc) == render.flag_rows(loc)
    parsed = render.parse_state(state)
    assert parsed.body == GOOD["body"]
    assert parsed.attrs == render.AttrsState("storage", "pro", "none")
    assert parsed.queue_rules == render.queue_rule_rows(loc)
    assert parsed.priority == render.priority_rows(loc)
    assert parsed.flags == render.flag_rows(loc)
    assert not parsed.is_empty


def test_state_carries_the_attr_row_not_the_text() -> None:
    """输入：一份合规草稿。输出：无。

    预期行为：属性三格只出现在那一行里，正文里一个词元都没有——这是「两边清单不重叠」
    在产物上的体现。
    """
    loc = locales.load("zh")
    state = render.to_state(GOOD, TAGS, loc)
    assert render.attr_lines(TAGS, loc)[0] in state
    body = render.body_of(state, loc)
    for token in (TAGS.area, TAGS.tier, TAGS.compliance):
        assert token not in body


def test_queue_lines_follow_the_code_order() -> None:
    """输入：两语言派单表。输出：无。

    预期行为：渲染出来的行序 == `features.QUEUE_RULES`——派单表是「从上往下第一条命中」。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        prefix = str(loc["meta"]["row_prefix"])
        lines = render.queue_lines(loc)
        assert len(lines) == len(features.QUEUE_RULES)
        names = [line.split(locales.CELL_SEP)[0].removeprefix(prefix).strip() for line in lines]
        assert names == list(features.QUEUE_RULES)


def test_priority_rows_follow_the_table() -> None:
    """输入：两语言优先级表。输出：无。

    预期行为：每行的三个档位依次是免费版 / 专业版 / 企业版，值取自 `features.PRIORITY_TABLE`。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        rows = render.priority_rows(loc)
        assert [row[0] for row in rows] == list(features.URGENCIES)
        for urgency, free, pro, enterprise in rows:
            assert free == str(features.PRIORITY_TABLE[(urgency, "free")])
            assert pro == str(features.PRIORITY_TABLE[(urgency, "pro")])
            assert enterprise == str(features.PRIORITY_TABLE[(urgency, "enterprise")])


def test_flag_rows_render_the_critical_areas_in_declaration_order() -> None:
    """输入：两语言附加规则。输出：无。

    预期行为：关键域清单按 `features.AREAS` 的声明顺序写出来（支付、身份），
    不是按 `frozenset` 的迭代顺序——否则同一份输入在不同进程里会渲染出不同字序。
    """
    loc = locales.load("zh")
    rows = dict(render.flag_rows(loc))
    first = str(loc["area"]["payments"])
    second = str(loc["area"]["identity"])
    assert rows["critical_areas"].index(first) < rows["critical_areas"].index(second)


def test_parse_state_tolerates_junk() -> None:
    """输入：非文本 / 空文本 / 只有段落头。输出：无。预期行为：不抛异常，返回空票据。"""
    assert render.parse_state(None).attrs is None
    assert render.parse_state("").is_empty
    assert render.body_of(None, locales.load("zh")) == ""
    assert render.attrs_of(None, locales.load("zh")) is None
    assert render.parse_state("【工单属性】").attrs is None


def test_attrs_of_rejects_an_unknown_token() -> None:
    """输入：属性行里写了个不在取值表里的词。输出：无。

    预期行为：解不出来（None）——判别器会把它当成「属性行对不上」，
    而不是把串味的词元当成合法取值放过去。
    """
    loc = locales.load("zh")
    state = render.to_state(GOOD, TAGS, loc).replace(
        render.attr_lines(TAGS, loc)[0], "- storage | gold | none"
    )
    assert render.attrs_of(state, loc) is None


def test_parsers_tolerate_non_text_states() -> None:
    """输入：非文本 state。输出：无。预期行为：一律给空结果，不抛异常。"""
    loc = locales.load("zh")
    assert render.attrs_of(None, loc) is None
    assert render.queue_rules_of(None, loc) == ()
    assert render.priority_of(None, loc) == ()
    assert render.flags_of(None, loc) == ()


def test_attrs_of_rejects_a_wrong_cell_count() -> None:
    """输入：属性行多了一格。输出：无。预期行为：解不出来（None）。"""
    loc = locales.load("zh")
    head = locales.header(loc, loc["meta"]["attrs_label"])
    assert render.attrs_of(f"{head}\n- storage | pro | none | extra", loc) is None


def test_queue_rules_of_skips_a_wrong_cell_count() -> None:
    """输入：派单表里混进一行缺格的。输出：无。预期行为：那一行被跳过，其余照收。"""
    loc = locales.load("zh")
    head = locales.header(loc, loc["meta"]["queue_label"])
    state = f"{head}\n- defect | 问题是缺陷 | engineering\n- howto | 问题是咨询"
    assert render.queue_rules_of(state, loc) == (("defect", "问题是缺陷", "engineering"),)


def test_priority_of_skips_unrecognized_rows() -> None:
    """输入：优先级表里混进缺格 / 首格不认识 / 档位不是单个数字的行。输出：无。

    预期行为：三类坏行都被跳过，只留下那一行合法的。
    """
    loc = locales.load("zh")
    head = locales.header(loc, loc["meta"]["priority_label"])
    rows = [
        "- can_wait | 0 | 0 | 1",
        "- nope | 0 | 0 | 1",
        "- blocked | 1 | 2",
        "- deadline | 2 | x | 3",
    ]
    assert render.priority_of(f"{head}\n" + "\n".join(rows), loc) == (
        ("can_wait", "0", "0", "1"),
    )


def test_flags_of_skips_a_wrong_cell_count_and_an_unknown_name() -> None:
    """输入：附加规则里混进缺格与名字不认识的行。输出：无。预期行为：都跳过。"""
    loc = locales.load("zh")
    head = locales.header(loc, loc["meta"]["flags_label"])
    rows = ["- needs_human | 说明", "- needs_human", "- bogus | 说明"]
    assert render.flags_of(f"{head}\n" + "\n".join(rows), loc) == (("needs_human", "说明"),)


def test_queue_rules_of_drops_a_tampered_row() -> None:
    """输入：派单表某行的队列被改成不认识的值。输出：无。

    预期行为：那一行被跳过，于是解析出来的行数变少——判别器会判「与规范表对不上」。
    """
    loc = locales.load("zh")
    state = render.to_state(GOOD, TAGS, loc).replace("| incident_response", "| nowhere", 1)
    assert len(render.queue_rules_of(state, loc)) == len(features.QUEUE_RULES) - 1


def test_banned_echo_is_case_insensitive() -> None:
    """输入：大小写不同的属性词。输出：无。预期行为：都能认出来。"""
    loc = locales.load("en")
    assert render.banned_echo("we are on the Enterprise plan", loc)
    assert render.banned_echo("we are on the enterprise plan", loc)
    assert render.banned_echo("a plain sentence", loc) == []
