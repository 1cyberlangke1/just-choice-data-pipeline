"""浏览器 / GUI 操作的文本外壳测试：提示词、草稿校验、state 拼装与反解。"""

from __future__ import annotations

import pytest

from workflows.browser import features, locales, render
from workflows.browser.features import TagSet

TAGS = TagSet(intent="fill", state="ready", sensitive="yes", decoy="same_role", screen="form")
LABEL = features.wanted_label(TAGS)
GOOD = {"request": f"帮我把{LABEL}填一下，别的先别动。"}


def state_of(data: dict[str, str] | None = None, tags: TagSet = TAGS, lang: str = "zh") -> str:
    """输入：可选草稿覆盖、特征、语言。输出：拼好的 state。"""
    return render.to_state({**GOOD, **(data or {})}, tags, locales.load(lang))


def test_prompt_lists_only_the_sentence_dimension_and_the_element() -> None:
    """输入：一组特征、语言字典。输出：无。

    预期行为：提示词的特征块只提那一维意图与要指的那个元素，绝不提状态 / 角色 / 敏感 /
    干扰行 / 屏型——提了模型就会把它们写进那句话，与那几张表抢同一件事（范式 §9.13）。
    """
    loc = locales.load("zh")
    prompt = render.build_prompt(TAGS, loc, 0)
    feature_block = prompt.split("\n\n")[1]
    assert loc["dims"]["intent"] in feature_block
    assert LABEL in feature_block
    for token in (*features.STATES, *features.ROLES, *features.SENSITIVES, *features.DECOYS):
        assert f" {token} " not in feature_block, token
    for key in render.SECTION_KEYS:
        assert str(loc["meta"][key]) not in feature_block
    assert "{" not in prompt


def test_prompt_rotates_rules() -> None:
    """输入：三个问法编号。输出：无。预期行为：规则顺序轮换，但条数不变。"""
    loc = locales.load("zh")
    first = render.build_prompt(TAGS, loc, 0).splitlines()
    second = render.build_prompt(TAGS, loc, 1).splitlines()
    assert first != second
    assert len(first) == len(second)


def test_build_task_carries_the_response_model_and_tags() -> None:
    """输入：一组特征、语言字典、任务 id。输出：无。

    预期行为：任务带 pydantic 草稿模型（结构化输出要用）、带 tags（假后端与事后切片要用）、
    而且提示词里没有答案。
    """
    loc = locales.load("zh")
    task = render.build_task(TAGS, loc, "browser-train-0000000-render")
    assert task.response_model is render.BrowserDraft
    assert task.spec["tags"] == TAGS.as_dict()
    assert task.task_id == "browser-train-0000000-render"


def test_check_draft_rejects_blank_multiline_and_fences() -> None:
    """输入：三种坏草稿。输出：无。预期行为：都抛 ValueError。"""
    loc = locales.load("zh")
    for bad in ("   ", f"{GOOD['request']}\n第二行", f"```{GOOD['request']}```"):
        with pytest.raises(ValueError):
            render.check_draft({"request": bad}, TAGS, loc)


def test_check_draft_rejects_a_repeated_section_name() -> None:
    """输入：那句话开头又写一遍「用户请求：」。输出：无。预期行为：抛 ValueError。"""
    loc = locales.load("zh")
    with pytest.raises(ValueError):
        render.check_draft({"request": f"用户请求：{GOOD['request']}"}, TAGS, loc)


def test_check_draft_rejects_state_words_in_the_sentence() -> None:
    """输入：那句话里写了界面状态。输出：无。

    预期行为：两种语言都抛 ValueError——状态归元素表管，那句话里再写一遍就可能与它打架。
    """
    for lang, bad in (("zh", f"{GOOD['request']}（那个按钮是置灰的）"), ("en", "click the disabled button")):
        with pytest.raises(ValueError):
            render.check_draft({"request": bad}, TAGS, locales.load(lang))


def test_check_draft_rejects_a_missing_element_name() -> None:
    """输入：那句话里没有原样出现目标元素名。输出：无。

    预期行为：抛 ValueError——反解判别器要拿那个名字比对，少了它这道题连
    「用户指的是谁」都读不出来。
    """
    loc = locales.load("zh")
    with pytest.raises(ValueError):
        render.check_draft({"request": "帮我填一下那个东西。"}, TAGS, loc)


def test_check_draft_passes_a_clean_sentence() -> None:
    """输入：一份合规草稿。输出：无。预期行为：不抛异常（两种语言都过）。"""
    render.check_draft(GOOD, TAGS, locales.load("zh"))
    render.check_draft(GOOD, TAGS, locales.load("en"))


def test_label_leak_only_fires_at_the_start() -> None:
    """输入：那句话中间提到段落名。输出：无。

    预期行为：只有「段落名 + 冒号」开头才算残留；那句话里正常提到这些词不算。
    """
    loc = locales.load("zh")
    assert render.label_leak(f"用户请求：{GOOD['request']}", loc) == "用户请求"
    assert render.label_leak("他说的「用户请求」不是这个意思", loc) == ""
    assert render.label_leak(GOOD["request"], loc) == ""


def test_state_round_trips() -> None:
    """输入：一份合规草稿。输出：无。预期行为：拼出来的 state 能逐段反解回原文与三张表。"""
    loc = locales.load("zh")
    state = state_of()
    assert render.request_of(state, loc) == GOOD["request"]
    assert render.elements_of(state, loc) == render.canonical_elements(TAGS)
    assert render.gates_of(state, loc) == render.gate_rows(loc)
    assert render.criteria_of(state, loc) == render.criteria_rows()
    parsed = render.parse_state(state)
    assert parsed.request == GOOD["request"]
    assert parsed.elements == render.canonical_elements(TAGS)
    assert parsed.gates == render.gate_rows(loc)
    assert parsed.criteria == render.criteria_rows()
    assert not parsed.is_empty


def test_state_carries_the_tables_not_the_draft() -> None:
    """输入：两份不同的草稿。输出：无。

    预期行为：只有那句话随草稿变，元素表与两张判据表**完全由特征渲染**——
    模型碰不到它们（这正是「标签先行」的落地）。
    """
    left = state_of().splitlines()[1:]
    right = state_of({"request": f"麻烦把{LABEL}给我填上。"}).splitlines()[1:]
    assert left == right


def test_state_has_four_sections_in_a_fixed_order() -> None:
    """输入：一份合规草稿。输出：无。预期行为：四段的段落头按固定顺序各出现一次。"""
    loc = locales.load("zh")
    state = state_of()
    heads = [
        str(loc["meta"][key])
        for key in render.SECTION_KEYS
    ]
    positions = [state.index(locales.header(loc, head)) for head in heads]
    assert positions == sorted(positions)
    for head in heads:
        assert state.count(locales.header(loc, head)) == 1


def test_elements_are_numbered_from_one_in_display_order() -> None:
    """输入：一份合规草稿。输出：无。预期行为：行号就是显示顺序里的 1 起编号。"""
    loc = locales.load("zh")
    rows = render.elements_of(state_of(), loc)
    assert rows is not None
    assert [row[0] for row in rows] == list(range(1, len(rows) + 1))
    assert rows[features.wanted_index(TAGS) - 1][2] == LABEL


def test_parse_state_is_language_independent() -> None:
    """输入：中文与英文各一份 state。输出：无。

    预期行为：两边的反解结果相同（`parse_state` 没有语言参数，只能把各语言的段落名
    都试一遍）——这句话是 `Workflow.reparse` 的接线前提。
    """
    zh = render.parse_state(state_of(lang="zh"))
    en = render.parse_state(state_of(lang="en"))
    assert zh.elements == en.elements
    assert zh.criteria == en.criteria
    assert zh.request == en.request == GOOD["request"]
    # 门控表那几行里，判据词元与说明文字分开看：词元与语言无关，说明文字按语言写。
    assert zh.gates is not None and en.gates is not None
    assert [token for token, _ in zh.gates] == [token for token, _ in en.gates]
    assert [text for _, text in zh.gates] != [text for _, text in en.gates]


def test_parse_state_handles_non_text_and_unknown_states() -> None:
    """输入：非文本 / 认不出的 state。输出：无。预期行为：回空，不炸。"""
    for bad in (123, {"any": "object"}, "完全不是这个格式的一段话"):
        parsed = render.parse_state(bad)
        assert parsed.is_empty
        assert parsed.elements is None


def test_elements_of_rejects_a_tampered_table() -> None:
    """输入：行号跳号 / 角色不认识 / 状态不认识。输出：无。

    预期行为：都解不出来（回 None）——判别器与审计拿它跟代码算出来的规范表逐格比对，
    串味会当场暴露。
    """
    loc = locales.load("zh")
    state = state_of()
    assert render.elements_of(state.replace("- 1 | ", "- 9 | ", 1), loc) is None
    assert render.elements_of(state.replace("| field |", "| bogus |", 1), loc) is None
    assert render.elements_of(state.replace("| ready", "| bogus", 1), loc) is None


def test_gates_and_criteria_reject_tampered_rows() -> None:
    """输入：门控表 / 意图表被改坏。输出：无。预期行为：都解不出来（回 None）。"""
    loc = locales.load("zh")
    state = state_of()
    assert render.gates_of(state.replace("- done |", "- bogus |", 1), loc) is None
    assert render.criteria_of(state.replace("- fill | field |", "- bogus | field |", 1), loc) is None
    assert render.gates_of(123, loc) is None
    assert render.criteria_of(123, loc) is None
    assert render.elements_of(123, loc) is None


def test_elements_of_rejects_wrong_cell_counts_and_bad_indices() -> None:
    """输入：元素表某行少一格 / 行号不是数字。输出：无。预期行为：解不出来（回 None）。"""
    loc = locales.load("zh")
    state = state_of()
    first = render.element_lines(TAGS, loc)[0]
    assert render.elements_of(state.replace(first, "- 1 | field | 收件地址"), loc) is None
    assert render.elements_of(state.replace(first, "- x | field | 收件地址 | ready"), loc) is None


def test_gates_and_criteria_reject_wrong_cell_counts() -> None:
    """输入：门控表 / 意图表某行多一格或少一格。输出：无。预期行为：解不出来（回 None）。"""
    loc = locales.load("zh")
    state = state_of()
    assert render.gates_of(state.replace("- done | ", "- done | x | ", 1), loc) is None
    assert (
        render.criteria_of(state.replace("- fill | field | type | 1 | 2", "- fill | field | type"), loc)
        is None
    )


def test_request_lines_handles_a_non_text_state() -> None:
    """输入：非文本 state。输出：无。预期行为：回空列表 / 空串，不炸。"""
    loc = locales.load("zh")
    assert render.request_lines(123, loc) == []
    assert render.request_of(None, loc) == ""


def test_canonical_tables_come_from_the_feature_layer() -> None:
    """输入：无。输出：无。

    预期行为：意图表就是 `features` 里那三张表的摊平（角色 / 动作 / 风险都由代码给），
    门控表的判据词元与 `GATE_TOKENS` 一致。
    """
    rows = render.criteria_rows()
    assert [row[0] for row in rows] == list(features.INTENTS)
    for intent, role, action, risk_no, risk_yes in rows:
        assert role == features.role_for_intent(intent)
        assert action == features.action_for_intent(intent)
        assert int(risk_no) == features.RISK_TABLE[(intent, "no")]
        assert int(risk_yes) == features.RISK_TABLE[(intent, "yes")]
    assert len(render.gate_rows(locales.load("zh"))) == len(render.GATE_TOKENS)
