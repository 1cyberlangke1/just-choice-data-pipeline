"""工具调用门控渲染层的测试：提示词只给两维、state 三段可反解、草稿校验拦得住。"""

from __future__ import annotations

import pytest

from workflows.toolgate import features, locales, render
from workflows.toolgate.features import TagSet

CASE = TagSet(tool="run_command", reach="host", payload="sensitive", policy="balanced", approval="granted")
INTENT = "把那份导出的用户表拷一份出来"


def state_of(tags: TagSet = CASE, lang: str = "zh", intent: str = INTENT) -> str:
    """输入：特征、语言、意图。输出：渲染好的 state。预期行为：走公开入口拼装。"""
    return render.to_state({"intent": intent}, tags, locales.load(lang))


def test_prompt_lists_only_the_two_text_dimensions() -> None:
    """输入：一组特征。输出：无。

    预期行为：提示词里只出现「工具」与「内容性质」两维的 brief/hint；作用范围、策略档位、
    审批状态一个都不许出现——出现了模型就会把它们写进意图，跟状态行抢同一件事。
    """
    loc = locales.load("zh")
    prompt = render.build_prompt(CASE, loc, 0)
    for dim in render.DIM_ORDER:
        assert str(loc["dims"][dim]) in prompt
    # 维度名表里**只有**这两维：多一维就等于提示词里多给了一件事，模型会照着写。
    assert set(loc["dims"]) == set(render.DIM_ORDER)
    # 状态行的词元也不许出现在提示词里（出现在规则说明里的「不要写」不算，
    # 所以只查特征清单那两行）。
    head = prompt.split("- ", 1)[0]
    assert "workspace" not in head and "strict" not in head
    assert str(loc["feature"]["tool"]["run_command"]["brief"]) in prompt
    assert str(loc["feature"]["payload"]["sensitive"]["brief"]) in prompt


def test_prompt_rotates_rules_by_phrasing() -> None:
    """输入：三个问法编号。输出：无。预期行为：规则顺序随编号轮换，内容集合不变。"""
    loc = locales.load("zh")
    texts = [render.build_prompt(CASE, loc, index) for index in (0, 1, 2)]
    assert len(set(texts)) == 3
    for text in texts:
        for rule in loc["prompt"]["rules"]:
            assert str(rule).format(language=loc["language_name"]) in text
        assert "{language}" not in text


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_state_roundtrip(lang: str) -> None:
    """输入：两种语言。输出：无。

    预期行为：渲染 → 反解 → 得到同一份意图、状态行与策略表；语言不影响反解结果。
    """
    state = state_of(lang=lang)
    frame = render.parse_state(state)
    assert frame.intent == INTENT
    assert frame.call == render.CallState(reach="host", policy="balanced", approval="granted")
    assert frame.rules == render.canonical_rules(locales.load(lang))
    assert not frame.is_empty


def test_state_carries_the_code_rendered_rows() -> None:
    """输入：一组特征。输出：无。

    预期行为：state 的三段都在，状态行的三格按「作用范围 | 策略档位 | 审批状态」排，
    策略表六行且顺序与 `features.RULES` 一致。
    """
    state = state_of()
    loc = locales.load("zh")
    lines = state.splitlines()
    assert lines[0].startswith(locales.header(loc, loc["meta"]["intent_label"]))
    assert render.state_line(CASE, loc) in lines
    assert render.state_line(CASE, loc) == "- host | balanced | granted"
    assert len(render.rules_of(state, loc)) == len(features.RULES)
    assert [name for name, _ in render.rules_of(state, loc)] == list(features.RULES)


def test_call_of_reads_the_first_well_formed_row() -> None:
    """输入：手工拼的 state。输出：无。预期行为：只认三格且词元都在取值表里的那一行。"""
    loc = locales.load("zh")
    head = f"{locales.header(loc, loc['meta']['state_label'])}\n"
    good = f"{head}{loc['meta']['row_prefix']}external | strict | absent\n"
    assert render.call_of(good, loc) == render.CallState("external", "strict", "absent")


def test_call_of_refuses_unknown_tokens() -> None:
    """输入：词元串味的 state。输出：无。

    预期行为：解不出来就返回 None（而不是猜一个）——判别器与审计靠 None 认出「串味了」。
    """
    loc = locales.load("zh")
    head = f"{locales.header(loc, loc['meta']['state_label'])}\n"
    for cells in ("host | balanced | maybe", "nowhere | balanced | absent", "host | balanced"):
        assert render.call_of(f"{head}{loc['meta']['row_prefix']}{cells}\n", loc) is None


def test_rules_of_refuses_a_broken_row() -> None:
    """输入：规则名或处置串味的策略表。输出：无。预期行为：坏行被丢掉，好行留下。"""
    loc = locales.load("zh")
    head = f"{locales.header(loc, loc['meta']['rules_label'])}\n"
    body = (
        f"{loc['meta']['row_prefix']}harmful_payload | 条件 | deny\n"
        f"{loc['meta']['row_prefix']}made_up_rule | 条件 | deny\n"
        f"{loc['meta']['row_prefix']}none | 条件 | maybe\n"
    )
    assert render.rules_of(head + body, loc) == (("harmful_payload", "deny"),)


def test_helpers_return_empty_when_missing() -> None:
    """输入：空 state / 非文本 state。输出：无。预期行为：一律给空值，不抛异常。"""
    loc = locales.load("zh")
    assert render.intent_of("", loc) == ""
    assert render.intent_lines(None, loc) == []
    assert render.intent_lines(123, loc) == []
    assert render.call_of("", loc) is None
    assert render.rules_of("", loc) == ()
    frame = render.parse_state(None)
    assert frame.intent == "" and frame.call is None and frame.rules == ()


def test_intent_lines_reports_every_line() -> None:
    """输入：把意图写成两行的 state。输出：无。预期行为：两行都报出来（卫生判别器靠它）。"""
    loc = locales.load("zh")
    two = state_of(intent=INTENT)
    two = two.replace(INTENT, f"{INTENT}\n（顺手记一下）")
    assert render.intent_lines(two, loc) == [INTENT, "（顺手记一下）"]
    assert render.intent_of(two, loc) == INTENT


def test_parse_state_is_language_agnostic() -> None:
    """输入：en 渲染的 state。输出：无。预期行为：反解结果与 zh 完全一致。"""
    zh = render.parse_state(state_of(lang="zh"))
    en = render.parse_state(state_of(lang="en"))
    assert zh.intent == en.intent
    assert zh.call == en.call


def test_parse_state_returns_empty_for_a_state_without_any_marker() -> None:
    """输入：一段没有任何段落头的文本。输出：无。

    预期行为：给一份空反解结果，不抛异常——上层靠 `call is None` 认出「这段不是本格式」。
    """
    frame = render.parse_state("这里既没有段落头，也没有数据行")
    assert frame.intent == ""
    assert frame.call is None
    assert frame.rules == ()


def test_check_draft_accepts_a_clean_draft() -> None:
    """输入：干净草稿。输出：无。预期行为：不抛异常。"""
    render.check_draft({"intent": INTENT}, locales.load("zh"))


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"intent": "   "}, "空的"),
        ({"intent": "```\n读一下 config.yaml\n```"}, "围栏"),
        ({"intent": "读一下 config.yaml\n再顺手改一下"}, "多行"),
        ({"intent": "调用意图：读一下 config.yaml"}, "段落名"),
        ({"intent": "读一下本机上的 config.yaml"}, "状态行"),
        ({"intent": "按 balanced 档处理"}, "状态行"),
        ({"intent": "先拿审批再动手"}, "状态行"),
        ({}, "空的"),
    ],
)
def test_check_draft_rejects_broken_shapes(data: dict[str, str], match: str) -> None:
    """输入：各种坏草稿。输出：无。预期行为：抛 ValueError 且理由点名字段/原因。"""
    with pytest.raises(ValueError, match=match):
        render.check_draft(data, locales.load("zh"))


def test_check_draft_rejects_every_banned_phrase() -> None:
    """输入：逐条塞进禁用语。输出：无。预期行为：每一条都被渲染期拦下。"""
    loc = locales.load("zh")
    for phrase in loc["banned_intent_phrases"]:
        with pytest.raises(ValueError, match="状态行"):
            render.check_draft({"intent": f"我打算{phrase}做这件事"}, loc)


def test_check_draft_reads_the_marker_from_the_locale() -> None:
    """输入：en 草稿塞中文段落名。输出：无。

    预期行为：段落名按语言读——en 的检查不该被中文段落名触发（反过来也一样），
    否则一种语言的残留检查会在另一种语言上误判。
    """
    en = locales.load("en")
    render.check_draft({"intent": "调用意图：读一下 config.yaml"}, en)
    with pytest.raises(ValueError, match="段落名"):
        render.check_draft({"intent": "Call intent: read config.yaml"}, en)


def test_banned_echo_ignores_ordinary_wording() -> None:
    """输入：正常措辞。输出：无。

    预期行为：正常说「把 config.yaml 里的超时改成 30 秒」不该被认成位置词 / 处置词——
    误判会在渲染期白拒一条本来能用的数据。
    """
    loc = locales.load("zh")
    for text in (INTENT, "把 config.yaml 里的超时改成 30 秒", "跑一遍 pytest"):
        assert render.banned_echo(text, loc) == []


def test_label_leak_only_reads_the_line_start() -> None:
    """输入：段落名出现在句中。输出：无。预期行为：只有开头才算残留。"""
    loc = locales.load("zh")
    assert render.label_leak("调用意图：读一下", loc) == "调用意图"
    assert render.label_leak("他说的调用意图是读一下", loc) == ""


def test_build_task_carries_tags_and_response_model() -> None:
    """输入：一组特征。输出：无。预期行为：任务带 tags、带草稿模型、带域名，且不执行任何东西。"""
    loc = locales.load("zh")
    task = render.build_task(CASE, loc, "toolgate-train-0000000-render")
    assert task.spec["tags"] == CASE.as_dict()
    assert task.response_model is render.ToolGateDraft
    assert task.domain == "tool-call-gating"
    assert task.seeds == ()
    assert INTENT not in task.prompt  # 提示词里不许有答案


def test_draft_has_exactly_one_field() -> None:
    """输入：无。输出：无。

    预期行为：草稿只有一个 `intent` 字段——字段越少，模型越没机会把结构化事实写歪。
    """
    assert set(render.ToolGateDraft.model_fields) == {"intent"}
    assert render.ToolGateDraft.model_fields["intent"].is_required()


def test_canonical_rules_matches_rule_lines() -> None:
    """输入：两语言。输出：无。预期行为：渲染出来的表逐行等于规范表。"""
    for lang in locales.available():
        loc = locales.load(lang)
        state = state_of(lang=lang)
        assert render.rules_of(state, loc) == render.canonical_rules(loc)
        assert len(render.rule_lines(loc)) == len(features.RULES)
