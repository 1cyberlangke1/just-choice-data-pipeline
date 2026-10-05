"""文本外壳测试：提示词该说的话都说了、state 能来回转、写歪的草稿在渲染期就被拦下。"""

from __future__ import annotations

import pytest

from jc.jsontypes import JsonValue
from jc.workflow import DraftError

from tests.workflows.chatgate.conftest import (
    BASE,
    CLEAN,
    FLAW_FRAGMENTS,
    FRAGMENTS,
    candidates_of,
    message_of,
    sample_of,
    spec_for,
)
from workflows.chatgate import features, locales, render
from workflows.chatgate.features import ChatCase

#: 一个普通 case：命中两条。
CASE = ChatCase(degree=2, scene=features.SCENE_COLLEAGUE, slot=0)


def _plan(case: ChatCase, phrasing: int = 0) -> features.ChatPlan:
    """输入：case、问法编号。输出：计划。"""
    return features.plan_for(case, phrasing)


def test_prompt_lists_required_and_forbidden_criteria() -> None:
    """输入：一个命中两条的 case。输出：无。

    预期行为：要求出现的两条各自印上**判据原文**（那是模型唯一的规格来源，示例片段一律不给）；
    不许出现的那两条各自带着反面写法——少列一条，模型就可能顺手把那条也写上。
    """
    loc = locales.load("zh")
    plan = _plan(CASE)
    prompt = render.build_prompt(loc, CASE, plan)
    for tag in plan.hits:
        assert str(render.tag_copy(loc, tag)["rule"]) in prompt
    for tag in features.TAG_IDS:
        if tag not in plan.hits:
            assert str(render.tag_copy(loc, tag)["avoid"]) in prompt


def test_prompt_assigns_each_candidate_its_own_flaws() -> None:
    """输入：一个普通 case。输出：无。

    预期行为：三条候选的毛病逐条写明——该踩的给出毛病原文，该发的那条明说「一个毛病都不许踩」
    并把三条毛病的反面写法一起列出来。少写一条，模型就会随手把毛病安错位置。
    """
    loc = locales.load("zh")
    plan = _plan(CASE)
    prompt = render.build_prompt(loc, CASE, plan)
    for index, flaws in enumerate(plan.reply.flaws):
        assert f"第 {index + 1} 条候选" in prompt
        for flaw in flaws:
            assert str(render.flaw_copy(loc, flaw)["rule"]) in prompt
        if not flaws:
            assert str(render.flaw_copy(loc, "F1")["avoid"]) in prompt


def test_prompt_says_none_may_be_hit_when_the_plan_is_empty() -> None:
    """输入：空命中的 case。输出：无。预期行为：改成「一条都不许踩」的说法，且不出现要求段落。"""
    loc = locales.load("zh")
    case = ChatCase(degree=0, scene=features.SCENE_GROUP, slot=1)
    prompt = render.build_prompt(loc, case, _plan(case))
    assert str(loc["prompt"]["no_hits"]) in prompt
    assert str(loc["prompt"]["hits_header"]) not in prompt


def test_prompt_states_the_scene_shape() -> None:
    """输入：一种语言 × 一种场景。输出：无。预期行为：提示词里点明这条消息从哪儿来。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in ["en", "zh"]:
        for scene in ["colleague", "group", "stranger"]:
            try:
                loc = locales.load(lang)
                case = ChatCase(degree=1, scene=scene, slot=2)
                assert render.scene_copy(loc, scene) in render.build_prompt(loc, case, _plan(case))
            except AssertionError as exc:
                bad.append(f"{(lang, scene)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_prompt_has_no_unfilled_slots() -> None:
    """输入：全空间 45 组 case。输出：无。预期行为：渲染完一个花括号都不剩。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        prompt = render.build_prompt(loc, case, _plan(case))
        assert "{" not in prompt and "}" not in prompt, case.describe()


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_state_round_trips(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：写出去再读回来，三张表、消息与三条候选一字不差。"""
    loc = locales.load(lang)
    plan = _plan(CASE)
    text = message_of(plan.hits)
    candidates = candidates_of(plan.reply.flaws)
    state = render.to_state(render.MessageDraft(text=text, candidates=candidates), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.tags == features.TAG_IDS
    assert parsed.flaws == features.FLAW_IDS
    assert parsed.message == text
    assert parsed.replies == tuple(candidates)
    assert render.detect_locale(state) == lang


def test_state_prints_all_three_tables_and_the_criteria_text() -> None:
    """输入：一段 state。输出：无。

    预期行为：判据表、毛病表、档位表都印在 state 里，而且每行都带上**原文**——只印名字时反解
    模型认不出「把决定推给我」指的是什么形状（范式 §9.44）。
    """
    loc = locales.load("zh")
    plan = _plan(CASE)
    state = render.to_state(
        render.MessageDraft(text=message_of(plan.hits), candidates=candidates_of(plan.reply.flaws)),
        loc,
    )
    for tag in features.TAG_IDS:
        assert render.tag_line(loc, tag) in state
        assert str(render.tag_copy(loc, tag)["rule"]) in state
    for flaw in features.FLAW_IDS:
        assert render.flaw_line(loc, flaw) in state
        assert str(render.flaw_copy(loc, flaw)["rule"]) in state
    for level in features.HANDLING_LEVELS:
        assert render.threshold_line(loc, level) in state


def test_parse_state_returns_empty_for_garbage() -> None:
    """输入：认不出抬头的各种输入。输出：无。预期行为：返回空结果，不抛异常。"""
    for garbage in ("这里什么抬头都没有", "", 42, None, {"state": "x"}):
        parsed = render.parse_state(garbage)
        assert parsed.message is None
        assert parsed.tags == () and parsed.flaws == () and parsed.replies == ()


def test_check_draft_accepts_every_faithful_draft() -> None:
    """输入：全空间 45 组 case 的忠实草稿。输出：无。预期行为：一条都不报错。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = _plan(case)
        draft = render.MessageDraft(
            text=message_of(plan.hits), candidates=candidates_of(plan.reply.flaws)
        )
        render.check_draft(draft, plan, loc)


def test_check_draft_rejects_a_broken_message() -> None:
    """输入：少写一条 / 多写一条命中的消息。输出：无。

    预期行为：抛 `DraftError`，理由里说清「算出来的组合」与「计划要的」各是什么。
    """
    loc = locales.load("zh")
    plan = _plan(CASE)
    candidates = candidates_of(plan.reply.flaws)
    with pytest.raises(DraftError, match="命中组合"):
        render.check_draft(render.MessageDraft(text=BASE, candidates=candidates), plan, loc)
    spare = next(tag for tag in features.TAG_IDS if tag not in plan.hits)
    extra = message_of(plan.hits) + FRAGMENTS[spare]
    with pytest.raises(DraftError, match="命中组合"):
        render.check_draft(render.MessageDraft(text=extra, candidates=candidates), plan, loc)


def test_check_draft_rejects_a_wrong_candidate_count() -> None:
    """输入：候选不是三条。输出：无。预期行为：抛 `DraftError`。"""
    loc = locales.load("zh")
    plan = _plan(CASE)
    with pytest.raises(DraftError, match="候选回复要有"):
        render.check_draft(
            render.MessageDraft(text=message_of(plan.hits), candidates=[CLEAN, CLEAN]), plan, loc
        )


@pytest.mark.parametrize("switch", ["swap", "clean_dirty"])
def test_check_draft_rejects_wrong_candidate_flaws(switch: str) -> None:
    """输入：把三条候选的毛病安错位置。输出：无。预期行为：抛 `DraftError`，点名第几条。"""
    loc = locales.load("zh")
    plan = _plan(CASE)
    candidates = candidates_of(plan.reply.flaws)
    if switch == "swap":
        candidates = list(reversed(candidates))
    else:
        candidates[plan.reply.clean] = FLAW_FRAGMENTS["F1"]
    with pytest.raises(DraftError, match="候选的毛病"):
        render.check_draft(
            render.MessageDraft(text=message_of(plan.hits), candidates=candidates), plan, loc
        )


def test_check_draft_rejects_empty_overlong_and_fenced_bodies() -> None:
    """输入：空消息 / 超长消息 / 带围栏的消息 / 过短的候选。输出：无。预期行为：都当场抛错。"""
    loc = locales.load("zh")
    plan = _plan(CASE)
    candidates = candidates_of(plan.reply.flaws)
    with pytest.raises(DraftError, match="空"):
        render.check_draft(render.MessageDraft(text="   ", candidates=candidates), plan, loc)
    with pytest.raises(DraftError, match="太长"):
        render.check_draft(
            render.MessageDraft(text="会" * (render.MAX_MESSAGE_CHARS + 1), candidates=candidates),
            plan,
            loc,
        )
    with pytest.raises(DraftError, match="围栏"):
        render.check_draft(
            render.MessageDraft(text=f"```{BASE}```", candidates=candidates), plan, loc
        )
    with pytest.raises(DraftError, match="太短"):
        render.check_draft(render.MessageDraft(text=BASE, candidates=[CLEAN, CLEAN, "短"]), plan, loc)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_check_draft_rejects_a_pasted_rule(lang: str) -> None:
    """输入：把判据原文**整段贴**进消息。输出：无。

    预期行为：抛 `DraftError`，理由是贴判据原文——提示词里唯一可抄的具体文字就是它，贴进去标签
    照样算命中（正则只看子串），但那条消息读起来不是消息（范式 §9.45）。
    """
    loc = locales.load(lang)
    case = ChatCase(degree=1, scene=features.SCENE_GROUP, slot=1)
    plan = _plan(case)
    candidates = candidates_of(plan.reply.flaws)
    rule = str(render.tag_copy(loc, features.TAG_IDS[0])["rule"])
    text = message_of(plan.hits) + " " + rule
    with pytest.raises(DraftError, match="原文"):
        render.check_draft(render.MessageDraft(text=text, candidates=candidates), plan, loc)
    flaw = str(render.flaw_copy(loc, features.FLAW_IDS[0])["rule"])
    with pytest.raises(DraftError, match="原文"):
        render.check_draft(
            render.MessageDraft(text=message_of(plan.hits), candidates=[flaw, *candidates[1:]]),
            plan,
            loc,
        )


def test_build_task_carries_the_plan() -> None:
    """输入：真值规格。输出：无。

    预期行为：任务 id 可重算、`spec` 里带上注入特征与候选毛病计划（假后端与真后端都能照着它答）、
    响应模型是草稿模型。
    """
    spec = spec_for(CASE)
    task = render.build_task(spec, locales.load("zh"))
    assert task.task_id == "chatgate-train-0000001-render"
    assert task.domain == "chatgate"
    assert task.response_model is render.MessageDraft
    assert task.spec["case"] == {"degree": 2, "scene": "colleague", "slot": 0}
    assert task.spec["hits"] == list(CASE.tags())
    assert task.spec["clean"] == 0
    assert task.spec["flaws"] == [list(flaws) for flaws in spec.plan.reply.flaws]


def test_threshold_line_reuses_the_route_copy() -> None:
    """输入：一种语言 × 四个档位。输出：无。预期行为：档位表与 `route` 那一族的选项文案同源。"""
    loc = locales.load("zh")
    for level in features.HANDLING_LEVELS:
        assert str(loc["fam"]["route"]["opt"][level]) in render.threshold_line(loc, level)


def test_copies_report_missing_fields() -> None:
    """输入：缺字段的语言字典。输出：无。预期行为：抛 ValueError，点名缺了哪一张表的哪一条。"""
    with pytest.raises(ValueError, match="缺 tag 的 C1"):
        render.tag_copy({"tag": {}}, "C1")
    broken: dict[str, JsonValue] = {"tag": {"C1": {"name": "把决定推给我"}}}
    with pytest.raises(ValueError, match="缺字段 rule"):
        render.tag_copy(broken, "C1")
    with pytest.raises(ValueError, match="缺 flaw 的 F1"):
        render.flaw_copy({"flaw": {}}, "F1")
    with pytest.raises(ValueError, match="缺 scene 的 group"):
        render.scene_copy({"scene": {"colleague": {"desc": "同事私聊"}}}, "group")


def test_sample_of_keeps_the_candidates_in_order() -> None:
    """输入：三条候选。输出：无。预期行为：解析回来还是那个顺序（`reply` 的选项就指这个）。"""
    plan = _plan(CASE)
    candidates = candidates_of(plan.reply.flaws)
    parsed = render.parse_state(sample_of(message_of(plan.hits), candidates)["state"])
    assert parsed.replies == tuple(candidates)
