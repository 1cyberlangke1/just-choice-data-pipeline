"""文本外壳测试：提示词该说的话都说了、state 能来回转、写歪的轨迹在渲染期就被拦下。"""

from __future__ import annotations

import pytest

from jc.jsontypes import JsonValue
from jc.workflow import DraftError

from tests.workflows.tracescore.conftest import BASE, FRAGMENTS, spec_for, text_of
from workflows.tracescore import features, locales, render
from workflows.tracescore.features import TraceScoreCase

#: 一个普通 case：命中两条。
CASE = TraceScoreCase("S1+S3", features.SOURCE_CLI)


def test_prompt_says_none_may_be_hit_when_the_plan_is_empty() -> None:
    """输入：空命中的 case。输出：无。预期行为：改成「一条都不许命中」的说法，且不出现要求段落。"""
    loc = locales.load("zh")
    case = TraceScoreCase("", features.SOURCE_CLI)
    prompt = render.build_prompt(loc, case)
    assert str(loc["prompt"]["no_hits"]) in prompt
    assert str(loc["prompt"]["hits_header"]) not in prompt


def test_prompt_states_the_source_shape() -> None:
    """输入：一种语言 × 一种入口。输出：无。预期行为：提示词里点明这段轨迹从哪种入口来。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in ["en", "zh"]:
        for source in ["cli", "editor", "chat"]:
            try:
                loc = locales.load(lang)
                case = TraceScoreCase("S2", source)
                prompt = render.build_prompt(loc, case)
                assert render.source_copy(loc, source) in prompt
            except AssertionError as exc:
                bad.append(f"{(lang, source)!r}: {exc}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_state_round_trips(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：写出去再读回来，信号表与轨迹一字不差。"""
    loc = locales.load(lang)
    text = text_of(CASE)
    state = render.to_state(render.ContentDraft(text=text), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.tags == features.TAG_IDS
    assert parsed.text == text
    assert render.detect_locale(state) == lang


def test_state_keeps_the_tag_table_and_the_thresholds() -> None:
    """输入：一段 state。输出：无。

    预期行为：信号表四行与档位表三行都印在 state 里，档位表那几行用的是与题面**同一份**处置
    文案。
    """
    loc = locales.load("zh")
    state = render.to_state(render.ContentDraft(text=text_of(CASE)), loc)
    for tag in features.TAG_IDS:
        assert render.tag_line(loc, tag) in state
    for level in features.ROUTE_LEVELS:
        assert render.threshold_line(loc, level) in state


def test_state_prints_each_criterion_rule() -> None:
    """输入：两种语言 × 一段 state。输出：无。

    预期行为：信号表每行都带上该信号的**原文**——只印信号名时，反解模型认不出「反复翻旧账」
    指的是什么形状（范式 §9.44，在 `memorygate` 上量过 1/8 → 8/8）。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        state = render.to_state(render.ContentDraft(text=text_of(CASE)), loc)
        for tag in features.TAG_IDS:
            assert str(render.tag_copy(loc, tag)["rule"]) in state


def test_multiline_content_round_trips() -> None:
    """输入：多行轨迹。输出：无。预期行为：多行原样保住。"""
    loc = locales.load("zh")
    text = text_of(CASE) + "\n$ pytest -q\n2 failed, 8 passed\n  at handler.py:41"
    state = render.to_state(render.ContentDraft(text=text), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.text == text
    assert parsed.tags == features.TAG_IDS


def test_content_mentioning_a_rule_does_not_break_the_parse() -> None:
    """输入：正文里出现一行「像信号表」的内容。输出：无。

    预期行为：轨迹抬头之后的一切都当正文——否则轨迹里出现一行 `- S9：假的` 就会把信号表数歪。
    """
    loc = locales.load("zh")
    text = text_of(CASE) + "\n- S9：这一行是正文，不是信号表"
    parsed = render.parse_with(render.to_state(render.ContentDraft(text=text), loc), loc)
    assert parsed.tags == features.TAG_IDS
    assert "- S9：这一行是正文，不是信号表" in str(parsed.text)


def test_check_draft_accepts_every_faithful_text() -> None:
    """输入：全空间 48 组 case 的忠实轨迹。输出：无。预期行为：一条都不报错。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        render.check_draft(render.ContentDraft(text=text_of(case)), case, loc)


@pytest.mark.parametrize(
    "text",
    [BASE, BASE + FRAGMENTS["S2"], BASE + FRAGMENTS["S4"]],
)
def test_check_draft_rejects_broken_text(text: str) -> None:
    """输入：少写一条 / 多写一条的轨迹。输出：无。

    预期行为：抛 `DraftError`，理由里说清「算出来的组合」与「计划要的」各是什么。
    """
    with pytest.raises(DraftError, match="命中组合"):
        render.check_draft(render.ContentDraft(text=text), CASE, locales.load("zh"))


def test_check_draft_rejects_empty_and_overlong_text() -> None:
    """输入：空轨迹 / 超长轨迹 / 带围栏的轨迹。输出：无。预期行为：四种都当场抛错。"""
    case = TraceScoreCase("", features.SOURCE_CLI)
    loc = locales.load("zh")
    with pytest.raises(DraftError, match="空"):
        render.check_draft(render.ContentDraft(text="   "), case, loc)
    with pytest.raises(DraftError, match="太长"):
        render.check_draft(
            render.ContentDraft(text="会" * (render.MAX_CONTENT_CHARS + 1)), case, loc
        )
    with pytest.raises(DraftError, match="围栏"):
        render.check_draft(render.ContentDraft(text=f"```{BASE}```"), case, loc)
    with pytest.raises(DraftError, match="太短"):
        render.check_draft(render.ContentDraft(text="短"), case, loc)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_check_draft_rejects_a_pasted_rule(lang: str) -> None:
    """输入：把判据原文**整段贴**进轨迹。输出：无。

    预期行为：抛 `DraftError`，理由是贴判据原文——提示词里唯一可抄的具体文字就是它，贴进去
    标签照样算命中（正则只看子串），但那段文本读起来不是轨迹（范式 §9.45）。
    """
    loc = locales.load(lang)
    rule = str(render.tag_copy(loc, "S1")["rule"])
    case = TraceScoreCase("S1", features.SOURCE_CLI)
    with pytest.raises(DraftError, match="判据原文"):
        render.check_draft(render.ContentDraft(text=f"{BASE} {rule}"), case, loc)


def test_pasted_rules_only_flags_verbatim_text() -> None:
    """输入：整段贴 / 改了一个词的两种轨迹。输出：无。

    预期行为：只认**逐字相同**的整段判据原文——改一个词就不再算，否则正常产物会被误伤。
    """
    loc = locales.load("zh")
    rule = str(render.tag_copy(loc, "S1")["rule"])
    assert render.pasted_rules(rule, loc) == [rule]
    # 换掉的是**夹在中间**的那个词：把末尾的词加长反而仍包含原串（子串匹配不要求到结尾），
    # 那样这条断言就测了个寂寞。
    assert render.pasted_rules(rule.replace("pytest", "pytestx"), loc) == []
    assert render.pasted_rules(BASE, loc) == []


def test_threshold_line_reuses_the_route_copy() -> None:
    """输入：一种语言 × 三个档位。输出：无。预期行为：档位表与 `route` 那一族的选项文案同源。"""
    loc = locales.load("zh")
    for level in features.ROUTE_LEVELS:
        assert str(loc["fam"]["route"]["opt"][level]) in render.threshold_line(loc, level)


def test_source_copy_reports_a_missing_shape() -> None:
    """输入：缺入口的语言字典。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError, match="缺入口 editor"):
        render.source_copy({"source": {"cli": {"desc": "命令行代理的轨迹"}}}, "editor")
