"""文本外壳测试：提示词该说的话都说了、state 能来回转、写歪的日志在渲染期就被拦下。"""

from __future__ import annotations

import pytest

from jc.jsontypes import JsonValue
from jc.workflow import DraftError

from tests.workflows.logtriage.conftest import BASE, FRAGMENTS, spec_for, text_of
from workflows.logtriage import features, locales, render
from workflows.logtriage.features import LogTriageCase

#: 一个普通 case：命中两条。
CASE = LogTriageCase("N1+N3", features.SOURCE_APP)


def test_prompt_says_none_may_be_hit_when_the_plan_is_empty() -> None:
    """输入：空命中的 case。输出：无。预期行为：改成「一条都不许踩」的说法，且不出现要求段落。"""
    loc = locales.load("zh")
    case = LogTriageCase("", features.SOURCE_APP)
    prompt = render.build_prompt(loc, case)
    assert str(loc["prompt"]["no_hits"]) in prompt
    assert str(loc["prompt"]["hits_header"]) not in prompt


def test_prompt_states_the_source_shape() -> None:
    """输入：一种语言 × 一种通道。输出：无。预期行为：提示词里点明这行日志从哪种通道来。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in ["en", "zh"]:
        for source in ["app", "infra", "access"]:
            try:
                loc = locales.load(lang)
                case = LogTriageCase("N2", source)
                prompt = render.build_prompt(loc, case)
                assert render.source_copy(loc, source) in prompt
            except AssertionError as exc:
                bad.append(f"{(lang, source)!r}: {exc}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_state_round_trips(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：写出去再读回来，判据表与日志一字不差。"""
    loc = locales.load(lang)
    text = text_of(CASE)
    state = render.to_state(render.ContentDraft(text=text), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.tags == features.TAG_IDS
    assert parsed.text == text
    assert render.detect_locale(state) == lang


def test_state_keeps_the_tag_table_and_the_thresholds() -> None:
    """输入：一段 state。输出：无。

    预期行为：判据表四行与档位表四行都印在 state 里，档位表那几行用的是与题面**同一份**处置文案。
    """
    loc = locales.load("zh")
    state = render.to_state(render.ContentDraft(text=text_of(CASE)), loc)
    for tag in features.TAG_IDS:
        assert render.tag_line(loc, tag) in state
    for level in features.HANDLING_LEVELS:
        assert render.threshold_line(loc, level) in state


def test_state_prints_each_criterion_rule() -> None:
    """输入：两种语言 × 一段 state。输出：无。

    预期行为：判据表每行都带上该判据的**原文**——只印判据名时，反解模型认不出「已自愈的重试与
    降级」指的是什么形状（范式 §9.44，在 `memorygate` 上量过 1/8 → 8/8）。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        state = render.to_state(render.ContentDraft(text=text_of(CASE)), loc)
        for tag in features.TAG_IDS:
            assert str(render.tag_copy(loc, tag)["rule"]) in state


def test_multiline_content_round_trips() -> None:
    """输入：带堆栈的日志（多行）。输出：无。预期行为：多行原样保住。"""
    loc = locales.load("zh")
    text = text_of(CASE) + "\n  at handler.py:41\n  at main.py:7"
    state = render.to_state(render.ContentDraft(text=text), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.text == text
    assert parsed.tags == features.TAG_IDS


def test_content_mentioning_a_rule_does_not_break_the_parse() -> None:
    """输入：正文里出现一行「像判据表」的内容。输出：无。

    预期行为：日志抬头之后的一切都当正文——否则堆栈里出现一行 `- N9：假的` 就会把判据表数歪。
    """
    loc = locales.load("zh")
    text = text_of(CASE) + "\n- N9：这一行是正文，不是判据表"
    parsed = render.parse_with(render.to_state(render.ContentDraft(text=text), loc), loc)
    assert parsed.tags == features.TAG_IDS
    assert "- N9：这一行是正文，不是判据表" in str(parsed.text)


def test_check_draft_accepts_every_faithful_text() -> None:
    """输入：全空间 48 组 case 的忠实日志。输出：无。预期行为：一条都不报错。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        render.check_draft(render.ContentDraft(text=text_of(case)), case, loc)


@pytest.mark.parametrize(
    "text",
    [BASE, BASE + FRAGMENTS["N2"], BASE + FRAGMENTS["N4"]],
)
def test_check_draft_rejects_broken_text(text: str) -> None:
    """输入：少写一条 / 多写一条的日志。输出：无。

    预期行为：抛 `DraftError`，理由里说清「算出来的组合」与「计划要的」各是什么。
    """
    with pytest.raises(DraftError, match="命中组合"):
        render.check_draft(render.ContentDraft(text=text), CASE, locales.load("zh"))


def test_check_draft_rejects_empty_and_overlong_text() -> None:
    """输入：空日志 / 超长日志 / 带围栏的日志。输出：无。预期行为：三种都当场抛错。"""
    case = LogTriageCase("", features.SOURCE_APP)
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
    """输入：把判据原文**整段贴**进日志。输出：无。

    预期行为：抛 `DraftError`，理由是贴判据原文——提示词里唯一可抄的具体文字就是它，贴进去
    标签照样算命中（正则只看子串），但那条日志读起来不是日志（范式 §9.45）。
    """
    loc = locales.load(lang)
    rule = str(render.tag_copy(loc, "N1")["rule"])
    case = LogTriageCase("N1", features.SOURCE_APP)
    with pytest.raises(DraftError, match="判据原文"):
        render.check_draft(render.ContentDraft(text=f"{BASE} {rule}"), case, loc)


def test_pasted_rules_only_flags_verbatim_text() -> None:
    """输入：整段贴 / 改了一个词的两种日志。输出：无。

    预期行为：只认**逐字相同**的整段判据原文——改一个词就不再算，否则正常产物会被误伤。
    """
    loc = locales.load("zh")
    rule = str(render.tag_copy(loc, "N1")["rule"])
    assert render.pasted_rules(rule, loc) == [rule]
    assert render.pasted_rules(rule.replace("healthz", "healthzzz"), loc) == []
    assert render.pasted_rules(BASE, loc) == []


def test_source_copy_reports_a_missing_shape() -> None:
    """输入：缺通道的语言字典。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError, match="通道 infra"):
        render.source_copy({"source": {"app": {"desc": "应用日志"}}}, "infra")
