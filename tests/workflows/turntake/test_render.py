"""文本外壳测试：提示词该说的话都说了、state 能来回转、写歪的输入在渲染期就被拦下。"""

from __future__ import annotations

import pytest

from jc.jsontypes import JsonValue
from jc.workflow import DraftError

from tests.workflows.turntake.conftest import BASE, spec_for, text_of
from workflows.turntake import features, locales, render
from workflows.turntake.features import TurnTakeCase

#: 一个普通 case：命中两条，通道是聊天软件。
CASE = TurnTakeCase("T1+T3", features.CHANNEL_IM)


def test_prompt_says_none_may_be_tripped_when_the_plan_is_empty() -> None:
    """输入：空命中的 case。输出：无。预期行为：改成「一条都不许踩」的说法，且不出现要求段落。"""
    loc = locales.load("zh")
    case = TurnTakeCase("", features.CHANNEL_IM)
    prompt = render.build_prompt(loc, case)
    assert str(loc["prompt"]["no_hits"]) in prompt
    assert str(loc["prompt"]["hits_header"]) not in prompt


def test_prompt_states_the_channel_shape() -> None:
    """输入：一种语言 × 一种通道。输出：无。预期行为：提示词里点明这段输入从哪种通道来。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in ["en", "zh"]:
        for channel in ["im", "voice", "note"]:
            try:
                loc = locales.load(lang)
                case = TurnTakeCase("T2", channel)
                prompt = render.build_prompt(loc, case)
                assert render.channel_copy(loc, channel) in prompt
            except AssertionError as exc:
                bad.append(f"{(lang, channel)!r}: {exc}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_state_round_trips(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：写出去再读回来，判据表与输入一字不差。"""
    loc = locales.load(lang)
    text = text_of(CASE.tags())
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
    state = render.to_state(render.ContentDraft(text=text_of(CASE.tags())), loc)
    for tag in features.TAG_IDS:
        assert render.tag_line(loc, tag) in state
    for level in features.ACTS:
        assert render.threshold_line(loc, level) in state


def test_state_prints_each_criterion_rule() -> None:
    """输入：两种语言 × 一段 state。输出：无。

    预期行为：判据表每行都带上该判据的**原文**——只印判据名时，反解模型认不出「话头悬在连接词上」
    指的是什么形状（范式 §9.44，在 `memorygate` 上量过 1/8 → 8/8）。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        state = render.to_state(render.ContentDraft(text=text_of(CASE.tags())), loc)
        for tag in features.TAG_IDS:
            assert str(render.tag_copy(loc, tag)["rule"]) in state


def test_multiline_content_round_trips() -> None:
    """输入：带换行的输入。输出：无。预期行为：多行原样保住。"""
    loc = locales.load("zh")
    text = text_of(CASE.tags()) + "\n我接着说一下，然后"
    state = render.to_state(render.ContentDraft(text=text), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.text == text
    assert parsed.tags == features.TAG_IDS


def test_content_mentioning_a_rule_does_not_break_the_parse() -> None:
    """输入：正文里出现一行「像判据表」的内容。输出：无。

    预期行为：输入抬头之后的一切都当正文——否则正文里出现一行 `- T9：假的` 就会把判据表数歪。
    """
    loc = locales.load("zh")
    text = text_of(CASE.tags()) + "\n- T9：这一行是正文，不是判据表"
    parsed = render.parse_with(render.to_state(render.ContentDraft(text=text), loc), loc)
    assert parsed.tags == features.TAG_IDS
    assert "- T9：这一行是正文，不是判据表" in str(parsed.text)


def test_check_draft_accepts_every_faithful_text() -> None:
    """输入：全空间 48 组 case 的忠实输入。输出：无。预期行为：一条都不报错。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        render.check_draft(render.ContentDraft(text=text_of(case.tags())), case, loc)


@pytest.mark.parametrize("tags", [("T1",), ("T1", "T3", "T4")])
def test_check_draft_rejects_broken_text(tags: tuple[str, ...]) -> None:
    """输入：少写一条 / 多写一条的输入。输出：无。

    预期行为：抛 `DraftError`，理由里说清「算出来的组合」与「计划要的」各是什么。
    """
    with pytest.raises(DraftError, match="命中组合"):
        render.check_draft(render.ContentDraft(text=text_of(tags)), CASE, locales.load("zh"))


def test_check_draft_rejects_empty_and_overlong_text() -> None:
    """输入：空输入 / 超长输入 / 带围栏的输入 / 太短的输入。输出：无。预期行为：四种都当场抛错。"""
    case = TurnTakeCase("", features.CHANNEL_IM)
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
    """输入：把判据原文**整段贴**进输入。输出：无。

    预期行为：抛 `DraftError`，理由是贴判据原文——提示词里唯一可抄的具体文字就是它，贴进去
    标签照样算命中（正则只看子串），但那段输入读起来不是人话（范式 §9.45）。
    """
    loc = locales.load(lang)
    rule = str(render.tag_copy(loc, "T4")["rule"])
    case = TurnTakeCase("T4", features.CHANNEL_IM)
    with pytest.raises(DraftError, match="判据原文"):
        render.check_draft(render.ContentDraft(text=f"{BASE} {rule}"), case, loc)


def test_channel_copy_reports_a_missing_shape() -> None:
    """输入：缺通道的语言字典。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError, match="通道 voice"):
        render.channel_copy({"channel": {"im": {"desc": "聊天消息"}}}, "voice")


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_states_the_ending_rule(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：提示词里有一句**结尾规格**——它是这一族最难的约束（真机首轮 8 条拒稿全落在
    「整段最后几个字符」上），所以必须逐条印出来：连接词之后只能有标点、要求以标点收尾时
    最后一个字符就是那个标点、两条都不要求时必须以句号收尾。这条是规格不是示例，不带可抄的成品句。
    """
    loc = locales.load(lang)
    prompt = render.build_prompt(loc, CASE)
    assert str(loc["prompt"]["end_rule"]) in prompt
    for tag in features.TAG_IDS:
        assert str(render.tag_copy(loc, tag)["rule"]) in prompt
