"""文本外壳测试：提示词该说的话都说了、state 能来回转、写歪的会话在渲染期就被拦下。"""

from __future__ import annotations

import pytest

from jc.jsontypes import JsonValue
from jc.workflow import DraftError

from tests.workflows.debugstep.conftest import session_of, spec_for
from workflows.debugstep import features, locales, render
from workflows.debugstep.features import DebugStepCase

#: 一个普通 case：命中两条，会话来自 Python 调试器。
CASE = DebugStepCase("D1+D3", features.RUNTIME_PYTHON)


def test_prompt_lists_required_rules_and_forbidden_shapes() -> None:
    """输入：一个命中两条的 case。输出：无。

    预期行为：要求出现的两条各自印上**判据原文**（那是模型唯一的规格来源，示例会话一律不给，
    见 `build_prompt`）；不许出现的那两条各自带着反面写法——少列一条，模型就可能顺手把那条也
    写上，命中组合当场多出来。
    """
    loc = locales.load("zh")
    prompt = render.build_prompt(loc, CASE)
    for tag in CASE.tags():
        assert str(render.tag_copy(loc, tag)["rule"]) in prompt
        assert "example" not in render.tag_copy(loc, tag)
    for tag in ("D2", "D4"):
        assert str(render.tag_copy(loc, tag)["avoid"]) in prompt


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_format_spec_only_offers_placeholders(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：格式规格那五行**只有占位符、没有一个能整段照抄的成品**——每一行的取值都得模型自己
    编，规格里只印键名与 `<...>`；行数正好五个结构键。
    """
    loc = locales.load(lang)
    lines = render.format_lines(loc)
    assert len(lines) == len(features.KEY_LINES)
    for key, line in zip(features.KEY_LINES, lines, strict=True):
        assert line.startswith(render.LINE_PREFIX)
        assert key in line.lower(), f"{lang}.{key}"
        assert "<" in line and ">" in line, f"{lang}.{key}"


def test_prompt_says_none_may_be_tripped_when_the_plan_is_empty() -> None:
    """输入：空命中的 case。输出：无。预期行为：改成「一条都不许踩」的说法，且不出现要求段落。"""
    loc = locales.load("zh")
    case = DebugStepCase("", features.RUNTIME_PYTHON)
    prompt = render.build_prompt(loc, case)
    assert str(loc["prompt"]["no_hits"]) in prompt
    assert str(loc["prompt"]["hits_header"]) not in prompt


def test_prompt_states_the_runtime_shape() -> None:
    """输入：一种语言 × 一种运行时。输出：无。预期行为：提示词里点明这屏会话来自哪种调试器。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in ["en", "zh"]:
        for runtime in ["python", "node", "go"]:
            try:
                loc = locales.load(lang)
                case = DebugStepCase("D2", runtime)
                prompt = render.build_prompt(loc, case)
                assert render.runtime_copy(loc, runtime) in prompt
            except AssertionError as exc:
                bad.append(f"{(lang, runtime)!r}: {exc}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_state_round_trips(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：写出去再读回来，判据表与会话一字不差。"""
    loc = locales.load(lang)
    session = session_of(CASE.tags())
    state = render.to_state(render.ContentDraft(session=session), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.tags == features.TAG_IDS
    assert parsed.text == session
    assert render.detect_locale(state) == lang


def test_state_keeps_the_tag_table_and_the_thresholds() -> None:
    """输入：一段 state。输出：无。

    预期行为：判据表四行与档位表四行都印在 state 里，档位表那几行用的是与题面**同一份**处置文案。
    """
    loc = locales.load("zh")
    state = render.to_state(render.ContentDraft(session=session_of(CASE.tags())), loc)
    for tag in features.TAG_IDS:
        assert render.tag_line(loc, tag) in state
    for level in features.ACTS:
        assert render.threshold_line(loc, level) in state


def test_state_prints_each_criterion_rule() -> None:
    """输入：两种语言 × 一段 state。输出：无。

    预期行为：判据表每行都带上该判据的**原文**——只印判据名时，反解模型认不出「空值变量」
    指的是什么形状（范式 §9.44，在 `memorygate` 上量过 1/8 → 8/8）。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        state = render.to_state(render.ContentDraft(session=session_of(CASE.tags())), loc)
        for tag in features.TAG_IDS:
            assert str(render.tag_copy(loc, tag)["rule"]) in state


def test_multiline_session_round_trips() -> None:
    """输入：会话末尾多一行。输出：无。预期行为：多行原样保住。"""
    loc = locales.load("zh")
    session = session_of(CASE.tags()) + "\nLocals: extra=1"
    state = render.to_state(render.ContentDraft(session=session), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.text == session
    assert parsed.tags == features.TAG_IDS


def test_session_mentioning_a_rule_does_not_break_the_parse() -> None:
    """输入：会话里出现一行「像判据表」的内容。输出：无。

    预期行为：会话抬头之后的一切都当正文——否则会话里出现一行 `- D9：假的` 就会把判据表数歪。
    """
    loc = locales.load("zh")
    session = session_of(CASE.tags()) + "\n- D9：这一行是正文，不是判据表"
    parsed = render.parse_with(render.to_state(render.ContentDraft(session=session), loc), loc)
    assert parsed.tags == features.TAG_IDS
    assert "- D9：这一行是正文，不是判据表" in str(parsed.text)


def test_check_draft_accepts_every_faithful_session() -> None:
    """输入：全空间 48 组 case 的忠实会话。输出：无。预期行为：一条都不报错。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        render.check_draft(render.ContentDraft(session=session_of(case.tags())), case, loc)


@pytest.mark.parametrize("tags", [("D1",), ("D1", "D2", "D4")])
def test_check_draft_rejects_broken_session(tags: tuple[str, ...]) -> None:
    """输入：少写一条 / 多写一条的会话。输出：无。

    预期行为：抛 `DraftError`，理由里说清「算出来的组合」与「计划要的」各是什么。
    """
    with pytest.raises(DraftError, match="命中组合"):
        render.check_draft(
            render.ContentDraft(session=session_of(tags)), CASE, locales.load("zh")
        )


def test_check_draft_rejects_empty_overlong_fence_and_too_short() -> None:
    """输入：空 / 超长 / 带围栏 / 太短 / 缺一行 的五种会话。输出：无。

    预期行为：五种都当场抛错，且各自命中该有的那道闸（顺序：空 → 围栏 → 太短 → 太长 → 缺键）
    ——闸门顺序搞反的话，报出来的理由会指错方向。
    """
    case = DebugStepCase("", features.RUNTIME_PYTHON)
    loc = locales.load("zh")
    session = session_of(())
    with pytest.raises(DraftError, match="空"):
        render.check_draft(render.ContentDraft(session="   "), case, loc)
    with pytest.raises(DraftError, match="围栏"):
        render.check_draft(render.ContentDraft(session=f"```{session}```"), case, loc)
    with pytest.raises(DraftError, match="太短"):
        render.check_draft(render.ContentDraft(session="短"), case, loc)
    with pytest.raises(DraftError, match="太长"):
        render.check_draft(
            render.ContentDraft(session="会" * (render.MAX_CONTENT_CHARS + 10)), case, loc
        )
    without_note = "\n".join(
        line for line in session.splitlines() if not line.startswith("Note:")
    )
    with pytest.raises(DraftError, match="缺这几行"):
        render.check_draft(render.ContentDraft(session=without_note), case, loc)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_check_draft_rejects_a_pasted_rule(lang: str) -> None:
    """输入：把判据原文**整段贴**进会话。输出：无。

    预期行为：抛 `DraftError`，理由是贴判据原文——提示词里唯一可抄的具体文字就是它，贴进去
    标签照样算命中（正则只看子串），但那段会话读起来不是调试器输出（范式 §9.45）。用 D3 那条
    判据：它的原文里同时出现 `Stack:` 与帧分隔符，贴进去之后按字面算仍然命中 D3，正好落在
    贴判据那道闸上（而不是先被命中组合那道闸拦下）；它也是四条里最短的一条，贴进去之后整屏
    仍然在长度区间内。
    """
    loc = locales.load(lang)
    rule = str(render.tag_copy(loc, "D3")["rule"])
    case = DebugStepCase("D3", features.RUNTIME_PYTHON)
    session = f"{session_of(('D3',))} {rule}"
    with pytest.raises(DraftError, match="判据原文"):
        render.check_draft(render.ContentDraft(session=session), case, loc)


def test_pasted_rules_only_flags_verbatim_text() -> None:
    """输入：整段贴 / 少了一个字的两种会话。输出：无。

    预期行为：只认**逐字相同**的整段判据原文——改一个字符就不再算，否则正常产物会被误伤。
    """
    loc = locales.load("zh")
    rule = str(render.tag_copy(loc, "D1")["rule"])
    assert render.pasted_rules(rule, loc) == [rule]
    assert render.pasted_rules(rule[:-2], loc) == []
    assert render.pasted_rules(session_of(()), loc) == []


def test_runtime_copy_reports_a_missing_shape() -> None:
    """输入：缺运行时的语言字典。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError, match="运行时 go"):
        render.runtime_copy({"runtime": {"python": {"desc": "Python"}}}, "go")


@pytest.mark.parametrize(
    ("session", "expect"),
    [
        (session_of(()), "手动暂停，先看看当前这一帧里的取值喵"),
        ("Stopped: pause requested (app.py:42, in run)", ""),
        ("note: 小写键名也认", "小写键名也认"),
    ],
)
def test_note_text_reports_only_the_note_line(session: str, expect: str) -> None:
    """输入：几种会话。输出：无。

    预期行为：只取 `Note:` 那一行的说明文字（没有那一行给空串）——这是这一族里唯一的自然语言
    正文，语言尺子只喂它（范式 §9.50）。
    """
    assert render.note_text(session) == expect
