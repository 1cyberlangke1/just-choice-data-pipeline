"""文本外壳测试：提示词该说的话都说了、state 能来回转、写歪的字幕对在渲染期就被拦下。"""

from __future__ import annotations

import re

import pytest

from jc.jsontypes import JsonValue
from jc.workflow import DraftError

from tests.workflows.subqa.conftest import cue_of, spec_for
from workflows.subqa import features, locales, render
from workflows.subqa.features import SubQaCase

#: 一个普通 case：命中两条，字幕对来自电影。
CASE = SubQaCase("S1+S3", features.TRACK_MOVIE)

#: 提示词里填的开始时间码（取池子里的第一个）：旋钮的取值本身由 `test_locales` 与 `test_pipeline`
#: 各自钉住，这里只需要一个合法值。
TIMECODE = features.TIMECODE_HINTS[0]


def test_prompt_lists_required_rules_and_forbidden_shapes() -> None:
    """输入：一个命中两条的 case。输出：无。

    预期行为：要求出现的两条各自印上**判据原文**（那是模型唯一的规格来源，示例字幕对一律不给，
    见 `build_prompt`）；不许出现的那两条各自带着反面写法——少列一条，模型就可能顺手把那条也
    写上，命中组合当场多出来。
    """
    loc = locales.load("zh")
    prompt = render.build_prompt(loc, CASE, TIMECODE)
    for tag in CASE.tags():
        assert str(render.tag_copy(loc, tag)["rule"]) in prompt
        assert "example" not in render.tag_copy(loc, tag)
    for tag in ("S2", "S4"):
        assert str(render.tag_copy(loc, tag)["avoid"]) in prompt


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_tells_the_model_to_compose_its_own_cue(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：提示词里明说判据里那些词只是**形状**、字幕对要自己组织、判据原文不许原样贴进去
    （范式 §9.45：给成品句就会被整段照抄，所以只给形状 + 明说不许贴）。
    """
    loc = locales.load(lang)
    assert str(loc["prompt"]["vary_rule"]) in render.build_prompt(loc, CASE, TIMECODE)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_format_spec_offers_no_copyable_timecode(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：格式规格里**不许出现填好的时间码**——给一对成品时间码，模型照抄下去正好是
    「零时长」（范式 §9.52）。规格只印占位符。
    """
    loc = locales.load(lang)
    for line in render.format_lines(loc):
        assert not re.search(r"\d{1,2}:\d{2}:\d{2}[,.]\d{3}", line), line


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_note_spec_forbids_restating_the_format(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：`Note:` 那一行的规格里**明说不要复述格式与要求**。真机上量到过：模型把备注写成
    「译文保留了音效标记，时间码保持零时长以便闪现提醒」这种复述合成规格的句子（`sq-10b` zh 侧
    9 条里 3 条、en 侧 10 条里 2 条）。备注是这一族两段自然语言正文之一，写成复述规格的句子之后
    整条样本读起来就是「写手在照着规格交作业」，而不是一对真字幕。
    """
    loc = locales.load(lang)
    spec = str(loc["prompt"]["format"]["note"])
    assert "格式" in spec or "format" in spec.lower()
    assert "不要" in spec or "do **not**" in spec.lower()


def test_prompt_says_none_may_be_tripped_when_the_plan_is_empty() -> None:
    """输入：空命中的 case。输出：无。预期行为：改成「一条都不许踩」的说法，且不出现要求段落。"""
    loc = locales.load("zh")
    case = SubQaCase("", features.TRACK_MOVIE)
    prompt = render.build_prompt(loc, case, TIMECODE)
    assert str(loc["prompt"]["no_hits"]) in prompt
    assert str(loc["prompt"]["hits_header"]) not in prompt


def test_prompt_states_the_track_shape() -> None:
    """输入：一种语言 × 一种素材。输出：无。预期行为：提示词里点明这对字幕来自哪种素材。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in ["en", "zh"]:
        for track in ["movie", "series", "lecture"]:
            try:
                loc = locales.load(lang)
                case = SubQaCase("S2", track)
                prompt = render.build_prompt(loc, case, TIMECODE)
                assert render.track_copy(loc, track) in prompt
            except AssertionError as exc:
                bad.append(f"{(lang, track)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_prompt_has_no_unfilled_slots() -> None:
    """输入：全空间 48 组 case。输出：无。预期行为：渲染完一个花括号都不剩。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        prompt = render.build_prompt(loc, case, TIMECODE)
        assert "{" not in prompt and "}" not in prompt, case.describe()


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_state_round_trips(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：写出去再读回来，判据表与字幕对一字不差。"""
    loc = locales.load(lang)
    cue = cue_of(CASE.tags())
    state = render.to_state(render.ContentDraft(cue=cue), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.tags == features.TAG_IDS
    assert parsed.text == cue
    assert render.detect_locale(state) == lang


def test_state_keeps_the_tag_table_and_the_thresholds() -> None:
    """输入：一段 state。输出：无。

    预期行为：判据表四行与档位表四行都印在 state 里，档位表那几行用的是与题面**同一份**处置文案。
    """
    loc = locales.load("zh")
    state = render.to_state(render.ContentDraft(cue=cue_of(CASE.tags())), loc)
    for tag in features.TAG_IDS:
        assert render.tag_line(loc, tag) in state
    for level in features.ACTS:
        assert render.threshold_line(loc, level) in state


def test_state_prints_each_criterion_rule() -> None:
    """输入：两种语言 × 一段 state。输出：无。

    预期行为：判据表每行都带上该判据的**原文**——只印判据名时，反解模型认不出「零时长」指的是
    什么形状（范式 §9.44）。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        state = render.to_state(render.ContentDraft(cue=cue_of(CASE.tags())), loc)
        for tag in features.TAG_IDS:
            assert str(render.tag_copy(loc, tag)["rule"]) in state


def test_multiline_cue_round_trips() -> None:
    """输入：字幕对末尾多一行。输出：无。预期行为：多行原样保住。"""
    loc = locales.load("zh")
    cue = cue_of(CASE.tags()) + "\nNote: 末尾又补了一句备注"
    state = render.to_state(render.ContentDraft(cue=cue), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.text == cue
    assert parsed.tags == features.TAG_IDS


def test_cue_mentioning_a_rule_does_not_break_the_parse() -> None:
    """输入：字幕对里出现一行「像判据表」的内容。输出：无。

    预期行为：字幕对抬头之后的一切都当正文——否则字幕里出现一行 `- S9：假的` 就会把判据表数歪。
    """
    loc = locales.load("zh")
    cue = cue_of(CASE.tags()) + "\n- S9：这一行是正文，不是判据表"
    parsed = render.parse_with(render.to_state(render.ContentDraft(cue=cue), loc), loc)
    assert parsed.tags == features.TAG_IDS
    assert "- S9：这一行是正文，不是判据表" in str(parsed.text)


def test_check_draft_accepts_every_faithful_cue() -> None:
    """输入：全空间 48 组 case 的忠实字幕对。输出：无。预期行为：一条都不报错。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        render.check_draft(render.ContentDraft(cue=cue_of(case.tags())), case, loc)


@pytest.mark.parametrize("tags", [("S1",), ("S1", "S2", "S4")])
def test_check_draft_rejects_broken_cue(tags: tuple[str, ...]) -> None:
    """输入：少写一条 / 多写一条的字幕对。输出：无。

    预期行为：抛 `DraftError`，理由里说清「算出来的组合」与「计划要的」各是什么。
    """
    with pytest.raises(DraftError, match="命中组合"):
        render.check_draft(render.ContentDraft(cue=cue_of(tags)), CASE, locales.load("zh"))


def test_check_draft_rejects_empty_overlong_fence_and_too_short() -> None:
    """输入：空 / 超长 / 带围栏 / 太短 / 缺一行 的五种字幕对。输出：无。

    预期行为：五种都当场抛错，且各自命中该有的那道闸（顺序：空 → 围栏 → 太短 → 太长 → 缺键）
    ——闸门顺序搞反的话，报出来的理由会指错方向。
    """
    case = SubQaCase("", features.TRACK_MOVIE)
    loc = locales.load("zh")
    cue = cue_of(())
    with pytest.raises(DraftError, match="空"):
        render.check_draft(render.ContentDraft(cue="   "), case, loc)
    with pytest.raises(DraftError, match="围栏"):
        render.check_draft(render.ContentDraft(cue=f"```{cue}```"), case, loc)
    with pytest.raises(DraftError, match="太短"):
        render.check_draft(render.ContentDraft(cue="短"), case, loc)
    with pytest.raises(DraftError, match="太长"):
        render.check_draft(
            render.ContentDraft(cue="会" * (render.MAX_CONTENT_CHARS + 10)), case, loc
        )
    without_note = "\n".join(
        line for line in cue.splitlines() if not line.startswith("Note:")
    )
    with pytest.raises(DraftError, match="缺这几行"):
        render.check_draft(render.ContentDraft(cue=without_note), case, loc)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_check_draft_rejects_a_pasted_rule(lang: str) -> None:
    """输入：把判据原文**整段贴**进字幕对。输出：无。

    预期行为：抛 `DraftError`，理由是贴判据原文——提示词里唯一可抄的具体文字就是它，贴进去
    标签照样算命中（正则只看子串），但那段文字读起来不是一条字幕（范式 §9.45）。用 S3 那条
    判据：贴进去之后按字面算仍然只命中 S3（它自己举的例子是 `HH:MM:SS,mmm` 这种占位写法，
    不是真时间码），正好落在贴判据那道闸上（而不是先被命中组合那道闸拦下）；它也是四条里
    最短的一条，贴进去之后整条仍在长度区间内。
    """
    loc = locales.load(lang)
    rule = str(render.tag_copy(loc, "S3")["rule"])
    case = SubQaCase("S3", features.TRACK_MOVIE)
    cue = f"{cue_of(('S3',))} {rule}"
    with pytest.raises(DraftError, match="判据原文"):
        render.check_draft(render.ContentDraft(cue=cue), case, loc)


def test_pasted_rules_only_flags_verbatim_text() -> None:
    """输入：整段贴 / 少了一个字的两种字幕对。输出：无。

    预期行为：只认**逐字相同**的整段判据原文——改一个字符就不再算，否则正常产物会被误伤。
    """
    loc = locales.load("zh")
    rule = str(render.tag_copy(loc, "S1")["rule"])
    assert render.pasted_rules(rule, loc) == [rule]
    assert render.pasted_rules(rule[:-2], loc) == []
    assert render.pasted_rules(cue_of(()), loc) == []


def test_track_copy_reports_a_missing_shape() -> None:
    """输入：缺素材类型的语言字典。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError, match="素材类型 lecture"):
        render.track_copy({"track": {"movie": {"desc": "电影"}}}, "lecture")


@pytest.mark.parametrize(
    ("cue", "expect"),
    [
        (cue_of(()), "如果还方便的话，我们得把会议挪到周四。"),
        ("Source: x", ""),
        ("target: 小写键名也认", "小写键名也认"),
    ],
)
def test_target_text_reports_only_the_target_line(cue: str, expect: str) -> None:
    """输入：几种字幕对。输出：无。

    预期行为：只取 `Target:` 那一行的译文（没有那一行给空串）——这是这一族里两段自然语言正文
    之一，语言尺子只喂它（范式 §9.50）。
    """
    assert render.target_text(cue) == expect


@pytest.mark.parametrize(
    ("cue", "expect"),
    [
        (cue_of(()), "这句直译就行，没有什么术语要查喵"),
        ("Target: 只有译文这一行", ""),
        ("note: 小写键名也认", "小写键名也认"),
    ],
)
def test_note_text_reports_only_the_note_line(cue: str, expect: str) -> None:
    """输入：几种字幕对。输出：无。

    预期行为：只取 `Note:` 那一行的备注（没有那一行给空串）——另一段自然语言正文（范式 §9.50）。
    """
    assert render.note_text(cue) == expect
