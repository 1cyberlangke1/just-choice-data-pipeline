"""文本外壳测试：提示词该说的话都说了、state 能来回转、写歪的记忆在渲染期就被拦下。"""

from __future__ import annotations

import pytest

from jc.jsontypes import JsonValue
from jc.workflow import DraftError

from tests.workflows.memorygate.conftest import BASE, FRAGMENTS, spec_for, text_of
from workflows.memorygate import features, locales, render
from workflows.memorygate.features import MemoryGateCase

#: 一个普通 case：命中两条。
CASE = MemoryGateCase("M1+M3", features.FORM_SENTENCE)


def test_prompt_lists_required_and_forbidden_fragments() -> None:
    """输入：一个命中两条的 case。输出：无。

    预期行为：要求出现的两条各自带着例子；不许出现的那两条各自带着反面写法——少列一条，模型就
    可能顺手把那条也写上，命中组合当场多出来。
    """
    loc = locales.load("zh")
    prompt = render.build_prompt(loc, CASE)
    for tag in CASE.tags():
        assert str(render.tag_copy(loc, tag)["example"]) in prompt
    for tag in ("M2", "M4"):
        assert str(render.tag_copy(loc, tag)["avoid"]) in prompt


def test_prompt_says_none_may_be_hit_when_the_plan_is_empty() -> None:
    """输入：空命中的 case。输出：无。预期行为：改成「一条都不许踩」的说法，且不出现要求段落。"""
    loc = locales.load("zh")
    case = MemoryGateCase("", features.FORM_SENTENCE)
    prompt = render.build_prompt(loc, case)
    assert str(loc["prompt"]["no_hits"]) in prompt
    assert str(loc["prompt"]["hits_header"]) not in prompt


def test_prompt_states_the_form_shape() -> None:
    """输入：一种语言 × 一种写法。输出：无。预期行为：提示词里点明这条记忆写成什么形态。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in ["en", "zh"]:
        for form in ["sentence", "bullet", "quote"]:
            try:
                loc = locales.load(lang)
                case = MemoryGateCase("M2", form)
                prompt = render.build_prompt(loc, case)
                assert render.form_copy(loc, form) in prompt
            except AssertionError as exc:
                bad.append(f"{(lang, form)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_multiline_content_round_trips() -> None:
    """输入：一条分点写法的记忆。输出：无。预期行为：多行原样保住（分点形态就是多行）。"""
    loc = locales.load("zh")
    text = "- 会议挪到了周四\n- 以后每次评审都照这个时间\n- 会议室还是老地方"
    state = render.to_state(render.ContentDraft(text=text), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.text == text
    assert parsed.tags == features.TAG_IDS


def test_check_draft_accepts_every_faithful_text() -> None:
    """输入：全空间 48 组 case 的忠实记忆。输出：无。预期行为：一条都不报错。"""
    for case in features.build_cases(512):
        render.check_draft(render.ContentDraft(text=text_of(case)), case)


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        (BASE, "少"),
        (BASE + FRAGMENTS["M2"], "多"),
        (BASE + FRAGMENTS["M3"], "多"),
    ],
)
def test_check_draft_rejects_broken_text(text: str, fragment: str) -> None:
    """输入：少写一条 / 多写一条的记忆。输出：无。

    预期行为：抛 `DraftError`，理由里说清「算出来的组合」与「计划要的」各是什么。
    """
    with pytest.raises(DraftError, match="命中组合"):
        render.check_draft(render.ContentDraft(text=text), CASE)


def test_check_draft_rejects_empty_and_overlong_text() -> None:
    """输入：空记忆 / 超长记忆 / 带围栏的记忆。输出：无。预期行为：三种都当场抛错。"""
    case = MemoryGateCase("", features.FORM_SENTENCE)
    with pytest.raises(DraftError, match="空"):
        render.check_draft(render.ContentDraft(text="   "), case)
    with pytest.raises(DraftError, match="太长"):
        render.check_draft(
            render.ContentDraft(text="会" * (render.MAX_CONTENT_CHARS + 1)), case
        )
    with pytest.raises(DraftError, match="围栏"):
        render.check_draft(render.ContentDraft(text=f"```{BASE}```"), case)
    with pytest.raises(DraftError, match="太短"):
        render.check_draft(render.ContentDraft(text="短"), case)


def test_threshold_line_reuses_the_gate_copy() -> None:
    """输入：一种语言 × 四个档位。输出：无。预期行为：档位表与 `gate` 那一族的选项文案同源。"""
    loc = locales.load("zh")
    for level in features.HANDLING_LEVELS:
        assert str(loc["fam"]["gate"]["opt"][level]) in render.threshold_line(loc, level)


def test_form_copy_reports_a_missing_shape() -> None:
    """输入：缺写法的语言字典。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError, match="写法 bullet"):
        render.form_copy({"form": {"sentence": {"desc": "一句话"}}}, "bullet")
