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


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_says_the_fragments_are_only_examples(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：提示词里明说那些片段只是例子、要自己另编。"""
    loc = locales.load(lang)
    assert str(loc["prompt"]["vary_rule"]) in render.build_prompt(loc, CASE)


def test_prompt_says_none_may_be_hit_when_the_plan_is_empty() -> None:
    """输入：空命中的 case。输出：无。预期行为：改成「一条都不许踩」的说法，且不出现要求段落。"""
    loc = locales.load("zh")
    case = MemoryGateCase("", features.FORM_SENTENCE)
    prompt = render.build_prompt(loc, case)
    assert str(loc["prompt"]["no_hits"]) in prompt
    assert str(loc["prompt"]["hits_header"]) not in prompt


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("form", ["sentence", "bullet", "quote"])
def test_prompt_states_the_form_shape(lang: str, form: str) -> None:
    """输入：一种语言 × 一种写法。输出：无。预期行为：提示词里点明这条记忆写成什么形态。"""
    loc = locales.load(lang)
    case = MemoryGateCase("M2", form)
    prompt = render.build_prompt(loc, case)
    assert render.form_copy(loc, form) in prompt


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_state_round_trips(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：写出去再读回来，判据表与记忆一字不差。"""
    loc = locales.load(lang)
    text = text_of(CASE)
    state = render.to_state(render.ContentDraft(text=text), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.tags == features.TAG_IDS
    assert parsed.text == text
    assert render.detect_locale(state) == lang


def test_state_prints_each_criterion_rule() -> None:
    """输入：两种语言 × 一段 state。输出：无。

    预期行为：判据表每行都带上该判据的**原文**——只印判据名时，反解模型认不出「作用域膨胀」
    指的是什么形状（真机第一批 zh 侧 8 条拒稿全是这个原因，见 `render.tag_line` 的说明）。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        state = render.to_state(render.ContentDraft(text=text_of(CASE)), loc)
        for tag in features.TAG_IDS:
            assert str(render.tag_copy(loc, tag)["rule"]) in state


def test_multiline_content_round_trips() -> None:
    """输入：一条分点写法的记忆。输出：无。预期行为：多行原样保住（分点形态就是多行）。"""
    loc = locales.load("zh")
    text = "- 会议挪到了周四\n- 以后每次评审都照这个时间\n- 会议室还是老地方"
    state = render.to_state(render.ContentDraft(text=text), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.text == text
    assert parsed.tags == features.TAG_IDS


def test_content_mentioning_a_rule_does_not_break_the_parse() -> None:
    """输入：正文里出现一行「像判据表」的内容。输出：无。

    预期行为：记忆抬头之后的一切都当正文——否则记忆里提一句「- M9：假的」就会把判据表数歪。
    """
    loc = locales.load("zh")
    text = text_of(CASE) + "\n- M9：这一行是正文，不是判据表"
    parsed = render.parse_with(render.to_state(render.ContentDraft(text=text), loc), loc)
    assert parsed.tags == features.TAG_IDS
    assert "- M9：这一行是正文，不是判据表" in str(parsed.text)


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


def test_build_task_carries_the_injected_features() -> None:
    """输入：真值规格。输出：无。

    预期行为：任务 id 可重算、`spec` 里带上注入特征（假后端与真后端都能照着它答）、响应模型是
    草稿模型。
    """
    spec = spec_for(CASE)
    task = render.build_task(spec, locales.load("zh"))
    assert task.task_id == "memorygate-train-0000001-render"
    assert task.domain == "memorygate"
    assert task.response_model is render.ContentDraft
    assert task.spec["case"] == {"hits": "M1+M3", "form": "sentence"}
    assert task.spec["hits"] == ["M1", "M3"]
    assert task.spec["form"] == "sentence"


def test_threshold_line_reuses_the_gate_copy() -> None:
    """输入：一种语言 × 四个档位。输出：无。预期行为：档位表与 `gate` 那一族的选项文案同源。"""
    loc = locales.load("zh")
    for level in features.HANDLING_LEVELS:
        assert str(loc["fam"]["gate"]["opt"][level]) in render.threshold_line(loc, level)


def test_tag_copy_reports_missing_fields() -> None:
    """输入：缺字段的语言字典。输出：无。预期行为：抛 ValueError，点名缺了哪个字段。"""
    with pytest.raises(ValueError, match="缺判据 M1"):
        render.tag_copy({"tag": {}}, "M1")
    broken: dict[str, JsonValue] = {"tag": {"M1": {"name": "作用域膨胀"}}}
    with pytest.raises(ValueError, match="缺字段 rule"):
        render.tag_copy(broken, "M1")


def test_form_copy_reports_a_missing_shape() -> None:
    """输入：缺写法的语言字典。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError, match="写法 bullet"):
        render.form_copy({"form": {"sentence": {"desc": "一句话"}}}, "bullet")
