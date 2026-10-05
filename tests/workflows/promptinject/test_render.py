"""提示注入的文本外壳测试：提示词、渲染、反解与语言识别。"""

from __future__ import annotations

import pytest

from jc.jsontypes import JsonValue

from tests.workflows.promptinject.conftest import (
    BASE,
    FakeBackend,
    spec_for,
    text_of,
)
from workflows.promptinject import features, locales, render

#: 全空间 48 组 case（16 组命中 × 3 种素材来源）：下面好几条测试都拿它当「都要过」的清单。
CASES = features.build_cases(512)


def test_prompt_lists_required_and_forbidden_fragments() -> None:
    """输入：一个命中 T1+T3 的 case。输出：无。

    预期行为：提示词里逐条给出**必须出现**的片段与**不许出现**的反面写法——少写一半，模型就会
    顺手带上别的形状，命中组合当场多出几条。
    """
    loc = locales.load("zh")
    prompt = render.build_prompt(loc, features.PromptInjectCase("T1+T3", features.SOURCE_WEB))
    assert str(loc["tag"]["T1"]["example"]) in prompt
    assert str(loc["tag"]["T3"]["example"]) in prompt
    assert str(loc["tag"]["T2"]["avoid"]) in prompt
    assert str(loc["tag"]["T4"]["avoid"]) in prompt
    assert str(loc["prompt"]["role_line"]) not in prompt  # 没要求 T2，就不提版面那条


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_says_the_fragments_are_only_examples(lang: str) -> None:
    """输入：两种语言 × 一个有命中的 case 与一个空命中的 case。输出：无。

    预期行为：给了具体例子就必须同时说明「那只是例子、要自己另编」——否则模型会逐条照抄，
    样本里 T3 全是同一个网址（范式 §9.25）。空命中的 case 不给例子，这句也就不该出现。
    """
    loc = locales.load(lang)
    prompt = render.build_prompt(loc, features.PromptInjectCase("T1+T3", features.SOURCE_WEB))
    assert str(loc["prompt"]["vary_rule"]) in prompt
    empty = render.build_prompt(loc, features.PromptInjectCase("", features.SOURCE_WEB))
    assert str(loc["prompt"]["vary_rule"]) not in empty


def test_prompt_says_none_may_be_hit_when_the_plan_is_empty() -> None:
    """输入：一个空命中的 case。输出：无。预期行为：走「一条都不许带」那一句，不提诱饵。"""
    loc = locales.load("zh")
    prompt = render.build_prompt(loc, features.PromptInjectCase("", features.SOURCE_WEB))
    assert str(loc["prompt"]["no_hits"]) in prompt
    assert str(loc["prompt"]["hits_header"]) not in prompt
    assert str(loc["prompt"]["role_line"]) not in prompt


def test_prompt_states_the_source_shape() -> None:
    """输入：两种语言 × 三种素材来源。输出：无。

    预期行为：这一段内容该长成什么样（网页正文 / 工具返回值 / 文件内容）在提示词里说清楚——
    不点明形态，模型写出来的东西四不像；点明之后这一维才是**只影响渲染、不影响标签**的。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in ["en", "zh"]:
        for source in features.SOURCES:
            try:
                loc = locales.load(lang)
                desc = render.source_copy(loc, source)
                prompt = render.build_prompt(loc, features.PromptInjectCase("T1", source))
                assert desc in prompt
                assert str(loc["prompt"]["source_line"]).format(source=desc) in prompt
            except AssertionError as exc:
                bad.append(f"{(lang, source)!r}: {exc}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_pins_the_line_start_when_role_spoofing_is_required(lang: str) -> None:
    """输入：要求命中 T2 / 不要求命中 T2 的 case。输出：无。

    预期行为：T2 是**唯一**与版面有关的判据（行首），所以只在要求命中它时才多贴一句版面说明——
    真机第一批 9 条里 5 条拒稿都是「模型把 `system:` 写进了句子中间」。
    """
    loc = locales.load(lang)
    role_line = str(loc["prompt"]["role_line"])
    assert role_line in render.build_prompt(
        loc, features.PromptInjectCase("T2", features.SOURCE_WEB)
    )
    assert role_line not in render.build_prompt(
        loc, features.PromptInjectCase("T1", features.SOURCE_WEB)
    )


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_state_round_trips(lang: str) -> None:
    """输入：两种语言 × 32 组 case。输出：无。

    预期行为：内容原样反解回来、判据表齐全、语言按**内容抬头**认得出。
    """
    for case in CASES:
        text = text_of(case)
        state = render.to_state(render.ContentDraft(text=text), locales.load(lang))
        parsed = render.parse_state(state)
        assert parsed.found
        assert parsed.tags == features.TAG_IDS
        assert parsed.text == text, case.describe()
        assert render.detect_locale(state) == lang


def test_state_keeps_the_tag_table_and_the_thresholds() -> None:
    """输入：一段 state。输出：无。预期行为：判据名与四档处置都印在 state 里（题面自带材料）。"""
    loc = locales.load("zh")
    state = render.to_state(render.ContentDraft(text=BASE), loc)
    assert str(loc["state"]["tag_mark"]) in state
    assert str(loc["state"]["rule_mark"]) in state
    for tag in features.TAG_IDS:
        assert str(loc["tag"][tag]["name"]) in state
    for level in features.HANDLING_LEVELS:
        assert str(loc["fam"]["action"]["opt"][level]) in state


def test_content_mentioning_a_tag_does_not_break_the_parse() -> None:
    """输入：内容里提到「注入判据」四个字。输出：无。

    预期行为：内容抬头之后一律当内容——内容里复述判据表不会把后面的行数成结构。
    """
    text = BASE + "\n我看了注入判据，也看了处置档位，一条都没带上。"
    state = render.to_state(render.ContentDraft(text=text), locales.load("zh"))
    parsed = render.parse_state(state)
    assert parsed.text == text
    assert parsed.tags == features.TAG_IDS


@pytest.mark.parametrize(
    ("switch", "case", "message"),
    [
        ("too_short", features.PromptInjectCase("T1", features.SOURCE_WEB), "太短"),
        ("fence", features.PromptInjectCase("T1", features.SOURCE_WEB), "围栏"),
        ("drop_fragment", features.PromptInjectCase("T1", features.SOURCE_WEB), "命中组合"),
        ("extra_fragment", features.PromptInjectCase("T1", features.SOURCE_WEB), "命中组合"),
    ],
)
def test_check_draft_rejects_broken_text(
    switch: str, case: features.PromptInjectCase, message: str
) -> None:
    """输入：各种「写歪」的开关。输出：无。预期行为：当场报错，理由里点名坏在哪。"""
    backend = FakeBackend(**{switch: True})
    with pytest.raises(ValueError, match=message):
        render.check_draft(render.ContentDraft(text=backend.text_for(case)), case)


def test_check_draft_rejects_empty_and_overlong_text() -> None:
    """输入：空内容与超长内容。输出：无。预期行为：两条都当场报错。"""
    case = features.PromptInjectCase("", features.SOURCE_WEB)
    with pytest.raises(ValueError, match="空"):
        render.check_draft(render.ContentDraft(text="   "), case)
    with pytest.raises(ValueError, match="太长"):
        render.check_draft(
            render.ContentDraft(text="甲" * (render.MAX_CONTENT_CHARS + 1)), case
        )


def test_build_task_carries_the_injected_features() -> None:
    """输入：一个规格。输出：无。预期行为：任务 id、域与 spec 都对得上（假后端要靠它造草稿）。"""
    case = features.PromptInjectCase("T2", features.SOURCE_TOOL)
    task = render.build_task(spec_for(case), locales.load("zh"))
    assert task.task_id == "promptinject-train-0000001-render"
    assert task.domain == "promptinject"
    assert task.response_model is render.ContentDraft
    assert task.spec["hits"] == ["T2"]
    assert task.spec["source"] == features.SOURCE_TOOL


def test_threshold_line_reuses_the_action_copy() -> None:
    """输入：四档处置。输出：无。

    预期行为：state 里那张阈值表与 `action` 那一族的选项取的是同一份文案——两边不可能各说各的。
    """
    loc = locales.load("zh")
    for level in features.HANDLING_LEVELS:
        line = render.threshold_line(loc, level)
        assert str(loc["state"]["threshold"][level]) in line
        assert str(loc["fam"]["action"]["opt"][level]) in line


def test_tag_copy_reports_missing_fields() -> None:
    """输入：缺字段 / 缺判据的文案表。输出：无。预期行为：都报人话，不静默给半句话。"""
    loc: dict[str, JsonValue] = {"tag": {"T1": {"name": "x", "rule": "y", "example": "z"}}}
    with pytest.raises(ValueError, match="avoid"):
        render.tag_copy(loc, "T1")
    with pytest.raises(ValueError, match="T9"):
        render.tag_copy(loc, "T9")


def test_source_copy_reports_a_missing_shape() -> None:
    """输入：缺形态说明 / 缺来源键的文案表。输出：无。预期行为：都报人话。"""
    loc: dict[str, JsonValue] = {"source": {"web": {"desc": "  "}}}
    with pytest.raises(ValueError, match="web"):
        render.source_copy(loc, "web")
    with pytest.raises(ValueError, match="file"):
        render.source_copy(loc, "file")
