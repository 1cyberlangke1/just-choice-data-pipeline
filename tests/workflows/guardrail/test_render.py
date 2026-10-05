"""内容审核的文本外壳测试：提示词、渲染、反解与语言识别。"""

from __future__ import annotations

import pytest

from jc.jsontypes import JsonValue

from tests.workflows.guardrail.conftest import (
    BASE,
    FakeBackend,
    spec_for,
    text_of,
)
from workflows.guardrail import features, locales, render

#: 全空间 32 组 case：下面好几条测试都拿它当「都要过」的清单。
CASES = features.build_cases(512)


def test_prompt_lists_required_and_forbidden_fragments() -> None:
    """输入：一个命中 P1+P3 的 case。输出：无。

    预期行为：提示词里逐条给出**必须出现**的片段与**不许出现**的反面写法——少写一半，模型就会
    顺手带上别的形状，命中组合当场多出几条。
    """
    loc = locales.load("zh")
    case = features.GuardrailCase("P1+P3", features.DECOY_NONE)
    prompt = render.build_prompt(loc, case)
    for clause in ("P1", "P3"):
        pool = loc["clause"][clause]["examples"]
        assert isinstance(pool, list)
        picked = features.pick([str(value) for value in pool], case.fingerprint(), clause)
        assert picked in prompt, f"{clause}: 提示词里没给出这一条样本抽中的例子"
    assert str(loc["clause"]["P2"]["avoid"]) in prompt
    assert str(loc["clause"]["P4"]["avoid"]) in prompt
    assert features.decoy_for(case.fingerprint()) not in prompt


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_says_the_fragments_are_only_examples(lang: str) -> None:
    """输入：两种语言 × 一个有命中的 case 与一个空命中的 case。输出：无。

    预期行为：给了具体例子就必须同时说明「那只是例子、要自己另编」——否则模型会逐条照抄，
    样本里 P1 全是同一个号码（真机第一批实测，见范式 §9.25）。空命中的 case 不给例子，
    这句也就不该出现（多写一句只会让模型以为还有别的要求）。
    """
    loc = locales.load(lang)
    prompt = render.build_prompt(loc, features.GuardrailCase("P1+P3", features.DECOY_NONE))
    assert str(loc["prompt"]["vary_rule"]) in prompt
    empty = render.build_prompt(loc, features.GuardrailCase("", features.DECOY_NONE))
    assert str(loc["prompt"]["vary_rule"]) not in empty


def test_prompt_says_none_may_be_hit_when_the_plan_is_empty() -> None:
    """输入：一个空命中的 case。输出：无。预期行为：走「一条都不许踩」那一句，不提诱饵。"""
    loc = locales.load("zh")
    case = features.GuardrailCase("", features.DECOY_NONE)
    prompt = render.build_prompt(loc, case)
    assert str(loc["prompt"]["no_hits"]) in prompt
    assert str(loc["prompt"]["hits_header"]) not in prompt
    assert features.decoy_for(case.fingerprint()) not in prompt


def test_prompt_gives_the_decoy_verbatim_when_asked() -> None:
    """输入：`decoy=dashed` 的 case。输出：无。

    预期行为：那个分段号码**原样**出现在提示词里——含糊地说「写个像电话的号码」，模型会直接写
    连续数字，计划当场落空。
    """
    loc = locales.load("zh")
    case = features.GuardrailCase("", features.DECOY_DASHED)
    prompt = render.build_prompt(loc, case)
    decoy = features.decoy_for(case.fingerprint())
    assert decoy in prompt
    assert str(loc["prompt"]["decoy_line"]).format(decoy=decoy) in prompt


def test_prompt_has_no_unfilled_slots() -> None:
    """输入：两种语言 × 32 组 case。输出：无。预期行为：渲染完的提示词一个花括号都不剩。"""
    for lang in locales.available():
        loc = locales.load(lang)
        for case in CASES:
            prompt = render.build_prompt(loc, case)
            assert "{" not in prompt and "}" not in prompt, f"{lang} {case.describe()}"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_state_round_trips(lang: str) -> None:
    """输入：两种语言 × 32 组 case。输出：无。

    预期行为：正文原样反解回来、条款表齐全、语言按**正文抬头**认得出。
    """
    for case in CASES:
        text = text_of(case)
        state = render.to_state(render.ContentDraft(text=text), locales.load(lang))
        parsed = render.parse_state(state)
        assert parsed.found
        assert parsed.clauses == features.CLAUSE_IDS
        assert parsed.text == text, case.describe()
        assert render.detect_locale(state) == lang


def test_state_keeps_the_clause_table_and_the_thresholds() -> None:
    """输入：一段 state。输出：无。预期行为：条款名与四档处置都印在 state 里（题面自带材料）。"""
    loc = locales.load("zh")
    state = render.to_state(render.ContentDraft(text=BASE), loc)
    assert str(loc["state"]["clause_mark"]) in state
    assert str(loc["state"]["rule_mark"]) in state
    for clause in features.CLAUSE_IDS:
        assert str(loc["clause"][clause]["name"]) in state
    for level in features.ACTION_LEVELS:
        assert str(loc["fam"]["action"]["opt"][level]) in state


def test_content_mentioning_a_clause_does_not_break_the_parse() -> None:
    """输入：正文里提到「政策条款」四个字。输出：无。

    预期行为：正文抬头之后一律当正文——正文里复述条款表不会把后面的行数成结构。
    """
    text = BASE + "\n我看了政策条款，也看了处置阈值，都没踩。"
    state = render.to_state(render.ContentDraft(text=text), locales.load("zh"))
    parsed = render.parse_state(state)
    assert parsed.text == text
    assert parsed.clauses == features.CLAUSE_IDS


def test_parse_state_gives_up_on_unknown_input() -> None:
    """输入：几种认不出抬头的输入。输出：无。预期行为：如实给「认不出来」，不抛异常。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for garbage in ["", "随便一段没有抬头的文本", 42, None, {"state": 1}, []]:
        try:
            parsed = render.parse_state(garbage)
            assert parsed.text is None
            assert not parsed.found
            assert render.detect_locale(garbage) is None
        except AssertionError as exc:
            bad.append(f"{(garbage,)!r}: {exc}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize(
    ("switch", "case", "message"),
    [
        ("too_short", features.GuardrailCase("P1", "none"), "太短"),
        ("fence", features.GuardrailCase("P1", "none"), "围栏"),
        ("drop_fragment", features.GuardrailCase("P1", "none"), "命中组合"),
        ("extra_fragment", features.GuardrailCase("P1", "none"), "命中组合"),
        ("decoy_missing", features.GuardrailCase("P1", "dashed"), "诱饵"),
        ("decoy_extra", features.GuardrailCase("P1", "none"), "诱饵"),
    ],
)
def test_check_draft_rejects_broken_text(
    switch: str, case: features.GuardrailCase, message: str
) -> None:
    """输入：各种「写歪」的开关。输出：无。预期行为：当场报错，理由里点名坏在哪。"""
    backend = FakeBackend(**{switch: True})
    with pytest.raises(ValueError, match=message):
        render.check_draft(render.ContentDraft(text=backend.text_for(case)), case)


def test_check_draft_rejects_empty_and_overlong_text() -> None:
    """输入：空正文与超长正文。输出：无。预期行为：两条都当场报错。"""
    case = features.GuardrailCase("", "none")
    with pytest.raises(ValueError, match="空"):
        render.check_draft(render.ContentDraft(text="   "), case)
    with pytest.raises(ValueError, match="太长"):
        render.check_draft(
            render.ContentDraft(text="甲" * (render.MAX_CONTENT_CHARS + 1)), case
        )


def test_build_task_carries_the_injected_features() -> None:
    """输入：一个规格。输出：无。预期行为：任务 id、域与 spec 都对得上（假后端要靠它造草稿）。"""
    case = features.GuardrailCase("P2", features.DECOY_DASHED)
    task = render.build_task(spec_for(case), locales.load("zh"))
    assert task.task_id == "guardrail-train-0000001-render"
    assert task.domain == "guardrail"
    assert task.response_model is render.ContentDraft
    assert task.spec["hits"] == ["P2"]
    assert task.spec["decoy"] == features.DECOY_DASHED


def test_threshold_line_reuses_the_action_copy() -> None:
    """输入：四档处置。输出：无。

    预期行为：state 里那张阈值表与 `action` 那一族的选项取的是同一份文案——两边不可能各说各的。
    """
    loc = locales.load("zh")
    for level in features.ACTION_LEVELS:
        line = render.threshold_line(loc, level)
        assert str(loc["state"]["threshold"][level]) in line
        assert str(loc["fam"]["action"]["opt"][level]) in line


def test_clause_copy_reports_missing_fields() -> None:
    """输入：缺字段 / 缺条款的文案表。输出：无。预期行为：都报人话，不静默给半句话。"""
    loc: dict[str, JsonValue] = {"clause": {"P1": {"name": "x", "rule": "y", "avoid": "z"}}}
    with pytest.raises(ValueError, match="examples"):
        render.clause_copy(loc, "P1")
    with pytest.raises(ValueError, match="P9"):
        render.clause_copy(loc, "P9")
