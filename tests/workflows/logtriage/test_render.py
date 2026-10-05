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
