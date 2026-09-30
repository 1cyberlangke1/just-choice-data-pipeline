"""文本外壳测试：三张表怎么渲染、state 怎么拼、草稿的六道闸各自拦什么。"""

from __future__ import annotations

import pytest

from jc.workflow import DraftError

from tests.workflows.alertrisk.conftest import (
    BASE_LINES,
    HIT_LINES,
    lines_of,
    packet_of,
    render_lines,
    sample_of,
)
from workflows.alertrisk import features, judges, locales, render
from workflows.alertrisk.features import AlertRiskCase

CASE = AlertRiskCase("R1+R2", features.SOURCE_WAF)


def draft_of(packet: str) -> render.ContentDraft:
    """输入：一屏记录。输出：草稿对象。预期行为：只包一层，不做校验。"""
    return render.ContentDraft(packet=packet)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_fact_lines_line_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：三条因子行两两不同、编号排在分隔符前面（解析就是按这个切回编号的）、判据原文在里面。
    """
    loc = locales.load(lang)
    sep = str(loc["state"]["tag_sep"])
    lines = [render.fact_line(loc, tag) for tag in features.TAG_IDS]
    assert len(set(lines)) == len(lines)
    for tag, line in zip(features.TAG_IDS, lines, strict=True):
        assert line.partition(sep)[0] == tag
        assert str(render.fact_copy(loc, tag)["rule"]) in line


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_matrix_and_action_tables_are_complete(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：矩阵八行、档位表五行，行行不同；矩阵左边就是 `features.HIT_SETS` 的顺序，右边是
    该组合查出来的等级名。
    """
    loc = locales.load(lang)
    matrix = render.matrix_lines(loc)
    assert len(matrix) == len(features.HIT_SETS) == 8
    assert len(set(matrix)) == len(matrix)
    for value, line in zip(features.HIT_SETS, matrix, strict=True):
        highs = features.parse_hits(value)
        assert line.startswith(features.combo_text_key(highs))
        assert render.level_text(loc, features.level_of(highs)) in line
    actions = render.action_lines(loc)
    assert len(actions) == len(features.LEVELS) == 5
    assert len(set(actions)) == len(actions)
    for level, line in zip(features.LEVELS, actions, strict=True):
        assert render.act_text(loc, features.act_for(level)) in line


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_format_spec_lines_up_with_key_lines(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：格式规格的键与 `KEY_LINES` 一一对应，而且**每一行都带占位符**——规格只印「键名 +
    占位符」，不给能整段照抄的成品（范式 §9.52）。
    """
    loc = locales.load(lang)
    assert sorted(loc["prompt"]["format"]) == sorted(features.KEY_LINES)
    lines = render.format_lines(loc)
    assert len(lines) == len(features.KEY_LINES)
    for key, line in zip(features.KEY_LINES, lines, strict=True):
        assert line.startswith(f"{render.LINE_PREFIX}`{key}: ")
        assert "<" in line and ">" in line, line


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_lists_both_required_and_forbidden_shapes(lang: str) -> None:
    """输入：一种语言 × 一个「取到两条」的 case。输出：无。

    预期行为：取到的那两条印出判据原文与「必须落在它那一行」，没取到的那一条印出反面写法；
    长度区间、五行格式规格与自查都在；**一个示例记录都不给**（不出现任何一行的成品取值）。
    """
    loc = locales.load(lang)
    prompt = render.build_prompt(loc, CASE, 0)
    for tag in CASE.tags():
        assert str(render.fact_copy(loc, tag)["rule"]) in prompt
        assert str(render.fact_copy(loc, tag)["name"]) in prompt
    missed = [tag for tag in features.TAG_IDS if tag not in CASE.tags()]
    for tag in missed:
        assert str(render.fact_copy(loc, tag)["avoid"]) in prompt
    assert str(render.MIN_CONTENT_CHARS) in prompt
    for key in features.KEY_LINES:
        assert f"`{key}: " in prompt
    assert "hosts=24" not in prompt and "reachable=public --" not in prompt


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_has_no_leftover_placeholder(lang: str) -> None:
    """输入：一种语言 × 全空间 24 组 case。输出：无。

    预期行为：提示词与反解提示词渲染完之后**一个花括号都不剩**——漏填会在 `format` 上抛错，多写的
    花括号会原样漏给模型，让它以为那是要填的槽。
    """
    loc = locales.load(lang)
    for case in features.build_cases(512):
        prompt = render.build_prompt(loc, case, 0)
        assert "{" not in prompt and "}" not in prompt, case.describe()
    trace = judges.build_trace_prompt("[state]", loc)
    assert "{" not in trace and "}" not in trace


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_state_round_trips(lang: str) -> None:
    """输入：一种语言 × 一屏兑现计划的记录。输出：无。

    预期行为：反解出来的因子表、矩阵、档位表与正文与渲染时逐字相同。
    """
    loc = locales.load(lang)
    packet = packet_of(CASE.tags(), lang)
    state = render.to_state(draft_of(packet), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.found
    assert parsed.facts == features.TAG_IDS
    assert parsed.matrix == tuple(render.matrix_lines(loc))
    assert parsed.actions == tuple(render.action_lines(loc))
    assert parsed.text == packet


def test_detect_locale_tells_the_two_languages_apart() -> None:
    """输入：zh / en 各一段 state。输出：无。预期行为：各自认回自己的语言，认不出的给 None。"""
    zh = render.to_state(draft_of(packet_of(("R1",), "zh")), locales.load("zh"))
    en = render.to_state(draft_of(packet_of(("R1",), "en")), locales.load("en"))
    assert render.detect_locale(zh) == "zh"
    assert render.detect_locale(en) == "en"
    assert render.detect_locale("这里什么抬头都没有") is None
    assert render.detect_locale(None) is None


def test_parse_state_survives_junk() -> None:
    """输入：非字符串、认不出的字符串。输出：无。预期行为：给空结果，不抛异常。"""
    for value in (None, 42, {"a": 1}, "没有抬头的文本"):
        parsed = render.parse_state(value)
        assert parsed.text is None
        assert parsed.facts == ()
        assert not parsed.found


def test_detail_text_reads_the_prose_line() -> None:
    """输入：一屏记录。输出：无。预期行为：取 `detail:` 那一行；没有那一行就给空串。"""
    assert render.detail_text(packet_of(())) == BASE_LINES["detail"].removeprefix("detail: ")
    assert render.detail_text(render_lines({"rule": "rule: X"})) == ""


@pytest.mark.parametrize(
    ("packet", "needle"),
    [
        ("   ", "空的"),
        (render_lines(BASE_LINES) + "\n```", "围栏"),
        ("短", "太短"),
        (render_lines(BASE_LINES) + "x" * 700, "太长"),
        ("detail: " + "只写一行" * 20, "缺这几行"),
        (render_lines({**BASE_LINES, "detail": "detail:"}), "唯一一段自然语言正文"),
        (render_lines({**BASE_LINES, "impact": HIT_LINES["R3"]}), "命中组合"),
        (render_lines({**BASE_LINES, "impact": BASE_LINES["impact"], "rule": "rule: 见 exposure: reachable=public 那条"}), "命中组合"),
    ],
)
def test_check_draft_blocks_each_broken_shape(packet: str, needle: str) -> None:
    """输入：八种「写歪」的记录。输出：无。预期行为：每一种都被 `check_draft` 拦下，理由说得清。"""
    case = AlertRiskCase("", features.SOURCE_EDR)
    with pytest.raises(DraftError, match=needle):
        render.check_draft(draft_of(packet), case, locales.load("zh"))


def test_check_draft_accepts_a_faithful_packet() -> None:
    """输入：一屏兑现计划的记录。输出：无。预期行为：放行（闸的下限）。"""
    case = AlertRiskCase("R2+R3", features.SOURCE_EDR)
    render.check_draft(draft_of(packet_of(case.tags())), case, locales.load("zh"))


def test_pasted_rule_text_is_caught() -> None:
    """输入：把因子原文整段贴进记录。输出：无。

    预期行为：报出来——提示词里唯一能整段照抄的就是因子原文，贴进去说明写手在照抄规格。
    """
    loc = locales.load("zh")
    rule = str(render.fact_copy(loc, "R1")["rule"])
    packet = render_lines({**BASE_LINES, "rule": f"rule: {rule}"})
    assert render.pasted_rules(packet, loc) == [rule]
    assert render.pasted_rules(packet_of(("R1",)), loc) == []


def test_state_is_self_evident_for_every_combination() -> None:
    """输入：全空间 8 组。输出：无。

    预期行为：任一组记录渲染出来的 state 里，因子表、矩阵与档位表都在（三张表齐全），而且按字面
    重算 state 里的正文得到的组合就是那一组——题面的答案确实能从 state 里读出来。
    """
    loc = locales.load("zh")
    for value in features.HIT_SETS:
        tags = features.parse_hits(value)
        state = render.to_state(draft_of(packet_of(tags)), loc)
        parsed = render.parse_state(state)
        assert parsed.facts == features.TAG_IDS
        assert parsed.matrix and parsed.actions
        assert features.hits_of(parsed.text or "") == value
        assert features.level_of(tags) in features.LEVELS


def test_sample_of_helper_builds_a_parseable_state() -> None:
    """输入：一屏记录。输出：无。预期行为：夹具造出来的样本能被 `language_texts` 读出一段正文。"""
    row = sample_of(packet_of(("R3",)))
    assert render.parse_state(row["state"]).text is not None
    assert render.detail_text(render.parse_state(row["state"]).text or "")


def test_lines_of_keeps_every_key() -> None:
    """输入：任意命中组合。输出：无。预期行为：五行齐全（夹具自己不许漏行，否则测试会假绿）。"""
    for value in features.HIT_SETS:
        lines = lines_of(features.parse_hits(value))
        assert sorted(lines) == sorted(features.KEY_LINES)
