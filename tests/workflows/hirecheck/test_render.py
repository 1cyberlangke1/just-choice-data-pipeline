"""hirecheck 文本外壳：提示词、草稿校验、state 往返与语言识别。"""

from __future__ import annotations

import pytest

from jc.workflow import DraftError
from tests.workflows.hirecheck.conftest import BASE_LINES, packet_of
from workflows.hirecheck import features, locales, render
from workflows.hirecheck.features import HireCheckCase
from workflows.hirecheck.spec import HireCheckSpec

LANGS: tuple[str, ...] = ("zh", "en")


def case_for(tags: tuple[str, ...] = ()) -> HireCheckCase:
    """输入：该命中的判据。输出：一个挂在 `board` 渠道上的 case。"""
    return HireCheckCase(flags="+".join(tags), channel=features.CHANNEL_BOARD)


def test_format_lines_cover_every_key_in_order() -> None:
    """输入：语言字典。输出：无。预期行为：格式规格一行一个键、顺序与 `KEY_LINES` 一致。"""
    lines = render.format_lines(locales.load("zh"))
    assert len(lines) == len(features.KEY_LINES)
    for key, line in zip(features.KEY_LINES, lines, strict=True):
        assert key in line, (key, line)


def test_format_lines_report_a_missing_key() -> None:
    """输入：少一个键的格式规格。输出：无。预期行为：当场报人话，点出缺的是哪个键。"""
    loc = dict(locales.load("zh"))
    prompt = dict(loc["prompt"])
    prompt["format"] = {key: value for key, value in prompt["format"].items() if key != "degree"}
    loc["prompt"] = prompt
    with pytest.raises(ValueError, match="degree"):
        render.format_lines(loc)


def test_fact_copy_reports_missing_entry_and_field() -> None:
    """输入：缺判据 / 缺字段的语言字典。输出：无。预期行为：两种都当场报人话。"""
    loc = locales.load("zh")
    stripped = dict(loc)
    stripped["fact"] = {key: value for key, value in loc["fact"].items() if key != "H2"}
    with pytest.raises(ValueError, match="H2"):
        render.fact_copy(stripped, "H2")
    partial = dict(loc)
    partial["fact"] = dict(loc["fact"])
    partial["fact"]["H2"] = {key: value for key, value in loc["fact"]["H2"].items() if key != "avoid"}
    with pytest.raises(ValueError, match="avoid"):
        render.fact_copy(partial, "H2")


def test_level_act_and_channel_copy_report_missing_entries() -> None:
    """输入：缺等级 / 缺档位 / 缺渠道的语言字典。输出：无。预期行为：三种都当场报人话。"""
    loc = dict(locales.load("zh"))
    loc["level"] = {key: value for key, value in loc["level"].items() if key != "L3"}
    loc["act"] = {key: value for key, value in loc["act"].items() if key != "hold"}
    loc["channel"] = {
        key: value for key, value in loc["channel"].items() if key != features.CHANNEL_CAMPUS
    }
    with pytest.raises(ValueError, match="L3"):
        render.level_text(loc, 3)
    with pytest.raises(ValueError, match="hold"):
        render.act_text(loc, "hold")
    with pytest.raises(ValueError, match="campus"):
        render.channel_copy(loc, features.CHANNEL_CAMPUS)


def test_weight_line_matches_the_code_weights() -> None:
    """输入：无。输出：无。

    预期行为：权重行是**代码渲染的**、与 `LEVEL_WEIGHTS` 逐条一致——它是 state 格式的一部分，
    语言文件里没有它，所以两处不可能各说各的。
    """
    line = render.weight_line()
    for tag in features.TAG_IDS:
        assert f"{tag}={features.LEVEL_WEIGHTS[tag]}" in line


def test_rule_and_action_lines_cover_the_whole_table() -> None:
    """输入：语言字典。输出：无。预期行为：规则两行（公式 + 权重）、档位五行，一行不少。"""
    for lang in LANGS:
        loc = locales.load(lang)
        assert len(render.rule_lines(loc)) == 2
        assert render.action_lines(loc) == [render.action_line(loc, level) for level in features.LEVELS]
        assert len(render.action_lines(loc)) == len(features.LEVELS)


@pytest.mark.parametrize("lang", LANGS)
def test_check_draft_accepts_every_faithful_packet(lang: str) -> None:
    """输入：全空间每一条「完全兑现计划」的材料。输出：无。

    预期行为：十六条一条都不许被打回——校验尺子必须放过**正确**的样本，否则重试会把整批烧光。
    """
    loc = locales.load(lang)
    for value in features.HIT_SETS:
        tags = features.parse_hits(value)
        packet = packet_of(tags, lang)
        render.check_draft(render.ContentDraft(packet=packet), case_for(tags), loc)


def test_check_draft_rejects_a_wrong_hit_combination() -> None:
    """输入：材料的实际命中组合与计划不一致。输出：无。预期行为：报人话，把两个组合都写出来。"""
    with pytest.raises(DraftError, match="命中组合"):
        render.check_draft(
            render.ContentDraft(packet=packet_of(())), case_for(("H1",)), locales.load("zh")
        )


def test_check_draft_rejects_a_verdict_in_the_note() -> None:
    """输入：备注里替候选人下了初筛结论。输出：无。

    预期行为：当场打回（范式 §9.59）——结论一旦写进自由文字，考生就能绕过结构键去读备注。
    """
    lines = dict(BASE_LINES)
    lines["note"] = "note: 这个人年限不足，先放着吧"
    with pytest.raises(DraftError, match="初筛结论"):
        render.check_draft(
            render.ContentDraft(packet="\n".join(lines.values())), case_for(()), locales.load("zh")
        )


def test_check_draft_rejects_a_skill_leak_in_the_note() -> None:
    """输入：备注里提到了岗位的硬性技能。输出：无。

    预期行为：当场打回——`H2` 考的就是技能表里有没有它，备注里再提一次材料就自相矛盾。
    """
    lines = dict(BASE_LINES)
    lines["note"] = "note: 他在上一家公司搭过 kubernetes 平台"
    with pytest.raises(DraftError, match="硬性技能"):
        render.check_draft(
            render.ContentDraft(packet="\n".join(lines.values())), case_for(()), locales.load("zh")
        )


def test_check_draft_rejects_pasted_rules() -> None:
    """输入：把判据原文整段贴进材料的备注。输出：无。

    预期行为：当场打回——提示词里唯一能整段照抄的就是判据原文，抄进去说明写手在交作业而不是在写材料。
    """
    lines = dict(BASE_LINES)
    # 挑 `H1` 的原文：它里面没有硬性技能的名字，所以先撞上的一定是「贴原文」那道闸。
    lines["note"] = f"note: {render.fact_copy(locales.load('zh'), 'H1')['rule']}"
    with pytest.raises(DraftError, match="判据原文"):
        render.check_draft(
            render.ContentDraft(packet="\n".join(lines.values())), case_for(()), locales.load("zh")
        )


@pytest.mark.parametrize(
    ("packet", "needle"),
    [
        ("   ", "空"),
        ("```\n" + packet_of(()) + "\n```", "围栏"),
        ("太短了喵", "太短"),
        (packet_of(()).replace("permit: citizen", "note: " + "很长的备注" * 200), "太长"),
        (packet_of(()).replace("permit: citizen\n", ""), "缺这几行"),
    ],
)
def test_check_draft_rejects_broken_packets(packet: str, needle: str) -> None:
    """输入：各种写歪的材料。输出：无。预期行为：每一类都当场打回，理由点明是哪一类。"""
    with pytest.raises(DraftError, match=needle):
        render.check_draft(render.ContentDraft(packet=packet), case_for(()), locales.load("zh"))


def test_check_draft_rejects_an_empty_note_explicitly() -> None:
    """输入：`note:` 那一行是空的。输出：无。

    预期行为：打回并点名那一行——它是这一族唯一的自然语言正文，空了语言尺子就空转。
    """
    lines = dict(BASE_LINES)
    lines["note"] = "note:"
    with pytest.raises(DraftError, match="note"):
        render.check_draft(
            render.ContentDraft(packet="\n".join(lines.values())), case_for(()), locales.load("zh")
        )


@pytest.mark.parametrize("lang", LANGS)
def test_state_round_trip(lang: str) -> None:
    """输入：渲染出来的 state。输出：无。

    预期行为：判据表列出的编号、定级规则与档位表的规范行、以及材料那一块都能原样反解回来——
    抬头之外的内容一律不当结构。
    """
    loc = locales.load(lang)
    packet = packet_of(("H1", "H3"), lang)
    state = render.to_state(render.ContentDraft(packet=packet), loc)
    parsed = render.parse_state(state)
    assert parsed.found
    assert parsed.facts == features.TAG_IDS
    assert parsed.rule == tuple(render.rule_lines(loc))
    assert parsed.actions == tuple(render.action_lines(loc))
    assert parsed.text is not None
    assert features.hits_of(parsed.text) == "H1+H3"
    assert render.detect_locale(state) == lang


@pytest.mark.parametrize("value", [None, 42, {}, "", "没有任何抬头的文本"])
def test_parse_state_gives_up_quietly(value: object) -> None:
    """输入：不是 state 的东西。输出：无。预期行为：返回空结果，不抛异常。"""
    parsed = render.parse_state(value)
    assert not parsed.found
    assert parsed.facts == ()


def test_detect_locale_separates_the_two_languages() -> None:
    """输入：两种语言的 state。输出：无。

    预期行为：按**材料抬头**认语言（编号是结构键，认不出语言）——一次 run 里混了 zh / en 时，
    用调用方给的那一种去比会平白报一堆「文案对不上」。
    """
    for lang in LANGS:
        state = render.to_state(
            render.ContentDraft(packet=packet_of((), lang)), locales.load(lang)
        )
        assert render.detect_locale(state) == lang
    assert render.detect_locale("随便一段文本") is None


def test_note_text_reads_the_prose_line() -> None:
    """输入：材料。输出：无。预期行为：只取 `note:` 那一行的取值；没有就给空串。"""
    assert render.note_text(packet_of(())) == BASE_LINES["note"].removeprefix("note: ")
    assert render.note_text("candidate: 林素") == ""


def test_note_verdicts_reports_the_words_that_hit() -> None:
    """输入：两句备注。输出：无。预期行为：命中的结论词全报出来；干净的备注给空表。"""
    loc = locales.load("zh")
    assert render.note_verdicts("这个人年限不足，学历也不符合岗位要求", loc) == ["年限不足", "不符合"]
    assert render.note_verdicts("他在做清结算链路", loc) == []


def test_pasted_rules_only_matches_the_whole_rule() -> None:
    """输入：整段判据原文与它的一段截取。输出：无。预期行为：只有**整段**才算贴进去。"""
    loc = locales.load("zh")
    rule = str(render.fact_copy(loc, "H1")["rule"])
    assert render.pasted_rules(f"note: {rule}", loc) == [rule]
    assert render.pasted_rules("note: years 那一行里的数字", loc) == []


def test_build_task_carries_the_case_and_the_task_id() -> None:
    """输入：真值规格。输出：无。预期行为：任务 id 由规格算、`spec` 里带上注入特征。"""
    case = case_for(("H2",))
    spec = HireCheckSpec(
        case=case,
        phrasing=0,
        role=1,
        years=5,
        name=2,
        split="train",
        seq=3,
        plan=features.plan_for(case, 0),
    )
    task = render.build_task(spec, locales.load("zh"))
    assert task.task_id == spec.render_task_id == "hirecheck-train-0000003-render"
    assert task.spec["hits"] == ["H2"]
    assert task.spec["channel"] == features.CHANNEL_BOARD
    assert spec.trace_task_id == "hirecheck-train-0000003-trace"
