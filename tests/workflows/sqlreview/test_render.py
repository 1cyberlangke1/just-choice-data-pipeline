"""文本外壳测试：提示词该说的话都说了、state 能来回转、写歪的评审记录在渲染期就被拦下。"""

from __future__ import annotations

import pytest

from jc.jsontypes import JsonValue
from jc.workflow import DraftError

from tests.workflows.sqlreview.conftest import packet_of, spec_for
from workflows.sqlreview import features, locales, render
from workflows.sqlreview.features import SqlReviewCase

#: 一个普通 case：命中两条，变更发生在事务型库上。
CASE = SqlReviewCase("Q1+Q3", features.WORKLOAD_OLTP)


def test_prompt_lists_required_rules_and_forbidden_shapes() -> None:
    """输入：一个命中两条的 case。输出：无。

    预期行为：要求出现的两条各自印上**判据原文**（那是模型唯一的规格来源，示例记录一律不给，
    见 `build_prompt`）；不许出现的那两条各自带着反面写法——少列一条，模型就可能顺手把那条也
    写上，命中组合当场多出来。
    """
    loc = locales.load("zh")
    prompt = render.build_prompt(loc, CASE)
    for tag in CASE.tags():
        assert str(render.tag_copy(loc, tag)["rule"]) in prompt
    for tag in ("Q2", "Q4"):
        assert str(render.tag_copy(loc, tag)["avoid"]) in prompt


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_format_spec_only_offers_placeholders(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：格式规格那四行**只有占位符、没有一个能整段照抄的成品**——每一行的取值都得模型自己
    编，规格里只印键名与 `<...>`；行数正好四个结构键。
    """
    loc = locales.load(lang)
    lines = render.format_lines(loc)
    assert len(lines) == len(features.KEY_LINES)
    for key, line in zip(features.KEY_LINES, lines, strict=True):
        assert line.startswith(render.LINE_PREFIX)
        assert key in line.lower(), f"{lang}.{key}"
        assert "<" in line and ">" in line, f"{lang}.{key}"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_note_spec_forbids_restating_the_format(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：`note:` 那一行的规格里**明说不要复述格式与要求**。`subqa` 上量到过：模型把备注写成
    复述合成规格的句子（`sq-10b` zh 侧 9 条里 3 条）。备注是这一族唯一一段自然语言正文，写成
    复述规格的句子之后整屏记录读起来就是「写手在照着规格交作业」，而不是一次真实变更。
    """
    loc = locales.load(lang)
    spec = str(loc["prompt"]["format"]["note"])
    assert "格式" in spec or "format" in spec.lower()
    assert "不要" in spec or "do **not**" in spec.lower()


@pytest.mark.parametrize(
    ("lang", "needle"),
    [("zh", "怎么执行"), ("en", "how to run")],
)
def test_note_spec_forbids_describing_the_shape(lang: str, needle: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：`note:` 那一行的规格里**明说备注写的是「怎么执行」**，并且禁止描述语句 / 计划 /
    结构变更**本身有什么性质**。真机上量到过：首版只禁了「复述格式」，模型就换个写法把判据
    复述一遍——`sqr-10a-zh` 9 条里 6 条是「这条语句没有加 WHERE 条件会直接修改全表」「结构变更
    带了防重入保护」这种句子。那等于把标签写进了材料：考生不用看 SQL，读备注就能答题（范式 §9.55
    在 `subqa` 上量到的是同一种失效）。
    """
    loc = locales.load(lang)
    spec = str(loc["prompt"]["format"]["note"])
    assert needle in spec
    assert "不要" in spec or "Do **not**" in spec


def test_prompt_says_none_may_be_tripped_when_the_plan_is_empty() -> None:
    """输入：空命中的 case。输出：无。预期行为：改成「一条都不许踩」的说法，且不出现要求段落。"""
    loc = locales.load("zh")
    case = SqlReviewCase("", features.WORKLOAD_OLTP)
    prompt = render.build_prompt(loc, case)
    assert str(loc["prompt"]["no_hits"]) in prompt
    assert str(loc["prompt"]["hits_header"]) not in prompt


def test_prompt_states_the_workload_shape() -> None:
    """输入：一种语言 × 一种变更场景。输出：无。预期行为：提示词里点明这次变更发生在哪类业务上。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in ["en", "zh"]:
        for workload in ["oltp", "analytics", "migration"]:
            try:
                loc = locales.load(lang)
                case = SqlReviewCase("Q2", workload)
                prompt = render.build_prompt(loc, case)
                assert render.workload_copy(loc, workload) in prompt
            except AssertionError as exc:
                bad.append(f"{(lang, workload)!r}: {exc}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_state_round_trips(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：写出去再读回来，判据表与记录一字不差。"""
    loc = locales.load(lang)
    packet = packet_of(CASE.tags())
    state = render.to_state(render.ContentDraft(packet=packet), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.tags == features.TAG_IDS
    assert parsed.text == packet
    assert render.detect_locale(state) == lang


def test_state_keeps_the_tag_table_and_the_thresholds() -> None:
    """输入：一段 state。输出：无。

    预期行为：判据表四行与档位表四行都印在 state 里，档位表那几行用的是与题面**同一份**处置文案。
    """
    loc = locales.load("zh")
    state = render.to_state(render.ContentDraft(packet=packet_of(CASE.tags())), loc)
    for tag in features.TAG_IDS:
        assert render.tag_line(loc, tag) in state
    for level in features.ACTS:
        assert render.threshold_line(loc, level) in state


def test_state_prints_each_criterion_rule() -> None:
    """输入：两种语言 × 一段 state。输出：无。

    预期行为：判据表每行都带上该判据的**原文**——只印判据名时，反解模型认不出「大表全表扫描」
    指的是什么形状（范式 §9.44）。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        state = render.to_state(render.ContentDraft(packet=packet_of(CASE.tags())), loc)
        for tag in features.TAG_IDS:
            assert str(render.tag_copy(loc, tag)["rule"]) in state


def test_multiline_packet_round_trips() -> None:
    """输入：记录末尾多一行。输出：无。预期行为：多行原样保住。"""
    loc = locales.load("zh")
    packet = packet_of(CASE.tags()) + "\nnote: 末尾又补了一句备注"
    state = render.to_state(render.ContentDraft(packet=packet), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.text == packet
    assert parsed.tags == features.TAG_IDS


def test_packet_mentioning_a_rule_does_not_break_the_parse() -> None:
    """输入：记录里出现一行「像判据表」的内容。输出：无。

    预期行为：记录抬头之后的一切都当正文——否则记录里出现一行 `- Q9：假的` 就会把判据表数歪。
    """
    loc = locales.load("zh")
    packet = packet_of(CASE.tags()) + "\n- Q9：这一行是正文，不是判据表"
    parsed = render.parse_with(render.to_state(render.ContentDraft(packet=packet), loc), loc)
    assert parsed.tags == features.TAG_IDS
    assert "- Q9：这一行是正文，不是判据表" in str(parsed.text)


def test_check_draft_accepts_every_faithful_packet() -> None:
    """输入：全空间 48 组 case 的忠实记录。输出：无。预期行为：一条都不报错。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        render.check_draft(render.ContentDraft(packet=packet_of(case.tags())), case, loc)


@pytest.mark.parametrize("tags", [("Q1",), ("Q1", "Q2", "Q4")])
def test_check_draft_rejects_broken_packet(tags: tuple[str, ...]) -> None:
    """输入：少写一条 / 多写一条的记录。输出：无。

    预期行为：抛 `DraftError`，理由里说清「算出来的组合」与「计划要的」各是什么。
    """
    with pytest.raises(DraftError, match="命中组合"):
        render.check_draft(render.ContentDraft(packet=packet_of(tags)), CASE, locales.load("zh"))


def test_check_draft_rejects_empty_overlong_fence_and_too_short() -> None:
    """输入：空 / 超长 / 带围栏 / 太短 / 缺一行 的五种记录。输出：无。

    预期行为：五种都当场抛错，且各自命中该有的那道闸（顺序：空 → 围栏 → 太短 → 太长 → 缺键）
    ——闸门顺序搞反的话，报出来的理由会指错方向。
    """
    case = SqlReviewCase("", features.WORKLOAD_OLTP)
    loc = locales.load("zh")
    packet = packet_of(())
    with pytest.raises(DraftError, match="空"):
        render.check_draft(render.ContentDraft(packet="   "), case, loc)
    with pytest.raises(DraftError, match="围栏"):
        render.check_draft(render.ContentDraft(packet=f"```{packet}```"), case, loc)
    with pytest.raises(DraftError, match="太短"):
        render.check_draft(render.ContentDraft(packet="短"), case, loc)
    with pytest.raises(DraftError, match="太长"):
        render.check_draft(
            render.ContentDraft(packet="会" * (render.MAX_CONTENT_CHARS + 10)), case, loc
        )
    without_note = "\n".join(
        line for line in packet.splitlines() if not line.startswith("note:")
    )
    with pytest.raises(DraftError, match="缺这几行"):
        render.check_draft(render.ContentDraft(packet=without_note), case, loc)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_check_draft_rejects_a_pasted_rule(lang: str) -> None:
    """输入：把判据原文**整段贴**进记录。输出：无。

    预期行为：抛 `DraftError`，理由是贴判据原文——提示词里唯一可抄的具体文字就是它，贴进去
    标签照样算命中（正则只看子串），但那段文字读起来不是一屏变更记录（范式 §9.45）。用 `Q1`
    那条判据：贴进去之后按字面算仍然只命中 `Q1`（它自己举的例子带着 `WHERE`，正好把这条判据
    的负向条件解掉），而且整屏仍在长度区间内——正好落在贴判据那道闸上。
    """
    loc = locales.load(lang)
    rule = str(render.tag_copy(loc, "Q1")["rule"])
    case = SqlReviewCase("Q1", features.WORKLOAD_OLTP)
    with pytest.raises(DraftError, match="判据原文"):
        render.check_draft(render.ContentDraft(packet=f"{packet_of(('Q1',))} {rule}"), case, loc)


def test_pasted_rules_only_flags_verbatim_text() -> None:
    """输入：整段贴 / 少了一个字的两种记录。输出：无。

    预期行为：只认**逐字相同**的整段判据原文——改一个字符就不再算，否则正常产物会被误伤。
    """
    loc = locales.load("zh")
    rule = str(render.tag_copy(loc, "Q2")["rule"])
    assert render.pasted_rules(rule, loc) == [rule]
    assert render.pasted_rules(rule[:-2], loc) == []
    assert render.pasted_rules(packet_of(()), loc) == []


def test_workload_copy_reports_a_missing_shape() -> None:
    """输入：缺变更场景的语言字典。输出：无。预期行为：抛 ValueError。"""
    with pytest.raises(ValueError, match="变更场景 migration"):
        render.workload_copy({"workload": {"oltp": {"desc": "事务型"}}}, "migration")


@pytest.mark.parametrize(
    ("packet", "expect"),
    [
        (packet_of(()), "这次只改一行、走主键索引，评审过一遍，直接执行没问题喵"),
        ("statement: x", ""),
        ("note: 小写键名也认", "小写键名也认"),
    ],
)
def test_note_text_reports_only_the_note_line(packet: str, expect: str) -> None:
    """输入：几种记录。输出：无。

    预期行为：只取 `note:` 那一行的备注（没有那一行给空串）——这是这一族里唯一一段自然语言正文
    （范式 §9.50）。
    """
    assert render.note_text(packet) == expect
