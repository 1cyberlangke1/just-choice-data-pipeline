"""文本外壳测试：三张表怎么渲染、state 怎么拼、草稿的八道闸各自拦什么。"""

from __future__ import annotations

import pytest

from jc.workflow import DraftError

from tests.workflows.depcheck.conftest import (
    BASE_LINES,
    HIT_LINES,
    lines_of,
    packet_of,
    render_lines,
    sample_of,
)
from workflows.depcheck import features, judges, locales, render
from workflows.depcheck.features import DepCheckCase

CASE = DepCheckCase("D1+D2", features.ECO_PYPI)

#: 这个 case 相容的许可证标识符（`D1` 命中 → 传染性那一半的第一个）。
LICENSE_FOR_CASE = features.license_pool(CASE.tags())[0]


def draft_of(packet: str) -> render.ContentDraft:
    """输入：一屏记录。输出：草稿对象。预期行为：只包一层，不做校验。"""
    return render.ContentDraft(packet=packet)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_fact_lines_line_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：四条标记行两两不同、编号排在分隔符前面（解析就是按这个切回编号的）、判据原文在里面。
    """
    loc = locales.load(lang)
    sep = str(loc["state"]["tag_sep"])
    lines = [render.fact_line(loc, tag) for tag in features.TAG_IDS]
    assert len(set(lines)) == len(lines)
    for tag, line in zip(features.TAG_IDS, lines, strict=True):
        assert line.partition(sep)[0] == tag
        assert str(render.fact_copy(loc, tag)["rule"]) in line


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_rule_and_action_tables_are_complete(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：定级规则两行（公式 + 权重）、档位表五行，行行不同；公式里带上限、权重行就是
    `features.LEVEL_WEIGHTS` 的展开。
    """
    loc = locales.load(lang)
    rule = render.rule_lines(loc)
    assert len(rule) == 2
    assert len(set(rule)) == len(rule)
    assert str(features.MAX_LEVEL) in rule[0]
    assert rule[1] == render.weight_line() == "D1=2 D2=1 D3=2 D4=1"
    actions = render.action_lines(loc)
    assert len(actions) == len(features.LEVELS) == 5
    assert len(set(actions)) == len(actions)
    for level, line in zip(features.LEVELS, actions, strict=True):
        assert render.level_text(loc, level) in line
        assert render.act_text(loc, features.act_for(level)) in line


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_lists_both_required_and_forbidden_shapes(lang: str) -> None:
    """输入：一种语言 × 一个「取到两条」的 case。输出：无。

    预期行为：取到的那两条印出判据原文与「必须落在它那一行」，没取到的那两条印出反面写法；
    长度区间、六行格式规格、生态、依赖类别与自查都在；**一个示例记录都不给**（不出现任何成品的
    取值）。
    """
    loc = locales.load(lang)
    prompt = render.build_prompt(loc, CASE, 0, 47, LICENSE_FOR_CASE)
    for tag in CASE.tags():
        assert str(render.fact_copy(loc, tag)["rule"]) in prompt
        assert str(render.fact_copy(loc, tag)["name"]) in prompt
    missed = [tag for tag in features.TAG_IDS if tag not in CASE.tags()]
    for tag in missed:
        assert str(render.fact_copy(loc, tag)["avoid"]) in prompt
    assert str(render.MIN_CONTENT_CHARS) in prompt
    assert render.eco_copy(loc, CASE.ecosystem) in prompt
    assert locales.pick(loc["prompt"]["categories"], 0) in prompt
    assert render.version_shape(47) in prompt
    for key in features.KEY_LINES:
        assert f"`{key}: " in prompt
    # 许可证标识符是**样本级旋钮**（范式 §9.61）：提示词里就该出现那**一个**要照抄的值，所以它
    # 不算「可照抄的成品」；除它之外仍然一个成品取值都不许有（照抄 `count=3` 就直接命中 D3）。
    assert LICENSE_FOR_CASE in prompt
    assert "count=3" not in prompt and "0.9.1" not in prompt


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_has_no_leftover_placeholder(lang: str) -> None:
    """输入：一种语言 × 全空间 48 组 case。输出：无。

    预期行为：提示词与反解提示词渲染完之后**一个花括号都不剩**——漏填会在 `format` 上抛错，多写的
    花括号会原样漏给模型，让它以为那是要填的槽。
    """
    loc = locales.load(lang)
    for case in features.build_cases(512):
        prompt = render.build_prompt(loc, case, 0, 47, features.license_pool(case.tags())[0])
        assert "{" not in prompt and "}" not in prompt, case.describe()
    trace = judges.build_trace_prompt("[state]", loc)
    assert "{" not in trace and "}" not in trace


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_state_round_trips(lang: str) -> None:
    """输入：一种语言 × 一屏兑现计划的记录。输出：无。

    预期行为：反解出来的标记表、定级规则、档位表与正文与渲染时逐字相同。
    """
    loc = locales.load(lang)
    packet = packet_of(CASE.tags(), lang)
    state = render.to_state(draft_of(packet), loc)
    parsed = render.parse_with(state, loc)
    assert parsed.found
    assert parsed.facts == features.TAG_IDS
    assert parsed.rule == tuple(render.rule_lines(loc))
    assert parsed.actions == tuple(render.action_lines(loc))
    assert parsed.text == packet


def test_detect_locale_tells_the_two_languages_apart() -> None:
    """输入：zh / en 各一段 state。输出：无。预期行为：各自认回自己的语言，认不出的给 None。"""
    zh = render.to_state(draft_of(packet_of(("D1",), "zh")), locales.load("zh"))
    en = render.to_state(draft_of(packet_of(("D1",), "en")), locales.load("en"))
    assert render.detect_locale(zh) == "zh"
    assert render.detect_locale(en) == "en"
    assert render.detect_locale("这里什么抬头都没有") is None
    assert render.detect_locale(None) is None


def test_note_text_reads_the_prose_line() -> None:
    """输入：一屏记录。输出：无。预期行为：取 `note:` 那一行；没有那一行就给空串。"""
    assert render.note_text(packet_of(())) == BASE_LINES["note"].removeprefix("note: ")
    assert render.note_text(render_lines({"package": "package: X"})) == ""


@pytest.mark.parametrize(
    ("packet", "needle"),
    [
        ("   ", "空的"),
        (render_lines(BASE_LINES) + "\n```", "围栏"),
        ("短", "太短"),
        (render_lines(BASE_LINES) + "x" * 700, "太长"),
        ("note: " + "只写一行" * 30, "缺这几行"),
        (
            render_lines(
                {**BASE_LINES, "package": "package: " + "x" * 40, "note": "note:"}
            ),
            "唯一一段自然语言正文",
        ),
        (render_lines({**BASE_LINES, "advisories": HIT_LINES["D3"]}), "命中组合"),
        (
            render_lines(
                {**BASE_LINES, "package": "package: 见 advisories: count=3 那条"}
            ),
            "命中组合",
        ),
    ],
)
def test_check_draft_blocks_each_broken_shape(packet: str, needle: str) -> None:
    """输入：八种「写歪」的记录。输出：无。预期行为：每一种都被 `check_draft` 拦下，理由说得清。"""
    case = DepCheckCase("", features.ECO_NPM)
    with pytest.raises(DraftError, match=needle):
        render.check_draft(draft_of(packet), case, locales.load("zh"))


def test_check_draft_accepts_a_faithful_packet() -> None:
    """输入：一屏兑现计划的记录。输出：无。预期行为：放行（闸的下限）。"""
    case = DepCheckCase("D2+D3", features.ECO_NPM)
    render.check_draft(draft_of(packet_of(case.tags())), case, locales.load("zh"))


def test_pasted_rule_text_is_caught() -> None:
    """输入：把标记原文整段贴进记录。输出：无。

    预期行为：报出来——提示词里唯一能整段照抄的就是标记原文，贴进去说明写手在照抄规格。
    """
    loc = locales.load("zh")
    rule = str(render.fact_copy(loc, "D1")["rule"])
    packet = render_lines({**BASE_LINES, "license": f"license: {rule}"})
    assert render.pasted_rules(packet, loc) == [rule]
    assert render.pasted_rules(packet_of(("D1",)), loc) == []


def test_note_verdicts_finds_the_conclusion_words() -> None:
    """输入：几种备注。输出：无。

    预期行为：只认**结论词**（「归档」「漏洞」这种），不认**要求写**的内容（「维护」「维护者」）——
    词表要挡的是「把结构键上的标签搬进自由文字」，不是正常描述（范式 §9.59）。
    """
    loc = locales.load("zh")
    assert render.note_verdicts("内部工具，上游已经归档不再维护喵", loc)
    assert render.note_verdicts("目前没有已知漏洞", loc) == ["漏洞"]
    assert render.note_verdicts("上游已经停止更新了", loc)
    assert render.note_verdicts("由社区维护者长期维护喵", loc) == []
    assert render.note_verdicts("a small helper, no longer maintained", locales.load("en"))


def test_check_draft_blocks_a_note_with_a_risk_verdict() -> None:
    """输入：把风险结论写进 `note:` 那一行（别的闸全合格）。输出：无。

    预期行为：被**结论闸**拦下——命中组合、长度、结构键、抄规格全都合格，只有「这一行不许下结论」
    这一条不成立，所以拦下它的只能是 `note_verdicts`。
    """
    loc = locales.load("zh")
    case = DepCheckCase("D2", features.ECO_NPM)
    packet = render_lines(
        {
            **BASE_LINES,
            "status": HIT_LINES["D2"],
            "note": "note: 这个包用来把结构化日志写进收集端，上游已经归档不再维护，但功能稳定所以引入喵",
        }
    )
    assert features.missing_keys(packet) == ()
    assert features.hits_of(packet) == "D2"
    assert render.MIN_CONTENT_CHARS <= len(packet) <= render.MAX_CONTENT_CHARS
    assert render.pasted_rules(packet, loc) == []
    with pytest.raises(DraftError, match="风险结论"):
        render.check_draft(draft_of(packet), case, loc)


def test_verdict_words_rejects_a_missing_table() -> None:
    """输入：没有 `[hygiene]` 那一张表的语言字典。输出：无。预期行为：当场报人话。"""
    with pytest.raises(ValueError, match="verdict_words"):
        render.verdict_words({"prompt": {}})


def test_state_is_self_evident_for_every_combination() -> None:
    """输入：全空间 16 组。输出：无。

    预期行为：任一组记录渲染出来的 state 里，标记表、定级规则与档位表都在（三张表齐全），而且按
    字面重算 state 里的正文得到的组合就是那一组——题面的答案确实能从 state 里读出来。
    """
    loc = locales.load("zh")
    for value in features.HIT_SETS:
        tags = features.parse_hits(value)
        state = render.to_state(draft_of(packet_of(tags)), loc)
        parsed = render.parse_state(state)
        assert parsed.facts == features.TAG_IDS
        assert parsed.rule and parsed.actions
        assert features.hits_of(parsed.text or "") == value
        assert features.level_of(tags) in features.LEVELS


def test_sample_of_helper_builds_a_parseable_state() -> None:
    """输入：一屏记录。输出：无。预期行为：夹具造出来的样本能被 `language_texts` 读出一段正文。"""
    row = sample_of(packet_of(("D3",)))
    assert render.parse_state(row["state"]).text is not None
    assert render.note_text(render.parse_state(row["state"]).text or "")


def test_lines_of_keeps_every_key() -> None:
    """输入：任意命中组合。输出：无。预期行为：六行齐全（夹具自己不许漏行，否则测试会假绿）。"""
    for value in features.HIT_SETS:
        lines = lines_of(features.parse_hits(value))
        assert sorted(lines) == sorted(features.KEY_LINES)
