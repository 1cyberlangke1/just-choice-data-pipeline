"""渲染与反解：任务表逐格对得上、坏 state 当场报人话、两种语言都走得通。"""

from __future__ import annotations

from typing import Final

import pytest

from tests.workflows.planstep.conftest import _make_spec
from workflows.planstep import features as ft
from workflows.planstep import locales, parse, render

LANGS: Final[tuple[str, ...]] = ("zh", "en")
ALL_CASES: Final[tuple[ft.PlanCase, ...]] = tuple(ft.build_cases(512))


def state_of(case: ft.PlanCase, locale: str = "zh") -> str:
    """输入：一个 case、语言。输出：渲染出来的 `state` 全文。"""
    return render.build(_make_spec(case.graph, case.done), locales.load(locale))


def _state(task_lines: list[str], locale: str = "zh") -> str:
    """输入：任务行原文、语言。输出：一段最小 state（抬头 + 那几行 + 一行规则）。

    预期行为：坏 state 的测试自己拼文本，不去篡改渲染结果——这样「渲染坏了」与
    「反解太宽松」是两条独立的证据链。
    """
    loc = locales.load(locale)
    meta = loc["meta"]
    head = [f"{meta['request']} P-TEST", f"[{meta['tasks']}]", *task_lines]
    head.extend([f"[{meta['rules']}]", f"{meta['rules_prefix']}规则一"])
    return "\n".join(head) + "\n"


def test_render_layout_is_the_expected_shape() -> None:
    """输入：链式图、什么都没做的 case。输出：无。

    预期行为：抬头一行、任务表五行（编号 1 起、三格）、规则段四条——结构与语言无关。
    """
    loc = locales.load("zh")
    meta = loc["meta"]
    text = state_of(ft.PlanCase(graph=0, done=0))
    lines = text.splitlines()
    assert lines[0].startswith(str(meta["request"]))
    assert lines[1] == f"[{meta['tasks']}]"
    assert lines[2:7] == [
        "  1. 需求梳理 | 待开始 | 无前置",
        "  2. 接口设计 | 待开始 | 前置=1",
        "  3. 核心实现 | 待开始 | 前置=2",
        "  4. 集成联调 | 待开始 | 前置=3",
        "  5. 上线发布 | 待开始 | 前置=4",
    ]
    assert lines[7] == f"[{meta['rules']}]"
    assert len(lines[8:]) == len(loc["rules"]["lines"])
    assert text.endswith("\n")


def test_render_is_deterministic() -> None:
    """输入：同一个 case 渲染两次。输出：无。预期行为：逐字节一致（纯函数，不掷骰子）。"""
    case = ft.PlanCase(graph=5, done=2)
    assert state_of(case) == state_of(case)
    assert state_of(case, "en") == state_of(case, "en")


def test_deps_text_writes_the_none_word_or_a_prefixed_list() -> None:
    """输入：空前置与两项前置。输出：无。预期行为：两种形状各一种写法，编号从 1 起。"""
    loc = locales.load("zh")
    meta = loc["meta"]
    assert render.deps_text(loc, ()) == str(meta["deps_none"])
    assert render.deps_text(loc, (0, 2)) == f"{meta['deps_prefix']}1{meta['deps_sep']}3"


def test_task_name_reports_a_missing_key() -> None:
    """输入：语言表里没有的任务键。输出：无。预期行为：报错——悄悄退回键名会把「没翻译」藏进产物。"""
    loc = locales.load("zh")
    with pytest.raises(ValueError, match="T9"):
        render.task_name(loc, "T9")


@pytest.mark.parametrize("lang", LANGS)
def test_roundtrip_is_exact_for_every_case(lang: str) -> None:
    """输入：全空间 case × 两种语言。输出：无。

    预期行为：反解出来的依赖表、已完成集合、请求号、规则段与真值规格逐项一致——这是
    「渲染把任务写歪了」那条路的第一道闸。
    """
    tables = parse.tables_of(locales.load(lang))
    for case in ALL_CASES:
        spec = _make_spec(case.graph, case.done)
        parsed = parse.parse_state(state_of(case, lang), tables)
        assert parsed.deps == case.deps, case.describe()
        assert parsed.done_set == case.done_set, case.describe()
        assert parsed.count == ft.TASK_COUNT
        assert parsed.request_id == spec.query_id
        assert parsed.rules == tuple(str(line) for line in locales.load(lang)["rules"]["lines"])


def test_parse_reads_a_minimal_state() -> None:
    """输入：手拼的最小 state。输出：无。预期行为：两项、依赖与状态都对得上。"""
    parsed = parse.parse_state(
        _state(["  1. 需求梳理 | 待开始 | 无前置", "  2. 接口设计 | 已完成 | 前置=1"]), 
        parse.merged_tables(),
    )
    assert parsed.request_id == "P-TEST"
    assert parsed.deps == ((), (0,))
    assert parsed.done_set == frozenset({1})
    assert parsed.tasks[0].name == "需求梳理"


def test_parse_skips_blank_lines_inside_the_table() -> None:
    """输入：任务行之间夹空行。输出：无。预期行为：空行跳过，编号仍然连续。"""
    parsed = parse.parse_state(
        _state(["  1. 需求梳理 | 待开始 | 无前置", "", "  2. 接口设计 | 待开始 | 前置=1"]),
        parse.merged_tables(),
    )
    assert parsed.count == 2


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        ([], "一项都没有"),
        (["  2. 需求梳理 | 待开始 | 无前置"], "跳号"),
        (["  1. 需求梳理 | 待开始"], "格"),
        (["  1. 需求梳理 | 乱写 | 无前置"], "状态词"),
        (["  1. 需求梳理 | 待开始 | 后面"], "认不出来"),
        (["  1. 需求梳理 | 待开始 | 前置=9"], "越界"),
        (["  1. 需求梳理 | 待开始 | 前置=1,1"], "重复"),
        (["  1. 需求梳理 | 待开始 | 前置=1"], "自己列成了前置"),
        (["  1. 需求梳理 | 待开始 | 前置=x"], "不是整数"),
    ],
)
def test_parse_reports_broken_task_rows(lines: list[str], message: str) -> None:
    """输入：各种写歪的任务行。输出：无。预期行为：每一条都报人话，不静默算出一个假答案。"""
    with pytest.raises(ValueError, match=message):
        parse.parse_state(_state(lines), parse.merged_tables())


def test_parse_reports_a_broken_header_and_short_input() -> None:
    """输入：抬头认不出的文本、太短的文本、非字符串。输出：无。预期行为：三种都当场报人话。"""
    with pytest.raises(ValueError, match="抬头认不出来"):
        parse.parse_state(_state([]).replace("计划推进请求号", "乱写"), parse.merged_tables())
    with pytest.raises(ValueError, match="太短"):
        parse.parse_state("a\nb", parse.merged_tables())
    with pytest.raises(ValueError, match="不是字符串"):
        parse.parse_state(123, parse.merged_tables())


def test_parse_reports_content_before_any_section() -> None:
    """输入：区块抬头之前就有一行内容。输出：无。预期行为：报错——那说明渲染漏了抬头。"""
    text = "计划推进请求号 P-TEST\n  1. 需求梳理 | 待开始 | 无前置\n[推进规则]\n- 规则一\n"
    with pytest.raises(ValueError, match="任何区块抬头之前"):
        parse.parse_state(text, parse.merged_tables())


def test_tables_of_rejects_empty_labels() -> None:
    """输入：抬头 / 格分隔符被写成空白。输出：无。预期行为：当场报错，不拖到解析阶段。"""
    loc = dict(locales.load("zh"))
    loc["meta"] = dict(loc["meta"], request="  ")
    with pytest.raises(ValueError, match="抬头不能是空串"):
        parse.tables_of(loc)
    loc["meta"] = dict(locales.load("zh")["meta"], cell_sep=" ")
    with pytest.raises(ValueError, match="格分隔符不能是空白"):
        parse.tables_of(loc)


def test_merged_tables_cover_both_languages() -> None:
    """输入：无。输出：无。

    预期行为：合并表收齐两种语言的抬头、区块、状态词、前置写法，且结构词元只有一份
    ——`Workflow.reparse` 没有语言参数，全靠它认行。
    """
    tables = parse.merged_tables()
    assert set(tables.request) == {"计划推进请求号", "Plan step request"}
    assert set(tables.sections) == {"[任务表]", "[推进规则]", "[Tasks]", "[Advance rules]"}
    assert tables.done_words == frozenset({"已完成", "done"})
    assert tables.pending_words == frozenset({"待开始", "pending"})
    assert tables.deps_none == frozenset({"无前置", "no deps"})
    assert set(tables.deps_prefix) == {"前置=", "deps="}
    assert tables.cell_sep == "|" and tables.deps_sep == "," and tables.rules_prefix == "- "


def test_merged_tables_reparse_works_for_both_languages() -> None:
    """输入：两种语言渲染的同一个 case。输出：无。预期行为：合并表能把两边都反解成同一张依赖表。"""
    case = ft.PlanCase(graph=4, done=3)
    tables = parse.merged_tables()
    zh = parse.parse_state(state_of(case, "zh"), tables)
    en = parse.parse_state(state_of(case, "en"), tables)
    assert zh.deps == en.deps == case.deps
    assert zh.done_set == en.done_set == case.done_set
