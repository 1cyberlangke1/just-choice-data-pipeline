"""渲染的测试：纯函数、四种版式、日期/金额格式、与反解的往返一致。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.invoice import locales, parse, render, spec as sp

LOCALES = locales.available()
SHAPES = [(layout, style) for layout in sp.LAYOUTS for style in sp.LABEL_STYLES]


@pytest.mark.parametrize("lang", LOCALES)
@pytest.mark.parametrize("fmt", sp.DATE_FORMATS)
def test_format_date_puts_the_year_first(lang: str, fmt: str) -> None:
    """输入：日期、格式名、语言。输出：文本。

    预期行为：年在前（反解按「四位年 + 月 + 日」读），且能原样反解回原日期。
    """
    text = render.format_date((2026, 3, 7), fmt, locales.load(lang))
    assert text.startswith("2026")
    assert parse.as_date(text) == (2026, 3, 7)


@pytest.mark.parametrize("lang", LOCALES)
def test_format_money_uses_the_locale_symbol_and_thousands(lang: str) -> None:
    """输入：金额、货币、语言。输出：文本。预期行为：符号取自语言文件，带千分位。"""
    loc = locales.load(lang)
    assert render.format_money(1234567, "cny", loc) == "¥1,234,567"
    assert render.format_money(0, "eur", loc) == "€0"
    assert render.format_money(98, "usd", loc) == "$98"


@pytest.mark.parametrize("lang", LOCALES)
def test_unknown_currency_date_format_or_slot_raises(lang: str, make_spec: Any) -> None:
    """输入：语言文件里没有的货币 / 日期格式 / 槽名。输出：KeyError。

    预期行为：一律报错——静默回落会让「渲染出来的值」与「真值」悄悄分叉。
    """
    loc = locales.load(lang)
    with pytest.raises(KeyError):
        render.format_money(1, "jpy", loc)
    with pytest.raises(KeyError):
        render.format_date((2026, 1, 1), "iso", loc)
    with pytest.raises(KeyError):
        render.field_text(make_spec(21), "nope", loc)


@pytest.mark.parametrize("lang", LOCALES)
def test_visible_fields_drops_tax_without_a_tax_row(lang: str, make_spec: Any) -> None:
    """输入：有/没有税额行的规格。输出：会写出来的字段槽。

    预期行为：没有税额行就整行不写 `tax`——写出来会被往返判别器判成「多出字段」。
    """
    loc = locales.load(lang)
    assert "tax" in render.visible_fields(make_spec(22, tax_state="ok"))
    without = make_spec(22, tax_state="none")
    assert "tax" not in render.visible_fields(without)
    assert set(render.expected_fields(without, loc)) == set(render.visible_fields(without))


@pytest.mark.parametrize("lang", LOCALES)
@pytest.mark.parametrize("layout,labels", SHAPES)
def test_build_roundtrips_through_the_parser(
    lang: str, layout: str, labels: str, make_spec: Any
) -> None:
    """输入：规格与语言。输出：单据文本。

    预期行为：四种版式 × 两种标签风格渲染出来的字段与明细，反解后与真值**逐字**相同
    （货币符号、千分位、日期写法都算在内）。
    """
    loc = locales.load(lang)
    spec = make_spec(31, layout=layout, labels=labels, lines=3, defects=1, tax_state="ok")
    parsed = parse.parse_document(render.build(spec, loc), loc)
    assert parsed.fields == render.expected_fields(spec, loc)
    assert [(i.name, i.qty, i.unit, i.amount) for i in parsed.items] == render.expected_items(
        spec, loc
    )


@pytest.mark.parametrize("lang", LOCALES)
def test_build_is_a_pure_function(lang: str, make_spec: Any) -> None:
    """输入：同一份规格渲染两次。输出：两段文本。预期行为：逐字节相同（渲染不用 RNG）。"""
    loc = locales.load(lang)
    spec = make_spec(41)
    assert render.build(spec, loc) == render.build(spec, loc)


def test_paragraph_layout_prints_the_intro_once(make_spec: Any) -> None:
    """输入：段落版式的一份单据。输出：文本。

    预期行为：引言只出现一次（抬头与金额区各渲染一次，引言只能出在第一段），
    正文与金额区各压成一行，字段之间用语言文件里的分隔符。
    """
    loc = locales.load("zh")
    spec = make_spec(51, layout="paragraph", tax_state="ok")
    lines = render.build(spec, loc).split("\n")
    assert render.build(spec, loc).count(loc["meta"]["intro"]) == 1
    assert lines[1] == loc["meta"]["intro"]
    assert loc["meta"]["sep"] in lines[2]
    assert all(render.field_text(spec, slot, loc) in lines[2] for slot in render.HEAD_FIELDS)


@pytest.mark.parametrize("layout", sp.LAYOUTS)
def test_english_text_carries_no_cjk_punctuation(layout: str, make_spec: Any) -> None:
    """输入：英文语言文件。输出：四种版式的单据文本。

    预期行为：英文单据里不出现「：」「；」「。」——标点由语言文件给，
    英文用 `: ` / `; `，否则这批数据看着就不像英文。
    """
    loc = locales.load("en")
    text = render.build(make_spec(61, layout=layout, tax_state="ok"), loc)
    assert not (set("：；。") & set(text))
    assert loc["meta"]["colon"] in text


def test_display_width_counts_cjk_as_two() -> None:
    """输入：文本。输出：显示宽度。

    预期行为：东亚宽字符按 2 格算——按字符数补齐会让等宽表参差不齐
    （`len("票号") == 2` 但实际占 4 格）。
    """
    assert render.display_width("票号") == 4
    assert render.display_width("INV-1") == 5
    assert render.display_width("") == 0
    assert render.display_width("税a") == 3


def test_aligned_table_starts_every_value_in_the_same_column(make_spec: Any) -> None:
    """输入：中文（双宽标签）的对齐表格单据。输出：文本。

    预期行为：所有字段行的值从同一显示列开始——按显示宽度判断，而不是字符数。
    """
    loc = locales.load("zh")
    spec = make_spec(71, layout="aligned_table", tax_state="ok")
    text = render.build(spec, loc)
    starts = set()
    for slot in render.visible_fields(spec):
        label = locales.label_of(loc, slot, spec.case.labels)
        value = render.field_text(spec, slot, loc)
        line = next(line for line in text.split("\n") if line.startswith(label))
        assert line.endswith(value)
        starts.add(render.display_width(line[: -len(value)]))
    assert len(starts) == 1
