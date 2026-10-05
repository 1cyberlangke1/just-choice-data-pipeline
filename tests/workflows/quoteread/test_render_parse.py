"""渲染与反解：编号、抬头、两种语言读得回来、坏 state 当场报人话。"""

from __future__ import annotations

import random
from typing import Final

import pytest

from tests.workflows.quoteread.conftest import _make_spec
from workflows.quoteread import features as ft
from workflows.quoteread import locales, parse, render
from workflows.quoteread import spec as sp

LANGS: Final[tuple[str, ...]] = ("zh", "en")


def test_state_has_a_header_and_a_section() -> None:
    """输入：一条规格。输出：无。预期行为：第一行是请求号、第二行是读数表区块抬头。"""
    loc = locales.load("zh")
    meta = loc["meta"]
    spec = _make_spec(0)
    lines = render.build(spec, loc).splitlines()
    assert lines[0].startswith(str(meta["request"]))
    assert spec.query_id in lines[0]
    assert lines[1] == f"[{meta['table']}]"


def test_row_numbering_starts_at_one() -> None:
    """输入：一条样本。输出：无。预期行为：行号从 1 起、连续、与行数一致。"""
    state = render.build(_make_spec(0), locales.load("zh"))
    numbers = [
        int(line.strip().split(".")[0]) for line in state.splitlines() if line.strip()[:1].isdigit()
    ]
    assert numbers == list(range(1, ft.ROWS + 1))


@pytest.mark.parametrize("lang", LANGS)
def test_parse_reads_back_every_case(lang: str) -> None:
    """输入：全空间 case × 两种语言。输出：无。

    预期行为：反解出来的标的名与读数**逐行逐字**等于渲染输入——只认一种语言的表会把
    另一种语言的产物整片读成「表外写法」。
    """
    loc = locales.load(lang)
    for case in ft.build_cases(512):
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        parsed = parse.parse_state(render.build(spec, loc), parse.tables_of(loc))
        assert parsed.request_id == spec.query_id
        assert parsed.count == ft.ROWS
        want_names = [str(loc["name"][name_id]) for name_id in case.name_ids]
        want_values = [
            render.format_value(loc, bp, unit)
            for bp, unit in zip(case.bp_values, case.unit_seq, strict=True)
        ]
        assert [row[0] for row in parsed.rows] == want_names
        assert [row[1] for row in parsed.rows] == want_values


@pytest.mark.parametrize("lang", LANGS)
def test_structural_separator_matches_the_parser(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：语言文件里的竖线必须落在反解认的那几套里，且渲染出来的行真的能被反解认下来
    ——差一个全角 / 半角就会把好产物整片判成坏数据。
    """
    loc = locales.load(lang)
    meta = loc["meta"]
    assert str(meta["sep"]).strip() in {half for _full, half in parse.SEPARATORS}
    line = render.row_line(loc, 1, ft.NAME_IDS[0], 25, "pct")
    state = f"{meta['request']} Q-000000\n[{meta['table']}]\n{line}\n"
    parsed = parse.parse_state(state, parse.tables_of(loc))
    assert parsed.count == 1
    assert parsed.rows[0][0] == str(loc["name"][ft.NAME_IDS[0]])


def test_format_value_uses_integer_arithmetic() -> None:
    """输入：几个基准点数与两种单位。输出：无。

    预期行为：百分比按两位小数拼、基准点按整数拼，符号只在非零时出现——经过浮点会把
    `-0.45` 变成 `-0.44999999999999996`，反解再乘回去就成了另一个数。
    """
    loc = locales.load("zh")
    assert render.format_value(loc, -150, "pct") == "-1.50%"
    assert render.format_value(loc, 25, "pct") == "+0.25%"
    assert render.format_value(loc, 0, "pct") == "0.00%"
    assert render.format_value(loc, 8, "pct") == "+0.08%"
    assert render.format_value(loc, -45, "bp") == "-45bp"
    assert render.format_value(loc, 200, "bp") == "+200bp"
    assert render.format_value(loc, 0, "bp") == "0bp"


def test_format_value_reports_an_unknown_unit() -> None:
    """输入：表外的单位键。输出：无。预期行为：报人话，不静默当成基准点。"""
    with pytest.raises(ValueError, match="不在语言表"):
        render.format_value(locales.load("zh"), 10, "permille")


def test_render_reports_an_unknown_name() -> None:
    """输入：表外的标的名键。输出：无。预期行为：报错，不悄悄退回键名。"""
    with pytest.raises(ValueError, match="不在语言表"):
        render.name_text(locales.load("zh"), "I9")


def test_parse_rejects_a_gap_in_row_numbers() -> None:
    """输入：行号跳了一号的 state。输出：无。预期行为：报「跳号」，不把两行认成一行。"""
    loc = locales.load("zh")
    state = render.build(_make_spec(2), loc).replace("  2. ", "  3. ", 1)
    with pytest.raises(ValueError, match="跳号"):
        parse.parse_state(state, parse.tables_of(loc))


def test_parse_rejects_an_unknown_row_line() -> None:
    """输入：区块里多了一行认不出来的内容。输出：无。

    预期行为：报「认不出的读数行」——静默跳过会把渲染写歪的一行当成不存在，行数就对不上了。
    """
    loc = locales.load("zh")
    state = render.build(_make_spec(3), loc).replace("[读数表]\n", "[读数表]\n  这是多出来的一行\n")
    with pytest.raises(ValueError, match="认不出的读数行"):
        parse.parse_state(state, parse.tables_of(loc))


def test_merged_tables_read_both_languages() -> None:
    """输入：中英两条 state。输出：无。

    预期行为：`Workflow.reparse` 没有语言参数，合并表必须两种语言都认得出——产物按
    `locales[seq % len]` 轮转渲染，反解只认一种语言就会把另一种整片判成坏数据。
    """
    spec = _make_spec(5, values=1, units=2)
    for lang in LANGS:
        parsed = parse.parse_state(render.build(spec, locales.load(lang)), parse.merged_tables())
        assert parsed.request_id == spec.query_id
        assert parsed.count == ft.ROWS


def test_tables_of_rejects_empty_labels() -> None:
    """输入：抬头写成空串的语言字典。输出：无。预期行为：报错，不静默收下一张认不出的表。"""
    loc = dict(locales.load("zh"))
    loc["meta"] = {"request": "  ", "table": "读数表", "sep": "|"}
    with pytest.raises(ValueError, match="抬头不能是空串"):
        parse.tables_of(loc)
    loc["meta"] = {"request": "行情读数请求号", "table": "", "sep": "|"}
    with pytest.raises(ValueError, match="区块抬头不能是空串"):
        parse.tables_of(loc)


def test_merged_tables_rejects_a_prefix_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：一份把抬头写成另一份前缀的假语言表。输出：无。

    预期行为：报错——反解靠最长前缀认标签，互为前缀时短的会把长的吃掉。
    """
    real = locales.load("zh")
    fake = dict(real)
    fake["meta"] = {"request": "行情读数请求", "table": "读数列表", "sep": "|"}
    monkeypatch.setattr(locales, "load", lambda lang: fake if lang == "en" else real)
    with pytest.raises(ValueError, match="前缀"):
        parse.merged_tables()
