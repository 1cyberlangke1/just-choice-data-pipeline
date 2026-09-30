"""渲染与反解：编号、抬头、两种语言读得回来、坏 state 当场报人话。"""

from __future__ import annotations

import random
from typing import Final

import pytest

from tests.workflows.clipcut.conftest import _make_spec
from workflows.clipcut import features as ft
from workflows.clipcut import locales, parse, render
from workflows.clipcut import spec as sp

LANGS: Final[tuple[str, ...]] = ("zh", "en")


def test_state_has_a_header_and_a_section() -> None:
    """输入：一条规格。输出：无。预期行为：第一行是请求号、第二行是片段表区块抬头。"""
    loc = locales.load("zh")
    meta = loc["meta"]
    lines = render.build(_make_spec(0), loc).splitlines()
    assert lines[0].startswith(str(meta["request"]))
    assert _make_spec(0).query_id in lines[0]
    assert lines[1] == f"[{meta['segments']}]"


def test_segment_numbering_starts_at_one() -> None:
    """输入：一条样本。输出：无。预期行为：片段号从 1 起、连续、与段数一致。"""
    state = render.build(_make_spec(0), locales.load("zh"))
    numbers = [
        int(line.strip().split(".")[0]) for line in state.splitlines() if line.strip()[:1].isdigit()
    ]
    assert numbers == list(range(1, ft.SEGMENTS + 1))


@pytest.mark.parametrize("lang", LANGS)
def test_parse_reads_back_every_case(lang: str) -> None:
    """输入：全空间 case × 两种语言。输出：无。

    预期行为：反解出来的主题名与高亮词**逐段逐字**等于渲染输入——只认一种语言的表会把
    另一种语言的产物整片读成「表外写法」。
    """
    loc = locales.load(lang)
    for case in ft.build_cases(512):
        spec = sp.sample_spec(random.Random(0), case, "train", 0, 0)
        parsed = parse.parse_state(render.build(spec, loc), parse.tables_of(loc))
        assert parsed.request_id == spec.query_id
        assert parsed.count == ft.SEGMENTS
        want_topics = [str(loc["topic"][topic]) for topic in case.topic_seq]
        want_flags = [str(loc["flag"]["yes" if flag else "no"]) for flag in case.flag_seq]
        assert [segment[0] for segment in parsed.segments] == want_topics
        assert [segment[1] for segment in parsed.segments] == want_flags


@pytest.mark.parametrize("lang", LANGS)
def test_structural_parens_match_the_parser(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：语言文件里的圆括号必须落在反解认的那两套里，且渲染出来的行真的能被反解
    认下来——差一个全角 / 半角就会把好产物整片判成坏数据。
    """
    loc = locales.load(lang)
    meta = loc["meta"]
    assert str(meta["paren_open"]).strip() in {pair[0] for pair in parse.PARENS}
    assert str(meta["paren_close"]).strip() in {pair[1] for pair in parse.PARENS}
    line = render.segment_line(loc, 1, ft.TOPIC_IDS[0], True)
    state = f"{meta['request']} K-000000\n[{meta['segments']}]\n{line}\n"
    parsed = parse.parse_state(state, parse.tables_of(loc))
    assert parsed.count == 1
    assert parsed.segments[0][0] == str(loc["topic"][ft.TOPIC_IDS[0]])


def test_parse_rejects_a_short_state() -> None:
    """输入：只有两行的 state。输出：无。预期行为：报「太短」，不给一个空片段表。"""
    with pytest.raises(ValueError, match="太短"):
        parse.parse_state("片段筛选请求号 K-000000\n[片段表]\n", parse.merged_tables())


def test_parse_rejects_a_missing_section() -> None:
    """输入：少了区块抬头的 state。输出：无。预期行为：报「出现在任何区块抬头之前」。"""
    loc = locales.load("zh")
    lines = render.build(_make_spec(1), loc).splitlines()
    without_section = "\n".join([lines[0], *lines[2:]]) + "\n"
    with pytest.raises(ValueError, match="区块抬头之前"):
        parse.parse_state(without_section, parse.tables_of(loc))


def test_parse_rejects_a_gap_in_segment_numbers() -> None:
    """输入：片段号跳了一号的 state。输出：无。预期行为：报「跳号」，不把两段认成一段。"""
    loc = locales.load("zh")
    state = render.build(_make_spec(2), loc).replace("  2. ", "  3. ", 1)
    with pytest.raises(ValueError, match="跳号"):
        parse.parse_state(state, parse.tables_of(loc))


def test_parse_rejects_an_unknown_segment_line() -> None:
    """输入：区块里多了一行认不出来的内容。输出：无。

    预期行为：报「认不出的片段行」——静默跳过会把渲染写歪的一段当成不存在，段数就对不上了。
    """
    loc = locales.load("zh")
    state = render.build(_make_spec(3), loc).replace("[片段表]\n", "[片段表]\n  这是多出来的一行\n")
    with pytest.raises(ValueError, match="认不出的片段行"):
        parse.parse_state(state, parse.tables_of(loc))


def test_parse_rejects_an_empty_table() -> None:
    """输入：区块下面一段都没有的 state。输出：无。预期行为：报「一段都没有」。"""
    loc = locales.load("zh")
    meta = loc["meta"]
    state = f"{meta['request']} K-000000\n[{meta['segments']}]\n"
    with pytest.raises(ValueError, match="太短|一段都没有"):
        parse.parse_state(state, parse.tables_of(loc))


def test_parse_rejects_an_unknown_header() -> None:
    """输入：抬头认不出来的 state。输出：无。预期行为：报「认不出来」，不静默当空值。"""
    loc = locales.load("zh")
    state = render.build(_make_spec(4), loc).replace("片段筛选请求号", "随便什么", 1)
    with pytest.raises(ValueError, match="认不出来"):
        parse.parse_state(state, parse.tables_of(loc))


def test_merged_tables_read_both_languages() -> None:
    """输入：中英两条 state。输出：无。

    预期行为：`Workflow.reparse` 没有语言参数，合并表必须两种语言都认得出——产物按
    `locales[seq % len]` 轮转渲染，反解只认一种语言就会把另一种整片判成坏数据。
    """
    spec = _make_spec(5, topics=1, flags=3)
    for lang in LANGS:
        parsed = parse.parse_state(render.build(spec, locales.load(lang)), parse.merged_tables())
        assert parsed.request_id == spec.query_id
        assert parsed.count == ft.SEGMENTS


def test_tables_of_rejects_empty_labels() -> None:
    """输入：抬头写成空串的语言字典。输出：无。预期行为：报错，不静默收下一张认不出的表。"""
    loc = dict(locales.load("zh"))
    loc["meta"] = {"request": "  ", "segments": "片段表"}
    with pytest.raises(ValueError, match="抬头不能是空串"):
        parse.tables_of(loc)
    loc["meta"] = {"request": "片段筛选请求号", "segments": ""}
    with pytest.raises(ValueError, match="区块抬头不能是空串"):
        parse.tables_of(loc)


def test_merged_tables_rejects_a_prefix_label(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：一份把抬头写成另一份前缀的假语言表。输出：无。

    预期行为：报错——反解靠最长前缀认标签，互为前缀时短的会把长的吃掉。
    """
    real = locales.load("zh")
    fake = dict(real)
    fake["meta"] = {"request": "片段筛选请求", "segments": "片段列表"}
    monkeypatch.setattr(locales, "load", lambda lang: fake if lang == "en" else real)
    with pytest.raises(ValueError, match="前缀"):
        parse.merged_tables()
