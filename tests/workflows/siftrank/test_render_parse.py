"""渲染与反解：一屏文本的两头必须严丝合缝，坏文本必须当场报人话。"""

from __future__ import annotations


import pytest

from tests.workflows.siftrank.conftest import _make_spec
from workflows.siftrank import locales, parse, render
from workflows.siftrank import spec as sp

LANGS: tuple[str, ...] = ("zh", "en")


@pytest.mark.parametrize("lang", LANGS)
def test_render_is_a_pure_function(lang: str) -> None:
    """输入：同一条规格、同一语言、两个 RNG。输出：无。

    预期行为：逐字节一致——渲染器不许掷骰子，否则「两次生成一致」这条没法查。
    """
    spec = _make_spec(1)
    loc = locales.load(lang)
    first = render.build(spec, loc)
    second = render.build(spec, loc)
    assert first == second
    assert first.endswith("\n")
    assert first.count("\n") == 3 + sp.TRIALS


@pytest.mark.parametrize("lang", LANGS)
def test_roundtrip_through_parse(lang: str) -> None:
    """输入：渲染出来的 state。输出：无。

    预期行为：反解出来的请求号、条目清单、四个批次与真值规格逐条一致——条目名按该语言的
    写法，批次按顺序。
    """
    spec = _make_spec(2, shape="swirl", perm="rotate2")
    loc = locales.load(lang)
    parsed = parse.parse_state(render.build(spec, loc), parse.tables_of(loc))
    assert parsed.request_id == spec.request_id
    assert parsed.prompt == loc["flavor"][spec.case.flavor]
    assert parsed.listing == tuple(render.item_text(loc, key) for key in sp.ITEMS)
    assert parsed.batches == tuple(
        tuple(render.item_text(loc, key) for key in batch) for batch in spec.batches
    )


def test_batch_numbering_starts_at_one() -> None:
    """输入：一条样本。输出：无。预期行为：批次号从 1 起、连续、与 `TRIALS` 一致。"""
    spec = _make_spec(3)
    state = render.build(spec, locales.load("zh"))
    numbers = [
        int(line.split("：", 1)[0].split()[-1])
        for line in state.splitlines()
        if line.startswith("批次")
    ]
    assert numbers == list(range(1, sp.TRIALS + 1))


def test_parse_rejects_a_short_state() -> None:
    """输入：只有两行的 state。输出：无。预期行为：报「太短」，不给一个空批次表。"""
    with pytest.raises(ValueError, match="太短"):
        parse.parse_state("排序请求号：S-000000\n排序提示：随便\n", parse.merged_tables())


def test_parse_rejects_a_gap_in_batch_numbers() -> None:
    """输入：批次号跳号的 state。输出：无。预期行为：报「跳号」，不把两个批次认成一个。"""
    loc = locales.load("zh")
    state = render.build(_make_spec(4), loc).replace("批次 2：", "批次 3：", 1)
    with pytest.raises(ValueError, match="跳号"):
        parse.parse_state(state, parse.tables_of(loc))


def test_parse_rejects_an_extra_batch() -> None:
    """输入：多了一个批次的 state。输出：无。

    预期行为：报「批次的个数不对」——批次被吞或多出来都会让聚合算错，必须当场拦住。
    """
    loc = locales.load("zh")
    state = render.build(_make_spec(5), loc)
    extra = "批次 5：概览说明 > 操作手册 > 变更记录 > 监控指标\n"
    with pytest.raises(ValueError, match="批次的个数不对"):
        parse.parse_state(state + extra, parse.tables_of(loc))


def test_parse_rejects_a_dropped_batch() -> None:
    """输入：少了最后一个批次的 state。输出：无。预期行为：当场报错（太短或个数不对）。"""
    loc = locales.load("zh")
    lines = render.build(_make_spec(5), loc).splitlines()
    with pytest.raises(ValueError, match="太短|个数不对"):
        parse.parse_state("\n".join(lines[:-1]) + "\n", parse.tables_of(loc))


def test_parse_rejects_a_wrong_batch_size() -> None:
    """输入：某个批次多了一个条目的 state。输出：无。预期行为：报条目数不对。"""
    loc = locales.load("zh")
    state = render.build(_make_spec(6), loc).replace("批次 1：", "批次 1：概览说明 > ", 1)
    with pytest.raises(ValueError, match="条目数不对"):
        parse.parse_state(state, parse.tables_of(loc))


def test_parse_rejects_an_empty_value() -> None:
    """输入：抬头写了但值为空的 state。输出：无。预期行为：报「没写值」。"""
    loc = locales.load("zh")
    state = render.build(_make_spec(8), loc).replace(
        "排序提示：", "排序提示：\n占位：", 1
    )
    with pytest.raises(ValueError, match="没写值"):
        parse.parse_state(state, parse.tables_of(loc))


def test_merged_tables_read_both_languages() -> None:
    """输入：中英两条 state。输出：无。

    预期行为：`Workflow.reparse` 没有语言参数，合并表必须两种语言都认得出——产物按
    `locales[seq % len]` 轮转渲染，反解只认一种语言就会把另一种整片判成坏数据。
    """
    spec = _make_spec(9, shape="chain", perm="reverse")
    for lang in LANGS:
        parsed = parse.parse_state(render.build(spec, locales.load(lang)), parse.merged_tables())
        assert parsed.request_id == spec.request_id
        assert len(parsed.batches) == sp.TRIALS


@pytest.mark.parametrize("lang", LANGS)
def test_structural_alphabet_matches_the_parser(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：语言文件里的分隔符必须落在反解认的那几套里——差一个空格（例如英文侧把箭头
    写成 `>` 而反解只认 ` > `），就会把好产物整片判成坏数据（范式 §9.71 四）。
    """
    meta = locales.load(lang)["meta"]
    assert str(meta["sep"]).strip() in parse.COLONS, meta["sep"]
    listing = render.listing_text(locales.load(lang))
    assert len(parse.LIST_SEP.split(listing)) == len(sp.ITEMS), listing
    line = render.batch_text(locales.load(lang), sp.ITEMS[: sp.BATCH_SIZE])
    assert len(parse.ARROW.split(line)) == sp.BATCH_SIZE, line
