"""渲染与反解的测试：版式、逐格往返、合并表、以及各类硬伤要报人话。"""

from __future__ import annotations

from typing import Any

import pytest

from tests.workflows.rubric.conftest import _make_spec
from workflows.rubric import locales, parse, render, spec as sp


def test_layout_is_fixed() -> None:
    """输入：一份规格。输出：无。

    预期行为：标题 / 单号 / 空行 / 六行四格——抬头一行与六行判据就是判据的全部输入，
    缺一行都不行。
    """
    spec = _make_spec()
    loc = locales.load("zh")
    meta = loc["meta"]
    lines = render.build(spec, loc).splitlines()
    assert lines[0] == meta["title"]
    assert lines[1] == f"{meta['request_label']}{meta['colon']}{spec.request_id}"
    assert lines[2] == ""
    assert len(lines) == 3 + len(sp.CRITERIA)
    for index, (line, row) in enumerate(zip(lines[3:], spec.rows), start=1):
        assert line == render.row_line(index, row, loc)


def test_render_is_a_pure_function() -> None:
    """输入：同一份规格渲染两次。输出：无。预期行为：逐字节一致。"""
    spec = _make_spec()
    loc = locales.load("zh")
    assert render.build(spec, loc) == render.build(spec, loc)


def test_row_line_joins_the_four_cells() -> None:
    """输入：评分单的一行。输出：无。预期行为：序号 + 四格（判据 · 权重 · 要求 · 观测）。"""
    loc = locales.load("zh")
    row = sp.rows_of("s0", "l2", "roomy")[0]
    assert render.row_line(1, row, loc) == "[1] " + str(loc["meta"]["cell_sep"]).join(
        (
            loc["criterion"][row.criterion],
            loc["weight"][row.weight],
            loc["need"][row.need],
            loc["count"][row.count],
        )
    )
    assert render.field_line("单号", "值", loc) == "单号" + str(loc["meta"]["colon"]) + "值"


def test_roundtrip_over_the_whole_grid_in_both_languages() -> None:
    """输入：全部 432 个 case × 两门语言。输出：无。

    预期行为：渲染出来的文本反解回来与真值逐格一致——「渲染写歪」会在这里当场现形。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        labels = parse.labels_of(loc)
        for case in sp.build_cases(sp.RubricConfig()):
            spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
            sheet = parse.parse_sheet(render.build(spec, loc), labels)
            assert sheet.request_id == spec.request_id
            assert [
                (row.criterion, row.weight, row.need, row.count) for row in sheet.rows
            ] == [
                (row.criterion, row.weight, row.need, row.count) for row in spec.rows
            ]


def test_merged_table_reads_every_language() -> None:
    """输入：两门语言渲染出来的同一份规格。输出：无。

    预期行为：合并表（`reparse` 用的那张）能把两门语言都反解成同一份键——语言无关的反解
    靠的就是这个。
    """
    spec = _make_spec(sheet="s13", layout="l3", slack="roomy", phrasing=1)
    merged = parse.merged()
    results = set()
    for lang in ("zh", "en"):
        sheet = parse.parse_sheet(render.build(spec, locales.load(lang)), merged)
        results.add(
            (
                sheet.request_id,
                tuple(
                    (row.criterion, row.weight, row.need, row.count) for row in sheet.rows
                ),
            )
        )
    assert len(results) == 1
    request_id, rows = results.pop()
    assert request_id == spec.request_id
    assert rows == tuple(
        (row.criterion, row.weight, row.need, row.count) for row in spec.rows
    )


def test_field_value_prefers_the_longest_label() -> None:
    """输入：两个互为前缀的候选标签。输出：无。

    预期行为：取最长的那个——否则「单」会把「单号」那一格提前吃掉。
    """
    assert parse._field_value("单号：G-1", ("单", "单号")) == "G-1"
    assert parse._field_value("单：G-1", ("单", "单号")) == "G-1"


def test_field_value_needs_a_separator_right_after_the_label() -> None:
    """输入：以标签开头但没有分隔符的一行。输出：无。

    预期行为：不算这一格——标题行确实以「评」开头之外的东西打头，收紧才不会误认。
    """
    assert parse._field_value("评分单", ("单号",)) is None
    assert parse._field_value("单号 G-1", ("单号",)) is None
    assert parse._field_value("今天天气不错。", ("单号",)) is None
    assert parse._field_value("Sheet ID: G-1", ("Sheet ID",)) == "G-1"


def test_parse_cells_reports_hard_damage() -> None:
    """输入：缺右括号 / 格数不对 / 认不出的格 的评分行。输出：无。预期行为：都报人话。"""
    labels = parse.labels_of(locales.load("zh"))
    with pytest.raises(ValueError, match="右方括号"):
        parse._parse_cells("1 例子 · 权重 3 · 要求 2 · 观测 2", labels)
    with pytest.raises(ValueError, match="格"):
        parse._parse_cells("[1] 例子 · 权重 3 · 要求 2", labels)
    with pytest.raises(ValueError, match="criterion"):
        parse._parse_cells("[1] 不存在 · 权重 3 · 要求 2 · 观测 2", labels)


def test_parse_sheet_reports_hard_damage() -> None:
    """输入：四种坏文本。输出：无。预期行为：各报人话，不静默猜。"""
    loc = locales.load("zh")
    labels = parse.labels_of(loc)
    good = render.build(_make_spec(), loc).splitlines()
    with pytest.raises(TypeError, match="字符串"):
        parse.parse_sheet(123, labels)
    with pytest.raises(ValueError, match="单号"):
        parse.parse_sheet("\n".join([*good[:1], *good[2:]]), labels)
    with pytest.raises(ValueError, match="评分单应该是"):
        parse.parse_sheet("\n".join(good[:5]), labels)
    with pytest.raises(ValueError, match="count"):
        broken = list(good)
        broken[3] = broken[3].replace(str(loc["count"]["n2"]), "观测 9")
        parse.parse_sheet("\n".join(broken), labels)


def test_parse_sheet_ignores_unknown_lines() -> None:
    """输入：混进两行噪声的评分单。输出：无。预期行为：噪声被忽略，其余照常读回。"""
    labels = parse.labels_of(locales.load("zh"))
    good = render.build(_make_spec(), locales.load("zh"))
    noisy = "\n".join(["随便一行说明", *good.splitlines(), "再来一行"])
    sheet = parse.parse_sheet(noisy, labels)
    assert len(sheet.rows) == len(sp.CRITERIA)


def test_labels_table_lookups_report_unknown_fields() -> None:
    """输入：两个不存在的字段名。输出：无。预期行为：都报 KeyError 并列出可用值。"""
    labels = parse.labels_of(locales.load("zh"))
    with pytest.raises(KeyError, match="词表"):
        labels.table("nope")
    with pytest.raises(KeyError, match="标签"):
        labels.labels_of("nope")
    assert labels.labels_of("request") == labels.request_labels
    assert labels.table("criterion") == labels.criterion
    assert labels.table("count") == labels.count


def test_labels_of_rejects_two_keys_sharing_one_text() -> None:
    """输入：把两档权重写成同一个原文。输出：无。预期行为：报人话（反解会认错键）。"""
    loc = dict(locales.load("zh"))
    loc["weight"] = dict(loc["weight"])
    loc["weight"]["w2"] = loc["weight"]["w1"]
    with pytest.raises(ValueError, match="共用同一个原文"):
        parse.labels_of(loc)


def _patched_merged(monkeypatch: pytest.MonkeyPatch, mutate: Any) -> None:
    """输入：monkeypatch、一个改语言字典的回调。输出：无。

    预期行为：把两门语言换成「zh + 改过的 en」，好单独验合并表的三种冲突。
    """
    zh = locales.load("zh")
    en = {
        key: (dict(value) if isinstance(value, dict) else value)
        for key, value in locales.load("en").items()
    }
    mutate(en, zh)
    monkeypatch.setattr(locales, "available", lambda: ["zh", "en"])
    monkeypatch.setattr(locales, "load", lambda lang: {"zh": zh, "en": en}[lang])


def test_merged_rejects_mismatched_separators(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言用不同的四格分隔符。输出：无。预期行为：报人话。"""
    _patched_merged(monkeypatch, lambda en, zh: en["meta"].update({"cell_sep": " | "}))
    with pytest.raises(ValueError, match="分隔符必须一致"):
        parse.merged()


def test_merged_rejects_colliding_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言的字段名写成一样。输出：无。预期行为：报人话（合并表有歧义）。"""
    _patched_merged(
        monkeypatch,
        lambda en, zh: en["meta"].update({"request_label": zh["meta"]["request_label"]}),
    )
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()


def test_merged_rejects_colliding_vocabularies(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言把同一条判据写成同一个词。输出：无。预期行为：报人话。"""
    _patched_merged(
        monkeypatch,
        lambda en, zh: en["criterion"].update({"example": zh["criterion"]["example"]}),
    )
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()


def test_merged_reports_no_locales(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：一个语言文件都没有。输出：无。预期行为：报人话。"""
    monkeypatch.setattr(locales, "available", lambda: [])
    with pytest.raises(ValueError, match="一个语言文件都没有"):
        parse.merged()
