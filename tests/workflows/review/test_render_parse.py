"""渲染与反解的测试：版式、逐格往返、合并表、以及各类硬伤要报人话。"""

from __future__ import annotations

from typing import Any

import pytest

from tests.workflows.review.conftest import _make_spec
from workflows.review import locales, parse, render, spec as sp


def test_layout_is_fixed() -> None:
    """输入：一条规格。输出：无。预期行为：标题 / 改动号 / 空行 / 六处改动各两行。"""
    spec = _make_spec()
    loc = locales.load("zh")
    lines = render.build(spec, loc).splitlines()
    assert lines[0] == loc["meta"]["title"]
    assert lines[1] == f"{loc['meta']['change_label']}{loc['meta']['colon']}{spec.change_id}"
    assert lines[2] == ""
    assert len(lines) == 3 + 2 * len(sp.SLOTS)
    for index, (header, body, row) in enumerate(
        zip(lines[3::2], lines[4::2], spec.rows), start=1
    ):
        assert header.startswith(f"[{index}] ")
        assert body.strip() == str(loc["body"][row.body]).strip()


def test_render_is_a_pure_function() -> None:
    """输入：同一份规格渲染两次。输出：无。预期行为：逐字节一致。"""
    spec = _make_spec()
    loc = locales.load("zh")
    assert render.build(spec, loc) == render.build(spec, loc)


def test_finding_helpers() -> None:
    """输入：一处改动。输出：无。预期行为：抬头三格按序拼；正文键认不出时报人话。"""
    spec = _make_spec()
    loc = locales.load("zh")
    row = spec.rows[0]
    assert render.finding_header(1, row, loc) == "[1] " + str(loc["meta"]["cell_sep"]).join(
        (loc["file"][row.file], loc["change"][row.change], loc["size"][row.size])
    )
    assert render.finding_body(row, loc).startswith(str(loc["meta"]["body_indent"]))
    with pytest.raises(KeyError, match="正文模板"):
        render.finding_body(sp.Finding(None, "src", "add", "small", "没有这个键"), loc)


def test_roundtrip_over_the_whole_grid_in_both_languages() -> None:
    """输入：全部 432 个 case × 两门语言。输出：无。

    预期行为：渲染出来的文本反解回来与真值逐格一致——「渲染写歪」会在这里当场现形。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        labels = parse.labels_of(loc)
        for case in sp.build_cases(sp.ReviewConfig()):
            spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
            change = parse.parse_change(render.build(spec, loc), labels)
            assert change.change_id == spec.change_id
            assert [(row.file, row.change, row.size, row.body) for row in change.rows] == [
                (row.file, row.change, row.size, str(loc["body"][row.body]).strip())
                for row in spec.rows
            ]


def test_merged_table_reads_every_language() -> None:
    """输入：两门语言渲染出来的同一份规格。输出：无。

    预期行为：合并表（`reparse` 用的那张）能把两门语言都反解成同一份键——
    语言无关的反解靠的就是这个：正文那一格留着各自语言的原文，所以这里比的是改动号
    与六处改动的抬头三格。
    """
    spec = _make_spec(triple="t1", noise="n2", change="c1", order="o5", phrasing=1)
    merged = parse.merged()
    results = set()
    for lang in ("zh", "en"):
        change = parse.parse_change(render.build(spec, locales.load(lang)), merged)
        results.add(
            (change.change_id, tuple((row.file, row.change, row.size) for row in change.rows))
        )
    assert len(results) == 1
    change_id, rows = results.pop()
    assert change_id == spec.change_id
    assert len(rows) == len(sp.SLOTS)


def test_field_value_prefers_the_longest_label() -> None:
    """输入：两个互为前缀的候选标签。输出：无。

    预期行为：取最长的那个——否则「改动」会把「改动号」那一格提前吃掉。
    """
    assert parse._field_value("改动号：R-1", ("改动", "改动号")) == "R-1"
    assert parse._field_value("改动：R-1", ("改动", "改动号")) == "R-1"


def test_field_value_needs_a_separator_right_after_the_label() -> None:
    """输入：以标签开头但没有分隔符的一行。输出：无。

    预期行为：不算这一格——标题行确实以「改动」开头，收紧才不会误认。
    """
    assert parse._field_value("改动审查单", ("改动",)) is None
    assert parse._field_value("今天天气不错。", ("改动",)) is None


def test_parse_cells_reports_hard_damage() -> None:
    """输入：缺右括号 / 格数不对 / 认不出的格 的抬头行。输出：无。预期行为：都报人话。"""
    labels = parse.labels_of(locales.load("zh"))
    with pytest.raises(ValueError, match="右方括号"):
        parse._parse_cells("1 源码 · 新增 · 小", labels)
    with pytest.raises(ValueError, match="格"):
        parse._parse_cells("[1] 源码 · 新增", labels)
    with pytest.raises(ValueError, match="file"):
        parse._parse_cells("[1] 不存在 · 新增 · 小", labels)


def test_parse_change_reports_hard_damage() -> None:
    """输入：五种坏文本。输出：无。预期行为：各报人话，不静默猜。"""
    loc = locales.load("zh")
    labels = parse.labels_of(loc)
    good = render.build(_make_spec(), loc)
    with pytest.raises(TypeError, match="字符串"):
        parse.parse_change(123, labels)
    with pytest.raises(ValueError, match="改动号"):
        parse.parse_change("\n".join(good.splitlines()[2:]), labels)
    with pytest.raises(ValueError, match="一处改动都没有"):
        parse.parse_change(
            f"{loc['meta']['title']}\n"
            f"{loc['meta']['change_label']}{loc['meta']['colon']}R-1",
            labels,
        )
    with pytest.raises(ValueError, match="没有正文"):
        parse.parse_change("\n".join(good.splitlines()[:-1]), labels)
    with pytest.raises(ValueError, match="还没写正文"):
        parse.parse_change("[1] 源码 · 新增 · 小\n[2] 测试 · 修改 · 小", labels)


def test_parse_change_ignores_unknown_lines() -> None:
    """输入：混进两行噪声的改动单。输出：无。预期行为：噪声被忽略，其余照常读回。"""
    labels = parse.labels_of(locales.load("zh"))
    good = render.build(_make_spec(), locales.load("zh"))
    noisy = "\n".join(["随便一行说明", *good.splitlines(), "再来一行"])
    change = parse.parse_change(noisy, labels)
    assert len(change.rows) == len(sp.SLOTS)


def test_labels_table_lookups_report_unknown_fields() -> None:
    """输入：两个不存在的字段名。输出：无。预期行为：都报 KeyError 并列出可用值。"""
    labels = parse.labels_of(locales.load("zh"))
    with pytest.raises(KeyError, match="词表"):
        labels.table("nope")
    with pytest.raises(KeyError, match="标签"):
        labels.labels_of("nope")
    assert labels.labels_of("change_id") == labels.change_labels
    assert labels.table("size") == labels.size


def test_labels_of_rejects_two_keys_sharing_one_text() -> None:
    """输入：把两个规模键写成同一个原文。输出：无。预期行为：报人话（反解会认错键）。"""
    loc = dict(locales.load("zh"))
    loc["size"] = dict(loc["size"])
    loc["size"]["big"] = loc["size"]["small"]
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
    """输入：两门语言用不同的三格分隔符。输出：无。预期行为：报人话。"""
    _patched_merged(monkeypatch, lambda en, zh: en["meta"].update({"cell_sep": " | "}))
    with pytest.raises(ValueError, match="分隔符必须一致"):
        parse.merged()


def test_merged_rejects_colliding_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言的改动号标签写成一样。输出：无。预期行为：报人话（合并表有歧义）。"""
    _patched_merged(
        monkeypatch,
        lambda en, zh: en["meta"].update({"change_label": zh["meta"]["change_label"]}),
    )
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()


def test_merged_rejects_colliding_vocabularies(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言把同一个文件档写成同一个词。输出：无。预期行为：报人话。"""
    _patched_merged(
        monkeypatch, lambda en, zh: en["file"].update({"src": zh["file"]["src"]})
    )
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()


def test_merged_reports_no_locales(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：一个语言文件都没有。输出：无。预期行为：报人话。"""
    monkeypatch.setattr(locales, "available", lambda: [])
    with pytest.raises(ValueError, match="一个语言文件都没有"):
        parse.merged()
