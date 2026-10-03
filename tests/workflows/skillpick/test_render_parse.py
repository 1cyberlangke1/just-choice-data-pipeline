"""渲染与反解的测试：版式、逐格往返、合并表、以及各类硬伤要报人话。"""

from __future__ import annotations
from jc.state_parse import read_field_value


import pytest

from tests.workflows.skillpick.conftest import _make_spec
from workflows.skillpick import locales, parse, render, spec as sp


def test_layout_is_fixed() -> None:
    """输入：一条规格。输出：无。

    预期行为：标题 / 请求号 / 标签 / 授权 / 已占用 / 空行 / 六行技能表——抬头四格与技能表
    就是判据的全部输入，缺一行都不行。
    """
    spec = _make_spec()
    loc = locales.load("zh")
    meta = loc["meta"]
    lines = render.build(spec, loc).splitlines()
    assert lines[0] == meta["title"]
    assert lines[1] == f"{meta['request_label']}{meta['colon']}{spec.request_id}"
    assert lines[2] == f"{meta['tag_label']}{meta['colon']}{loc['tag'][spec.case.tag]}"
    assert lines[3] == f"{meta['grant_label']}{meta['colon']}{loc['grant'][spec.case.grant]}"
    assert lines[4] == f"{meta['inuse_label']}{meta['colon']}{loc['tag'][spec.inuse_tag]}"
    assert lines[5] == ""
    assert len(lines) == 6 + len(sp.CATALOG)
    for index, (line, row) in enumerate(zip(lines[6:], spec.rows), start=1):
        assert line == render.row_line(index, row, loc)


def test_row_line_joins_the_three_cells() -> None:
    """输入：技能表的一行。输出：无。预期行为：序号 + 三格（适用场景 · 权限范围 · 资源）。"""
    loc = locales.load("zh")
    row = sp.rows_of("l2")[0]
    assert render.row_line(1, row, loc) == "[1] " + str(loc["meta"]["cell_sep"]).join(
        (loc["tag"][row.tag], loc["scope"][row.scope], loc["resource"][row.resource])
    )
    assert render.field_line("标签", "值", loc) == "标签" + str(loc["meta"]["colon"]) + "值"


def test_roundtrip_over_the_whole_grid_in_both_languages() -> None:
    """输入：全部 576 个 case × 两门语言。输出：无。

    预期行为：渲染出来的文本反解回来与真值逐格一致——「渲染写歪」会在这里当场现形。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        labels = parse.labels_of(loc)
        for case in sp.build_cases(sp.SkillConfig()):
            spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
            request = parse.parse_request(render.build(spec, loc), labels)
            assert request.request_id == spec.request_id
            assert (request.tag, request.grant) == (spec.case.tag, spec.case.grant)
            assert request.inuse == spec.inuse_tag
            assert [
                (row.tag, row.scope, row.resource) for row in request.rows
            ] == [(row.tag, row.scope, row.resource) for row in spec.rows]


def test_merged_table_reads_every_language() -> None:
    """输入：两门语言渲染出来的同一份规格。输出：无。

    预期行为：合并表（`reparse` 用的那张）能把两门语言都反解成同一份键——语言无关的反解
    靠的就是这个。
    """
    spec = _make_spec(tag="ship_release", layout="l3", grant="write", inuse="below", phrasing=1)
    merged = parse.merged()
    results: set[tuple[str, str, str, str, tuple[tuple[str, str, str], ...]]] = set()
    for lang in ("zh", "en"):
        request = parse.parse_request(render.build(spec, locales.load(lang)), merged)
        results.add(
            (
                request.request_id,
                request.tag,
                request.grant,
                request.inuse,
                tuple((row.tag, row.scope, row.resource) for row in request.rows),
            )
        )
    assert len(results) == 1
    request_id, tag, grant, inuse, rows = results.pop()
    assert (request_id, tag, grant, inuse) == (
        spec.request_id,
        spec.case.tag,
        spec.case.grant,
        spec.inuse_tag,
    )
    assert len(rows) == len(sp.CATALOG)


def test_field_value_prefers_the_longest_label() -> None:
    """输入：两个互为前缀的候选标签。输出：无。

    预期行为：取最长的那个——否则「标签」会把「标签号」那一格提前吃掉。
    """
    assert read_field_value("标签号：Q-1", ("标签", "标签号")) == "Q-1"
    assert read_field_value("标签：Q-1", ("标签", "标签号")) == "Q-1"


def test_field_value_needs_a_separator_right_after_the_label() -> None:
    """输入：以标签开头但没有分隔符的一行。输出：无。

    预期行为：不算这一格——标题行确实以「技能」开头，收紧才不会误认。
    """
    assert read_field_value("技能选择单", ("技能",)) is None
    assert read_field_value("今天天气不错。", ("标签",)) is None


def test_parse_cells_reports_hard_damage() -> None:
    """输入：缺右括号 / 格数不对 / 认不出的格 的技能行。输出：无。预期行为：都报人话。"""
    labels = parse.labels_of(locales.load("zh"))
    with pytest.raises(ValueError, match="右方括号"):
        parse._parse_cells("1 读代码 · 只读 · 代码", labels)
    with pytest.raises(ValueError, match="格"):
        parse._parse_cells("[1] 读代码 · 只读", labels)
    with pytest.raises(ValueError, match="tag"):
        parse._parse_cells("[1] 不存在 · 只读 · 代码", labels)


def test_parse_request_reports_hard_damage() -> None:
    """输入：四种坏文本。输出：无。预期行为：各报人话，不静默猜。"""
    loc = locales.load("zh")
    labels = parse.labels_of(loc)
    good = render.build(_make_spec(), loc).splitlines()
    with pytest.raises(TypeError, match="字符串"):
        parse.parse_request(123, labels)
    with pytest.raises(ValueError, match="请求号"):
        parse.parse_request("\n".join(good[2:]), labels)
    with pytest.raises(ValueError, match="标签"):
        parse.parse_request("\n".join([*good[:2], *good[3:]]), labels)
    with pytest.raises(ValueError, match="技能表应该是"):
        parse.parse_request("\n".join(good[:6]), labels)
    with pytest.raises(ValueError, match="授权"):
        parse.parse_request("\n".join([*good[:3], *good[4:]]), labels)


def test_parse_request_ignores_unknown_lines() -> None:
    """输入：混进两行噪声的选择单。输出：无。预期行为：噪声被忽略，其余照常读回。"""
    labels = parse.labels_of(locales.load("zh"))
    good = render.build(_make_spec(), locales.load("zh"))
    noisy = "\n".join(["随便一行说明", *good.splitlines(), "再来一行"])
    request = parse.parse_request(noisy, labels)
    assert len(request.rows) == len(sp.CATALOG)


def test_labels_table_lookups_report_unknown_fields() -> None:
    """输入：两个不存在的字段名。输出：无。预期行为：都报 KeyError 并列出可用值。"""
    labels = parse.labels_of(locales.load("zh"))
    with pytest.raises(KeyError, match="词表"):
        labels.table("nope")
    with pytest.raises(KeyError, match="标签"):
        labels.labels_of("nope")
    assert labels.labels_of("inuse") == labels.inuse_labels
    assert labels.table("tag") == labels.tag
    assert labels.table("grant") == labels.grant


def test_labels_of_rejects_two_keys_sharing_one_text() -> None:
    """输入：把两个权限范围写成同一个原文。输出：无。预期行为：报人话（反解会认错键）。"""
    loc = dict(locales.load("zh"))
    loc["scope"] = dict(loc["scope"])
    loc["scope"]["write"] = loc["scope"]["read"]
    with pytest.raises(ValueError, match="共用同一个原文"):
        parse.labels_of(loc)


def _patched_merged(monkeypatch: pytest.MonkeyPatch, mutate: object) -> None:
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
    """输入：两门语言的标签字段名写成一样。输出：无。预期行为：报人话（合并表有歧义）。"""
    _patched_merged(
        monkeypatch,
        lambda en, zh: en["meta"].update({"tag_label": zh["meta"]["tag_label"]}),
    )
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()


def test_merged_rejects_colliding_vocabularies(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言把同一个标签写成同一个词。输出：无。预期行为：报人话。"""
    _patched_merged(
        monkeypatch, lambda en, zh: en["tag"].update({"read_code": zh["tag"]["read_code"]})
    )
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()
