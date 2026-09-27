"""渲染与反解的测试：版式、逐格往返、合并表、以及各类硬伤要报人话。"""

from __future__ import annotations

from typing import Any

import pytest

from tests.workflows.browser.conftest import _make_case, _make_spec
from workflows.browser import locales, parse, render, spec as sp


def test_layout_is_fixed() -> None:
    """输入：一条规格。输出：无。预期行为：标题 / 窗口号 / 任务 / 空行 / 五行元素。"""
    spec = _make_spec()
    loc = locales.load("zh")
    lines = render.build(spec, loc).splitlines()
    assert lines[0] == loc["meta"]["title"]
    assert lines[1] == f"{loc['meta']['screen_label']}{loc['meta']['colon']}{spec.screen_id}"
    assert lines[2] == render.task_line(spec, loc)
    assert lines[3] == ""
    assert len(lines) == 4 + len(sp.SLOTS)
    for index, (line, row) in enumerate(zip(lines[4:], spec.rows), start=1):
        assert line.startswith(f"[{index}] ")
        for cell in (row.role, row.object, row.state):
            assert loc["role"].get(cell) or loc["object"].get(cell) or loc["state"].get(cell)


def test_render_is_a_pure_function() -> None:
    """输入：同一份规格渲染两次（一次带 RNG）。输出：无。预期行为：逐字节一致。"""
    spec = _make_spec()
    loc = locales.load("zh")
    assert render.build(spec, loc) == render.build(spec, loc)


def test_roundtrip_over_the_whole_grid_in_both_languages() -> None:
    """输入：全部 480 个 case × 两门语言。输出：无。

    预期行为：渲染出来的文本反解回来与真值逐格一致——「渲染写歪」会在这里当场现形。
    """
    for lang in locales.available():
        labels = parse.labels_of(locales.load(lang))
        for case in sp.build_cases(sp.ScreenConfig()):
            spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
            screen = parse.parse_screen(render.build(spec, locales.load(lang)), labels)
            assert screen.screen_id == spec.screen_id
            assert (screen.task, screen.object) == (spec.task, spec.object)
            assert [(row.role, row.object, row.state) for row in screen.rows] == [
                (row.role, row.object, row.state) for row in spec.rows
            ]


def test_merged_table_reads_every_language() -> None:
    """输入：两门语言渲染出来的同一份规格。输出：无。

    预期行为：合并表（`reparse` 用的那张）能把两门语言都反解成同一份键——
    语言无关的反解靠的就是这个。
    """
    spec = _make_spec(task="switch", form="sensitive", done=True, decoy_state="current")
    merged = parse.merged()
    results = {
        parse.parse_screen(render.build(spec, locales.load(lang)), merged) for lang in ("zh", "en")
    }
    assert len(results) == 1
    screen = results.pop()
    assert screen.screen_id == spec.screen_id and screen.object == spec.object


def test_field_value_prefers_the_longest_label() -> None:
    """输入：两个互为前缀的候选标签。输出：无。

    预期行为：取最长的那个——否则「窗口」会把「窗口号」那一格提前吃掉。
    """
    assert parse._field_value("窗口号：W-1", ("窗口", "窗口号")) == "W-1"
    assert parse._field_value("窗口：W-1", ("窗口", "窗口号")) == "W-1"


def test_field_value_needs_a_separator_right_after_the_label() -> None:
    """输入：以标签开头但没有分隔符的一行。输出：无。

    预期行为：不算这一格——标题行确实可能以某个标签开头，收紧才不会误认。
    """
    assert parse._field_value("窗口号错了", ("窗口号",)) is None
    assert parse._field_value("今天天气不错。", ("窗口号",)) is None


def test_split_task_reports_both_halves() -> None:
    """输入：意图词认不出 / 对象认不出。输出：无。预期行为：各报人话，不静默猜。"""
    labels = parse.labels_of(locales.load("zh"))
    assert parse._split_task("填写备注", labels) == ("fill", "note")
    assert parse._split_task("切换到支付记录", labels) == ("switch", "pay_history")
    with pytest.raises(ValueError, match="意图词"):
        parse._split_task("随便点一下", labels)
    with pytest.raises(ValueError, match="对象"):
        parse._split_task("填写不存在的对象", labels)


def test_parse_cells_reports_hard_damage() -> None:
    """输入：缺右括号 / 格数不对的元素行。输出：无。预期行为：都报人话。"""
    labels = parse.labels_of(locales.load("zh"))
    with pytest.raises(ValueError, match="右方括号"):
        parse._parse_cells("1 输入框 · 备注 · 可用", labels)
    with pytest.raises(ValueError, match="格"):
        parse._parse_cells("[1] 输入框 · 备注", labels)
    with pytest.raises(ValueError, match="role"):
        parse._parse_cells("[1] 不存在 · 备注 · 可用", labels)


def test_parse_screen_reports_missing_pieces() -> None:
    """输入：四种残缺文本。输出：无。预期行为：都报人话，不返回半成品。"""
    labels = parse.labels_of(locales.load("zh"))
    good = render.build(_make_spec(), locales.load("zh"))
    with pytest.raises(TypeError, match="字符串"):
        parse.parse_screen({"state": good}, labels)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="窗口号"):
        parse.parse_screen(
            "\n".join([good.splitlines()[0], *good.splitlines()[2:]]), labels
        )
    with pytest.raises(ValueError, match="任务"):
        parse.parse_screen(
            "\n".join([*good.splitlines()[:2], *good.splitlines()[3:]]), labels
        )
    with pytest.raises(ValueError, match="一个元素都没有"):
        parse.parse_screen("\n".join(good.splitlines()[:4]), labels)


def test_parse_screen_ignores_unknown_lines() -> None:
    """输入：混进两行噪声的界面文本。输出：无。预期行为：噪声被忽略，其余照常读回。"""
    labels = parse.labels_of(locales.load("zh"))
    good = render.build(_make_spec(), locales.load("zh"))
    noisy = "\n".join(["随便一行说明", *good.splitlines(), "再来一行"])
    screen = parse.parse_screen(noisy, labels)
    assert len(screen.rows) == len(sp.SLOTS)


def test_labels_table_lookups_report_unknown_fields() -> None:
    """输入：两个不存在的字段名。输出：无。预期行为：都报 KeyError 并列出可用值。"""
    labels = parse.labels_of(locales.load("zh"))
    with pytest.raises(KeyError, match="词表"):
        labels.table("nope")
    with pytest.raises(KeyError, match="标签"):
        labels.labels_of("nope")
    assert labels.labels_of("screen") == labels.screen_labels
    assert labels.table("role") == labels.role


def test_labels_of_rejects_two_keys_sharing_one_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：把两个对象键写成同一个原文。输出：无。预期行为：报人话（反解会认错键）。"""
    loc = dict(locales.load("zh"))
    loc["object"] = dict(loc["object"])
    loc["object"]["help"] = loc["object"]["note"]
    with pytest.raises(ValueError, match="共用同一个原文"):
        parse.labels_of(loc)


def _patched_merged(
    monkeypatch: pytest.MonkeyPatch, mutate: Any
) -> None:
    """输入：monkeypatch、一个改语言字典的回调。输出：无。

    预期行为：把两门语言换成「zh + 改过的 en」，好单独验合并表的三种冲突。
    """
    zh = locales.load("zh")
    en = {key: (dict(value) if isinstance(value, dict) else value) for key, value in locales.load("en").items()}
    mutate(en, zh)
    monkeypatch.setattr(locales, "available", lambda: ["zh", "en"])
    monkeypatch.setattr(locales, "load", lambda lang: {"zh": zh, "en": en}[lang])


def test_merged_rejects_mismatched_separators(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言用不同的三格分隔符。输出：无。预期行为：报人话。"""
    _patched_merged(monkeypatch, lambda en, zh: en["meta"].update({"cell_sep": " | "}))
    with pytest.raises(ValueError, match="分隔符必须一致"):
        parse.merged()


def test_merged_rejects_colliding_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言的窗口标签写成一样。输出：无。预期行为：报人话（合并表有歧义）。"""
    _patched_merged(
        monkeypatch, lambda en, zh: en["meta"].update({"screen_label": zh["meta"]["screen_label"]})
    )
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()


def test_merged_rejects_colliding_vocabularies(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言把同一个角色写成同一个词。输出：无。预期行为：报人话。"""
    _patched_merged(monkeypatch, lambda en, zh: en["role"].update({"field": zh["role"]["field"]}))
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()


def test_merged_reports_no_locales(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：一个语言文件都没有。输出：无。预期行为：报人话。"""
    monkeypatch.setattr(locales, "available", lambda: [])
    with pytest.raises(ValueError, match="一个语言文件都没有"):
        parse.merged()
