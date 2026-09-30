"""渲染与反解的测试：版式、逐格往返、合并表、以及各类硬伤要报人话。"""

from __future__ import annotations


import pytest

from tests.workflows.elementpick.conftest import _make_spec
from workflows.elementpick import locales, parse, render, spec as sp


def test_layout_is_fixed() -> None:
    """输入：一条规格。输出：无。

    预期行为：标题 / 屏号 / 动作 / 空行 / 六行元素表——动作那一行与六行四格就是判据的
    全部输入，缺一行都不行。
    """
    spec = _make_spec(extras=1, shadow=True)
    loc = locales.load("zh")
    meta = loc["meta"]
    lines = render.build(spec, loc).splitlines()
    assert lines[0] == meta["title"]
    assert lines[1] == f"{meta['screen_label']}{meta['colon']}{spec.screen_id}"
    assert lines[2] == f"{meta['action_label']}{meta['colon']}{loc['label'][spec.case.label]}"
    assert lines[3] == ""
    assert len(lines) == 4 + sp.SIZE
    for index, (line, row) in enumerate(zip(lines[4:], spec.rows), start=1):
        assert line == render.row_line(index, row, loc)


def test_render_is_a_pure_function() -> None:
    """输入：同一份规格渲染两次。输出：无。预期行为：逐字节一致。"""
    spec = _make_spec()
    loc = locales.load("zh")
    assert render.build(spec, loc) == render.build(spec, loc)


def test_row_line_joins_the_four_cells() -> None:
    """输入：元素表的一行。输出：无。预期行为：序号 + 四格（角色 · 可见性 · 可点性 · 文本）。"""
    loc = locales.load("zh")
    row = sp.rows_of(_make_spec().case)[0]
    assert render.row_line(1, row, loc) == "[1] " + str(loc["meta"]["cell_sep"]).join(
        (
            loc["role"][row.role],
            loc["visibility"][row.visibility],
            loc["clickable"][row.clickable],
            render.text_of(row.text, loc),
        )
    )
    assert render.field_line("屏号", "S-1", loc) == "屏号" + str(loc["meta"]["colon"]) + "S-1"


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_near_text_really_contains_the_label_text(lang: str) -> None:
    """输入：每种语言的标签与近似文本。输出：无。

    预期行为：近似文本**真的**把标签原文当子串含在里面、而且不完全一样——判据里
    「当子串含在里面但不完全一样」这句话靠这条成立，`shadow` 那一族才有对手。
    """
    loc = locales.load(lang)
    for label in sp.LABELS:
        plain = render.text_of(label, loc)
        near = render.text_of(f"{sp.SHADOW_PREFIX}{label}", loc)
        assert plain in near and plain != near, (lang, label)
        for other in sp.LABELS:
            if other != label:
                assert render.text_of(other, loc) not in near, (lang, label, other)


def test_roundtrip_over_the_whole_grid_in_both_languages() -> None:
    """输入：全部 864 个 case × 两门语言。输出：无。

    预期行为：渲染出来的文本反解回来与真值逐格一致——「渲染写歪」会在这里当场现形。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        labels = parse.labels_of(loc)
        for case in sp.build_cases(sp.ElementConfig()):
            spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
            screen = parse.parse_screen(render.build(spec, loc), labels)
            assert screen.screen_id == spec.screen_id
            assert screen.action == spec.case.label
            assert [
                (row.role, row.visibility, row.clickable, row.text) for row in screen.rows
            ] == [
                (row.role, row.visibility, row.clickable, row.text) for row in spec.rows
            ]


def test_merged_table_reads_every_language() -> None:
    """输入：两门语言渲染出来的同一份规格。输出：无。

    预期行为：合并表（`reparse` 用的那张）能把两门语言都反解成同一份键——语言无关的反解
    靠的就是这个。
    """
    spec = _make_spec(label="delete", layout="l3", extras=2, shadow=True, phrasing=1)
    merged = parse.merged()
    results: set[tuple[object, ...]] = set()
    for lang in ("zh", "en"):
        screen = parse.parse_screen(render.build(spec, locales.load(lang)), merged)
        results.add(
            (
                screen.screen_id,
                screen.action,
                tuple(
                    (row.role, row.visibility, row.clickable, row.text) for row in screen.rows
                ),
            )
        )
    assert len(results) == 1
    screen_id, action, rows = results.pop()
    assert (screen_id, action) == (spec.screen_id, spec.case.label)
    assert len(rows) == sp.SIZE


def test_decode_text_prefers_the_exact_match_then_strips_the_suffix() -> None:
    """输入：两门语言的原文。输出：无。

    预期行为：先查完全一致，再按各语言的近似后缀剥一层；两步都不认就报人话。
    """
    merged = parse.merged()
    assert parse.decode_text("保存", merged) == "save"
    assert parse.decode_text("保存草稿", merged) == "shadow:save"
    assert parse.decode_text("Save", merged) == "save"
    assert parse.decode_text("Save draft", merged) == "shadow:save"
    with pytest.raises(ValueError, match="文本认不出来"):
        parse.decode_text("查无此物", merged)


def test_field_value_prefers_the_longest_label() -> None:
    """输入：两个互为前缀的候选标签。输出：无。

    预期行为：取最长的那个——否则「屏」会把「屏号」那一格提前吃掉。
    """
    assert parse._field_value("屏号：S-1", ("屏", "屏号")) == "S-1"
    assert parse._field_value("屏：S-1", ("屏", "屏号")) == "S-1"


def test_field_value_needs_a_separator_right_after_the_label() -> None:
    """输入：以标签开头但没有分隔符的一行。输出：无。

    预期行为：不算这一格——标题行确实以「元素」开头，收紧才不会误认。
    """
    assert parse._field_value("元素清单", ("元素",)) is None
    assert parse._field_value("今天天气不错。", ("动作",)) is None


def test_parse_cells_reports_hard_damage() -> None:
    """输入：缺右括号 / 格数不对 / 认不出的格 的元素行。输出：无。预期行为：都报人话。"""
    labels = parse.labels_of(locales.load("zh"))
    with pytest.raises(ValueError, match="右方括号"):
        parse._parse_cells("1 按钮 · 可见 · 可点 · 保存", labels)
    with pytest.raises(ValueError, match="格"):
        parse._parse_cells("[1] 按钮 · 可见 · 可点", labels)
    with pytest.raises(ValueError, match="role"):
        parse._parse_cells("[1] 不存在 · 可见 · 可点 · 保存", labels)
    with pytest.raises(ValueError, match="文本认不出来"):
        parse._parse_cells("[1] 按钮 · 可见 · 可点 · 查无此物", labels)


def test_parse_screen_reports_hard_damage() -> None:
    """输入：四种坏文本。输出：无。预期行为：各报人话，不静默猜。"""
    loc = locales.load("zh")
    labels = parse.labels_of(loc)
    good = render.build(_make_spec(), loc).splitlines()
    with pytest.raises(TypeError, match="字符串"):
        parse.parse_screen(123, labels)
    with pytest.raises(ValueError, match="屏号"):
        parse.parse_screen("\n".join([good[0], *good[2:]]), labels)
    with pytest.raises(ValueError, match="动作"):
        parse.parse_screen("\n".join([good[0], good[1], *good[3:]]), labels)
    with pytest.raises(ValueError, match="元素表应该是"):
        parse.parse_screen("\n".join(good[:5]), labels)


def test_parse_screen_ignores_unknown_lines() -> None:
    """输入：混进两行噪声的清单。输出：无。预期行为：噪声被忽略，其余照常读回。"""
    labels = parse.labels_of(locales.load("zh"))
    good = render.build(_make_spec(), locales.load("zh"))
    noisy = "\n".join(["随便一行说明", *good.splitlines(), "再来一行"])
    screen = parse.parse_screen(noisy, labels)
    assert len(screen.rows) == sp.SIZE


def test_labels_table_lookups_report_unknown_fields() -> None:
    """输入：两个不存在的字段名。输出：无。预期行为：都报 KeyError 并列出可用值。"""
    labels = parse.labels_of(locales.load("zh"))
    with pytest.raises(KeyError, match="词表"):
        labels.table("nope")
    with pytest.raises(KeyError, match="标签"):
        labels.labels_of("nope")
    assert labels.labels_of("screen") == labels.screen_labels
    assert labels.table("role") == labels.role
    assert labels.table("label") == labels.label


def test_labels_of_rejects_two_keys_sharing_one_text() -> None:
    """输入：把两个角色写成同一个原文。输出：无。预期行为：报人话（反解会认错键）。"""
    loc = dict(locales.load("zh"))
    loc["role"] = dict(loc["role"])
    loc["role"]["toggle"] = loc["role"]["button"]
    with pytest.raises(ValueError, match="共用同一个原文"):
        parse.labels_of(loc)


def _patched_merged(monkeypatch: pytest.MonkeyPatch, mutate: object) -> None:
    """输入：monkeypatch、一个改语言字典的回调。输出：无。

    预期行为：把两门语言换成**副本**（`locales.load` 带缓存，改原字典会污染后面的用例），
    再交给回调去改，好单独验合并表的几种冲突。
    """
    def copy_of(lang: str) -> dict[str, object]:
        """输入：语言。输出：语言字典的浅副本（嵌套表也复制一层）。"""
        return {
            key: (dict(value) if isinstance(value, dict) else value)
            for key, value in locales.load(lang).items()
        }

    zh = copy_of("zh")
    en = copy_of("en")
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
        lambda en, zh: en["meta"].update({"screen_label": zh["meta"]["screen_label"]}),
    )
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()


def test_merged_rejects_colliding_vocabularies(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言把同一个标签写成同一个词。输出：无。预期行为：报人话。"""
    _patched_merged(
        monkeypatch, lambda en, zh: en["label"].update({"save": zh["label"]["save"]})
    )
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()


def test_merged_reports_no_locales_and_no_suffixes(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：一个语言文件都没有 / 所有语言都没有近似后缀。输出：无。预期行为：都报人话。"""
    monkeypatch.setattr(locales, "available", lambda: [])
    with pytest.raises(ValueError, match="一个语言文件都没有"):
        parse.merged()
    _patched_merged(
        monkeypatch,
        lambda en, zh: (
            en["meta"].update({"shadow_suffix": ""}),
            zh["meta"].update({"shadow_suffix": ""}),
        ),
    )
    with pytest.raises(ValueError, match="近似后缀"):
        parse.merged()
