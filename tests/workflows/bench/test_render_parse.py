"""渲染与反解的往返测试：逐格一致、两种语言、坏文本各报人话。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from itertools import product


import pytest

from tests.workflows.bench.conftest import _make_spec, TINY_TABLES
from workflows.bench import locales, parse, render
from workflows.bench import spec as sp

LANGS = ("zh", "en")
#: 原始语言字典：`_patched_merged` 每次都从它深拷贝，免得同一条测试里第二次改动叠在第一次上。
_ORIGINAL_LOCALES: dict[str, JsonValue] = {"zh": locales.load("zh"), "en": locales.load("en")}


def _copy(value: object) -> object:
    """输入：语言字典里的任意一层。输出：深拷贝（只为测试里改着玩）。"""
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy(item) for item in value]
    return value


def test_render_is_a_pure_function_of_the_spec_and_the_locale() -> None:
    """输入：同一份规格、同一份语言文件，渲染两次。输出：无。预期行为：逐字节一致。"""
    spec = _make_spec()
    loc = locales.load("zh")
    assert render.build(spec, loc) == render.build(spec, loc)
    assert render.build(spec, locales.load("en")) != render.build(spec, loc)


def test_roundtrip_is_exact_for_every_table_and_layout() -> None:
    """输入：三十张成绩表 × 六种列序 × 两种语言。输出：无。

    预期行为：反解出来的集号、门槛、行数、五行四格与真值规格**逐格一致**——渲染与反解
    各写一套、谁写歪了都会在这里现形。
    """
    for table, layout in product(TINY_TABLES, sp.LAYOUTS):
        spec = sp.sample_spec(sp.BenchCase(table, layout, 0), "train", 0, 0, None)
        for lang in LANGS:
            loc = locales.load(lang)
            record = parse.parse_record(render.build(spec, loc), parse.labels_of(loc))
            assert record.request_id == spec.request_id
            assert record.bar == spec.bar
            assert record.layout == sp.layout_perm(layout)
            assert len(record.rows) == sp.TOTAL
            for got, want in zip(record.rows, spec.rows, strict=True):
                assert got.ref == want.ref
                assert tuple((cell.cand, cell.letter) for cell in got.cells) == tuple(
                    (cell.cand, cell.letter) for cell in want.cells
                )


def test_render_writes_the_head_and_the_cells() -> None:
    """输入：一份规格。输出：无。

    预期行为：第一行是标题、接着是集与门槛两格、然后空行、再五行；每行第一格是参考格标签，
    四格由格分隔符连起来，格内用名字 / 字母分隔符连起来。
    """
    loc = locales.load("zh")
    spec = _make_spec()
    lines = render.build(spec, loc).splitlines()
    assert lines[0] == loc["meta"]["title"]
    assert lines[1] == render.field_line(str(loc["label"]["request"]), spec.request_id, loc)
    assert lines[2] == render.field_line(str(loc["label"]["bar"]), str(spec.bar), loc)
    assert lines[3] == ""
    assert len(lines) == 4 + sp.TOTAL
    assert lines[4].startswith("[1] ")
    sep = str(loc["meta"]["cell_sep"])
    assert len(lines[4].split(sep)) == 1 + len(sp.CANDIDATES)
    assert str(loc["cell"]["ref"]) in lines[4]


def test_parse_ignores_unknown_lines() -> None:
    """输入：混进两行噪声的记录。输出：无。预期行为：噪声被忽略，其余照常读回。"""
    loc = locales.load("zh")
    labels = parse.labels_of(loc)
    good = render.build(_make_spec(), loc)
    noisy = "\n".join(["随便一行说明", *good.splitlines(), "再来一行"])
    record = parse.parse_record(noisy, labels)
    assert len(record.rows) == sp.TOTAL
    assert record.request_id == _make_spec().request_id


def test_parse_reports_hard_damage() -> None:
    """输入：五种坏文本。输出：无。预期行为：各报人话，不静默猜。"""
    loc = locales.load("zh")
    labels = parse.labels_of(loc)
    good = render.build(_make_spec(), loc).splitlines()
    with pytest.raises(TypeError, match="字符串"):
        parse.parse_record(123, labels)
    with pytest.raises(ValueError, match="缺这几格"):
        parse.parse_record("\n".join([*good[:1], *good[3:]]), labels)
    with pytest.raises(ValueError, match="应该是"):
        parse.parse_record("\n".join(good[:8]), labels)
    with pytest.raises(ValueError, match="应该是 4 格"):
        parse.parse_record("\n".join([*good[:4], "[1] 参考 A", *good[5:]]), labels)
    with pytest.raises(ValueError, match="缺右方括号"):
        parse.parse_record("\n".join([*good[:4], "[1 参考 A · 甲 B · 乙 C · 丙 D", *good[5:]]), labels)
    with pytest.raises(ValueError, match="参考格标签|第一格"):
        parse.parse_record("\n".join([*good[:4], good[4].replace("参考", "别的"), *good[5:]]), labels)


def test_parse_reports_bad_cells() -> None:
    """输入：候选名认不出 / 字母认不出 / 同一行候选重复 / 各行列序不一致。输出：无。"""
    loc = locales.load("zh")
    labels = parse.labels_of(loc)
    good = render.build(_make_spec(), loc).splitlines()
    with pytest.raises(KeyError, match="候选词表里没有"):
        parse.parse_record("\n".join([*good[:4], good[4].replace("甲", "戊"), *good[5:]]), labels)
    with pytest.raises(ValueError, match="答案字母认不出来"):
        parse.parse_record("\n".join([*good[:4], good[4].replace(" A ", " Z "), *good[5:]]), labels)
    with pytest.raises(ValueError, match="候选重复"):
        parse.parse_record(
            "\n".join([*good[:4], good[4].replace("乙", "甲"), *good[5:]]), labels
        )
    swapped = good[4].split(" · ")
    swapped[1], swapped[2] = swapped[2], swapped[1]
    with pytest.raises(ValueError, match="列序必须一致"):
        parse.parse_record("\n".join([*good[:4], " · ".join(swapped), *good[5:]]), labels)


def test_parse_bar_reports_bad_values() -> None:
    """输入：不是整数的门槛 / 出了取值域的门槛。输出：无。预期行为：都报人话。"""
    with pytest.raises(ValueError, match="不是整数"):
        parse._parse_bar("三")
    with pytest.raises(ValueError, match="出了取值域"):
        parse._parse_bar("9")
    assert parse._parse_bar("3") == sp.BAR


def test_split_pair_reports_the_wrong_number_of_parts() -> None:
    """输入：没有分隔符 / 分隔符多了一个。输出：无。预期行为：都报人话。"""
    with pytest.raises(ValueError, match="两段"):
        parse._split_pair("甲B", " ", "候选格")
    with pytest.raises(ValueError, match="两段"):
        parse._split_pair("甲 B C", " ", "候选格")


def test_labels_table_lookups_report_unknown_fields() -> None:
    """输入：两个不存在的名字。输出：无。预期行为：都报 KeyError 并列出可用值。"""
    labels = parse.labels_of(locales.load("zh"))
    with pytest.raises(KeyError, match="抬头字段"):
        labels.labels_of("nope")
    with pytest.raises(KeyError, match="候选词表"):
        labels.cand_key("nope")
    assert labels.labels_of("bar") == (str(locales.load("zh")["label"]["bar"]),)
    assert labels.cand_key(str(locales.load("zh")["cand"]["m0"])) == "m0"


def test_labels_of_rejects_a_short_or_colliding_vocabulary() -> None:
    """输入：候选词表少一个键 / 两条共用同一个原文。输出：无。预期行为：都报人话。"""
    loc = {key: value for key, value in locales.load("zh").items()}
    loc["cand"] = {"m0": "甲", "m1": "乙"}
    with pytest.raises(KeyError, match="候选词表"):
        parse.labels_of(loc)
    loc["cand"] = {"m0": "甲", "m1": "甲", "m2": "丙"}
    with pytest.raises(ValueError, match="共用同一个原文"):
        parse.labels_of(loc)


def _patched_merged(monkeypatch: pytest.MonkeyPatch, mutate: object) -> None:
    """输入：monkeypatch、一个改语言字典的回调。输出：无。

    预期行为：把两门语言换成「zh + 改过的 en」，好单独验合并表的三种冲突。
    每次都从**原始**语言文件深拷贝——不然同一条测试里第二次改动会叠在第一次上面，
    验出来的就是上一条冲突而不是这一条。
    """
    zh = _copy(_ORIGINAL_LOCALES["zh"])
    en = _copy(_ORIGINAL_LOCALES["en"])
    mutate(en, zh)
    monkeypatch.setattr(locales, "available", lambda: ["zh", "en"])
    monkeypatch.setattr(locales, "load", lambda lang: {"zh": zh, "en": en}[lang])


def test_merged_rejects_mismatched_separators(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言用不同的格分隔符 / 不同的格内分隔符。输出：无。预期行为：都报人话。"""
    _patched_merged(monkeypatch, lambda en, zh: en["meta"].update({"cell_sep": " | "}))
    with pytest.raises(ValueError, match="格分隔符必须一致"):
        parse.merged()
    _patched_merged(monkeypatch, lambda en, zh: en["meta"].update({"pair_sep": "="}))
    with pytest.raises(ValueError, match="名字 / 字母分隔符必须一致"):
        parse.merged()


def test_merged_rejects_colliding_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言的抬头字段名 / 参考格标签 / 候选名指向不同键。输出：无。预期行为：都报人话。"""
    _patched_merged(monkeypatch, lambda en, zh: en["label"].update({"bar": zh["label"]["bar"]}))
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()
    _patched_merged(monkeypatch, lambda en, zh: en["cell"].update({"ref": zh["cell"]["ref"]}))
    with pytest.raises(ValueError, match="在多种语言里都出现"):
        parse.merged()
    _patched_merged(monkeypatch, lambda en, zh: en["cand"].update({"m1": zh["cand"]["m0"]}))
    with pytest.raises(ValueError, match="指向不同键"):
        parse.merged()


def test_merged_accepts_one_candidate_text_in_both_languages(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：两门语言把同一个候选写成同一个词。输出：无。

    预期行为：**不报错**——同一段原文指向同一个键时合并表没有歧义（真冲突是「同一段原文
    指向不同的键」，上一条测的就是那个）。两个**键**共用同一个原文才是硬伤，那条在
    `labels_of` 里拦（见 `test_labels_of_rejects_a_short_or_colliding_vocabulary`）。
    """
    _patched_merged(monkeypatch, lambda en, zh: en["cand"].update({"m0": zh["cand"]["m0"]}))
    assert parse.merged().cand_key(str(_ORIGINAL_LOCALES["zh"]["cand"]["m0"])) == "m0"


def test_merged_reports_no_locales(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：一个语言文件都没有。输出：无。预期行为：报人话。"""
    monkeypatch.setattr(locales, "available", lambda: [])
    with pytest.raises(ValueError, match="一个语言文件都没有"):
        parse.merged()


def test_merged_reads_both_languages() -> None:
    """输入：合并表。输出：无。

    预期行为：合并表把两种语言的写法都收进来，所以**任意一种语言**渲染出来的文本都能用它
    反解（`reparse` 没有语言参数，只能走这条）。
    """
    merged = parse.merged()
    for lang in LANGS:
        loc = locales.load(lang)
        spec = _make_spec()
        record = parse.parse_record(render.build(spec, loc), merged)
        assert record.request_id == spec.request_id
        assert record.layout == sp.layout_perm("l0")
