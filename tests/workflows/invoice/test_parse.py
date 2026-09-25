"""反解的测试：纯文本解析、标签最长匹配、噪声行忽略。"""

from __future__ import annotations

from typing import Any

from workflows.invoice import locales, parse, render, spec as sp
from workflows.invoice import workflow as inv_workflow

LOCALES = locales.available()


def test_as_int_drops_symbols_and_separators() -> None:
    """输入：带货币符号 / 千分位 / 百分号的文本。输出：整数。

    预期行为：金额与税率都是整数，符号与千分位一律丢掉；一个数字都没有时给 None。
    """
    assert parse.as_int("¥1,234,567") == 1234567
    assert parse.as_int("13%") == 13
    assert parse.as_int("$ 0") == 0
    assert parse.as_int("") is None
    assert parse.as_int("没有数字") is None


def test_as_date_requires_the_year_first() -> None:
    """输入：几种日期写法。输出：三元组。

    预期行为：年必须在前（三种日期格式的模板都保证这点）；倒序写法解不出来——
    宁可判「解不出来」，也不能把日当成年。
    """
    assert parse.as_date("2026-03-07") == (2026, 3, 7)
    assert parse.as_date("2026年3月7日") == (2026, 3, 7)
    assert parse.as_date("2026.12.05") == (2026, 12, 5)
    assert parse.as_date("07/03/2026") is None
    assert parse.as_date("没有日期") is None


def test_label_index_only_contains_field_labels() -> None:
    """输入：语言文件。输出：{标签: 槽}。

    预期行为：明细表的列名（品名 / 数量 / …）不算字段标签——否则表头会被当成字段读进来。
    """
    for lang in LOCALES:
        loc = locales.load(lang)
        index = parse.label_index(loc)
        for name in ("name", "qty", "unit", "amount"):
            for label in loc["column"][name]["labels"]:
                assert label not in index
        for slot, entry in loc["field"].items():
            for label in entry["labels"]:
                assert index[label] == slot


def test_match_label_prefers_the_longest_label() -> None:
    """输入：英文的两行（`Tax Rate` 与 `Tax`）。输出：匹配结果。

    预期行为：取最长匹配，且标签后面必须跟分隔符或两个以上空白——
    `Tax` 是 `Tax Rate` 的前缀，只看前缀会把 `Tax Rate: 6%` 读成 `Tax`；
    只看前缀又会把 `Total lines: 3` 这类噪声读成字段。
    """
    labels = parse.label_index(locales.load("en"))
    assert parse._match_label("Tax Rate: 6%", labels) == ("Tax Rate", "tax_rate")
    assert parse._match_label("Tax: 100", labels) == ("Tax", "tax")
    assert parse._match_label("Total lines: 3", labels) is None
    assert parse._match_label("这不是字段行", labels) is None


def test_parse_ignores_noise_lines(make_spec: Any) -> None:
    """输入：一份单据。输出：反解结果。

    预期行为：标题、明细表头、备注、段落引言都不算字段；解出来的槽正好是「会写出来的」那些。
    """
    loc = locales.load("zh")
    spec = make_spec(81, layout="paragraph", tax_state="ok")
    parsed = parse.parse_document(render.build(spec, loc), loc)
    assert set(parsed.fields) == set(render.visible_fields(spec))
    assert len(parsed.items) == spec.case.lines


def test_parse_empty_text_gives_an_empty_result() -> None:
    """输入：空文本。输出：反解结果。预期行为：没有字段也没有明细，不抛异常。"""
    for lang in LOCALES:
        parsed = parse.parse_document("", locales.load(lang))
        assert parsed.fields == {} and parsed.items == ()


def test_item_rows_are_recognised_by_shape_not_by_header() -> None:
    """输入：一行四格。输出：它是不是明细行。

    预期行为：判据是「恰好四格 + 后三格能解出数字」，所以表头行不会被误判成明细。
    """
    assert parse._looks_like_item(["系统运维", "9", "¥721", "¥6,489"])
    assert not parse._looks_like_item(["品名", "数量", "单价", "金额"])
    assert not parse._looks_like_item(["只有两格", "1"])
    assert not parse._looks_like_item(["五格", "1", "2", "3", "4"])


def test_merged_labels_is_language_agnostic() -> None:
    """输入：无。输出：合并标签表。

    预期行为：每份语言文件的标签都收进同一张表，且一个标签只指向一个槽
    （重叠会让语言无关的反解变得不可靠）。
    """
    merged = parse.merged_labels()["field"]
    for lang in LOCALES:
        for slot, entry in locales.load(lang)["field"].items():
            for label in entry["labels"]:
                assert label in merged[slot]["labels"]
    seen: dict[str, str] = {}
    for slot, entry in merged.items():
        for label in entry["labels"]:
            assert seen.setdefault(label, slot) == slot, label


def test_reparse_reads_both_languages(make_spec: Any) -> None:
    """输入：同一份规格的中文与英文文本。输出：反解结果。

    预期行为：`Workflow.reparse` 没有语言参数，靠合并标签表把两种语言都读出来；
    数值与明细的数量对得上（词表本身各语言不同，所以只比结构）。
    """
    spec = make_spec(91, lines=3)
    instance = inv_workflow.InvoiceWorkflow()
    zh = instance.reparse(render.build(spec, locales.load("zh")))
    en = instance.reparse(render.build(spec, locales.load("en")))
    assert set(zh.fields) == set(en.fields) == set(render.visible_fields(spec))
    assert zh.money("total") == en.money("total") == spec.total
    assert zh.date("due") == en.date("due") == spec.due_date
    assert [(i.qty, i.unit, i.amount) for i in zh.items] == [
        (i.qty, i.unit, i.amount) for i in en.items
    ]
