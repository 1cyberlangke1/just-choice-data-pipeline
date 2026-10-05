"""语言文件的契约测试：标签、日期格式、问法、词表、标点。

这组测试是**锁**：`spec` 的槽位常量与词表长度、`render` 的日期格式名、
`questions` 的题族名、`parse` 的最长匹配，只要有一处漂移就立刻红灯。
"""

from __future__ import annotations

import pytest

from workflows.invoice import locales, parse, questions, render, spec as sp

LOCALES = locales.available()


@pytest.mark.parametrize("lang", LOCALES)
def test_every_field_has_two_distinct_labels(lang: str) -> None:
    """输入：语言文件。输出：无。

    预期行为：字段集合与 `render.ALL_FIELDS` 一致，每个槽至少两个标签（正名 + 别名）
    且互不相同——「换个说法」是值留出的一档，别名撞上正名就等于没留出。
    """
    loc = locales.load(lang)
    assert set(loc["field"]) == set(render.ALL_FIELDS)
    for slot, entry in loc["field"].items():
        labels = entry["labels"]
        assert len(labels) >= 2, slot
        assert len(set(labels)) == len(labels), slot


@pytest.mark.parametrize("lang", LOCALES)
def test_labels_do_not_collide_across_slots(lang: str) -> None:
    """输入：语言文件。输出：无。

    预期行为：一个标签只指向一个槽（标签数 = 槽内标签数之和），且每个标签都能反解回自己的槽。
    """
    loc = locales.load(lang)
    index = parse.label_index(loc)
    assert len(index) == sum(len(entry["labels"]) for entry in loc["field"].values())
    for slot, entry in loc["field"].items():
        for label in entry["labels"]:
            assert index[label] == slot


@pytest.mark.parametrize("lang", LOCALES)
def test_prefix_labels_are_resolved_to_the_longest_match(lang: str) -> None:
    """输入：每个标签后面接一个值。输出：`_match_label` 的结果。

    预期行为：无论标签之间有没有前缀关系，带值的行都要认回自己的槽——
    英文的 `Tax` / `Tax Rate` 就是这种情形，反解靠「最长匹配 + 标签后必须有分隔符」区分。
    """
    loc = locales.load(lang)
    labels = parse.label_index(loc)
    colon = loc["meta"]["colon"]
    for slot, entry in loc["field"].items():
        for label in entry["labels"]:
            assert parse._match_label(f"{label}{colon}VALUE", labels) == (label, slot)


@pytest.mark.parametrize("lang", LOCALES)
def test_labels_do_not_overlap_across_languages(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：本语言的标签不出现在别的语言里——`Workflow.reparse` 用合并标签表，
    两个语言共用一个标签就会指向两个槽，语言无关的反解立刻不可靠。
    """
    mine = set(parse.label_index(locales.load(lang)))
    for other in LOCALES:
        if other != lang:
            assert not (mine & set(parse.label_index(locales.load(other))))


@pytest.mark.parametrize("lang", LOCALES)
def test_date_formats_are_year_first_and_roundtrip(lang: str) -> None:
    """输入：语言文件里的日期格式。输出：无。

    预期行为：三种格式都在、都年在前、都能被 `as_date` 反解回原日期。
    """
    loc = locales.load(lang)
    assert set(loc["date"]) >= set(sp.DATE_FORMATS)
    for fmt in sp.DATE_FORMATS:
        text = render.format_date((2026, 12, 5), fmt, loc)
        assert text.startswith("2026")
        assert parse.as_date(text) == (2026, 12, 5)


@pytest.mark.parametrize("lang", LOCALES)
def test_money_and_rate_templates_carry_their_placeholders(lang: str) -> None:
    """输入：语言文件。输出：无。预期行为：模板占位符与货币表齐全（少一个就渲染出错）。"""
    loc = locales.load(lang)
    assert "{symbol}" in loc["money"]["tpl"] and "{amount}" in loc["money"]["tpl"]
    assert "{rate}" in loc["rate"]["tpl"]
    assert set(loc["currency"]) >= set(sp.CURRENCIES)


@pytest.mark.parametrize("lang", LOCALES)
def test_word_pools_are_long_enough_and_unique(lang: str) -> None:
    """输入：语言文件。输出：无。

    预期行为：spec 里存的是词表**下标**，所以每种语言的词表都不能短于 `WORD_POOL`；
    词条还要两两不同——同一个单据里出现两个同名当事方会让那道题有两个正确答案。
    """
    loc = locales.load(lang)
    for group in ("vendor", "service", "handler"):
        items = loc["word"][group]["items"]
        assert len(items) >= sp.WORD_POOL, group
        assert len(set(items)) == len(items), group


@pytest.mark.parametrize("lang", LOCALES)
def test_ask_covers_every_slot_and_every_question_is_unique(lang: str) -> None:
    """输入：语言文件。输出：无。

    预期行为：每个可问的槽都有问法，且**所有问法文本两两不同**——
    `locate` 是拿题面反查槽名的，两句话撞车就会把字段认错（判别器随之失效）。
    """
    loc = locales.load(lang)
    assert set(loc["ask"]) == (
        set(sp.DATE_SLOTS) | set(sp.MONEY_SLOTS) | set(sp.PARTY_SLOTS) | set(sp.ID_SLOTS) | {"line_unit"}
    )
    texts: list[str] = []
    for slot, entry in loc["ask"].items():
        assert entry["q"], slot
        texts.extend(entry["q"])
    # 行号模板要展开后再比：`第 1 行的单价` 不能和别的问法撞车
    texts.extend(
        text.format(n=n) for text in loc["ask"]["line_unit"]["q"] for n in (1, 2, 3, 4)
    )
    fam_texts = [text for family in loc["fam"].values() for text in family["q"]]
    assert len(set(texts)) == len(texts)
    assert len(set(fam_texts)) == len(fam_texts)
    assert not (set(texts) & set(fam_texts))


@pytest.mark.parametrize("lang", LOCALES)
def test_family_texts_and_score_levels(lang: str) -> None:
    """输入：语言文件。输出：无。

    预期行为：`fam` 覆盖全部 noul/score 题族；noul 的 criteria 只有 true/false；
    score 的等级表是「0 / 1 / 2 / 3 及以上」四档，与 `DEFECT_COUNTS` 对得上。
    """
    loc = locales.load(lang)
    assert set(loc["fam"]) == set(questions.NOUL_FAMILIES) | set(questions.SCORE_FAMILIES)
    for family in questions.NOUL_FAMILIES:
        assert set(loc["fam"][family]["criteria"]) == {"true", "false"}
    levels = loc["fam"]["bad_lines"]["levels"]
    assert len(levels) == len(sp.DEFECT_COUNTS)
    assert levels[0].startswith("0") and levels[-1].startswith("3")


def test_label_of_requires_two_labels_and_pick_wraps() -> None:
    """输入：只有一个标签的假语言文件。输出：KeyError。

    预期行为：`label_of` 缺别名直接报错（别名与正名撞成同一串就等于没留出）；
    `pick` 越界回绕（问法池可能比列表长）。
    """
    with pytest.raises(KeyError, match="至少要两个"):
        locales.label_of({"field": {"due": {"labels": ["只有一个"]}}}, "due", "aliased")
    texts = locales.load("zh")["ask"]["due"]["q"]
    assert locales.pick(texts, len(texts)) == texts[0]
    assert locales.pick(texts, len(texts) + 1) == texts[1]


def test_word_of_rejects_out_of_range_index() -> None:
    """输入：越界下标。输出：IndexError。

    预期行为：取不到就报错——静默回绕会让词与下标错位，而 spec 存的正是下标。
    """
    loc = locales.load("zh")
    with pytest.raises(IndexError, match="取不到下标"):
        locales.word_of(loc, "vendor", 99)
    with pytest.raises(IndexError):
        locales.word_of(loc, "vendor", -1)
