"""语言文件：标签不撞车、题族两边齐、判据写全、条目名认得出。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import pytest

from workflows.siftrank import locales, parse, questions
from workflows.siftrank import spec as sp

LANGS: tuple[str, ...] = ("zh", "en")


def test_merged_words_cover_every_item() -> None:
    """输入：合并反查表。输出：无。

    预期行为：每个条目键在合并表里都至少有一个名字，而且指向的键都在条目池里——漏一个
    就会让那种语言的产物整片读成「表外名字」。
    """
    table = locales.words()["item"]
    assert isinstance(table, dict)
    assert set(table.values()) == set(sp.ITEMS), sorted(str(value) for value in table.values())


@pytest.mark.parametrize("lang", LANGS)
def test_labels_are_not_mutual_prefixes(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：四个抬头两两**不互为前缀**——反解靠最长前缀认标签，互为前缀时短的会把长的
    吃掉（「排序请求号」被认成「排序请求」）。
    """
    tables = parse.tables_of(locales.load(lang))
    labels = [*tables.request, *tables.prompt, *tables.items, *tables.batch]
    for left in labels:
        for right in labels:
            if left != right:
                assert not right.startswith(left), f"{lang}: {left!r} / {right!r}"


def test_languages_do_not_share_labels() -> None:
    """输入：两份语言文件。输出：无。预期行为：标签原文互不重叠（合并表才不炸）。"""
    zh = parse.tables_of(locales.load("zh"))
    en = parse.tables_of(locales.load("en"))
    for name in ("request", "prompt", "items", "batch"):
        assert not set(getattr(zh, name)) & set(getattr(en, name)), name
    merged = parse.merged_tables()
    for name in ("request", "prompt", "items", "batch"):
        assert set(getattr(merged, name)) == set(getattr(zh, name)) | set(getattr(en, name)), name


def test_tables_of_rejects_duplicate_labels() -> None:
    """输入：两个抬头写成一样的语言字典。输出：无。预期行为：报错，不静默认错行。"""
    loc: dict[str, JsonValue] = dict(locales.load("zh"))
    meta = dict(loc["meta"])
    meta["prompt"] = meta["request"]
    loc["meta"] = meta
    with pytest.raises(ValueError, match="四个抬头标签有重复"):
        parse.tables_of(loc)


@pytest.mark.parametrize("lang", LANGS)
def test_item_names_are_distinct_and_complete(lang: str) -> None:
    """输入：一份语言文件。输出：无。预期行为：六个条目名互不相同，键与条目池一一对应。"""
    table = locales.load(lang)["item"]
    assert set(table) == set(sp.ITEMS), sorted(table)
    names = [str(value) for value in table.values()]
    assert len(set(names)) == len(sp.ITEMS), names


@pytest.mark.parametrize("lang", LANGS)
def test_families_have_phrasings_and_valid_criteria(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：每个题族至少一条问法；`pair_order` 三个选项键、`stable` 真 / 假两行、
    `cut_size` 的等级数与 `spec.CUTS` 一致——选项键与代码里那一套完全一致。
    """
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        assert loc["fam"][family]["q"], family
    assert set(loc["fam"]["pair_order"]["criteria"]) == set(sp.RELATIONS)
    assert set(loc["fam"]["stable"]["criteria"]) == {"true", "false"}
    assert len(loc["fam"]["cut_size"]["levels"]) == len(sp.CUTS)


@pytest.mark.parametrize("lang", LANGS)
def test_pair_phrasings_carry_both_placeholders(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：`pair_order` 的每条问法都同时带 `{a}` 与 `{b}`——题面必须点名是哪两个条目，
    少一个占位符就会问「条目「甲」与条目「甲」谁更相关」。
    """
    for text in locales.load(lang)["fam"]["pair_order"]["q"]:
        assert "{a}" in text and "{b}" in text, text


@pytest.mark.parametrize("lang", LANGS)
def test_pair_criteria_name_exactly_one_item_each(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：`a_first` / `b_first` 两条描述各只点名**一个**条目名——判别器靠「每个选项
    描述里恰好一个条目名」把被比较的两个条目认出来，两边都写全就会认不出。
    """
    loc = locales.load(lang)
    names = [str(value) for value in loc["item"].values()]
    criteria = loc["fam"]["pair_order"]["criteria"]
    for key in ("a_first", "b_first"):
        text = str(criteria[key]).format(a=names[0], b=names[1])
        assert sum(1 for name in names if name in text) == 1, (key, text)
    tie = str(criteria["tie"]).format(a=names[0], b=names[1])
    assert not any(name in tie for name in names), tie


@pytest.mark.parametrize("lang", LANGS)
def test_rules_state_the_elbow_and_the_consistency_rule(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：判据原文里出现**三档肘部的档名**（「第 1 名与第 2 名之间」…）与「一致」的
    判据——判据必须让只读 state 与题面的读者自己算得出来（范式 §9.8），档名是读者要对照的
    那张表的入口。
    """
    loc = locales.load(lang)
    rules = "\n".join(loc["rules"]["lines"])
    for level in loc["fam"]["cut_size"]["levels"]:
        head = str(level).split("——")[0].split(" - ")[0].strip()
        assert head and head in rules, f"{lang}: 档位名 {head!r} 没写进判据"
    assert "并列" in rules or "tie" in rules.lower(), rules
