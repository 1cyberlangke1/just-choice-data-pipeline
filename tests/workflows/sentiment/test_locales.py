"""语言文件：结构词元两边一致、标签不撞车、词表完整、题族两边齐。"""

from __future__ import annotations

from typing import Final

import pytest

from workflows.sentiment import features as ft
from workflows.sentiment import locales, parse, questions

LANGS: Final[tuple[str, ...]] = ("zh", "en")


@pytest.mark.parametrize("lang", LANGS)
def test_topic_table_is_complete(lang: str) -> None:
    """输入：一份语言文件。输出：无。预期行为：每个场景都有名字与两个称呼，且都不重复。"""
    table = locales.load(lang)["topic"]
    assert isinstance(table, dict)
    assert set(table) == set(ft.TOPICS), sorted(table)
    words = [str(value) for entry in table.values() for value in entry.values()]
    assert len(set(words)) == len(words), words


@pytest.mark.parametrize("lang", LANGS)
def test_word_table_is_five_levels_of_four(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：`[word]` 恰好五个档位、每档 `WORD_SLOTS` 个词、二十个词互不相同——词数对不上
    会让 `word_slot` 取到表外下标。
    """
    table = locales.load(lang)["word"]
    assert isinstance(table, dict)
    assert set(table) == {str(level) for level in ft.LEVELS}, sorted(table)
    all_words: list[str] = []
    for level in ft.LEVELS:
        slot = table[str(level)]
        assert isinstance(slot, list) and len(slot) == ft.WORD_SLOTS, level
        all_words.extend(str(word) for word in slot)
    assert len(set(all_words)) == len(all_words), all_words


@pytest.mark.parametrize("lang", LANGS)
def test_labels_are_not_mutual_prefixes(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：抬头 / 场景抬头 / 区块抬头两两**不互为前缀**——反解靠最长前缀认标签，
    互为前缀时短的会把长的吃掉。
    """
    tables = parse.tables_of(locales.load(lang))
    labels = [*tables.request, *tables.topic_labels, *tables.sections]
    for left in labels:
        for right in labels:
            if left != right:
                assert not right.startswith(left), f"{lang}: {left!r} / {right!r}"


def test_languages_do_not_share_labels() -> None:
    """输入：两份语言文件。输出：无。预期行为：标签原文互不重叠（合并表才不炸）。"""
    zh = parse.tables_of(locales.load("zh"))
    en = parse.tables_of(locales.load("en"))
    assert not set(zh.request) & set(en.request)
    assert not set(zh.topic_labels) & set(en.topic_labels)
    assert not set(zh.sections) & set(en.sections)
    merged = parse.merged_tables()
    assert set(merged.request) == set(zh.request) | set(en.request)
    assert set(merged.topic_labels) == set(zh.topic_labels) | set(en.topic_labels)
    assert set(merged.sections) == set(zh.sections) | set(en.sections)


@pytest.mark.parametrize("lang", LANGS)
def test_families_have_phrasings_and_valid_criteria(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：每个题族至少一条问法；`tone` 三个选项、`pivot` 五个选项、`drop` 五档、
    `rebound` 真假两行——选项键与代码里那一套完全一致。
    """
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        assert loc["fam"][family]["q"], family
    assert set(loc["fam"]["tone"]["criteria"]) == {"pos", "neg", "flat"}
    assert set(loc["fam"]["pivot"]["criteria"]) == set(questions.PIVOT_KEYS)
    assert set(loc["fam"]["rebound"]["criteria"]) == {"true", "false"}
    assert len(loc["fam"]["drop"]["levels"]) == 5


def test_structural_tokens_are_identical_across_languages() -> None:
    """输入：两份语言文件。输出：无。

    预期行为：格分隔符 / 词分隔符 / 规则行前缀三种**结构词元**中英完全一致——反解只有一份
    合并表，各语言用不同的分隔符会让另一种语言的产物整片读不出来。
    """
    zh = locales.load("zh")["meta"]
    en = locales.load("en")["meta"]
    for key in ("cell_sep", "word_sep", "rules_prefix"):
        assert zh[key] == en[key], key
