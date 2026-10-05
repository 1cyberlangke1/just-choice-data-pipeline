"""语言文件：标签不撞车、题族两边齐、判据写全、主题名认得出。"""

from __future__ import annotations

import pytest

from workflows.clipcut import features as ft
from workflows.clipcut import locales, parse, questions

LANGS: tuple[str, ...] = ("zh", "en")


def test_merged_words_cover_every_key() -> None:
    """输入：合并反查表。输出：无。

    预期行为：每个主题键在合并表里都至少有一个名字、指向的键都在主题池里，高亮词只指向
    yes / no——漏一个就会让那种语言的产物整片读成「表外名字」。
    """
    words = locales.words()
    topics = words["topic"]
    flags = words["flag"]
    assert isinstance(topics, dict) and isinstance(flags, dict)
    assert set(topics.values()) == set(ft.TOPIC_IDS) | {ft.OOD_TOPIC_ID}
    assert set(flags.values()) == {"yes", "no"}


@pytest.mark.parametrize("lang", LANGS)
def test_topic_names_are_distinct_and_complete(lang: str) -> None:
    """输入：一份语言文件。输出：无。预期行为：五个主题名互不相同，键与主题池一一对应。"""
    table = locales.load(lang)["topic"]
    assert isinstance(table, dict)
    assert set(table) == set(ft.TOPIC_IDS) | {ft.OOD_TOPIC_ID}, sorted(table)
    names = [str(value) for value in table.values()]
    assert len(set(names)) == len(names), names


@pytest.mark.parametrize("lang", LANGS)
def test_flag_words_are_exactly_yes_and_no(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：`[flag]` 只有 yes / no 两个词、互不相同，而且都不含在主题名里——反查表把
    它们按整串比对，撞了就会认错。
    """
    loc = locales.load(lang)
    table = loc["flag"]
    assert isinstance(table, dict)
    assert set(table) == {"yes", "no"}
    words = [str(value) for value in table.values()]
    assert len(set(words)) == 2, words
    topics = [str(value) for value in loc["topic"].values()]
    for word in words:
        assert not any(word in topic or topic in word for topic in topics), word


@pytest.mark.parametrize("lang", LANGS)
def test_families_have_phrasings_and_valid_criteria(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：每个题族至少一条问法；`first_kept` 五个选项键、两个 noul 族真 / 假两行、
    `keep_count` 的等级数与 `features.SEGMENTS + 1` 一致——选项键与代码里那一套完全一致。
    """
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        assert loc["fam"][family]["q"], family
    assert set(loc["fam"]["first_kept"]["criteria"]) == {"s1", "s2", "s3", "s4", "none"}
    assert set(loc["fam"]["blocked_any"]["criteria"]) == {"true", "false"}
    assert set(loc["fam"]["dup_any"]["criteria"]) == {"true", "false"}
    assert len(loc["fam"]["keep_count"]["levels"]) == ft.SEGMENTS + 1


@pytest.mark.parametrize("lang", LANGS)
def test_rules_lines_are_not_empty(lang: str) -> None:
    """输入：一份语言文件。输出：无。预期行为：判据原文非空、每一条都不是空串。"""
    lines = locales.load(lang)["rules"]["lines"]
    assert lines
    for line in lines:
        assert str(line).strip(), lang


def test_meta_has_the_four_structural_labels() -> None:
    """输入：一份语言文件。输出：无。

    预期行为：`[meta]` 里抬头 / 区块抬头 / 圆括号对齐全——渲染要按它们拼行，缺一个就
    直接 KeyError（那是接线错误，不该拖到采样阶段）。
    """
    for lang in LANGS:
        meta = locales.load(lang)["meta"]
        assert isinstance(meta, dict)
        assert set(meta) == {"request", "segments", "paren_open", "paren_close"}
        assert str(meta["paren_open"]).strip() in {"（", "("}
        assert str(meta["paren_close"]).strip() in {"）", ")"}
