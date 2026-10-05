"""语言文件：标签不撞车、题族两边齐、判据写全、标的名与单位认得出。"""

from __future__ import annotations

import pytest

from workflows.quoteread import features as ft
from workflows.quoteread import locales, parse, questions

LANGS: tuple[str, ...] = ("zh", "en")


def test_merged_words_cover_every_key() -> None:
    """输入：合并反查表。输出：无。

    预期行为：每个名键在合并表里都至少有一个名字、指向的键都在名池里，单位词只指向
    pct / bp——漏一个就会让那种语言的产物整片读成「表外写法」。
    """
    words = locales.words()
    names = words["name"]
    units = words["unit"]
    assert isinstance(names, dict) and isinstance(units, dict)
    assert set(names.values()) == set(ft.NAME_IDS) | set(ft.OOD_NAME_IDS)
    assert set(units.values()) == set(ft.UNITS)


@pytest.mark.parametrize("lang", LANGS)
def test_name_keys_are_distinct_and_complete(lang: str) -> None:
    """输入：一份语言文件。输出：无。预期行为：六个标的名互不相同，键与名池一一对应。"""
    table = locales.load(lang)["name"]
    assert isinstance(table, dict)
    assert set(table) == set(ft.NAME_IDS) | set(ft.OOD_NAME_IDS), sorted(table)
    names = [str(value) for value in table.values()]
    assert len(set(names)) == len(names), names


@pytest.mark.parametrize("lang", LANGS)
def test_unit_words_are_exactly_pct_and_bp(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：`[unit]` 只有 pct / bp 两个键、写法互不相同，而且都不含在标的名里——反查表把
    它们按**整串后缀**比对，撞了就会认错。
    """
    loc = locales.load(lang)
    table = loc["unit"]
    assert isinstance(table, dict)
    assert set(table) == set(ft.UNITS)
    words = [str(value) for value in table.values()]
    assert len(set(words)) == len(ft.UNITS), words
    names = [str(value) for value in loc["name"].values()]
    for word in words:
        assert not any(word in name or name in word for name in names), word


@pytest.mark.parametrize("lang", LANGS)
def test_families_have_phrasings_and_valid_criteria(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：每个题族至少一条问法；`leader` 六个选项键（按 case 取其中三个）、两个 noul 族
    真 / 假两行、`tier` 的等级数与 `features.TIER_FLOORS` 一致——选项键与代码里那一套完全一致。
    """
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        assert loc["fam"][family]["q"], family
    assert set(loc["fam"]["leader"]["criteria"]) == set(ft.NAME_IDS) | set(ft.OOD_NAME_IDS)
    assert set(loc["fam"]["alert"]["criteria"]) == {"true", "false"}
    assert set(loc["fam"]["divergent"]["criteria"]) == {"true", "false"}
    assert len(loc["fam"]["tier"]["levels"]) == len(ft.TIER_FLOORS)


def test_meta_has_the_three_structural_labels() -> None:
    """输入：一份语言文件。输出：无。

    预期行为：`[meta]` 里抬头 / 区块抬头 / 竖线齐全——渲染要按它们拼行，缺一个就直接
    KeyError（那是接线错误，不该拖到采样阶段）。
    """
    for lang in LANGS:
        meta = locales.load(lang)["meta"]
        assert isinstance(meta, dict)
        assert set(meta) == {"request", "table", "sep"}
        assert str(meta["sep"]).strip() in {half for _full, half in parse.SEPARATORS}
