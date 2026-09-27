"""文案表的不变量：这些不是「文案好不好」，而是「答案对不对」的前提。"""

from __future__ import annotations

import pytest

from workflows.rubric import locales, parse, questions
from workflows.rubric import spec as sp

LANGS = locales.available()
#: 判据里必须印出来的四张词表。
VOCAB_TABLES = ("criterion", "weight", "need", "count")


def test_available_locales_are_sorted_and_nonempty() -> None:
    """输入：无。输出：无。预期行为：至少两种语言，顺序稳定。"""
    assert LANGS == sorted(LANGS)
    assert {"zh", "en"} <= set(LANGS)


def test_unknown_locale_reports_available_ones() -> None:
    """输入：不存在的语言。输出：无。预期行为：报错并列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="没有语言文件"):
        locales.load("xx")


def test_pick_wraps_around() -> None:
    """输入：长度为 2 的池、编号 3。输出：无。预期行为：回绕到第 1 个（不报错）。"""
    assert locales.pick(["a", "b"], 3) == "b"
    assert locales.pick(["a", "b"], 4) == "a"


@pytest.mark.parametrize("lang", LANGS)
def test_locales_share_the_same_shape(lang: str) -> None:
    """输入：每种语言。输出：无。

    预期行为：语言文件的**键集合完全一致**——少一格会让那种语言直接跑不动，少一道问法
    会让 `pick` 回绕到别的句面上（答案照样发出去，只是题变得莫名其妙）。
    """
    base = locales.load("zh")
    other = locales.load(lang)
    assert set(other["meta"]) == set(base["meta"])
    for table in VOCAB_TABLES:
        assert set(other[table]) == set(base[table])
    assert set(other["fam"]) == set(base["fam"])
    for family in base["fam"]:
        assert set(other["fam"][family]) == set(base["fam"][family])
        if "q" in base["fam"][family]:
            assert len(other["fam"][family]["q"]) == len(base["fam"][family]["q"])
        if "levels" in base["fam"][family]:
            assert other["fam"][family]["levels"] == base["fam"][family]["levels"] or (
                len(other["fam"][family]["levels"]) == len(base["fam"][family]["levels"])
            )
    assert set(other["option"]) == set(base["option"])
    for table in base["option"]:
        assert set(other["option"][table]) == set(base["option"][table])
    assert set(other["policy"]) == set(base["policy"])


@pytest.mark.parametrize("lang", LANGS)
def test_vocabularies_cover_the_spec_constants(lang: str) -> None:
    """输入：每种语言。输出：无。

    预期行为：每张词表的键就是 spec 里的常量——少一个取值会让渲染 `KeyError`，
    多一个取值会让反解出来的键落进规则函数不认识的洞里。
    """
    loc = locales.load(lang)
    assert set(loc["criterion"]) == set(sp.CRITERIA)
    assert set(loc["weight"]) == set(sp.WEIGHTS)
    assert set(loc["need"]) == set(sp.NEEDS)
    assert set(loc["count"]) == set(sp.COUNTS)
    assert set(loc["option"]["gate"]) == set(sp.GATES)
    assert set(loc["fam"]) == set(questions.FAMILIES)


@pytest.mark.parametrize("lang", LANGS)
def test_score_levels_and_choice_options_match_the_bands_and_gates(lang: str) -> None:
    """输入：每种语言的档位等级表与处置选项表。输出：无。

    预期行为：等级数组的长度就是档位数、选项表的键就是处置键——`band` 的答案是**下标**，
    长度不对会让「高档」落到别的编号上；处置少一个键会让 `gate` 直接 KeyError。
    """
    loc = locales.load(lang)
    assert len(loc["fam"]["band"]["levels"]) == len(sp.BANDS)
    assert len(set(loc["fam"]["band"]["levels"])) == len(sp.BANDS)
    assert set(loc["option"]["gate"]) == set(sp.GATES)


@pytest.mark.parametrize("lang", LANGS)
def test_policy_prints_every_table(lang: str) -> None:
    """输入：每种语言的判据原文。输出：无。

    预期行为：六条判据、三档权重、三档要求、五档观测都在判据里印出来——考生只能从题面读
    规则，漏印一项就等于把那一档的答案变成不可解。
    """
    loc = locales.load(lang)
    policy = str(loc["policy"]["text"])
    for table in VOCAB_TABLES:
        for text in loc[table].values():
            assert str(text) in policy, (lang, table, text)


@pytest.mark.parametrize("lang", LANGS)
def test_policy_prints_every_band_and_gate_word(lang: str) -> None:
    """输入：每种语言的判据原文。输出：无。

    预期行为：三档档位名与三个处置名都出现在判据里——选项 / 等级的描述与判据里的措辞
    对不上时，考生只能靠猜。
    """
    loc = locales.load(lang)
    policy = str(loc["policy"]["text"])
    for text in loc["fam"]["band"]["levels"]:
        head = str(text).split("：")[0].split(":")[0].strip()
        assert head and head in policy, (lang, head)
    for text in loc["option"]["gate"].values():
        head = str(text).split("：")[0].split(":")[0].strip()
        assert head and head in policy, (lang, head)


@pytest.mark.parametrize("lang", LANGS)
def test_policy_names_the_redline_criterion(lang: str) -> None:
    """输入：每种语言的判据原文。输出：无。

    预期行为：红线那条判据的原文必须印在判据里——不印的话「哪条是红线」就没法从题面读出来。
    """
    loc = locales.load(lang)
    assert str(loc["criterion"][sp.REDLINE]) in str(loc["policy"]["text"])


@pytest.mark.parametrize("lang", LANGS)
def test_noul_criteria_are_distinct_and_nonempty(lang: str) -> None:
    """输入：每种语言的 `met` 题族判据。输出：无。预期行为：真 / 假两行都不空且不同。"""
    criteria = locales.load(lang)["fam"]["met"]["criteria"]
    assert set(criteria) == {"true", "false"}
    assert all(criteria.values())
    assert criteria["true"] != criteria["false"]


@pytest.mark.parametrize("lang", LANGS)
def test_cell_separator_is_shared_across_languages(lang: str) -> None:
    """输入：每种语言。输出：无。预期行为：四格分隔符一致（合并表只按一个分隔符切行）。"""
    assert locales.load(lang)["meta"]["cell_sep"] == locales.load("zh")["meta"]["cell_sep"]


def test_merged_table_covers_every_key() -> None:
    """输入：合并表。输出：无。

    预期行为：四张词表把 spec 里的每一个键都覆盖到，而且语言之间不重叠（重叠会直接
    报错，这里验的是「合并之后仍然认得出所有键」）。
    """
    merged = parse.merged()
    assert set(merged.criterion.values()) == set(sp.CRITERIA)
    assert set(merged.weight.values()) == set(sp.WEIGHTS)
    assert set(merged.need.values()) == set(sp.NEEDS)
    assert set(merged.count.values()) == set(sp.COUNTS)

