"""文案表的不变量：这些不是「文案好不好」，而是「答案对不对」的前提。"""

from __future__ import annotations

import pytest

from workflows.sim import locales, parse, questions
from workflows.sim import spec as sp

LANGS = locales.available()
#: 判据里必须印出来的三张**档位**词表（动作名印在局面里，不在判据里）。
BAND_TABLES = ("cost", "gain", "cooldown")


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
    assert set(other["label"]) == set(base["label"])
    for table in ("action", "cost", "gain", "cooldown"):
        assert set(other[table]) == set(base[table])
    assert set(other["fam"]) == set(base["fam"])
    for family in base["fam"]:
        assert set(other["fam"][family]) == set(base["fam"][family])
        if "q" in base["fam"][family]:
            assert len(other["fam"][family]["q"]) == len(base["fam"][family]["q"])
        if "levels" in base["fam"][family]:
            assert len(other["fam"][family]["levels"]) == len(base["fam"][family]["levels"])
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
    assert set(loc["action"]) == set(sp.ACTIONS)
    assert set(loc["cost"]) == set(sp.COSTS)
    assert set(loc["gain"]) == set(sp.GAINS)
    assert set(loc["cooldown"]) == set(sp.COOLDOWNS)
    assert set(loc["fam"]) == set(questions.FAMILIES)


@pytest.mark.parametrize("lang", LANGS)
def test_score_levels_match_the_bands(lang: str) -> None:
    """输入：每种语言的档位等级表。输出：无。

    预期行为：等级数组的长度就是档位数、描述两两不同——`gain` / `cost` 的答案是**下标**，
    长度不对会让某一档落到别的编号上。
    """
    loc = locales.load(lang)
    assert len(loc["fam"]["gain"]["levels"]) == len(sp.GAINS)
    assert len(set(loc["fam"]["gain"]["levels"])) == len(sp.GAINS)
    assert len(loc["fam"]["cost"]["levels"]) == len(sp.COSTS)
    assert len(set(loc["fam"]["cost"]["levels"])) == len(sp.COSTS)


@pytest.mark.parametrize("lang", LANGS)
def test_policy_prints_every_band(lang: str) -> None:
    """输入：每种语言的判据原文。输出：无。

    预期行为：4 档耗、6 档收益、3 档冷却都印在判据里——考生只能从题面读规则，
    漏印一项就等于把那一族的答案变成不可解。
    """
    loc = locales.load(lang)
    policy = str(loc["policy"]["text"])
    for table in BAND_TABLES:
        for text in loc[table].values():
            assert str(text) in policy, (lang, table, text)


@pytest.mark.parametrize("lang", LANGS)
def test_policy_prints_the_two_thresholds(lang: str) -> None:
    """输入：每种语言的判据原文。输出：无。

    预期行为：战利品门槛 6 与回合上限 5 都印在判据里——结束条件只从这两格读，
    不印就等于「结没结束」不可解。
    """
    policy = str(locales.load(lang)["policy"]["text"])
    assert str(sp.TARGET) in policy
    assert str(sp.ROUND_CAP) in policy


@pytest.mark.parametrize("lang", LANGS)
def test_noul_criteria_are_distinct_and_nonempty(lang: str) -> None:
    """输入：每种语言的两族 noul 判据。输出：无。预期行为：真 / 假两行都不空且不同。"""
    fam = locales.load(lang)["fam"]
    for family in ("can_move", "finished"):
        criteria = fam[family]["criteria"]
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
    assert set(merged.action.values()) == set(sp.ACTIONS)
    assert set(merged.cost.values()) == set(sp.COSTS)
    assert set(merged.gain.values()) == set(sp.GAINS)
    assert set(merged.cooldown.values()) == set(sp.COOLDOWNS)
