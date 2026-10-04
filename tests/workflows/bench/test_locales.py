"""文案表的不变量：这些不是「文案好不好」，而是「答案对不对」的前提。"""

from __future__ import annotations

import pytest

from workflows.bench import locales, parse, questions
from workflows.bench import spec as sp

LANGS = locales.available()
#: 两族 noul 的判据（真 / 假两行文案）。
NOUL_FAMILIES = ("hit", "pass")


@pytest.mark.parametrize("lang", LANGS)
def test_locales_share_the_same_shape(lang: str) -> None:
    """输入：每种语言。输出：无。

    预期行为：语言文件的**键集合完全一致**——少一格会让那种语言直接跑不动，少一道问法
    会让 `pick` 回绕到别的句面上（答案照样发出去，只是题变得莫名其妙）。
    """
    base = locales.load("zh")
    other = locales.load(lang)
    for table in ("meta", "label", "cell", "cand", "policy"):
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


@pytest.mark.parametrize("lang", LANGS)
def test_vocabularies_cover_the_spec_constants(lang: str) -> None:
    """输入：每种语言。输出：无。

    预期行为：候选词表的键就是 spec 里的常量，题族的键集合也一致——少一个取值会让渲染
    `KeyError`，多一个取值会让反解出来的键落进规则函数不认识的洞里。
    """
    loc = locales.load(lang)
    assert set(loc["cand"]) == set(sp.CANDIDATES)
    assert set(loc["fam"]) == set(questions.FAMILIES)
    assert set(questions.FAMILIES) == {"hit", "count", "pass", "best", "gap"}


@pytest.mark.parametrize("lang", LANGS)
def test_score_levels_match_the_bands(lang: str) -> None:
    """输入：每种语言的档位等级表。输出：无。

    预期行为：等级数组的长度就是档位数、描述两两不同——`count` / `gap` 的答案是**下标**，
    长度不对会让某一档落到别的编号上。
    """
    loc = locales.load(lang)
    for family, values in (("count", sp.COUNTS), ("gap", sp.GAP_VALUES)):
        levels = loc["fam"][family]["levels"]
        assert len(levels) == len(values)
        assert len(set(levels)) == len(values)


@pytest.mark.parametrize("lang", LANGS)
def test_policy_prints_every_letter_and_band(lang: str) -> None:
    """输入：每种语言的判据原文。输出：无。

    预期行为：4 个选项字母、6 档答对条数、门槛的取值域、2 档差距都印在判据里——考生只能
    从题面读规则，漏印一项就等于把那一族的答案变成不可解。
    """
    loc = locales.load(lang)
    policy = str(loc["policy"]["text"])
    for letter in sp.LETTERS:
        assert letter in policy, (lang, letter)
    for level in loc["fam"]["count"]["levels"]:
        assert str(level) in policy, (lang, level)
    for level in loc["fam"]["gap"]["levels"]:
        assert str(level) in policy, (lang, level)
    for bar in sp.BARS:
        assert str(bar) in policy, (lang, bar)
    assert str(sp.TOTAL) in policy


@pytest.mark.parametrize("lang", LANGS)
def test_noul_criteria_are_distinct_and_nonempty(lang: str) -> None:
    """输入：每种语言的两族 noul 判据。输出：无。预期行为：真 / 假两行都不空且不同。"""
    fam = locales.load(lang)["fam"]
    for family in NOUL_FAMILIES:
        criteria = fam[family]["criteria"]
        assert set(criteria) == {"true", "false"}
        assert all(criteria.values())
        assert criteria["true"] != criteria["false"]


@pytest.mark.parametrize("lang", LANGS)
def test_separators_are_shared_across_languages(lang: str) -> None:
    """输入：每种语言。输出：无。

    预期行为：格分隔符与格内的名字 / 字母分隔符一致（合并表只按一个分隔符切行、切格），
    冒号**允许**不同（中文 `：`、英文 `: `，各自只在自己的语言里用）。
    """
    loc = locales.load(lang)
    base = locales.load("zh")
    assert loc["meta"]["cell_sep"] == base["meta"]["cell_sep"]
    assert loc["meta"]["pair_sep"] == base["meta"]["pair_sep"]


@pytest.mark.parametrize("lang", LANGS)
def test_candidate_names_do_not_contain_the_in_cell_separator(lang: str) -> None:
    """输入：每种语言的候选名与参考格标签。输出：无。

    预期行为：名字里不含格内分隔符——含了那一格就会被切成三段，反解直接报错（与其等到
    生成时炸，不如在这里点名）。
    """
    loc = locales.load(lang)
    sep = str(loc["meta"]["pair_sep"])
    names = [*loc["cand"].values(), loc["cell"]["ref"]]
    assert all(sep not in str(name) for name in names), names


@pytest.mark.parametrize("lang", LANGS)
def test_head_labels_do_not_collide_with_the_gold_cell(lang: str) -> None:
    """输入：每种语言的抬头字段名与参考格标签。输出：无。

    预期行为：抬头字段名与参考格标签互不相同——`_field_value` 按字段名认格，参考格标签要是
    撞上抬头字段名，反解会把记录行认成抬头（或反过来）。
    """
    loc = locales.load(lang)
    heads = {str(loc["label"][field]) for field in parse.LABEL_FIELDS}
    assert str(loc["cell"]["ref"]) not in heads
    assert len(heads) == len(parse.LABEL_FIELDS)


def test_merged_table_covers_every_key() -> None:
    """输入：合并表。输出：无。

    预期行为：候选词表把 spec 里的每一个键都覆盖到，而且语言之间不重叠（重叠会直接报错，
    这里验的是「合并之后仍然认得出所有键」）。
    """
    merged = parse.merged()
    assert set(merged.cand.values()) == set(sp.CANDIDATES)
    assert len(merged.cand) == len(sp.CANDIDATES) * len(LANGS)
    assert set(merged.label_texts) == set(parse.LABEL_FIELDS)
    assert merged.cell_sep == str(locales.load("zh")["meta"]["cell_sep"])
    assert merged.pair_sep == str(locales.load("zh")["meta"]["pair_sep"])
    assert {": ", "："} <= set(merged.colons)
