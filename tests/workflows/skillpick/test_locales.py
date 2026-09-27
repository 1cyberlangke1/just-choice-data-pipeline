"""文案表的不变量：这些不是「文案好不好」，而是「答案对不对」的前提。"""

from __future__ import annotations

import pytest

from workflows.skillpick import locales, parse, questions
from workflows.skillpick import spec as sp

LANGS = locales.available()
#: 判据里必须印出来的四张词表。
VOCAB_TABLES = ("tag", "scope", "resource", "grant")


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
        assert len(other["fam"][family]["q"]) == len(base["fam"][family]["q"])
        assert set(other["fam"][family]) == set(base["fam"][family])
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
    assert set(loc["tag"]) == set(sp.TAGS)
    assert set(loc["scope"]) == set(sp.SCOPES)
    assert set(loc["resource"]) == set(sp.RESOURCES)
    assert set(loc["grant"]) == set(sp.GRANTS)
    assert set(loc["option"]["scope"]) == set(sp.SCOPES)
    assert set(loc["fam"]) == set(questions.FAMILIES)


@pytest.mark.parametrize("lang", LANGS)
def test_policy_prints_every_table(lang: str) -> None:
    """输入：每种语言的判据原文。输出：无。

    预期行为：十二个标签、三档权限范围、三类资源、两种授权都在判据里印出来——考生只能
    从题面读规则，漏印一项就等于把那一档的答案变成不可解。
    """
    loc = locales.load(lang)
    policy = str(loc["policy"]["text"])
    for table in VOCAB_TABLES:
        for text in loc[table].values():
            assert str(text) in policy, (lang, table, text)


@pytest.mark.parametrize("lang", LANGS)
def test_policy_prints_every_scope_word(lang: str) -> None:
    """输入：每种语言的判据原文。输出：无。

    预期行为：权限三个选项的描述开头那个词都出现在判据里——选项名与判据里的措辞对不上时，
    考生只能靠猜。
    """
    loc = locales.load(lang)
    policy = str(loc["policy"]["text"])
    for text in loc["option"]["scope"].values():
        head = str(text).split("：")[0].split(":")[0].strip()
        assert head and head in policy, (lang, head)


@pytest.mark.parametrize("lang", LANGS)
def test_noul_criteria_are_distinct_and_nonempty(lang: str) -> None:
    """输入：每种语言的三个 noul 题族判据。输出：无。预期行为：真 / 假两行都不空且不同。"""
    fam = locales.load(lang)["fam"]
    for family in ("fits", "allow", "conflict"):
        criteria = fam[family]["criteria"]
        assert set(criteria) == {"true", "false"}
        assert all(criteria.values())
        assert criteria["true"] != criteria["false"]


@pytest.mark.parametrize("lang", LANGS)
def test_cell_separator_is_shared_across_languages(lang: str) -> None:
    """输入：每种语言。输出：无。预期行为：抬头分隔符一致（合并表只按一个分隔符切行）。"""
    assert locales.load(lang)["meta"]["cell_sep"] == locales.load("zh")["meta"]["cell_sep"]


@pytest.mark.parametrize("lang", LANGS)
def test_decoy_tags_are_substring_traps_in_every_language(lang: str) -> None:
    """输入：每种语言的标签表。输出：无。

    预期行为：每个干扰标签在**原文层面**也把对应的技能标签当子串含在里面——陷阱要跨语言
    成立（英文里 "read code" 也在 "read code notes" 里），否则那种语言的干扰标签就退化成
    普通噪声，「完全一致才算匹配」这条判据也就没有对手了。
    """
    loc = locales.load(lang)
    for orphan in sp.ORPHANS:
        parents = [tag for tag in sp.TAGS if tag != orphan and tag in orphan]
        assert parents, orphan
        for parent in parents:
            assert str(loc["tag"][parent]) in str(loc["tag"][orphan]), (lang, parent, orphan)


def test_merged_table_covers_every_key() -> None:
    """输入：合并表。输出：无。

    预期行为：四张词表把 spec 里的每一个键都覆盖到，而且语言之间不重叠（重叠会直接
    报错，这里验的是「合并之后仍然认得出所有键」）。
    """
    merged = parse.merged()
    assert set(merged.tag.values()) == set(sp.TAGS)
    assert set(merged.scope.values()) == set(sp.SCOPES)
    assert set(merged.resource.values()) == set(sp.RESOURCES)
    assert set(merged.grant.values()) == set(sp.GRANTS)
