"""文案表的不变量：这些不是「文案好不好」，而是「答案对不对」的前提。"""

from __future__ import annotations

import pytest

from workflows.browser import locales, parse
from workflows.browser import spec as sp
from workflows.browser.spec import ACTIONS, DECOYS, NEUTRALS, OBJECTS, ROLES, STATES, TASKS

LANGS = locales.available()


def _every_object_key() -> set[str]:
    """输入：无。输出：spec 里用到的全部对象键（陪衬 + 干扰 + 目标）。"""
    used: set[str] = set(NEUTRALS.values()) | set(DECOYS.values())
    for task in TASKS:
        used |= set(OBJECTS[task])
    return used


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
    for table in ("role", "state", "intent", "object"):
        assert set(other[table]) == set(base[table])
    assert set(other["fam"]) == set(base["fam"])
    for family in base["fam"]:
        assert len(other["fam"][family]["q"]) == len(base["fam"][family]["q"])
        assert set(other["fam"][family]) == set(base["fam"][family])
    assert set(other["option"]) == set(base["option"])
    for table in base["option"]:
        assert set(other["option"][table]) == set(base["option"][table])
    assert set(other["policy"]) == set(base["policy"])
    assert set(other["sensitive"]) == set(base["sensitive"])


@pytest.mark.parametrize("lang", LANGS)
def test_vocabularies_cover_the_spec_constants(lang: str) -> None:
    """输入：每种语言。输出：无。

    预期行为：每张词表的键就是 spec 里的常量——少一个取值会让渲染 `KeyError`，
    多一个取值会让反解出来的键落进规则函数不认识的洞里。
    """
    loc = locales.load(lang)
    assert set(loc["role"]) == set(ROLES)
    assert set(loc["state"]) == set(STATES)
    assert set(loc["intent"]) == set(TASKS)
    assert set(loc["object"]) == _every_object_key()
    assert set(loc["option"]["action"]) == set(ACTIONS)


@pytest.mark.parametrize("lang", LANGS)
def test_sensitive_labels_really_hit_the_list(lang: str) -> None:
    """输入：每种语言的每一条对象文案。输出：无。

    预期行为：**命中敏感清单的对象，文案里真的含清单里的词；其余对象一个都不含**——
    这是「risk 那一族答案对不对」的前提：题面里的判据是**子串匹配**，而构造侧用的是
    对象键上的布尔标记，两者不一致时产出的答案就是错的。
    """
    loc = locales.load(lang)
    words = loc["sensitive"]["words"]
    for key, text in loc["object"].items():
        hit = any(word in text for word in words)
        assert hit is sp.is_sensitive(key), (lang, key, text)


@pytest.mark.parametrize("lang", LANGS)
def test_policy_prints_the_sensitive_list(lang: str) -> None:
    """输入：每种语言的判据原文。输出：无。

    预期行为：敏感清单里的每个词都出现在判据里——考生只能从题面读到清单，漏印一个词
    就等于把那一档的答案变成不可解。
    """
    loc = locales.load(lang)
    for word in loc["sensitive"]["words"]:
        assert word in loc["policy"]["text"], (lang, word)


@pytest.mark.parametrize("lang", LANGS)
def test_cell_separator_is_shared_across_languages(lang: str) -> None:
    """输入：每种语言。输出：无。预期行为：三格分隔符一致（合并表只按一个分隔符切行）。"""
    assert locales.load(lang)["meta"]["cell_sep"] == locales.load("zh")["meta"]["cell_sep"]


def test_merged_table_covers_every_key() -> None:
    """输入：合并表。输出：无。

    预期行为：四张词表把 spec 里的每一个键都覆盖到，而且语言之间不重叠（重叠会直接
    报错，这里验的是「合并之后仍然认得出所有键」）。
    """
    merged = parse.merged()
    assert set(merged.role.values()) == set(ROLES)
    assert set(merged.state.values()) == set(STATES)
    assert set(merged.intent.values()) == set(TASKS)
    assert set(merged.object.values()) == _every_object_key()
