"""文案表的不变量：这些不是「文案好不好」，而是「答案对不对」的前提。"""

from __future__ import annotations

from itertools import product

import pytest

from workflows.context import locales, parse
from workflows.context.spec import ACTIONS, AGES, KINDS, SIZES, TOPICS

LANGS = locales.available()


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
    for table in ("kind", "size", "age", "topic", "content"):
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

    预期行为：每个词表的键就是 spec 里的常量——少一个取值会让渲染 `KeyError`，
    多一个取值会让反解出来的键落进规则函数不认识的洞里。
    """
    loc = locales.load(lang)
    assert set(loc["kind"]) == set(KINDS)
    assert set(loc["size"]) == set(SIZES)
    assert set(loc["age"]) == set(AGES)
    assert set(loc["topic"]) == set(TOPICS)
    assert set(loc["option"]["action"]) == set(ACTIONS)
    assert set(loc["content"]) == {
        f"{kind}_{size}_{age}_{topic}"
        for kind, size, age, topic in product(KINDS, SIZES, AGES, TOPICS)
    }


@pytest.mark.parametrize("lang", LANGS)
def test_content_templates_are_distinct(lang: str) -> None:
    """输入：每种语言的正文表。输出：无。

    预期行为：没有两格正文一模一样——正文按四元组取，两格一样说明语言文件写重复了，
    同一屏里就会出现两行相同的正文（看起来像渲染漏了块）。
    """
    table = locales.load(lang)["content"]
    assert len(set(table.values())) == len(table)


@pytest.mark.parametrize("lang", LANGS)
def test_policy_spells_out_every_rule_and_number(lang: str) -> None:
    """输入：每种语言。输出：无。

    预期行为：判据原文里点名了四个数字（180 / 1600 / 800 / 60）、报错块、两种尺寸与
    两个主题——判据写在题面里（范式 §9.8），少写一个，考生就只能靠猜。
    """
    loc = locales.load(lang)
    policy = loc["policy"]["text"]
    for number in ("180", "1600", "800", "60"):
        assert number in policy, f"{lang}: 判据里没有 {number}"
    assert loc["kind"]["err"] in policy, f"{lang}: 判据里没有报错块"
    for size in SIZES:
        assert loc["size"][size] in policy, f"{lang}: 判据里没有尺寸 {size}"
    for topic in TOPICS:
        assert loc["topic"][topic] in policy, f"{lang}: 判据里没有主题 {topic}"


@pytest.mark.parametrize("lang", LANGS)
def test_labels_are_distinct_within_a_locale(lang: str) -> None:
    """输入：每种语言。输出：无。预期行为：同一语言里没有两格标签文案相同。"""
    labels = parse.labels_of(locales.load(lang))
    texts = [*labels.session_labels, *labels.budget_labels, *labels.task_labels]
    assert len(set(texts)) == len(texts)


def test_merged_labels_cover_both_locales_without_collisions() -> None:
    """输入：无。输出：无。

    预期行为：三格标签各语言互不相同、四张属性词表也互不重叠，分隔符则**必须一致**——
    `reparse` 没有语言参数，只靠这张合并表。
    """
    merged = parse.merged()
    for labels in (merged.session_labels, merged.budget_labels, merged.task_labels):
        assert len(labels) == len(LANGS)
        assert len(set(labels)) == len(labels)
    assert len(merged.kind) == len(KINDS) * len(LANGS)
    assert len(merged.size) == len(SIZES) * len(LANGS)
    assert len(merged.age) == len(AGES) * len(LANGS)
    assert len(merged.topic) == len(TOPICS) * len(LANGS)
    separators = {locales.load(lang)["meta"]["block_sep"] for lang in LANGS}
    assert separators == {merged.block_sep}


@pytest.mark.parametrize("lang", LANGS)
def test_option_descriptions_are_distinct(lang: str) -> None:
    """输入：每种语言的选项表。输出：无。预期行为：同一张表里没有两个一样的描述。"""
    loc = locales.load(lang)
    for table in loc["option"].values():
        assert len(set(table.values())) == len(table)


@pytest.mark.parametrize("lang", LANGS)
def test_noul_criteria_are_distinct(lang: str) -> None:
    """输入：每种语言的 noul 判据。输出：无。预期行为：true / false 两行文案不一样。"""
    criteria = locales.load(lang)["fam"]["compact"]["criteria"]
    assert set(criteria) == {"true", "false"}
    assert criteria["true"] != criteria["false"]
