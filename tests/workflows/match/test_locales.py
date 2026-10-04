"""语言文件：键齐全、写法表自洽、标签不撞车、判据覆盖五栏。"""

from __future__ import annotations

import pytest

from workflows.match import judges, locales, parse, questions
from workflows.match import spec as sp

LANGS = ("zh", "en")


@pytest.mark.parametrize("lang", LANGS)
def test_every_required_key_is_present(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：栏名、额外栏、四张写法表、判据、八个题族全齐——少一个键就当场报出来
    （渲染是直接下标取值的，缺键会在生成期才炸）。
    """
    loc = locales.load(lang)
    assert set(loc["field"]) == set(sp.FIELDS)
    assert set(loc["name"]) == set(sp.PRODUCTS)
    assert set(loc["brand"]) == set(sp.PRODUCTS)
    assert set(loc["size"]) == {str(value) for value in sp.SIZES}
    assert set(loc["abv"]) == {str(value) for value in sp.ABVS}
    assert set(loc["pack"]) == set(sp.PACKS)
    assert set(loc["fam"]) == set(questions.FAMILIES)
    assert {"request", "left", "right", "sep"} <= set(loc["meta"])
    assert {"origin_label", "origin_left", "origin_right"} <= set(loc["extra"])
    assert loc["rules"]["lines"]


@pytest.mark.parametrize("lang", LANGS)
def test_spelling_tables_have_three_distinct_spellings(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：每个取值恰好三种**互不相同**的写法——`same` 与 `variant` 的区别全靠原文
    是否逐字相同，写法重复会让某一档永远出不来。
    """
    loc = locales.load(lang)
    for table in ("name", "brand", "size", "abv", "pack"):
        for key, spellings in loc[table].items():
            assert len(spellings) == len(sp.SPELLINGS) == 3, (table, key)
            assert len(set(spellings)) == 3, (table, key)


@pytest.mark.parametrize("lang", LANGS)
def test_spellings_normalize_to_the_same_value(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：同一取值的三种写法，按**判据里那四条规则**归一化后完全相等——写法表与
    `judges.normalize` 是同一套规则的两种载体，这一条把它们钉在一起。
    """
    loc = locales.load(lang)
    for field in sp.FIELDS:
        for key, spellings in loc[field].items():
            values = {judges.normalize(field, text, loc) for text in spellings}
            assert len(values) == 1, (field, key, spellings)


@pytest.mark.parametrize("lang", LANGS)
def test_different_values_normalize_differently(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：不同取值归一化后**不相等**——否则「真的不一致」那一栏会被读成一致，
    `diff` 就成了假答案。
    """
    loc = locales.load(lang)
    for field in sp.FIELDS:
        values = [judges.normalize(field, spellings[0], loc) for spellings in loc[field].values()]
        assert len(set(values)) == len(values), field


@pytest.mark.parametrize("lang", LANGS)
def test_labels_do_not_collide(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：栏标签两两之间没有前缀关系（否则反解会把一格认成另一格），侧标题也不与
    任何栏标签相同。
    """
    loc = locales.load(lang)
    tables = parse.tables_of(loc)
    labels = list(tables.labels)
    for left in labels:
        for right in labels:
            if left != right:
                assert not right.startswith(left), (left, right)
    assert not set(labels) & set(tables.sides)
    assert tables.sides[loc["meta"]["left"]] == "left"
    assert tables.sides[loc["meta"]["right"]] == "right"
    assert tables.request == (loc["meta"]["request"],)


def test_languages_do_not_share_labels() -> None:
    """输入：两份语言文件。输出：无。预期行为：标签原文互不重叠（合并表才不炸）。"""
    zh = parse.tables_of(locales.load("zh"))
    en = parse.tables_of(locales.load("en"))
    assert not set(zh.labels) & set(en.labels)
    assert not set(zh.sides) & set(en.sides)
    merged = parse.merged_tables()
    assert len(merged.labels) == len(zh.labels) + len(en.labels)
    assert len(merged.sides) == len(zh.sides) + len(en.sides)
    assert set(merged.request) == {"匹配请求", "Match request"}


@pytest.mark.parametrize("lang", LANGS)
def test_families_have_phrasings_and_valid_criteria(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：每个题族至少一条问法；noul 的判据恰好 true / false 两行且带 `{field}`
    占位符；`verdict` 三个选项、`variant_count` 四个选项、`band` 三档。
    """
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        assert loc["fam"][family]["q"], family
    for family in questions.NOUL_FIELDS:
        criteria = loc["fam"][family]["criteria"]
        assert set(criteria) == {"true", "false"}, family
        for text in criteria.values():
            assert "{field}" in text
    assert set(loc["fam"]["verdict"]["criteria"]) == {"same", "different", "review"}
    assert set(loc["fam"]["variant_count"]["criteria"]) == {
        f"v{count}" for count in questions.VARIANT_CHOICES
    }
    assert len(loc["fam"]["band"]["levels"]) == 3


@pytest.mark.parametrize("lang", LANGS)
def test_rules_text_covers_every_field_and_the_extra(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：判据原文里出现全部五栏的栏名、以及额外栏的栏名——「只比五栏、产地不算」
    这句话必须真的写在题面里，读者才可能算对。
    """
    loc = locales.load(lang)
    rules = "\n".join(loc["rules"]["lines"])
    for field in sp.FIELDS:
        assert loc["field"][field] in rules, field
    assert loc["extra"]["origin_label"] in rules


def test_extra_values_differ_between_the_two_records() -> None:
    """输入：两份语言文件。输出：无。

    预期行为：额外栏两侧的值必须不同——它是个「看着像可比、其实不在比较范围」的干扰项，
    两侧写一样就退化成普通栏了。
    """
    for lang in LANGS:
        extra = locales.load(lang)["extra"]
        assert extra["origin_left"] != extra["origin_right"]


@pytest.mark.parametrize("lang", LANGS)
def test_language_files_have_the_same_shape(lang: str) -> None:
    """输入：一份语言文件。输出：无。预期行为：与 zh 逐表逐键对齐（少一个键就报出来）。"""
    reference = locales.load("zh")
    other = locales.load(lang)
    for table in ("field", "name", "brand", "size", "abv", "pack"):
        assert set(other[table]) == set(reference[table]), table
    for family in questions.FAMILIES:
        assert set(other["fam"][family]) == set(reference["fam"][family]), family
    assert set(other["rules"]) == set(reference["rules"])
    assert len(other["rules"]["lines"]) == len(reference["rules"]["lines"])
    assert set(other["extra"]) == set(reference["extra"])
    assert set(other["meta"]) == set(reference["meta"])


