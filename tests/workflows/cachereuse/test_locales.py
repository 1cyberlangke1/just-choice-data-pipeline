"""语言文件的不变量：写法表、标签表、题族形状、与反解模块共用的结构字母表。"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from typing import Final

import pytest

from workflows.cachereuse import locales, parse, questions
from workflows.cachereuse import spec as sp

LANGS: Final[tuple[str, ...]] = ("zh", "en")

#: 三张写法表：动作、元素、来源。三张表都要「键 -> 至少一种写法」且写法互不撞车。
WORD_TABLES: Final[tuple[str, ...]] = ("action", "target", "flavor")


def test_reverse_rejects_duplicate_spellings() -> None:
    """输入：两个键共用一种写法。输出：无。预期行为：直接报错，不静默取其一。"""
    with pytest.raises(ValueError, match="同时属于"):
        locales.reverse({"a": ["x"], "b": ["x"]})
    assert locales.reverse({"a": ["x"], "b": ["y"]}) == {"x": "a", "y": "b"}
    assert locales.reverse({"a": "x"}) == {"x": "a"}


def test_word_tables_have_no_collisions() -> None:
    """输入：一份语言文件、一张写法表。输出：无。

    预期行为：`reverse` 不报错（不同键之间不撞词）——反解会把撞车的两种写法认成一个键，
    那是构造 bug，不是「数据难」。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for lang in LANGS:
        for table in WORD_TABLES:
            try:
                entries = locales.load(lang)[table]
                reversed_table = locales.reverse(entries)
                expected = sum(len(value) if isinstance(value, list) else 1 for value in entries.values())
                assert len(reversed_table) == expected, f"{lang}/{table}"
            except AssertionError as exc:
                bad.append(f"{(lang, table)!r}: {exc}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("table", ("action", "target"))
def test_equivalent_steps_need_two_distinct_spellings(lang: str, table: str) -> None:
    """输入：一份语言文件、一张写法表。输出：无。

    预期行为：每个键至少两种写法、且**互不相同**——同义写法是「写法不同但归一化后相同」
    那一层唯一的载体，缺了它 `synonym` 那一维渲染出来就是逐字相同。
    """
    for key, spellings in locales.load(lang)[table].items():
        assert isinstance(spellings, list), f"{lang}/{table}/{key}"
        assert len(spellings) >= 2, f"{lang}/{table}/{key}"
        assert len(set(spellings)) == len(spellings), f"{lang}/{table}/{key}"


@pytest.mark.parametrize("lang", LANGS)
def test_flavor_values_are_distinct_single_words(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：来源三档各一个词、互不相同——来源只改抬头那一行，两种写法会让反解认错人。
    """
    flavor = locales.load(lang)["flavor"]
    assert set(flavor) == set(sp.FLAVORS)
    values = list(flavor.values())
    assert all(isinstance(value, str) and value for value in values)
    assert len(set(values)) == len(values)


def test_language_files_share_the_same_keys() -> None:
    """输入：两份语言文件。输出：无。预期行为：三张写法表的键集合逐个对齐。"""
    reference = locales.load("zh")
    other = locales.load("en")
    for table in WORD_TABLES:
        assert set(other[table]) == set(reference[table]), table
    assert set(other["meta"]) == set(reference["meta"])
    assert set(other["rules"]) == set(reference["rules"])
    assert len(other["rules"]["lines"]) == len(reference["rules"]["lines"])


def test_words_covers_every_spelling_of_every_language() -> None:
    """输入：无。输出：合并反查表。

    预期行为：三张合并表把**每种语言的每种写法**都收进来——产物按 `locales[seq % len]`
    轮转渲染，审计拿到的语言未必是产物那一种，只认一种语言的写法会把另一种整片读成表外。
    """
    merged = locales.words()
    for table in WORD_TABLES:
        expected = {
            word
            for lang in LANGS
            for value in locales.load(lang)[table].values()
            for word in (value if isinstance(value, list) else [value])
        }
        assert set(merged[table]) == expected, table


def test_merged_words_point_each_spelling_at_one_key() -> None:
    """输入：无。输出：无。

    预期行为：同一个写法在两种语言里只能指向同一个键——指向不同键说明有一处写法写重了，
    反解会认错人（`words()` 自己会在那种情况下报错，这里把结论钉成断言）。
    """
    merged = locales.words()
    for table in WORD_TABLES:
        for key, spellings in locales.load("zh")[table].items():
            for word in spellings if isinstance(spellings, list) else [spellings]:
                assert merged[table][word] == key, f"{table}/{key}/{word}"


def test_structural_alphabet_matches_the_parser() -> None:
    """输入：两份语言文件的 `meta`。输出：无。

    预期行为：渲染器写出来的引号对与分隔符，必须落在反解模块认的那两张表里——引号与冒号是
    **结构**（反解按它们切分），两边对不上会把好产物整片判成坏数据。
    """
    for lang in LANGS:
        meta = locales.load(lang)["meta"]
        pair = (meta["quote_open"].lstrip(), meta["quote_close"])
        assert pair in parse.QUOTES, f"{lang}: {pair}"
        assert meta["sep"].strip() in parse.COLONS, f"{lang}: {meta['sep']!r}"


@pytest.mark.parametrize("lang", LANGS)
def test_labels_are_not_mutual_prefixes(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：抬头 / 来源 / 两个区块抬头两两**不互为前缀**——反解靠最长前缀认标签，互为前缀
    时短的会把长的吃掉（「缓存复用请求号」被认成「缓存复用请求」）。
    """
    loc = locales.load(lang)
    tables = parse.tables_of(loc)
    labels = [*tables.request, *tables.sources, *tables.sections]
    for left in labels:
        for right in labels:
            if left != right:
                assert not right.startswith(left), f"{lang}: {left!r} / {right!r}"
    assert not set(tables.sections) & set(tables.sources)
    assert tables.sections[f"[{loc['meta']['cached']}]"] == "cached"
    assert tables.sections[f"[{loc['meta']['current']}]"] == "current"


def test_languages_do_not_share_labels() -> None:
    """输入：两份语言文件。输出：无。预期行为：标签原文互不重叠（合并表才不炸）。"""
    zh = parse.tables_of(locales.load("zh"))
    en = parse.tables_of(locales.load("en"))
    assert not set(zh.request) & set(en.request)
    assert not set(zh.sources) & set(en.sources)
    assert not set(zh.sections) & set(en.sections)
    merged = parse.merged_tables()
    assert set(merged.request) == set(zh.request) | set(en.request)
    assert set(merged.sources) == set(zh.sources) | set(en.sources)
    assert set(merged.sections) == set(zh.sections) | set(en.sections)


def test_tables_of_rejects_identical_section_headers() -> None:
    """输入：两个区块抬头写成一样的语言字典。输出：无。预期行为：报错，不静默认错区块。"""
    loc: dict[str, JsonValue] = dict(locales.load("zh"))
    meta = dict(loc["meta"])
    meta["current"] = meta["cached"]
    loc["meta"] = meta
    with pytest.raises(ValueError, match="两个区块抬头的文案相同"):
        parse.tables_of(loc)


@pytest.mark.parametrize("lang", LANGS)
def test_word_table_keys_match_the_code_vocabulary(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：`[action]` / `[target]` 的键集合与 `spec.ACTIONS` / `spec.TARGETS` 逐字一致——
    代码词表少了谁都不会报错（渲染时那个取值永远挑不到，数据只是悄悄少一档），所以钉死在这里。
    """
    loc = locales.load(lang)
    assert set(loc["action"]) == set(sp.ACTIONS)
    assert set(loc["target"]) == set(sp.TARGETS)


@pytest.mark.parametrize("lang", LANGS)
def test_families_have_phrasings_and_valid_criteria(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：每个题族至少一条问法；`diverge_at` 四个位置档、`diff_kind` 两个原因、
    `reuse_half` 真 / 假两行、`reuse_band` 四档——选项键与代码里那一套完全一致。
    """
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        assert loc["fam"][family]["q"], family
    assert set(loc["fam"]["diverge_at"]["criteria"]) == {"first", "second", "middle", "last"}
    assert set(loc["fam"]["diff_kind"]["criteria"]) == set(sp.DIFF_KINDS)
    assert set(loc["fam"]["reuse_half"]["criteria"]) == {"true", "false"}
    assert len(loc["fam"]["reuse_band"]["levels"]) == len(sp.BANDS)


@pytest.mark.parametrize("lang", LANGS)
def test_diff_kind_phrasings_carry_the_step_placeholder(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：`diff_kind` 的每条问法都带 `{step}`——题面必须点名「第几步」，不然三种问法
    说的是同一件事，考生不知道在问哪一步。
    """
    for text in locales.load(lang)["fam"]["diff_kind"]["q"]:
        assert "{step}" in text, text


@pytest.mark.parametrize("lang", LANGS)
def test_rules_state_the_four_bands_and_the_prefix_rule(lang: str) -> None:
    """输入：一份语言文件。输出：无。

    预期行为：判据原文里出现**四档各自的档名**与四个位置档的档名——判据必须让只读 state
    与题面的读者自己算得出来（范式 §9.8），档名是「读者要对照的那张表」的入口。
    """
    loc = locales.load(lang)
    rules = "\n".join(loc["rules"]["lines"])
    heads = [
        level.split("——")[0].split(" - ")[0] for level in loc["fam"]["reuse_band"]["levels"]
    ]
    assert len(set(heads)) == len(sp.BANDS)
    for head in heads:
        assert head in rules, f"{lang}: 档名 {head!r} 没写进判据"
    lowered = rules.lower()
    for band in ("first", "second", "middle", "last"):
        head = loc["fam"]["diverge_at"]["criteria"][band].split("——")[0].split(" - ")[0]
        assert head.lower() in lowered, f"{lang}: 位置档 {head!r} 没写进判据"
