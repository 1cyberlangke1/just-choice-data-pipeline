"""语言文件：两张表必须形状一致、判据表必须互斥、模板必须满足读法的前提。"""

from __future__ import annotations

import pytest

from workflows.autoformat import locales, parse, spec as sp
from workflows.autoformat.workflow import AutoformatWorkflow

LANGS = tuple(locales.available())


def _keys(value: object, prefix: str = "") -> set[str]:
    """输入：语言文件里的一个值。输出：它下面全部键的路径集合（递归）。"""
    if not isinstance(value, dict):
        return set()
    out: set[str] = set()
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        out.add(path)
        out |= _keys(item, path)
    return out


def test_both_locales_have_the_same_shape() -> None:
    """输入：两个语言文件。输出：无。预期行为：键集合完全一致（少一格就会渲染时炸）。"""
    assert len(LANGS) >= 2
    base = _keys(locales.load(LANGS[0]))
    for lang in LANGS[1:]:
        assert _keys(locales.load(lang)) == base, lang


@pytest.mark.parametrize("lang", LANGS)
def test_callout_keywords_are_grouped_and_disjoint(lang: str) -> None:
    """输入：每种语言。输出：无。

    预期行为：同一个关键词只属于一个种类（否则反推种类有歧义）；
    三种种类都在、且都不是空表。
    """
    loc = locales.load(lang)
    index = parse.callout_index(loc)
    assert set(index.values()) == set(loc["callout_kind"])
    for kind, words in loc["callout_word"].items():
        assert words, kind
        assert all(word.strip() for word in words), kind


@pytest.mark.parametrize("lang", LANGS)
def test_no_text_outside_a_callout_contains_a_callout_keyword(lang: str) -> None:
    """输入：每种语言的全部模板（提示框模板除外）。输出：无。

    预期行为：普通段落 / 标题 / 列表项 / 引用 / 命令行里**一个关键词都不许出现**——
    判别器就是靠关键词认提示框的，漏进来一个就会把普通段落判成提示框。
    """
    loc = locales.load(lang)
    words = list(parse.callout_index(loc))
    pools: list[str] = []
    for region in ("head", "sent", "item", "code", "quote"):
        for value in loc[region].values():
            pools.extend([value] if isinstance(value, str) else value)
    for region in ("intro",):
        for value in loc[region].values():
            pools.extend(value)
    for template in pools:
        for word in words:
            assert word not in template, f"{lang}: {word!r} 出现在 {template!r}"


@pytest.mark.parametrize("lang", LANGS)
def test_only_quotes_carry_the_attribution_mark(lang: str) -> None:
    """输入：每种语言的全部模板。输出：无。

    预期行为：署名前缀只出现在引用模板里（判别器靠它认引用）。
    """
    loc = locales.load(lang)
    mark = parse.attribution_of(loc)
    assert mark
    for key, template in loc["quote"].items():
        assert mark in template, key
    pools: list[str] = []
    for region in ("head", "sent", "item", "code", "intro", "callout"):
        for value in loc[region].values():
            pools.extend(value)
    for template in pools:
        assert mark not in template, template


@pytest.mark.parametrize("lang", LANGS)
def test_headings_and_items_are_short_and_unpunctuated(lang: str) -> None:
    """输入：每种语言的标题池与列表项池。输出：无。

    预期行为：标题与列表项都**短且没有句末标点**——这是「短行」判据的前提；
    超长会被读成散文，整块切分就错位（实测踩过）。
    """
    loc = locales.load(lang)
    for pool in ("head", "item"):
        for key, values in loc[pool].items():
            for template in values:
                filled = _fill_all(template, loc)
                assert len(filled) <= parse.SHORT_LIMIT, (lang, pool, key, filled)
                assert not parse.ends_terminal(filled), (lang, pool, key, filled)


@pytest.mark.parametrize("lang", LANGS)
def test_prose_templates_end_with_punctuation(lang: str) -> None:
    """输入：每种语言的段落 / 引用 / 提示框模板。输出：无。

    预期行为：这些模板的**句子**以句末标点收尾——否则它们会被当成「短行」读成标题。
    引用块的署名写在句子之后，所以先切掉署名；句子外面还可能包着引号 / 括号，
    所以再把收尾的引号括号剥掉，剩下的尾巴才是要判的那一段。
    """
    loc = locales.load(lang)
    mark = parse.attribution_of(loc)
    pools: list[str] = []
    for region in ("sent", "intro", "quote", "callout"):
        for value in loc[region].values():
            pools.extend([value] if isinstance(value, str) else value)
    for template in pools:
        filled = _fill_all(template, loc)
        sentence = filled.split(mark)[0] if mark in filled else filled
        assert parse.ends_terminal(sentence.rstrip().rstrip(_CLOSERS)), (lang, filled)


@pytest.mark.parametrize("lang", LANGS)
def test_step_items_use_sequencing_words_and_plain_ones_do_not(lang: str) -> None:
    """输入：每种语言的两种列表项池。输出：无。

    预期行为：步骤项**每条**都带时序词，普通项**一条都不带**——判别器靠时序词判有序性。
    """
    loc = locales.load(lang)
    seq = parse.seq_words(loc)
    assert seq
    for template in loc["item"]["step"]:
        filled = _fill_all(template, loc)
        assert any(word in filled for word in seq), filled
    for template in loc["item"]["plain"]:
        filled = _fill_all(template, loc)
        assert not any(word in filled for word in seq), filled


@pytest.mark.parametrize("lang", LANGS)
def test_code_templates_start_with_a_command_word(lang: str) -> None:
    """输入：每种语言的命令行池。输出：无。

    预期行为：每条命令行的首词都在命令表里（判别器靠首词认命令行）。
    """
    loc = locales.load(lang)
    commands = parse.command_tokens(loc)
    for family, pool in loc["code"].items():
        for template in pool:
            assert template.split(" ", 1)[0] in commands, (family, template)


def test_merged_table_is_language_agnostic() -> None:
    """输入：合并表。输出：无。

    预期行为：两种语言的判据表合并后不冲突，且合并表能认出两种语言渲染出来的文档。
    """
    merged = locales.merged()
    assert parse.command_tokens(merged) == parse.command_tokens(locales.load("zh"))
    for lang in LANGS:
        spec = sp.sample_spec(sp.DocCase("build", 64, "mixed", "warning", "build"), "t", 0, 0)
        from workflows.autoformat import render

        parsed = parse.parse_state(render.render_state(spec, locales.load(lang)), merged)
        assert parsed.kinds() == ("heading", "paragraph", "paragraph", "list_item", "list_item",
                                  "list_item", "paragraph", "code", "quote", "callout", "paragraph")
    assert AutoformatWorkflow().available_locales() == list(LANGS)


def test_missing_locale_lists_the_available_ones() -> None:
    """输入：不存在的语言。输出：无。预期行为：报错并列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="可用"):
        locales.load("fr")


#: 包在句子外面的收尾符号（引号 / 括号）：判句末标点时要先剥掉。
_CLOSERS = "\"')]}」』）】"


def _fill_all(template: str, loc: dict) -> str:
    """输入：模板、语言字典。输出：把全部槽位填成最长候选后的文本。

    预期行为：用**最长**的候选值填，测出来的长度就是最坏情况——这样「短行」判据才稳。
    """
    out = template
    for name, pool in loc["slot"].items():
        out = out.replace("{" + name + "}", max(pool, key=len))
    return out
