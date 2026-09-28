"""会话上下文压缩的文案测试：选项名与特征表对齐、处置表顺序、禁用语不误伤。"""

from __future__ import annotations

import pytest

from workflows.compact import features, locales, questions, render

from tests.workflows.compact.conftest import body_for


def test_criteria_keys_match_the_feature_tables() -> None:
    """输入：两语言文案。输出：无。

    预期行为：每个 choice / noul 题族的选项名必须与特征层那一维的取值完全一致——
    对不上就会出现「选项里没有的答案」或「答不出选项里的答案」。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        assert list(loc["fam"]["kind"]["criteria"]) == list(features.KINDS)
        assert list(loc["fam"]["fact"]["criteria"]) == list(features.FACTS)
        assert list(loc["fam"]["rule"]["criteria"]) == list(features.RULES)
        assert list(loc["fam"]["disposition"]["criteria"]) == list(features.DISPOSITIONS)
        assert list(loc["fam"]["pin_overrode"]["criteria"]) == ["true", "false"]
        assert len(loc["fam"]["priority"]["levels"]) == features.PRIORITY_LEVELS


def test_rule_table_order_matches_the_code() -> None:
    """输入：两语言处置表。输出：无。

    预期行为：locale 里 `rules` 的书写顺序必须与 `features.RULES` 一致——
    处置表是「从上往下第一条命中」，顺序变了语义就变了。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        assert list(loc["rules"]) == list(features.RULES)
        for name in features.RULES:
            assert str(loc["rules"][name]["verdict"]) == features.RAW_DISPOSITIONS[name]
            assert str(loc["rules"][name]["condition"]).strip()


def test_prompt_only_covers_the_text_dimensions() -> None:
    """输入：两语言 `dims` 表。输出：无。

    预期行为：`dims` 里只有由文本承载的两维——多一维就会让提示词把「本块位置」那一行
    管的事也写进正文，两边一冲突这道题就有歧义了（范式 §9.13）。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        assert list(loc["dims"]) == list(render.DIM_ORDER)


def test_every_prompt_dimension_value_has_text() -> None:
    """输入：两语言文案。输出：无。预期行为：提示词里每一维的每个取值都有 brief 与 hint。"""
    for lang in locales.available():
        loc = locales.load(lang)
        for dim in render.DIM_ORDER:
            for value in features.DIMS[dim]:
                assert str(loc["feature"][dim][value]["brief"]).strip()
                assert str(loc["feature"][dim][value]["hint"]).strip()


def test_feature_line_raises_on_a_missing_phrase() -> None:
    """输入：一个没写进 locale 的取值。输出：无。预期行为：抛 KeyError，不静默给空串。"""
    loc = locales.load("zh")
    with pytest.raises(KeyError):
        locales.feature_line(loc, "kind", "bogus")


def test_banned_phrases_are_present_and_usable() -> None:
    """输入：两语言禁用语清单。输出：无。

    预期行为：每条禁用语都能被 `render.banned_echo` 认出来，而且清单非空——
    空清单等于这道防线不存在。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        phrases = loc["banned_intent_phrases"]
        assert phrases, f"{lang} 没有禁用语"
        for phrase in phrases:
            assert str(phrase) in render.banned_echo(f"这是{phrase}的说法", loc)


def test_banned_phrases_do_not_fire_on_the_fixture_bodies() -> None:
    """输入：夹具里那几段正文。输出：无。

    预期行为：一条都不许撞禁用语——撞了就说明禁用语写得过宽，真跑时会把本来能用的
    数据在渲染期白拒掉。
    """
    cases = features.build_cases(1024)
    for lang in locales.available():
        loc = locales.load(lang)
        for tags in cases:
            body = body_for(tags.as_dict())
            assert render.banned_echo(body, loc) == [], body


def test_every_family_has_three_phrasings() -> None:
    """输入：两语言文案。输出：无。预期行为：每题族三种问法，`pick` 在三个编号上都取得到。"""
    for lang in locales.available():
        loc = locales.load(lang)
        for family in questions.FAMILIES:
            assert len(loc["fam"][family]["q"]) >= 3
            for phrasing in (0, 1, 2):
                assert locales.pick(loc["fam"][family]["q"], phrasing)


def test_meta_placeholders_render() -> None:
    """输入：两语言 meta。输出：无。预期行为：段落头与括号模板都能格式化。"""
    for lang in locales.available():
        loc = locales.load(lang)
        assert "X" in locales.header(loc, "X")
        assert loc["meta"]["paren"].format(text="T").strip()
        for key in render.SECTION_KEYS:
            assert str(loc["meta"][key]).strip()


def test_unknown_locale_lists_the_available_ones() -> None:
    """输入：不存在的语言。输出：无。预期行为：抛 FileNotFoundError 并列出可用语言。"""
    with pytest.raises(FileNotFoundError) as info:
        locales.load("nope")
    assert "zh" in str(info.value)
