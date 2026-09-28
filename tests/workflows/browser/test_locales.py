"""浏览器 / GUI 操作的文案测试：选项与取值一一对应、提示词只覆盖那一维、禁用语可用。"""

from __future__ import annotations

import pytest

from workflows.browser import features, locales, questions, render
from workflows.browser.features import TagSet

from tests.workflows.browser.conftest import request_for

TAGS = TagSet(intent="fill", state="ready", sensitive="yes", decoy="same_role", screen="form")


def test_every_choice_and_noul_family_matches_the_feature_values() -> None:
    """输入：两语言的题族选项。输出：无。

    预期行为：每个 choice / noul 题族的选项名必须与特征层那一维的取值完全一致——
    对不上就会出现「选项里没有的答案」或「答不出选项里的答案」。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        assert list(loc["fam"]["intent"]["criteria"]) == list(features.INTENTS)
        assert list(loc["fam"]["action"]["criteria"]) == list(features.ACTIONS)
        for family in questions.NOUL_FAMILIES:
            assert list(loc["fam"][family]["criteria"]) == ["true", "false"]
        assert len(loc["fam"]["risk"]["levels"]) == features.RISK_LEVELS


def test_prompt_only_covers_the_sentence_dimension() -> None:
    """输入：两语言 `dims` 表。输出：无。

    预期行为：`dims` 里只有由那句话承载的那一维——多一维就会让提示词把元素表 / 门控表 /
    意图表管的事也写进那句话，两边一冲突这道题就有歧义了（范式 §9.13）。
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
        locales.feature_line(loc, "intent", "bogus")
    with pytest.raises(KeyError):
        locales.feature_line(loc, "bogus", "fill")


def test_target_line_carries_the_element_name() -> None:
    """输入：元素名、两语言文案。输出：无。

    预期行为：那一行必须原样带上元素名，而且不留占位符——留着 `{label}` 会把占位符
    原样发给模型，那样它只能自己编一个元素名。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        line = locales.target_line(loc, "收件地址")
        assert "收件地址" in line
        assert "{" not in line


def test_role_and_state_words_cover_every_token() -> None:
    """输入：两语言的词表。输出：无。预期行为：每个角色 / 状态词元都有对应的说法。"""
    for lang in locales.available():
        loc = locales.load(lang)
        assert set(loc["role"]) == set(features.ROLES)
        assert set(loc["state"]) == set(features.STATES)
        for role in features.ROLES:
            assert locales.role_word(loc, role).strip()
        for state in features.STATES:
            assert locales.state_word(loc, state).strip()


def test_gates_table_embeds_the_code_word_list() -> None:
    """输入：两语言门控表。输出：无。

    预期行为：敏感那一行的说明里必须逐个出现 `features.SENSITIVE_WORDS` 的每个词——
    词表由代码填进 `{words}`，不靠人抄一遍（抄一遍就会漂移）。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        rows = render.gate_rows(loc)
        assert [token for token, _ in rows] == list(render.GATE_TOKENS)
        sensitive_row = rows[0][1]
        for word in features.SENSITIVE_WORDS:
            assert word in sensitive_row, word
        assert "{" not in sensitive_row


def test_every_prompt_rule_formats_and_has_no_stray_placeholder() -> None:
    """输入：两语言规则。输出：无。预期行为：每条规则都能把 `{language}` 填掉、不留别的槽。"""
    for lang in locales.available():
        loc = locales.load(lang)
        for rule in loc["prompt"]["rules"]:
            filled = str(rule).format(language=loc["language_name"])
            assert "{" not in filled


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


def test_banned_phrases_do_not_fire_on_the_fixture_requests() -> None:
    """输入：夹具里那几句请求。输出：无。

    预期行为：一条都不许撞禁用语——撞了就说明禁用语写得过宽，真跑时会把本来能用的
    数据在渲染期白拒掉。
    """
    cases = features.build_cases(1024)
    for lang in locales.available():
        loc = locales.load(lang)
        for case in cases:
            request = request_for(case.as_dict(), lang)
            assert render.banned_echo(request, loc) == [], request


def test_every_family_has_three_phrasings() -> None:
    """输入：两语言文案。输出：无。预期行为：每题族三种问法，`pick` 在三个编号上都取得到。"""
    for lang in locales.available():
        loc = locales.load(lang)
        for family in questions.FAMILIES:
            assert len(loc["fam"][family]["q"]) >= 3
            for phrasing in (0, 1, 2):
                assert locales.pick(loc["fam"][family]["q"], phrasing)


def test_target_family_has_the_row_and_none_texts() -> None:
    """输入：两语言 `target` 题族。输出：无。预期行为：行号模板与「不操作」的说法都在。"""
    for lang in locales.available():
        loc = locales.load(lang)
        assert "{" in str(loc["fam"]["target"]["row_option"])
        assert str(loc["fam"]["target"]["none"]).strip()


def test_meta_placeholders_render() -> None:
    """输入：两语言 meta。输出：无。预期行为：段落头与括号模板都能格式化。"""
    for lang in locales.available():
        loc = locales.load(lang)
        assert "X" in locales.header(loc, "X")
        assert loc["meta"]["paren"].format(text="T").strip()
        for key in render.SECTION_KEYS:
            assert str(loc["meta"][key]).strip()
        for key in ("elements_columns", "gates_columns", "criteria_columns"):
            assert str(loc["meta"][key]).strip()


def test_unknown_locale_lists_the_available_ones() -> None:
    """输入：不存在的语言。输出：无。预期行为：抛 FileNotFoundError 并列出可用语言。"""
    with pytest.raises(FileNotFoundError) as info:
        locales.load("nope")
    assert "zh" in str(info.value)