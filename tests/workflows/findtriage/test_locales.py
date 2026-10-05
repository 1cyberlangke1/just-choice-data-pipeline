"""改动引入性判定的文案测试：选项名与特征表对齐、口径顺序、行号写法。"""

from __future__ import annotations

import re

from workflows.findtriage import features, locales, questions, render


def test_criteria_keys_match_the_feature_tables() -> None:
    """输入：两语言文案。输出：无。

    预期行为：每个 choice / noul 题族的选项名，必须与特征层那一维的取值完全一致——
    对不上就会出现「选项里没有的答案」或「答不出选项里的答案」。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        assert list(loc["fam"]["introduced"]["criteria"]) == list(features.INTRODUCED)
        assert list(loc["fam"]["supported"]["criteria"]) == list(features.SUPPORTED)
        assert list(loc["fam"]["triage"]["criteria"]) == list(features.TRIAGES)
        assert list(loc["fam"]["owner"]["criteria"]) == list(features.OWNERS)
        assert list(loc["fam"]["actionable"]["criteria"]) == ["true", "false"]
        assert list(loc["fam"]["decoy"]["criteria"]) == ["true", "false"]
        assert str(loc["fam"]["cite"]["opt"]["none"]).strip()


def test_rule_table_order_matches_the_code() -> None:
    """输入：两语言口径。输出：无。

    预期行为：locale 里 `rules` 的书写顺序必须与 `features.RULES` 一致——口径是
    **从上往下**读的，顺序变了语义就变了。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        assert list(loc["rules"]) == list(features.RULES)
        for name in features.RULES:
            assert str(loc["rules"][name]["text"]).strip()


def test_owner_table_order_matches_the_code() -> None:
    """输入：两语言归属口径。输出：无。

    预期行为：locale 里 `owners` 的书写顺序必须与 `features.OWNER_RULES` 一致——归属口径是
    **按优先级从上往下**读的，顺序变了「第一条命中的说了算」就换人了。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        assert list(loc["owners"]) == list(features.OWNER_RULES)
        for name in features.OWNER_RULES:
            assert str(loc["owners"][name]["text"]).strip()


def test_prompt_fact_lines_carry_their_placeholders() -> None:
    """输入：两语言提示词模板。输出：无。

    预期行为：场景事实那几行必须带 `{n}` / `{symbol}` 占位符——不带就填不进具体事实，
    模型写出来的行号与符号只能靠猜。
    """
    for lang in locales.available():
        prompt = locales.load(lang)["prompt"]
        assert "{n}" in str(prompt["cite_line"])
        assert "{symbol}" in str(prompt["symbol_line"])
        assert "{symbol}" in str(prompt["fix_symbol_line"])
        assert "{" not in str(prompt["fix_vague_line"]).replace("{language}", "")


def test_meta_placeholders_render() -> None:
    """输入：两语言 meta。输出：无。预期行为：段落头、括号、序号与列名模板都能格式化。"""
    for lang in locales.available():
        loc = locales.load(lang)
        assert "X" in locales.header(loc, "X")
        assert loc["meta"]["paren"].format(text="T").strip()
        assert str(loc["meta"]["row_ref"]).format(n=2).strip()
        assert str(loc["meta"]["record_columns"]).count("|") == 3
        assert str(loc["meta"]["no_cite"]) == features.NO_CITE


def test_cite_pattern_reads_the_line_number() -> None:
    """输入：两语言的行号写法。输出：无。

    预期行为：每语言的 `cite_pattern` 都能从一句自然的发现里抓出行号，而且抓的是**第一处**。
    """
    samples = {
        "zh": ("第 42 行的取值没夹紧，第 77 行也一样。", 42),
        "en": ("line 42 is not clamped, line 77 as well.", 42),
    }
    for lang, (text, want) in samples.items():
        loc = locales.load(lang)
        pattern = re.compile(str(loc["meta"]["cite_pattern"]))
        match = pattern.search(text)
        assert match is not None, lang
        assert int(match.group(1)) == want, lang


def test_both_locales_declare_the_same_section_keys() -> None:
    """输入：两语言文案。输出：无。预期行为：段落键、维度名、题族名一一对应。"""
    keys = [locales.load(lang) for lang in locales.available()]
    assert len(keys) >= 2
    for name in render.SECTION_KEYS:
        assert len({str(loc["meta"][name]) for loc in keys}) == len(keys)
    for dim in render.DIM_ORDER:
        assert len({str(loc["dims"][dim]) for loc in keys}) == len(keys)
