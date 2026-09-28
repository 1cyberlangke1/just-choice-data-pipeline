"""改动审查的多语言文案测试：词表与代码表必须一一对得上。"""

from __future__ import annotations

import pytest

from workflows.review import features, judges, locales, questions, render


def all_keys(node: object, path: str = "") -> set[str]:
    """输入：解析后的 TOML 节点、路径前缀。输出：全部叶子路径。

    预期行为：列表按「一个叶子」处理（不展开下标）——两个语言文件的**结构**要一致，
    值当然不一样。
    """
    if isinstance(node, dict):
        out: set[str] = set()
        for key, value in node.items():
            out |= all_keys(value, f"{path}.{key}")
        return out
    return {path}


def test_available_locales() -> None:
    """输入：无。输出：无。预期行为：就 zh / en 两个，按文件名排序。"""
    assert locales.available() == ["en", "zh"]


def test_missing_locale_reports_available() -> None:
    """输入：不存在的语言。输出：无。预期行为：报错并列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="zh"):
        locales.load("nope")


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_locale_covers_every_dimension_value(lang: str) -> None:
    """输入：一个语言。输出：无。

    预期行为：`dims` 与 `feature` 覆盖全部维度与取值——缺一条，`build_prompt` 会抛
    KeyError，而那条 KeyError 只有跑到渲染阶段才暴露。
    """
    loc = locales.load(lang)
    assert set(loc["dims"]) == set(features.DIMS)
    assert set(loc["feature"]) == set(features.DIMS)
    for dim, values in features.DIMS.items():
        assert set(loc["feature"][dim]) == set(values), dim
        for value in values:
            assert str(loc["feature"][dim][value]["brief"]).strip()
            assert str(loc["feature"][dim][value]["hint"]).strip()


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_locale_covers_every_rule_and_family(lang: str) -> None:
    """输入：一个语言。输出：无。

    预期行为：规则表与题族的文案齐全，且 choice 的选项名与代码里的取值表**完全一致**
    （多一个少一个都会让「答对了」与「标签对了」不是一回事）。
    """
    loc = locales.load(lang)
    assert set(loc["rules"]) == set(features.REVIEW_RULES)
    for rule in features.REVIEW_RULES:
        assert str(loc["rules"][rule]["condition"]).strip()
    assert set(loc["fam"]) == set(questions.FAMILIES)
    for family in questions.FAMILIES:
        spec = loc["fam"][family]
        assert [str(item).strip() for item in spec["q"]]
        if family in questions.CHOICE_FAMILIES:
            assert set(spec["criteria"]) == set(features.CHOICE_OPTIONS[family]), family
        elif family in questions.NOUL_FAMILIES:
            assert set(spec["criteria"]) == {"true", "false"}, family
        else:
            assert len(spec["levels"]) == features.SEVERITY_LEVELS, family


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_locale_meta_and_columns(lang: str) -> None:
    """输入：一个语言。输出：无。预期行为：段落名、行前缀与四张表的列名齐全。"""
    loc = locales.load(lang)
    meta = loc["meta"]
    for key in render.SECTION_KEYS:
        assert str(meta[key]).strip()
    for key in ("header", "row_prefix", "paren", "colon"):
        assert str(meta[key]).strip()
    for key in ("manifest_columns", "rules_columns", "severity_columns", "gate_columns"):
        assert str(meta[key]).strip()
    assert str(loc["verify"]["intro"]).strip()
    assert str(loc["verify"]["option_line"]).strip()


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_prompt_rules_format_without_leftover_placeholders(lang: str) -> None:
    """输入：一个语言。输出：无。

    预期行为：每条提示词规则都能按 `{language}` 格式化，且格式化后不再留占位符——
    留着占位符会把 `{language}` 原样发给模型。
    """
    loc = locales.load(lang)
    rules = [str(item) for item in loc["prompt"]["rules"]]
    assert rules
    for rule in rules:
        filled = rule.format(language=loc["language_name"])
        assert "{" not in filled and "}" not in filled, rule
    assert str(loc["prompt"]["intro"]).strip()
    assert "{dim}" in str(loc["prompt"]["bullet"])


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_banned_phrases_are_the_table_names(lang: str) -> None:
    """输入：一个语言。输出：无。

    预期行为：禁写清单里就是那几个表名——正文是一段代码，收进 `test` / `size` 这类
    标识符会大面积误伤（见 `render.banned_echo` 的口径）。
    """
    loc = locales.load(lang)
    banned = [str(item) for item in loc["banned_intent_phrases"]]
    assert banned
    # 比对时统一小写：英文的段落名首字母大写，而禁写清单按小写匹配（`banned_echo` 也是）
    titles = {str(loc["meta"][key]).lower() for key in render.SECTION_KEYS}
    assert {item.lower() for item in banned} == titles


def test_trace_draft_matches_the_dimension_tables() -> None:
    """输入：反解草稿的取值域。输出：无。

    预期行为：`TraceDraft` 的 Literal 与 `features.SECRETS` / `DEBUGS` 完全一致，
    而且反解问的就是 `TRACE_FAMILIES` 那两族。
    """
    secret_field = judges.TraceDraft.model_fields["secret"]
    debug_field = judges.TraceDraft.model_fields["debug"]
    assert set(secret_field.annotation.__args__) == set(features.SECRETS)
    assert set(debug_field.annotation.__args__) == set(features.DEBUGS)
    assert judges.TRACE_FAMILIES == ("secret", "debug")


def test_both_locales_have_the_same_structure() -> None:
    """输入：两个语言文件。输出：无。

    预期行为：叶子路径集合完全一致——加维度 / 加题族时漏翻一个语言，这条会当场报出来。
    """
    zh = all_keys(locales.load("zh"))
    en = all_keys(locales.load("en"))
    assert zh == en
