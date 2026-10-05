"""改动审查的多语言文案测试：词表与代码表必须一一对得上。"""

from __future__ import annotations

import re

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
def test_writer_copy_carries_no_finished_example(lang: str) -> None:
    """写手提示词里不许留成品例子（范式 §9.25 / §9.64：给了例子，模型就会逐字照抄）。

    输入：一个语言。输出：无。
    预期行为：把写手提示词的文案（`intro` + 每条 `brief` 与 `hint` + `rules`——`bullet` 把
    `brief` 与 `hint` 都拼进了提示词，所以两边都要查）拼起来，

    - 不许出现带 diff 前缀的成品代码行（`+ ` / `- ` 后面跟非空白字符）；
    - 不许出现成品赋值（`名字 = "字母数字开头的值"`）。

    形状要留（§9.19：行首前缀逼不出来，所以「前缀后面跟一个空格」这类口径必须写清），成品不许
    留——真机上 `+ timeout = 60` 被抄了 2/41 条、`db_password = "hunter2"` 被抄了 5/41 条。
    """
    loc = locales.load(lang)
    entries = [entry for values in loc["feature"].values() for entry in values.values()]
    copy_lines = [entry[key] for entry in entries for key in ("brief", "hint")]
    copy = "\n".join(
        [
            str(loc["prompt"]["intro"]),
            *[str(line) for line in copy_lines],
            *[str(rule) for rule in loc["prompt"]["rules"]],
        ]
    )
    assert re.search(r"(?m)^\s*[+-] \S", copy) is None
    assert re.search(r'=\s*"[A-Za-z0-9]', copy) is None


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
