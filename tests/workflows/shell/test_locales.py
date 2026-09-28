"""命令审批的文案：两种语言的键集合必须完全一致，且覆盖所有取值。"""

from __future__ import annotations

import pytest

from workflows.shell import features, locales, questions
from workflows.shell.judges import TraceDraft


def keys_of(node: object, path: str = "") -> set[str]:
    """输入：TOML 节点。输出：该节点下全部叶子路径（形如 `fam.verdict.criteria.allow`）。"""
    if not isinstance(node, dict):
        return {path}
    out: set[str] = set()
    for key, value in node.items():
        out |= keys_of(value, f"{path}.{key}" if path else str(key))
    return out


def test_both_locales_have_identical_key_sets() -> None:
    """输入：无。输出：无。

    预期行为：zh 与 en 的叶子路径集合**完全一致**——少一个键就会让另一种语言在渲染时
    KeyError，而那时已经在烧配额了。
    """
    zh = keys_of(locales.load("zh"))
    en = keys_of(locales.load("en"))
    assert zh == en, f"只在 zh 里：{sorted(zh - en)}；只在 en 里：{sorted(en - zh)}"


def test_every_feature_value_has_text() -> None:
    """输入：无。输出：无。预期行为：每个维度的每个取值都有 brief 与 hint，且非空。"""
    for lang in locales.available():
        loc = locales.load(lang)
        for dim, values in features.DIMS.items():
            for value in values:
                spec = loc["feature"][dim][value]
                assert spec["brief"].strip(), f"{lang}: {dim}.{value} 的 brief 是空的"
                assert spec["hint"].strip(), f"{lang}: {dim}.{value} 的 hint 是空的"


def test_every_family_has_questions_and_criteria() -> None:
    """输入：无。输出：无。

    预期行为：每个题族至少 3 种问法；choice 族的 criteria 覆盖该族全部选项；
    noul 族恰好 true/false；score 族 5 档。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        for family in questions.FAMILIES:
            spec = loc["fam"][family]
            assert len(spec["q"]) >= 3, f"{lang}: {family} 的问法少于 3 条"
            if family in questions.SCORE_FAMILIES:
                assert len(spec["levels"]) == 5, f"{lang}: {family} 的档位不是 5 档"
            elif family in questions.NOUL_FAMILIES:
                assert set(spec["criteria"]) == {"true", "false"}
            else:
                wanted = set(features.CHOICE_OPTIONS[family])
                assert wanted <= set(spec["criteria"]), f"{lang}: {family} 少了选项 {wanted - set(spec['criteria'])}"


def test_trace_options_match_features() -> None:
    """输入：无。输出：无。

    预期行为：反解草稿的取值域与 `features.TARGETS` 一致——两处各写一份迟早会漂移，
    漂了以后「反解答对」与「标签对了」就不是一回事。
    """
    literal = TraceDraft.model_fields["target_owner"].annotation
    assert set(literal.__args__) == set(features.TARGETS)


def test_rules_are_formattable_and_rotatable() -> None:
    """输入：无。输出：无。

    预期行为：每条规则都能用 `{language}` 填出来（规则里有别的花括号会在渲染时炸），
    且按问法编号旋转不会丢规则。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        rules = loc["prompt"]["rules"]
        assert len(rules) >= 3
        for rule in rules:
            text = rule.format(language=loc["language_name"])
            assert text.strip()
        assert sorted(rules) == sorted(list(rules[1:]) + list(rules[:1]))


def test_unknown_locale_reports_available() -> None:
    """输入：不存在的语言。输出：无。预期行为：报错里列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="zh"):
        locales.load("xx")