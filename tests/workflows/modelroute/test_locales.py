"""模型路由的文案：两种语言的键集合必须完全一致，且覆盖所有取值。"""

from __future__ import annotations

from workflows.modelroute import features, locales, questions
from workflows.modelroute.judges import TraceDraft


def keys_of(node: object, path: str = "") -> set[str]:
    """输入：TOML 节点。输出：该节点下全部叶子路径（形如 `fam.route.criteria.none`）。"""
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
    noul 族恰好 true/false；score 族 4 档（本 workflow 的价格档就是 0~3）。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        for family in questions.FAMILIES:
            spec = loc["fam"][family]
            assert len(spec["q"]) >= 3, f"{lang}: {family} 的问法少于 3 条"
            if family in questions.SCORE_FAMILIES:
                assert len(spec["levels"]) == 4, f"{lang}: {family} 的档位不是 4 档"
            elif family in questions.NOUL_FAMILIES:
                assert set(spec["criteria"]) == {"true", "false"}
            else:
                wanted = set(features.CHOICE_OPTIONS[family])
                missing = wanted - set(spec["criteria"])
                assert not missing, f"{lang}: {family} 少了选项 {missing}"


def test_trace_options_match_features() -> None:
    """输入：无。输出：无。

    预期行为：反解草稿的取值域与 `features.TASK_KINDS` 一致——两处各写一份迟早会漂移，
    漂了以后「反解答对」与「标签对了」就不是一回事。
    """
    literal = TraceDraft.model_fields["kind"].annotation
    assert set(literal.__args__) == set(features.TASK_KINDS)


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


def test_rules_forbid_naming_a_model() -> None:
    """输入：无。输出：无。

    预期行为：每种语言的规则里都有一条明说「请求里不许出现模型名」——渲染期的
    `render.check_draft` 是兜底，提示词本身也得说清楚，否则拒稿率会白高。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        joined = " ".join(str(rule) for rule in loc["prompt"]["rules"])
        assert "atlas" in joined
        assert any(
            marker in joined
            for marker in ("不许出现任何模型名", "never name any model")
        ), f"{lang}: 规则里没有「不许写模型名」那一条"


def test_meta_placeholders_render() -> None:
    """输入：无。输出：无。

    预期行为：段落头模板与括号模板都能填出来，且表格 / 门槛列名非空——渲染器直接
    `.format()` 它们，占位符写错会在第一次渲染时才炸。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        meta = loc["meta"]
        assert locales.header(loc, "X") != ""
        assert str(meta["paren"]).format(text="T").strip() == "T" or "T" in str(
            meta["paren"]
        ).format(text="T")
        assert str(meta["catalog_columns"]).count("|") == 5
        assert str(meta["needs_columns"]).count("|") == 4
        assert str(meta["row_prefix"]).strip()
        assert str(meta["none_mark"]).strip()


