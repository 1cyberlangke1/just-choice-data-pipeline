"""完成声明核验的语言文件：键集、取值覆盖、题族文案。"""

from __future__ import annotations


from jc.jsontypes import JsonValue
from workflows.verify import features, locales, questions, render
from workflows.verify.judges import TraceDraft


def keys_of(value: JsonValue, prefix: str = "") -> set[str]:
    """输入：一段语言字典。输出：全部叶子键的路径集合（表与数组算叶子）。"""
    out: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, dict):
                out |= keys_of(item, path)
            else:
                out.add(path)
    return out


def test_every_prompt_dimension_value_has_text() -> None:
    """输入：无。输出：无。

    预期行为：**进提示词的那两维**每个取值都有 brief 与 hint，且非空。
    只查这两维：另外三维由代码渲染进 state，不该出现在提示词里（见 `render.DIM_ORDER`）。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        for dim in render.DIM_ORDER:
            for value in features.DIMS[dim]:
                spec = loc["feature"][dim][value]
                assert spec["brief"].strip(), f"{lang}: {dim}.{value} 的 brief 是空的"
                assert spec["hint"].strip(), f"{lang}: {dim}.{value} 的 hint 是空的"


def test_every_family_has_questions_and_criteria() -> None:
    """输入：无。输出：无。

    预期行为：每个题族至少 3 种问法；choice 族的 criteria 覆盖该族全部选项；
    noul 族恰好 true/false；score 族 4 档（三条门槛就是 0~3）。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        for family in questions.FAMILIES:
            spec = loc["fam"][family]
            assert len(spec["q"]) >= 3, f"{lang}: {family} 的问法少于 3 条"
            if family in questions.SCORE_FAMILIES:
                assert len(spec["levels"]) == features.GATE_COUNT + 1, f"{lang}: {family} 档位不对"
            elif family in questions.NOUL_FAMILIES:
                assert set(spec["criteria"]) == {"true", "false"}
            else:
                wanted = set(features.CHOICE_OPTIONS[family])
                missing = wanted - set(spec["criteria"])
                assert not missing, f"{lang}: {family} 少了选项 {missing}"


def test_trace_options_match_features() -> None:
    """输入：无。输出：无。

    预期行为：反解草稿的取值域与 `features.CLAIM_KINDS` / `features.TONES` 一致——
    两处各写一份迟早会漂移，漂了以后「反解答对」与「标签对了」就不是一回事。
    """
    kind = TraceDraft.model_fields["kind"].annotation
    tone = TraceDraft.model_fields["tone"].annotation
    assert set(kind.__args__) == set(features.CLAIM_KINDS)
    assert set(tone.__args__) == set(features.TONES)


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
            assert rule.format(language=loc["language_name"]).strip()
        assert sorted(rules) == sorted(list(rules[1:]) + list(rules[:1]))


def test_rules_forbid_writing_evidence() -> None:
    """输入：无。输出：无。

    预期行为：每种语言的规则里都有一条明说「声明里不要写证据」——渲染期的
    `render.check_draft` 与卫生判别器是兜底，提示词本身也得说清楚，否则拒稿率会白高。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        joined = " ".join(str(rule) for rule in loc["prompt"]["rules"])
        label = str(loc["meta"]["facts_label"]).lower()
        assert label in joined.lower(), f"{lang}: 规则里没提「自报事实」那一行"
        assert any(marker in joined for marker in ("不要", "never", "do not")), (
            f"{lang}: 规则里没有「声明不许写证据」那一条"
        )


def test_banned_phrases_are_present_and_usable() -> None:
    """输入：无。输出：无。

    预期行为：每种语言都点名了几条证据话术，非空且小写化后仍非空——`evidence_echo`
    两边都小写化再比，空串会匹配一切（那就等于拒掉所有声明）。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        phrases = loc["banned_claim_phrases"]
        assert len(phrases) >= 3
        for phrase in phrases:
            assert str(phrase).strip()
            assert str(phrase).lower().strip()


def test_meta_placeholders_render() -> None:
    """输入：无。输出：无。

    预期行为：段落头模板与括号模板都能填出来；两行的列名各有两格分隔符（三列）；
    yes / no 两个记号非空——渲染器直接 `.format()` 它们，占位符写错会在第一次渲染时才炸。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        meta = loc["meta"]
        assert locales.header(loc, "X") != ""
        assert "T" in str(meta["paren"]).format(text="T")
        assert str(meta["facts_columns"]).count("|") == 2
        assert str(meta["gates_columns"]).count("|") == 2
        assert str(meta["row_prefix"]).strip()
        assert str(meta["yes"]).strip() and str(meta["no"]).strip()
        assert str(meta["yes"]) != str(meta["no"])
