"""语言文件测试：结构不变量（抬头、判据表、处置档位、占位符）逐条钉住。"""

from __future__ import annotations

import pytest

from workflows.promptinject import features, locales, questions, render


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_marks_are_non_empty_and_pairwise_non_prefix(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：三个抬头两两不互为前缀——`【注入判据】` 与 `【注入判据附注】` 那种关系会让解析先
    命中短的那个，整段 state 当场错位。
    """
    loc = locales.load(lang)
    marks = [str(loc["state"][key]) for key in ("tag_mark", "rule_mark", "content_mark")]
    assert all(marks)
    assert len(set(marks)) == len(marks)
    for left in marks:
        for right in marks:
            if left != right:
                assert not right.startswith(left), f"{left!r} 是 {right!r} 的前缀"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_action_and_threshold_tables_line_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：两张表都覆盖四个档位，动作文案两两不同。"""
    loc = locales.load(lang)
    options = loc["fam"]["action"]["opt"]
    assert isinstance(options, dict)
    assert sorted(options) == sorted(features.HANDLING_LEVELS)
    assert len({str(value) for value in options.values()}) == len(options)
    thresholds = loc["state"]["threshold"]
    assert isinstance(thresholds, dict)
    assert sorted(thresholds) == sorted(features.HANDLING_LEVELS)
    assert len({str(value) for value in thresholds.values()}) == len(thresholds)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_source_table_lines_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：素材来源表的键与 `SOURCES` 一一对应，每种形态的说明都非空、两两不同——
    三种来源写成同一句话，这一维的留出就白留了。
    """
    loc = locales.load(lang)
    table = loc["source"]
    assert isinstance(table, dict)
    assert sorted(table) == sorted(features.SOURCES)
    descs = [render.source_copy(loc, source) for source in features.SOURCES]
    assert all(desc.strip() for desc in descs)
    assert len(set(descs)) == len(descs)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_family_has_questions_and_the_right_shape(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：四族的问法池非空；逐判据那一族有 true / false 判据；count 与 combo 的选项表齐全。
    """
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        spec = loc["fam"][family]
        assert isinstance(spec["q"], list) and spec["q"], f"{family} 没有问法"
    assert str(loc["fam"]["tag"]["criteria_true"]).strip()
    assert str(loc["fam"]["tag"]["criteria_false"]).strip()
    assert sorted(loc["fam"]["count"]["opt"]) == [
        f"c{value}" for value in range(len(features.TAG_IDS) + 1)
    ]
    assert str(loc["fam"]["combo"]["none"]).strip()
    assert str(loc["fam"]["combo"]["joiner"])


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_pins_the_language(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：顶层有 `language_name`，提示词里有一句把语言钉死的约束——不写这一句，模型会照它
    自己的先验换语言（范式 §10.96 那条尺子的由来）。
    """
    loc = locales.load(lang)
    assert str(loc["language_name"]).strip()
    assert "{language}" in str(loc["prompt"]["lang_rule"])
    case = features.PromptInjectCase("T1", features.SOURCE_WEB)
    assert str(loc["language_name"]) in render.build_prompt(loc, case)
