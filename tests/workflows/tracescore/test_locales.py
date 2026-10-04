"""语言文件测试：结构不变量（抬头、信号表、处置档位、入口、占位符）逐条钉住。"""

from __future__ import annotations

import pytest

from workflows.tracescore import features, locales, questions, render


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_tag_table_lines_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：信号表的键与 `TAG_IDS` 一一对应，每条三个字段都非空、信号名与表行两两不同，
    而且表行里**编号排在分隔符前面**（解析就是按这个切回编号的）。
    """
    loc = locales.load(lang)
    table = loc["tag"]
    assert isinstance(table, dict)
    assert sorted(table) == sorted(features.TAG_IDS)
    for tag in features.TAG_IDS:
        entry = render.tag_copy(loc, tag)
        for key in ("name", "rule", "avoid"):
            assert str(entry[key]).strip(), f"{lang}.{tag}.{key}"
    names = [str(table[tag]["name"]) for tag in features.TAG_IDS]
    assert len(set(names)) == len(names)
    sep = str(loc["state"]["tag_sep"])
    lines = [render.tag_line(loc, tag) for tag in features.TAG_IDS]
    assert len(set(lines)) == len(lines)
    for tag, line in zip(features.TAG_IDS, lines, strict=True):
        assert line.partition(sep)[0] == tag


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_avoid_lists_trip_exactly_their_own_tag(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：每条信号的「不许出现」文案**只**命中它自己那一条——它是提示词里给模型看的反面
    清单，同时也写进 state 的信号表，同时踩中两条就会把模型往错的方向带（范式 §9.45 起，
    提示词里唯一的具体文字就是判据原文与这份反面清单，所以它更要钉住）。
    """
    loc = locales.load(lang)
    for tag in features.TAG_IDS:
        avoid = str(render.tag_copy(loc, tag)["avoid"])
        assert features.hit_set(avoid) == (tag,), f"{lang}.{tag}: {avoid!r}"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_route_and_threshold_tables_line_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：两张表都覆盖三个档位，动作文案两两不同。"""
    loc = locales.load(lang)
    options = loc["fam"]["route"]["opt"]
    assert isinstance(options, dict)
    assert sorted(options) == sorted(features.ROUTE_LEVELS)
    assert len({str(value) for value in options.values()}) == len(options)
    thresholds = loc["state"]["threshold"]
    assert isinstance(thresholds, dict)
    assert sorted(thresholds) == sorted(features.ROUTE_LEVELS)
    assert len({str(value) for value in thresholds.values()}) == len(thresholds)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_family_has_questions_and_the_right_shape(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：四族的问法池非空；逐信号那一族有 true / false 判据；count 与 combo 的选项表齐全。
    """
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        spec = loc["fam"][family]
        assert isinstance(spec["q"], list) and spec["q"], f"{family} 没有问法"
    assert str(loc["fam"]["flag"]["criteria_true"]).strip()
    assert str(loc["fam"]["flag"]["criteria_false"]).strip()
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
    case = features.TraceScoreCase("S1", features.SOURCE_CLI)
    assert str(loc["language_name"]) in render.build_prompt(loc, case)
