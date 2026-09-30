"""语言文件测试：结构不变量（抬头、判据表、处置档位、通道、占位符）逐条钉住。"""

from __future__ import annotations

import pytest

from workflows.logtriage import features, locales, render


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_marks_are_non_empty_and_pairwise_non_prefix(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：三个抬头两两不互为前缀——`【噪声判据】` 与 `【噪声判据附注】` 那种关系会让解析先
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
def test_tag_table_lines_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：判据表的键与 `TAG_IDS` 一一对应，每条四个字段都非空、判据名与表行两两不同，
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

    预期行为：每条判据的「不许出现」文案**只**命中它自己那一条——它是提示词里给模型看的反面
    清单，同时也写进 state 的判据表，同时踩中两条就会把模型往错的方向带（范式 §9.45 起，
    提示词里唯一的具体文字就是判据原文与这份反面清单，所以它更要钉住）。
    """
    loc = locales.load(lang)
    for tag in features.TAG_IDS:
        avoid = str(render.tag_copy(loc, tag)["avoid"])
        assert features.hit_set(avoid) == (tag,), f"{lang}.{tag}: {avoid!r}"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_route_and_threshold_tables_line_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：两张表都覆盖四个档位，动作文案两两不同。"""
    loc = locales.load(lang)
    options = loc["fam"]["route"]["opt"]
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

    预期行为：通道表的键与 `SOURCES` 一一对应，每种形态的说明都非空、两两不同——
    三种通道写成同一句话，这一维的留出就白留了。
    """
    loc = locales.load(lang)
    table = loc["source"]
    assert isinstance(table, dict)
    assert sorted(table) == sorted(features.SOURCES)
    descs = [render.source_copy(loc, source) for source in features.SOURCES]
    assert all(desc.strip() for desc in descs)
    assert len(set(descs)) == len(descs)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_pins_the_language(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：顶层有 `language_name`，提示词里有一句把语言钉死的约束——不写这一句，模型会照它
    自己的先验换语言（范式 §10.96 那条尺子的由来）。
    """
    loc = locales.load(lang)
    assert str(loc["language_name"]).strip()
    assert "{language}" in str(loc["prompt"]["lang_rule"])
    case = features.LogTriageCase("N1", features.SOURCE_APP)
    assert str(loc["language_name"]) in render.build_prompt(loc, case)
