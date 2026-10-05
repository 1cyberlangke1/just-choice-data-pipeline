"""语言文件测试：结构不变量（抬头、判据表、处置档位、页面类型、占位符）逐条钉住。"""

from __future__ import annotations

import pytest

from workflows.adblock import features, judges, locales, questions, render


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_avoid_lists_do_not_trip_other_tags(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：每条判据的「不许出现」文案**不许误触别的**判据——它是提示词里给模型看的反面清单，
    同时也写进 state 的判据表，踩到别的判据就会把模型往错的方向带（范式 §9.45 起，提示词里
    唯一的具体文字就是判据原文与这份反面清单，所以它更要钉住）。
    """
    loc = locales.load(lang)
    for tag in features.TAG_IDS:
        avoid = str(render.tag_copy(loc, tag)["avoid"])
        stray = set(features.hit_set(avoid)) - {tag}
        assert not stray, f"{lang}.{tag}: {avoid!r} 误触了 {sorted(stray)}"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_act_and_threshold_tables_line_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：两张表都覆盖四个档位，动作文案两两不同。"""
    loc = locales.load(lang)
    options = loc["fam"]["act"]["opt"]
    assert isinstance(options, dict)
    assert sorted(options) == sorted(features.ACTS)
    assert len({str(value) for value in options.values()}) == len(options)
    thresholds = loc["state"]["threshold"]
    assert isinstance(thresholds, dict)
    assert sorted(thresholds) == sorted(features.ACTS)
    assert len({str(value) for value in thresholds.values()}) == len(thresholds)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_surface_table_lines_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：页面类型表的键与 `SURFACES` 一一对应，每种形态的说明都非空、两两不同——
    三种页面写成同一句话，这一维的留出就白留了。
    """
    loc = locales.load(lang)
    table = loc["surface"]
    assert isinstance(table, dict)
    assert sorted(table) == sorted(features.SURFACES)
    descs = [render.surface_copy(loc, surface) for surface in features.SURFACES]
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
    case = features.AdBlockCase("A1", features.SURFACE_ARTICLE)
    assert str(loc["language_name"]) in render.build_prompt(loc, case)
