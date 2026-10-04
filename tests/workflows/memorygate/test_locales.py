"""语言文件测试：结构不变量（抬头、判据表、处置档位、写法、占位符）逐条钉住。"""

from __future__ import annotations

import pytest

from workflows.memorygate import features, locales, render


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
        for key in ("name", "rule", "example", "avoid"):
            assert str(entry[key]).strip(), f"{lang}.{tag}.{key}"
    names = [str(table[tag]["name"]) for tag in features.TAG_IDS]
    assert len(set(names)) == len(names)
    sep = str(loc["state"]["tag_sep"])
    lines = [render.tag_line(loc, tag) for tag in features.TAG_IDS]
    assert len(set(lines)) == len(lines)
    for tag, line in zip(features.TAG_IDS, lines, strict=True):
        assert line.partition(sep)[0] == tag


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_examples_hit_exactly_their_own_tag(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：每条的示例片段只命中它自己那一条——示例同时踩中两条，提示词就会把模型往错的方向带。
    """
    loc = locales.load(lang)
    for tag in features.TAG_IDS:
        example = str(render.tag_copy(loc, tag)["example"])
        assert features.hit_set(example) == (tag,), f"{lang}.{tag}: {example!r}"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_form_table_lines_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：写法表的键与 `FORMS` 一一对应，每种形态的说明都非空、两两不同——
    三种写法写成同一句话，这一维的留出就白留了。
    """
    loc = locales.load(lang)
    table = loc["form"]
    assert isinstance(table, dict)
    assert sorted(table) == sorted(features.FORMS)
    descs = [render.form_copy(loc, form) for form in features.FORMS]
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
    case = features.MemoryGateCase("M1", features.FORM_SENTENCE)
    assert str(loc["language_name"]) in render.build_prompt(loc, case)
