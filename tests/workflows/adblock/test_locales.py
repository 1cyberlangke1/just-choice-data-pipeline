"""语言文件测试：结构不变量（抬头、判据表、处置档位、页面类型、占位符）逐条钉住。"""

from __future__ import annotations

import pytest

from workflows.adblock import features, judges, locales, questions, render


def test_both_languages_exist() -> None:
    """输入：无。输出：无。预期行为：zh / en 都在（少一个就是漏分发）。"""
    assert locales.available() == ["en", "zh"]


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_marks_are_non_empty_and_pairwise_non_prefix(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：三个抬头两两不互为前缀——`【判定判据】` 与 `【判定判据附注】` 那种关系会让解析先
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

    预期行为：判据表的键与 `TAG_IDS` 一一对应，每条三个字段都非空、判据名与表行两两不同，
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
def test_every_family_has_questions_and_the_right_shape(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：四族的问法池非空；逐判据那一族有 true / false 判据；count 与 combo 的选项表齐全。
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
def test_prompts_and_questions_leave_no_placeholder(lang: str) -> None:
    """输入：一种语言 × 全空间 48 组 case。输出：无。

    预期行为：提示词、反解提示词、题面与选项渲染完之后**一个花括号都不剩**——漏填会在 `format`
    上抛错，而多写的花括号会原样漏给模型，让它以为那是要填的槽。
    """
    loc = locales.load(lang)
    for case in features.build_cases(512):
        prompt = render.build_prompt(loc, case)
        assert "{" not in prompt and "}" not in prompt, case.describe()
        for item in questions.build(features.plan_for(case, 0), loc, 0):
            instructions = str(item.question["instructions"])
            assert "{" not in instructions and "}" not in instructions
    trace = judges.build_trace_prompt("[state]", loc)
    assert "{" not in trace and "}" not in trace


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


def test_pick_wraps_around() -> None:
    """输入：越界的问法编号。输出：无。预期行为：回绕（不抛异常）。"""
    assert locales.pick(["a", "b"], 3) == "b"

