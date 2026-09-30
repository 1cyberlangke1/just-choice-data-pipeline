"""testimpact 的语言文件测试：两张表、题族、提示词三处都必须与代码同构。"""

from __future__ import annotations

from itertools import combinations

import pytest

from workflows.testimpact import features, locales, questions, render
from workflows.testimpact.features import ImpactCase

LANGS = ("zh", "en")


def test_both_languages_exist() -> None:
    """输入：无。输出：无。预期行为：`zh` 与 `en` 都在，且与语言文件目录一致。"""
    assert locales.available() == ["en", "zh"]


@pytest.mark.parametrize("lang", LANGS)
def test_marks_are_non_empty_and_pairwise_non_prefix(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：四个抬头两两不互为前缀——解析靠 `startswith`，互为前缀时会把别的块抢过来
    （「【改动】」与「【改动说明】」这种）。
    """
    marks = locales.load(lang)["state"]
    names = ("coverage_mark", "ladder_mark", "note_mark", "diff_mark")
    values = [str(marks[name]) for name in names]
    assert all(value.strip() for value in values)
    for left, right in combinations(values, 2):
        assert not left.startswith(right)
        assert not right.startswith(left)


@pytest.mark.parametrize("lang", LANGS)
def test_suite_table_lines_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：套件键与 `features.SUITES` 一一对应、名字两两不同。"""
    table = locales.load(lang)["suite"]
    assert set(table) == set(features.SUITES)
    for suite in features.SUITES:
        assert str(table[suite]["name"]).strip()
        assert str(table[suite]["desc"]).strip()
        assert render.suite_name(locales.load(lang), suite) == table[suite]["name"]
    names = [str(table[suite]["name"]) for suite in features.SUITES]
    assert len(set(names)) == len(names)


@pytest.mark.parametrize("lang", LANGS)
def test_flavor_table_lines_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：改动性质的键与 `features.ALL_FLAVORS` 一一对应。"""
    table = locales.load(lang)["flavor"]
    assert set(table) == set(features.ALL_FLAVORS)
    for flavor in features.ALL_FLAVORS:
        assert render.flavor_copy(locales.load(lang), flavor) == table[flavor]["desc"]


@pytest.mark.parametrize("lang", LANGS)
def test_threshold_and_route_tables_line_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：档位表的阈值文案与跑法动作的键都与 `features.LEVELS` 一一对应，且阈值文案两两
    不同（解析靠它反查档位）。
    """
    loc = locales.load(lang)
    thresholds = loc["state"]["threshold"]
    actions = loc["fam"]["route"]["opt"]
    assert set(thresholds) == set(features.LEVELS)
    assert set(actions) == set(features.LEVELS)
    values = [str(thresholds[level]) for level in features.LEVELS]
    assert len(set(values)) == len(values)
    for level in features.LEVELS:
        assert render.ladder_line(loc, level).startswith(values[features.LEVELS.index(level)])
        assert questions.route_text(loc, level) == actions[level]


@pytest.mark.parametrize("lang", LANGS)
def test_every_family_has_questions_and_the_right_shape(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：四族都有问法列表；`flag` 的问法带 `{name}`、`count` / `which` / `route` 的选项表
    与 `features` 对得上。
    """
    loc = locales.load(lang)
    fam = loc["fam"]
    assert set(fam) >= set(questions.FAMILIES)
    for family in questions.FAMILIES:
        assert isinstance(fam[family]["q"], list) and fam[family]["q"]
    for text in fam["flag"]["q"]:
        assert "{name}" in str(text)
    assert set(fam["count"]["opt"]) == {f"c{index}" for index in range(len(features.SUITES) + 1)}


@pytest.mark.parametrize("lang", LANGS)
def test_prompts_and_questions_leave_no_placeholder(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：按渲染期那套填槽跑一遍之后，题面与提示词里不许剩 `{`（漏填会静默留一个花括号
    给模型看）。
    """
    loc = locales.load(lang)
    case = ImpactCase(touch="P1+P2+P3", flavor="fix")
    plan = features.plan_for(case, 0)
    prompt = render.build_prompt(loc, case)
    assert "{" not in prompt and "}" not in prompt
    for item in questions.build(plan, loc, 0):
        assert "{" not in str(item.question["instructions"])
        for text in item.question["criteria"].values():
            assert "{" not in text


@pytest.mark.parametrize("lang", LANGS)
def test_prompt_pins_the_language(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：提示词里有一句 `{language}` 约束——通篇目标语言**不足以**让模型不换语言（范式
    里量过：rerank 的 zh 侧把查询写成了英文）。
    """
    loc = locales.load(lang)
    assert str(loc["language_name"]).strip()
    prompt = render.build_prompt(loc, ImpactCase(touch="P1", flavor="fix"))
    assert str(loc["language_name"]) in prompt


def test_unknown_language_reports_available() -> None:
    """输入：不存在的语言。输出：无。预期行为：报错里列出可用语言，不静默回退。"""
    with pytest.raises(FileNotFoundError, match="en, zh"):
        locales.load("fr")
