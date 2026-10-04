"""语言文件的结构不变量：两个语言必须说同一套结构，占位符必须都被填过。"""

from __future__ import annotations

import pytest

from tests.workflows.segmentation.conftest import pages_of
from workflows.segmentation import features, locales, questions, render
from workflows.segmentation.features import SegmentationCase

LANGS = ("en", "zh")


@pytest.mark.parametrize("lang", LANGS)
def test_structural_tables_match_the_code(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：`[class.*]` 与 `features.CLASS_IDS` 一一对应（每条都有 name / rule / avoid）、
    `[source.*]` 与 `features.SOURCES` 一一对应（每条都有 desc / trace）、
    `[fam.count.opt]` 与 `1..MAX_PACKET_CLASSES` 对应、`[fam.route.opt]` 与 `ROUTE_LEVELS` 对应。
    """
    loc = locales.load(lang)
    assert set(loc["class"]) == set(features.CLASS_IDS)
    for cls in features.CLASS_IDS:
        entry = render.class_copy(loc, cls)
        assert str(entry["name"]).strip()
        assert str(entry["rule"]).strip()
        assert str(entry["avoid"]).strip()
    assert set(loc["source"]) == set(features.SOURCES)
    for source in features.SOURCES:
        entry = render.source_copy(loc, source)
        assert str(entry["desc"]).strip()
        assert str(entry["trace"]).strip()
    count_keys = {f"n{value}" for value in range(1, features.MAX_PACKET_CLASSES + 1)}
    assert set(loc["fam"]["count"]["opt"]) == count_keys
    assert set(loc["fam"]["route"]["opt"]) == set(features.ROUTE_LEVELS)
    low, high = render.page_bounds(loc)
    assert 0 < low < high


@pytest.mark.parametrize("lang", LANGS)
def test_section_marks_are_distinguishable(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：三个抬头标记两两不互为前缀，页序抬头与它们都不同形——否则解析会把一段内容
    算进两个区块。
    """
    marks = locales.load(lang)["state"]
    names = ["table_mark", "rule_mark", "order_mark"]
    values = [str(marks[name]) for name in names]
    assert len(set(values)) == len(values)
    for left in values:
        for right in values:
            if left != right:
                assert not left.startswith(right)
    marker = render.page_marker_re(locales.load(lang))
    for value in values:
        assert not marker.match(value)


@pytest.mark.parametrize("lang", LANGS)
def test_prompt_pins_the_language(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：顶层有 `language_name`，提示词里有一句把语言钉死的约束——不写这一句，模型会照它
    自己的先验换语言（`rerank` 的 zh 侧曾 3/3 写成英文）。
    """
    loc = locales.load(lang)
    assert str(loc["language_name"]).strip()
    assert "{language}" in str(loc["prompt"]["lang_rule"])
    case = SegmentationCase("form+receipt", "3", features.SOURCE_SCAN)
    plan = features.plan_for(case, 0)
    assert str(loc["language_name"]) in render.build_prompt(loc, plan, case.source)


@pytest.mark.parametrize("lang", LANGS)
def test_every_template_fills_without_leftover_slots(lang: str) -> None:
    """输入：一种语言 × 全空间 126 组 case。输出：无。

    预期行为：提示词与题面模板里的每个占位符都被填过，填完不留 `{...}`——漏填会在渲染期
    直接抛 `KeyError`，这条尺子把它提前到毫秒级。
    """
    loc = locales.load(lang)
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        prompt = render.build_prompt(loc, plan, case.source)
        assert "{" not in prompt, case.describe()
        state = render.to_state(
            render.PacketDraft(pages=pages_of(plan, case.source)), loc
        )
        assert "{" not in state, case.describe()
        for phrasing in (0, 1):
            for item in questions.build(plan, loc, phrasing):
                assert "{" not in item.question["instructions"], case.describe()


def test_both_languages_define_the_same_families() -> None:
    """输入：两份语言文件。输出：无。预期行为：题族与选项键完全一致（只差文案）。"""
    zh = locales.load("zh")
    en = locales.load("en")
    assert set(zh["fam"]) == set(en["fam"])
    for family in questions.FAMILIES:
        assert len(zh["fam"][family]["q"]) == len(en["fam"][family]["q"])
    assert set(zh["fam"]["count"]["opt"]) == set(en["fam"]["count"]["opt"])
    assert set(zh["fam"]["route"]["opt"]) == set(en["fam"]["route"]["opt"])
    assert set(zh["state"]["rules"]) == set(en["state"]["rules"])

