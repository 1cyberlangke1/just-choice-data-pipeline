"""工具调用门控题族的测试：题号、题型、选项与答案形态。"""

from __future__ import annotations

import pytest

from workflows.toolgate import features, locales, questions
from workflows.toolgate.features import TagSet

from jc.jsontypes import JsonValue

CASE = TagSet(tool="write_file", reach="host", payload="sensitive", policy="balanced", approval="absent")


def build(tags: TagSet = CASE, phrasing: int = 0) -> list[object]:
    """输入：特征、问法编号。输出：题目列表。预期行为：走公开入口，不碰内部状态。"""
    return questions.build(tags, locales.load("zh"), phrasing)


def test_families_are_in_a_fixed_order() -> None:
    """输入：无。输出：无。

    预期行为：题族顺序固定（先文本承载的两维，再策略表命中，最后两族直读）；
    顺序变了题目顺序就变了，产物不再逐字节可重建。
    """
    assert questions.FAMILIES == (
        "tool",
        "payload",
        "rule",
        "decision",
        "overridden",
        "severity",
    )
    assert questions.families_for(CASE) == questions.FAMILIES


def test_types_match_the_declared_groups() -> None:
    """输入：一道题的每个题族。输出：无。预期行为：题号就是题族名，题型与分组一致。"""
    got = {item.family: item.question["type"] for item in build()}
    assert list(got) == list(questions.FAMILIES)
    for family in questions.CHOICE_FAMILIES:
        assert got[family] == "choice"
    for family in questions.NOUL_FAMILIES:
        assert got[family] == "noul"
    for family in questions.SCORE_FAMILIES:
        assert got[family] == "score"


def test_choice_answers_are_option_names() -> None:
    """输入：一道题。输出：无。预期行为：choice 的答案是选项名（编号由组装器最后映射）。"""
    for item in build():
        if item.question["type"] == "choice":
            assert isinstance(item.answer, str)
            assert item.answer in item.question["criteria"]


def test_choice_options_are_exactly_the_allowed_ones() -> None:
    """输入：六个题族 × 全空间。输出：无。

    预期行为：每道 choice 题的选项恰好是该族允许的取值，一个不多一个不少——
    多一个会让模型去选一个不可能的值，少一个会让正确答案不在选项里。
    """
    for tags in features.build_cases(10_000):
        for item in questions.build(tags, locales.load("zh"), 0):
            if item.question["type"] != "choice":
                continue
            assert set(item.question["criteria"]) == set(
                features.CHOICE_OPTIONS[item.family]
            ), f"{item.family} 的选项与取值表不一致"


def test_score_answer_is_a_level_index() -> None:
    """输入：全空间。输出：无。预期行为：score 的答案是 0~3 的档位下标。"""
    for tags in features.build_cases(10_000):
        for item in build(tags):
            if item.question["type"] == "score":
                assert item.answer == features.severity_for(tags)
                assert isinstance(item.answer, int)
                assert 0 <= item.answer < features.SEVERITY_LEVELS
                assert len(item.question["criteria"]) == features.SEVERITY_LEVELS


def test_noul_answer_is_a_boolean() -> None:
    """输入：全空间。输出：无。预期行为：noul 的答案是布尔，且两档都出得到。"""
    seen: set[bool] = set()
    for tags in features.build_cases(10_000):
        for item in build(tags):
            if item.question["type"] == "noul":
                assert isinstance(item.answer, bool)
                seen.add(item.answer)
                assert set(item.question["criteria"]) == {"true", "false"}
    assert seen == {True, False}


def test_choice_families_are_shuffled() -> None:
    """输入：一道题。输出：无。

    预期行为：四个 choice 题族都允许打乱（防「答案总在第 n 个」），
    score / noul 不允许打乱（有自然顺序）。
    """
    for item in build():
        if item.question["type"] == "choice":
            assert item.shuffle is True
        else:
            assert item.shuffle is False
    assert questions.ORDERED == frozenset()


def test_phrasing_rotates_the_instructions() -> None:
    """输入：同一特征、三个问法编号。输出：无。

    预期行为：编号不同则指令文案不同，且都在 locale 的问法表里——轮换必须真的生效，
    否则整批数据的题面一模一样。
    """
    loc = locales.load("zh")
    texts: list[list[JsonValue]] = []
    for phrasing in (0, 1, 2):
        items = questions.build(CASE, loc, phrasing)
        texts.append([item.question["instructions"] for item in items])
    assert len({tuple(row) for row in texts}) == 3
    for index, family in enumerate(questions.FAMILIES):
        allowed = set(loc["fam"][family]["q"])
        for row in texts:
            assert row[index] in allowed


def test_every_case_answers_every_family() -> None:
    """输入：全空间。输出：无。预期行为：每条样本六个题族一条不缺、题号唯一。"""
    for tags in features.build_cases(10_000):
        items = build(tags)
        assert [item.family for item in items] == list(questions.FAMILIES)
        assert len({item.family for item in items}) == len(questions.FAMILIES)


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_both_languages_build(lang: str) -> None:
    """输入：两种语言。输出：无。预期行为：都能建出六个题族、答案一致。"""
    items = questions.build(CASE, locales.load(lang), 1)
    assert [item.family for item in items] == list(questions.FAMILIES)
    assert all(item.question["instructions"].strip() for item in items)
