"""题族：齐全、答案与规格一致、选项合法、判据原文写进题面。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.rerank import locales, questions, render
from workflows.rerank import spec as sp

LANGS = ("zh", "en")


def build(spec: sp.RerankSpec, lang: str, phrasing: int = 0) -> list[Any]:
    """输入：规格、语言、问法编号。输出：题目列表。预期行为：纯函数（不掷骰子）。"""
    return questions.build(spec, locales.load(lang), phrasing, random.Random(0))


def test_families_are_complete_and_in_order(make_spec: Any) -> None:
    """输入：一份规格。输出：无。预期行为：七个题族各一题，顺序与 `FAMILIES` 一致。"""
    built = build(make_spec(), "zh")
    assert [item.family for item in built] == list(questions.FAMILIES)
    assert len(built) == 7


@pytest.mark.parametrize("lang", LANGS)
def test_answers_match_the_spec(lang: str, make_spec: Any) -> None:
    """输入：一份规格。输出：无。

    预期行为：choice 的答案指向规格算出来的那一条候选（组装前的答案存的是**选项名**）、
    noul 的布尔与规格一致、score 的下标就是相关度档位。
    """
    spec = make_spec(traps=("forwarded",), wording="titled")
    for item in build(spec, lang):
        if item.question["type"] == "choice":
            want = {
                "first": render.cand_id(spec.first()),
                "nth": render.cand_id(spec.nth()),
                "last": render.cand_id(spec.last()),
            }[item.family]
            assert item.answer == want.lower()
            assert item.answer in item.question["criteria"]
        elif item.question["type"] == "noul":
            want = {
                "ahead": spec.ahead(),
                "same_relevance": spec.same_relevance(),
                "on_topic": spec.on_topic(),
            }[item.family]
            assert item.answer is want
        else:
            assert item.family == "relevance"
            assert item.answer == spec.probe_relevance()


def test_choice_options_are_distinct_and_shufflable(make_spec: Any) -> None:
    """输入：一份规格。输出：无。

    预期行为：choice 三族允许打乱、四条选项描述两两不同；noul 与 score 不打乱。
    """
    for item in build(make_spec(), "zh"):
        if item.question["type"] == "choice":
            assert item.shuffle is True
            options = item.question["criteria"]
            assert len(options) == 4
            assert len(set(options.values())) == 4
        else:
            assert item.shuffle is False


def test_noul_criteria_are_exactly_true_and_false(make_spec: Any) -> None:
    """输入：一份规格。输出：无。预期行为：noul 的判据恰好 true / false 两行、都有正文。"""
    for item in build(make_spec(), "zh"):
        if item.question["type"] == "noul":
            assert set(item.question["criteria"]) == {"true", "false"}
            assert all(item.question["criteria"].values())


def test_score_criteria_are_the_five_levels(make_spec: Any) -> None:
    """输入：一份规格。输出：无。预期行为：档位描述就是语言文件里的五档，顺序不动。"""
    loc = locales.load("zh")
    item = next(item for item in build(make_spec(), "zh") if item.family == "relevance")
    assert item.question["criteria"] == list(loc["fam"]["relevance"]["levels"])


def test_rules_block_is_written_into_every_question(make_spec: Any) -> None:
    """输入：一份规格（缓存主题、常见故障方面）。输出：无。

    预期行为：判据原文（四条必要条件 + 四条排序规则）拼在**每一道**题面后面——
    只读 state 与题面的读者必须能自己算出答案。
    """
    loc = locales.load("zh")
    spec = make_spec(topic="cache", aspect="failures")
    block = questions.rules_block(spec, loc)
    assert loc["topic"]["cache"] in block
    assert loc["aspect"]["failures"] in block
    assert sp.DEADLINE in block
    assert loc["source"]["official"] in block
    for item in build(spec, "zh"):
        assert item.question["instructions"].endswith(block)


def test_named_families_name_their_candidates(make_spec: Any) -> None:
    """输入：一份规格。输出：无。

    预期行为：逐对 / 单条题族把编号写进题面——判别器靠从题面读编号来重算，
    所以「题面里必须写清判据」这条纪律的落地就在这里。
    """
    spec = make_spec()
    shape = spec.case.shape
    by_family = {item.family: item for item in build(spec, "zh")}
    left, right = shape.ahead
    assert render.cand_id(left) in by_family["ahead"].question["instructions"]
    assert render.cand_id(right) in by_family["ahead"].question["instructions"]
    left, right = shape.tie
    assert render.cand_id(left) in by_family["same_relevance"].question["instructions"]
    assert render.cand_id(right) in by_family["same_relevance"].question["instructions"]
    probe = render.cand_id(shape.probe)
    assert probe in by_family["on_topic"].question["instructions"]
    assert probe in by_family["relevance"].question["instructions"]
    assert str(shape.nth) in by_family["nth"].question["instructions"]


def test_phrasing_pool_rotates(make_spec: Any) -> None:
    """输入：同一份规格、三个问法编号。输出：无。预期行为：三档问法互不相同。"""
    spec = make_spec()
    texts = {index: build(spec, "zh", index)[0].question["instructions"] for index in (0, 1, 2)}
    assert len(set(texts.values())) == 3


def test_duplicate_option_descriptions_are_rejected() -> None:
    """输入：两个描述相同的选项。输出：无。预期行为：构造期报错（考生没法分辨）。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("first", "题面", {"c1": "一样", "c2": "一样"}, "c1")


def test_unknown_answer_is_rejected() -> None:
    """输入：答案不在选项里。输出：无。预期行为：构造期报错。"""
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("first", "题面", {"c1": "甲", "c2": "乙"}, "c3")


def test_build_is_pure(make_spec: Any) -> None:
    """输入：同一份规格构建两次。输出：无。预期行为：题目与答案完全一致。"""
    spec = make_spec()
    first = build(spec, "zh")
    second = build(spec, "zh")
    assert [item.question for item in first] == [item.question for item in second]
    assert [item.answer for item in first] == [item.answer for item in second]


def test_options_carry_the_candidate_ids(make_spec: Any) -> None:
    """输入：一份规格。输出：无。

    预期行为：四条选项描述里都带编号——选项会被打乱，没有编号就读不出指的是哪一条。
    """
    loc = locales.load("zh")
    options = questions.options_of(make_spec(), loc)
    assert set(options) == {"c1", "c2", "c3", "c4"}
    for cid, text in options.items():
        assert cid.upper() in text
