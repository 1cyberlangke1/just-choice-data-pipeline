"""题族：出哪些题、选项与答案、条件题族（`next`）、题面里的判据。"""

from __future__ import annotations

import random
from typing import Any

import pytest

from workflows.modelroute import locales, questions
from workflows.modelroute.spec import (
    COMPLEXITIES,
    LATENCIES,
    STEPS,
    TASKS,
    TIERS,
    RouteCase,
    sample_spec,
)

ZH = locales.load("zh")
EN = locales.load("en")


def build(task: str = "codefix", phrasing: int = 0, lang: str = "zh", **overrides: Any) -> list[Any]:
    """输入：任务、问法、语言与维度覆盖。输出：题目列表（走生产路径的出题）。"""
    fields: dict[str, Any] = {
        "task": task,
        "complexity": "mid",
        "latency": "normal",
        "cost": "normal",
        "retry": "first",
        "data": "public",
        "phrasing": phrasing,
    }
    fields.update(overrides)
    loc = locales.load(lang)
    spec = sample_spec(RouteCase(**fields), "train", 0, phrasing)
    return questions.build(spec, loc, phrasing, random.Random(0))


def test_families_for_drops_next_when_there_is_no_chain() -> None:
    """输入：四个任务。输出：无。

    预期行为：只有 `reason` 不出 `next`（首选档已经是 max，上面没有档）——这条与
    `build` 的出题规则一一对应，审计拿它算「应有题族」。
    """
    assert questions.families_for("classify") == (
        "cache",
        "degrade",
        "effort",
        "is_first",
        "next",
        "tier",
    )
    assert "next" not in questions.families_for("reason")
    assert set(questions.FAMILIES) == {
        "cache",
        "degrade",
        "effort",
        "is_first",
        "next",
        "tier",
    }
    assert set(questions.CHOICE_FAMILIES) | set(questions.NOUL_FAMILIES) == set(questions.FAMILIES)


def test_build_emits_the_families_of_the_task() -> None:
    """输入：四个任务。输出：无。预期行为：出的题族与 `families_for` 完全一致。"""
    for task in TASKS:
        families = {item.family for item in build(task=task)}
        assert families == set(questions.families_for(task))


def test_build_order_is_stable() -> None:
    """输入：一个 case 造两次。输出：无。

    预期行为：题目完全由规格与语言文件决定（不掷骰子），顺序也稳定：先选档与升级链，
    再推理深度，最后三个判断。
    """
    first = build(task="extract")
    assert [item.family for item in first] == ["tier", "next", "effort", "cache", "degrade", "is_first"]
    assert [item.answer for item in first] == [item.answer for item in build(task="extract")]


def test_choice_answers_are_always_inside_the_options() -> None:
    """输入：全部任务 × 全部维度取值。输出：无。

    预期行为：每道 choice 的金答案都是它自己选项表里的键——答案落在选项外是构造 bug，
    会在 `_choice` 里当场炸掉，这条把「四个档位表都覆盖到了」也一起钉住。
    """
    for task in TASKS:
        for complexity in COMPLEXITIES:
            for latency in LATENCIES:
                for item in build(task=task, complexity=complexity, latency=latency):
                    if item.question["type"] != "choice":
                        continue
                    assert item.answer in item.question["criteria"]


def test_option_keys_cover_the_whole_space() -> None:
    """输入：无。输出：无。

    预期行为：`tier` 的选项就是四个档、`next` 的选项是没有 nano 的三个档、
    `effort` 的选项就是三个深度档——选项少了会让某档答案无处可落。
    """
    assert set(ZH["option"]["tier"]) == set(TIERS)
    assert set(ZH["option"]["next"]) == set(TIERS) - {"nano"}
    assert set(ZH["option"]["effort"]) == set(STEPS)


def test_noul_questions_never_shuffle_and_carry_criteria() -> None:
    """输入：三个 noul 族。输出：无。

    预期行为：`shuffle=False`（noul 的判据是有序的 true/false），且两行判据文案都在。
    """
    built = [item for item in build() if item.question["type"] == "noul"]
    assert {item.family for item in built} == set(questions.NOUL_FAMILIES)
    for item in built:
        assert item.shuffle is False
        assert set(item.question["criteria"]) == {"true", "false"}
        assert all(item.question["criteria"].values())


def test_choice_questions_shuffle_by_default() -> None:
    """输入：三个 choice 族。输出：无。预期行为：允许打乱（选项名没有自然顺序）。"""
    built = [item for item in build() if item.question["type"] == "choice"]
    assert {item.family for item in built} == set(questions.CHOICE_FAMILIES)
    assert all(item.shuffle is True for item in built)


def test_option_descriptions_are_mutually_distinct() -> None:
    """输入：四张选项表。输出：无。预期行为：同一张表里没有两个一样的描述。"""
    for table in ZH["option"].values():
        assert len(set(table.values())) == len(table)


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_every_question_carries_the_policy(lang: str) -> None:
    """输入：两种语言。输出：无。

    预期行为：判据整段写在每道题的题面里（范式 §9.8）——只读 state 与题面的读者必须能
    自己算出来，所以任何一道题少了判据都是坏数据。
    """
    loc = locales.load(lang)
    for task in TASKS:
        for item in build(task=task, lang=lang):
            assert loc["policy"]["text"] in item.question["instructions"]


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_no_question_carries_an_unfilled_placeholder(lang: str) -> None:
    """输入：全部任务 × 两套句面。输出：无。

    预期行为：题面、判据与选项里都不留 `{xxx}`——题面少填一个槽会变成一道读不懂的题，
    而答案照样发出去。
    """
    for task in TASKS:
        for phrasing in (0, 1):
            for item in build(task=task, phrasing=phrasing, lang=lang):
                texts = [str(item.question["instructions"])]
                texts.extend(str(value) for value in item.question.get("criteria", {}).values())
                for text in texts:
                    assert "{" not in text and "}" not in text, text


def test_choice_rejects_duplicate_descriptions() -> None:
    """输入：两个描述一样的选项。输出：无。预期行为：报错（考生没法分辨）。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("tier", "题面", {"a": "一样", "b": "一样"}, "a")


def test_choice_rejects_an_answer_outside_the_options() -> None:
    """输入：答案不在选项里。输出：无。预期行为：报错。"""
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("tier", "题面", {"a": "甲", "b": "乙"}, "c")


def test_phrasings_change_the_wording_only() -> None:
    """输入：同一个 case 的两套句面。输出：无。预期行为：答案一样、题面不一样。"""
    first = {item.family: item.answer for item in build(phrasing=0)}
    second = {item.family: item.answer for item in build(phrasing=1)}
    assert first == second
    text = {
        item.family: item.question["instructions"] for item in build(phrasing=0)
    }["tier"]
    other = {item.family: item.question["instructions"] for item in build(phrasing=1)}["tier"]
    assert text != other
