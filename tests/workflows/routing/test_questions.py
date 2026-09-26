"""题族：数量、题型、金答案与题面形状。"""

from __future__ import annotations

import random

import pytest

from jc import assemble
from tests.workflows.routing.conftest import _make_spec
from workflows.routing import locales, questions, spec as sp


def test_family_set_is_the_declared_one() -> None:
    """输入：题族清单。输出：无。

    预期行为：八个题族、三族 choice、四族 noul、一族 score，且没有重名——
    重名会让组装器给同族加 `_2` 后缀，判别器再按族名找答案就找错了。
    """
    assert questions.FAMILIES == (
        "severity",
        "queue",
        "escalation",
        "critical_area",
        "needs_human",
        "sla_breach",
        "data_risk",
        "priority",
    )
    assert len(set(questions.FAMILIES)) == len(questions.FAMILIES) == 8
    assert len(questions.CHOICE_FAMILIES) == 3
    assert len(questions.NOUL_FAMILIES) == 4
    assert questions.SCORE_FAMILIES == ("priority",)


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_build_returns_one_question_per_family(lang: str) -> None:
    """输入：一条规格 + 一种语言。输出：无。

    预期行为：题族顺序与 `FAMILIES` 一致，题型与声明一致，金答案与规格一致。
    """
    loc = locales.load(lang)
    spec = _make_spec(tier="enterprise", compliance="security", issue="bug")
    built = questions.build(spec, loc, 0, random.Random(1))
    assert [item.family for item in built] == list(questions.FAMILIES)
    want = {
        "severity": spec.severity,
        "queue": spec.queue,
        "escalation": spec.escalation,
        "critical_area": spec.critical_area,
        "needs_human": spec.needs_human,
        "sla_breach": spec.sla_breach,
        "data_risk": spec.data_risk,
        "priority": spec.priority,
    }
    for item in built:
        assert item.answer == want[item.family]
        expected_type = (
            "choice"
            if item.family in questions.CHOICE_FAMILIES
            else "noul"
            if item.family in questions.NOUL_FAMILIES
            else "score"
        )
        assert item.question["type"] == expected_type


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_choice_questions_carry_their_option_sets(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：choice 的 `criteria` 是 {选项名: 描述}、描述两两不同、金答案是选项名；
    答案不在选项里、或描述撞车都必须在出题时就炸掉。
    """
    loc = locales.load(lang)
    spec = _make_spec()
    for item in questions.build(spec, loc, 0, random.Random(2)):
        if item.question["type"] != "choice":
            continue
        criteria = item.question["criteria"]
        assert isinstance(criteria, dict)
        assert len(set(criteria.values())) == len(criteria)
        assert item.answer in criteria


@pytest.mark.parametrize("lang", ("zh", "en"))
def test_noul_and_score_questions_carry_criteria(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：noul 带 true / false 两行判据（官方支持描述，带上它才知道真假意味着什么）；
    score 的 `criteria` 是五档描述数组，下标从 0 起。
    """
    loc = locales.load(lang)
    spec = _make_spec()
    for item in questions.build(spec, loc, 0, random.Random(3)):
        criteria = item.question["criteria"]
        if item.question["type"] == "noul":
            assert set(criteria) == {"true", "false"}
            assert criteria["true"] != criteria["false"]
        elif item.question["type"] == "score":
            assert isinstance(criteria, list)
            assert len(criteria) == len(sp.PRIORITY_LEVELS)
            assert 0 <= item.answer < len(criteria)


def test_phrasing_index_wraps_around() -> None:
    """输入：比问法池大的问法编号。输出：无。

    预期行为：回绕而不是报错（题面措辞不是判据），而且回绕后的题面与第 0 号一致。
    这里用 6 当编号：它是各题族问法条数（2 与 3）的公倍数，所以每一族都回绕到第 0 条。
    """
    loc = locales.load("zh")
    spec = _make_spec()
    first = questions.build(spec, loc, 0, random.Random(4))
    wrapped = questions.build(spec, loc, 6, random.Random(4))
    assert [item.question["instructions"] for item in first] == [
        item.question["instructions"] for item in wrapped
    ]


def test_assembled_sample_has_exactly_the_declared_families() -> None:
    """输入：一条组装好的样本。输出：无。

    预期行为：`questions` 与 `answers` 的键集合都等于八个题族名——
    这条抓的是「题族名与判别器键对不上」那类错（判别器会把每条样本都判成「未知题族」）。
    """
    loc = locales.load("zh")
    spec = _make_spec()
    state = "state"
    labeled = questions.build(spec, loc, 0, random.Random(5))
    sample = assemble.assemble_sample("routing-train-0000001", state, labeled, random.Random(6))
    assert set(sample["questions"]) == set(sample["answers"]) == set(questions.FAMILIES)


def test_choice_answers_survive_shuffling() -> None:
    """输入：同一个 case 用两个不同 RNG 组装。输出：无。

    预期行为：选项顺序会变，但**答案指向的描述不变**——编号只是「选项名的书写顺序」，
    这条错了整批数据的标签就全错位了。
    """
    loc = locales.load("zh")
    spec = _make_spec(compliance="gdpr", tier="enterprise")
    labeled = questions.build(spec, loc, 0, random.Random(7))
    first = assemble.assemble_sample("routing-train-0000001", "s", labeled, random.Random(8))
    second = assemble.assemble_sample("routing-train-0000002", "s", labeled, random.Random(9))
    for qid, question in first["questions"].items():
        if question["type"] != "choice":
            continue
        names = list(question["criteria"])
        picked = names[first["answers"][qid]["choice"]]
        other_names = list(second["questions"][qid]["criteria"])
        other_picked = other_names[second["answers"][qid]["choice"]]
        assert picked == other_picked
        assert question["criteria"][picked] == second["questions"][qid]["criteria"][other_picked]
