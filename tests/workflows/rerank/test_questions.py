"""题族测试：出题顺序、选项、答案类型与两两比较那一族的配平。"""

from __future__ import annotations

import pytest

from workflows.rerank import features, locales, questions, render


def _plan(terms: int = 3, relevant: int = 3, best: str = "first") -> features.RerankPlan:
    """输入：三个特征取值。输出：名次计划。"""
    return features.plan_for(features.RerankCase(terms, relevant, best), 0)


def test_families_skip_top_when_nothing_is_relevant() -> None:
    """输入：相关条数为 0 与大于 0 的两份计划。输出：无。

    预期行为：一条都不相关时 `top` 不出（没有唯一解），其余三族照出。
    """
    assert questions.families_for(_plan(relevant=0)) == ("count", "pair", "order")
    assert questions.families_for(_plan(relevant=1)) == questions.FAMILIES


def test_question_counts_and_order() -> None:
    """输入：一份有相关候选的计划。输出：无。预期行为：1 + 1 + 20 + 1 道，顺序按题族。"""
    plan = _plan(terms=3, relevant=2, best="middle")
    labeled = questions.build(plan, locales.load("zh"), 0)
    families = [item.family for item in labeled]
    assert families[0] == "top"
    assert families[1] == "count"
    assert families.count("pair") == len(features.PAIRS)
    assert families[-1] == "order"
    assert len(labeled) == 1 + 1 + len(features.PAIRS) + 1


def test_choice_answers_are_option_names_and_noul_answers_are_booleans() -> None:
    """输入：一份计划。输出：无。预期行为：choice 的答案是选项名、noul 的答案是布尔。"""
    plan = _plan(terms=2, relevant=3, best="last")
    labels = features.label_for(plan)
    for item in questions.build(plan, locales.load("zh"), 0):
        criteria = item.question["criteria"]
        if item.question["type"] == "choice":
            assert isinstance(criteria, dict)
            assert isinstance(item.answer, str) and item.answer in criteria
        else:
            assert isinstance(item.answer, bool)
    out = {item.family: item for item in questions.build(plan, locales.load("zh"), 0)}
    assert out["top"].answer == f"s{labels.top + 1}"
    assert out["count"].answer == f"n{labels.relevant}"
    assert out["order"].answer == "o0"


def test_top_options_cover_every_candidate() -> None:
    """输入：一份计划。输出：无。预期行为：5 个候选全出（只出答案那一个会泄露答案）。"""
    item = next(
        entry
        for entry in questions.build(_plan(relevant=1), locales.load("zh"), 0)
        if entry.family == "top"
    )
    criteria = item.question["criteria"]
    assert isinstance(criteria, dict)
    assert list(criteria) == ["s1", "s2", "s3", "s4", "s5"]


def test_count_options_cover_every_tier() -> None:
    """输入：一份计划。输出：无。预期行为：0~4 五个档位全出，与 `features.RELEVANT` 对齐。"""
    item = next(
        entry
        for entry in questions.build(_plan(relevant=2), locales.load("zh"), 0)
        if entry.family == "count"
    )
    criteria = item.question["criteria"]
    assert isinstance(criteria, dict)
    assert list(criteria) == [f"n{value}" for value in features.RELEVANT]


@pytest.mark.parametrize("relevant", features.RELEVANT)
def test_pair_family_is_balanced_by_construction(relevant: int) -> None:
    """输入：每一档相关条数。输出：无。

    预期行为：每个无序对两个方向都问，所以 20 道里 True 恰好 10 道——noul 的 50% 是**算出来
    的**，不是靠采样凑的。
    """
    plan = _plan(terms=3, relevant=relevant, best="first")
    answers = [item.answer for item in questions.build(plan, locales.load("zh"), 0) if item.family == "pair"]
    assert len(answers) == len(features.PAIRS)
    assert answers.count(True) == len(features.PAIRS) // 2


def test_pair_answers_match_the_recomputed_order() -> None:
    """输入：一份计划。输出：无。预期行为：每道 pair 的答案就是「左边是否排在右边前面」。"""
    plan = _plan(terms=2, relevant=4, best="middle")
    labels = features.label_for(plan)
    pairs = [item for item in questions.build(plan, locales.load("zh"), 0) if item.family == "pair"]
    for index, (left, right) in enumerate(features.PAIRS):
        assert pairs[index].answer is labels.above(left, right)


def test_order_options_are_four_distinct_permutations() -> None:
    """输入：一份计划。输出：无。

    预期行为：四个选项两两不同，正确的那一个（`o0`）就是重算出来的前三名。
    """
    plan = _plan(terms=3, relevant=3, best="last")
    labels = features.label_for(plan)
    item = next(
        entry
        for entry in questions.build(plan, locales.load("zh"), 0)
        if entry.family == "order"
    )
    criteria = item.question["criteria"]
    assert isinstance(criteria, dict)
    assert len(set(criteria.values())) == 4
    loc = locales.load("zh")
    head = labels.order[:3]
    expected = str(loc["fam"]["order"]["opt"]["row"]).format(
        a=render.cand_ref(loc, head[0]),
        b=render.cand_ref(loc, head[1]),
        c=render.cand_ref(loc, head[2]),
    )
    assert criteria["o0"] == expected


def test_choice_helper_rejects_broken_options() -> None:
    """输入：重复的选项描述 / 答案不在选项里。输出：无。预期行为：都当场报错。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("x", "q", {"a": "same", "b": "same"}, "a")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("x", "q", {"a": "one", "b": "two"}, "c")


def test_unknown_family_reports_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：一个假题族名。输出：无。预期行为：`build` 报 `KeyError`，不静默跳过。"""
    plan = _plan(relevant=1)
    monkeypatch.setattr(questions, "families_for", lambda _plan: ("nope",))
    with pytest.raises(KeyError, match="未知题族"):
        questions.build(plan, locales.load("zh"), 0)


def test_phrasing_switches_the_wording() -> None:
    """输入：两种问法。输出：无。预期行为：题面真的换了说法（问法池不是摆设）。"""
    plan = _plan(terms=2, relevant=2, best="first")
    loc = locales.load("zh")
    first = questions.build(plan, loc, 0)[0].question["instructions"]
    second = questions.build(plan, loc, 1)[0].question["instructions"]
    assert first != second
