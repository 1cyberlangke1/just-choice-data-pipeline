"""题族测试：出题顺序、两阶段依赖、选项、答案类型。"""

from __future__ import annotations

import pytest

from workflows.citation import features, locales, questions, render
from tests.workflows.citation.conftest import spec_for


def _plan(
    form: str = "exact", where: str = "same", place: str = "middle", target: str = "first"
) -> features.CitationPlan:
    """输入：四个特征取值。输出：引文计划。预期行为：测试里随手造计划的便捷入口。"""
    return spec_for(features.CitationCase(form, where, place, target)).plan


def test_section_family_only_shows_up_when_the_quote_is_in_the_document() -> None:
    """输入：能查到 / 查不到两种计划。输出：无。

    预期行为：查不到引文时**不出** `section` 那一族——「它出自第几节」没有唯一解，硬出就是
    一条错数据。
    """
    assert "section" in questions.families_for(_plan(form="exact"))
    assert "section" in questions.families_for(_plan(form="partial"))
    assert "section" not in questions.families_for(_plan(form="altered"))
    assert questions.families_for(_plan(form="altered")) == ("verdict", "in_doc", "in_claimed")


def test_family_order_is_fixed() -> None:
    """输入：一份计划。输出：无。预期行为：题族顺序就是 `FAMILIES`（报告与审计按它读）。"""
    families = [item.family for item in questions.build(_plan(), locales.load("zh"), 0)]
    assert families == list(questions.FAMILIES)


def test_answer_types_and_names() -> None:
    """输入：一份计划。输出：无。预期行为：choice 的答案是选项名、noul 的答案是布尔。"""
    plan = _plan(where="other", place="last")
    labels = features.label_for(plan)
    for item in questions.build(plan, locales.load("zh"), 0):
        criteria = item.question["criteria"]
        assert isinstance(criteria, dict)
        if item.question["type"] == "choice":
            assert isinstance(item.answer, str) and item.answer in criteria
        else:
            assert isinstance(item.answer, bool)
    out = {item.family: item for item in questions.build(plan, locales.load("zh"), 0)}
    assert out["verdict"].answer == f"v{features.VERDICTS.index(labels.verdict)}"
    assert out["section"].answer == f"s{labels.section}"
    assert out["in_doc"].answer is labels.in_doc
    assert out["in_claimed"].answer is labels.at_claimed


def test_verdict_options_cover_every_verdict() -> None:
    """输入：一份计划。输出：无。预期行为：三种结论全出（只出答案那一个会泄露答案）。"""
    item = next(
        entry
        for entry in questions.build(_plan(), locales.load("zh"), 0)
        if entry.family == "verdict"
    )
    criteria = item.question["criteria"]
    assert isinstance(criteria, dict)
    assert list(criteria) == [f"v{index}" for index in range(len(features.VERDICTS))]
    assert set(criteria.values()) == {
        str(locales.load("zh")["fam"]["verdict"]["opt"][name]) for name in features.VERDICTS
    }


def test_section_options_cover_every_section() -> None:
    """输入：一份计划。输出：无。预期行为：三节全出，选项文案就是节的可读指代。"""
    loc = locales.load("zh")
    item = next(
        entry
        for entry in questions.build(_plan(), loc, 0)
        if entry.family == "section"
    )
    criteria = item.question["criteria"]
    assert isinstance(criteria, dict)
    assert list(criteria) == [f"s{index}" for index in range(features.SECTIONS)]
    assert [criteria[f"s{index}"] for index in range(features.SECTIONS)] == [
        render.sec_ref(loc, index) for index in range(features.SECTIONS)
    ]


def test_yes_no_families_carry_their_criteria() -> None:
    """输入：两份 noul 题。输出：无。预期行为：判据都有 true / false 两句文案，且不为空。"""
    out = {
        item.family: item for item in questions.build(_plan(form="altered"), locales.load("zh"), 0)
    }
    for family in ("in_doc", "in_claimed"):
        criteria = out[family].question["criteria"]
        assert isinstance(criteria, dict)
        assert sorted(criteria) == ["false", "true"]
        assert all(str(value).strip() for value in criteria.values())
        assert out[family].shuffle is False


@pytest.mark.parametrize("form", features.FORMS)
def test_in_claimed_is_the_conjunction(form: str) -> None:
    """输入：三种造法。输出：无。预期行为：`in_claimed` 的答案就是「在文档里且落在声称那节」。"""
    for where in features.WHERES:
        plan = _plan(form=form, where=where)
        labels = features.label_for(plan)
        item = next(
            entry
            for entry in questions.build(plan, locales.load("zh"), 0)
            if entry.family == "in_claimed"
        )
        assert item.answer is labels.at_claimed
        assert item.answer is (labels.in_doc and labels.section == labels.claimed)


def test_choice_helper_rejects_broken_options() -> None:
    """输入：重复的选项描述 / 答案不在选项里。输出：无。预期行为：都当场报错。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("x", "q", {"a": "same", "b": "same"}, "a")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("x", "q", {"a": "one", "b": "two"}, "c")


def test_unknown_family_reports_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """输入：一个假题族名。输出：无。预期行为：`build` 报 `KeyError`，不静默跳过。"""
    monkeypatch.setattr(questions, "families_for", lambda _plan: ("nope",))
    with pytest.raises(KeyError, match="未知题族"):
        questions.build(_plan(), locales.load("zh"), 0)


def test_phrasing_switches_the_wording() -> None:
    """输入：两种问法。输出：无。预期行为：题面真的换了说法（问法池不是摆设）。"""
    plan = _plan()
    loc = locales.load("zh")
    first = questions.build(plan, loc, 0)[0].question["instructions"]
    second = questions.build(plan, loc, 1)[0].question["instructions"]
    assert first != second
