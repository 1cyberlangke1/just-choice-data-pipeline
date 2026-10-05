"""会话上下文压缩的题族测试：题型、选项、金答案与打乱规则。"""

from __future__ import annotations

from jc.workflow import LabeledQuestion

from workflows.compact import features, locales, questions
from workflows.compact.features import TagSet

TAGS = TagSet(kind="note", fact="chatter", age="old", pins="unpinned", budget="critical")


def by_family(tags: TagSet = TAGS) -> dict[str, LabeledQuestion]:
    """输入：可选特征。输出：{题族名: 题目}。"""
    return {item.family: item for item in questions.build(tags, locales.load("zh"), 0)}


def test_families_are_fixed_and_typed() -> None:
    """输入：一组特征。输出：无。

    预期行为：六个题族恒出；choice / noul / score 三类的划分与题面 `type` 对得上。
    """
    built = questions.build(TAGS, locales.load("zh"), 0)
    assert [item.family for item in built] == list(questions.FAMILIES)
    assert questions.CHOICE_FAMILIES + questions.NOUL_FAMILIES + questions.SCORE_FAMILIES == (
        questions.FAMILIES
    )
    families = by_family()
    for family in questions.CHOICE_FAMILIES:
        assert families[family].question["type"] == "choice"
    for family in questions.NOUL_FAMILIES:
        assert families[family].question["type"] == "noul"
    for family in questions.SCORE_FAMILIES:
        assert families[family].question["type"] == "score"


def test_answers_match_the_labels() -> None:
    """输入：一组特征。输出：无。预期行为：每族金答案等于特征层算出来的那一格。"""
    labels = TAGS.label()
    families = by_family()
    assert families["kind"].answer == labels.kind
    assert families["fact"].answer == labels.fact
    assert families["rule"].answer == labels.rule
    assert families["disposition"].answer == labels.disposition
    assert families["pin_overrode"].answer is labels.pin_overrode
    assert families["priority"].answer == labels.priority


def test_text_borne_answers_follow_the_tags() -> None:
    """输入：两个只差文本承载两维的组合。输出：无。

    预期行为：`kind` / `fact` 两族的金答案就是注入取值本身——它们由正文承载，
    判据在标签里，不在文本里（标签先行，文本后写）。
    """
    other = TagSet(kind="system", fact="fact", age="old", pins="unpinned", budget="critical")
    left, right = by_family(TAGS), by_family(other)
    assert left["kind"].answer == "note" and right["kind"].answer == "system"
    assert left["fact"].answer == "chatter" and right["fact"].answer == "fact"


def test_priority_depends_only_on_kind_age_and_pins() -> None:
    """输入：两个只差「持久事实」与「剩余预算」的组合。输出：无。

    预期行为：优先级只由（这一块是什么 × 轮次位置）与是否钉住决定——改文本承载那一维
    或改预算都不动它。这条钉住「哪个标签由哪几个特征算出」的分工。
    """
    other = TagSet(kind="note", fact="fact", age="old", pins="unpinned", budget="ample")
    left, right = by_family(TAGS), by_family(other)
    assert left["fact"].answer != right["fact"].answer
    expected = features.BASE_PRIORITY[("note", "old")]
    assert left["priority"].answer == right["priority"].answer == expected


def test_choice_options_are_only_the_allowed_ones() -> None:
    """输入：一组特征。输出：无。预期行为：每道 choice 的选项名就是该族允许的取值。"""
    families = by_family()
    for family in questions.CHOICE_FAMILIES:
        criteria = families[family].question["criteria"]
        assert list(criteria) == list(features.CHOICE_OPTIONS[family])
