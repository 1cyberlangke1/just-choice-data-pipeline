"""工单分诊的题族测试：题型、选项、金答案与打乱规则。"""

from __future__ import annotations

from jc.workflow import LabeledQuestion

from workflows.routing import features, locales, questions
from workflows.routing.features import TagSet

TAGS = TagSet(issue="data_loss", urgency="blocked", tier="free", compliance="none", area="payments")


def by_family(tags: TagSet = TAGS) -> dict[str, LabeledQuestion]:
    """输入：可选特征。输出：{题族名: 题目}。"""
    return {item.family: item for item in questions.build(tags, locales.load("zh"), 0)}


def test_families_are_fixed_and_typed() -> None:
    """输入：一组特征。输出：无。

    预期行为：六个题族恒出；choice / noul / score 三类的划分与题面 `type` 对得上。
    """
    built = questions.build(TAGS, locales.load("zh"), 0)
    assert [item.family for item in built] == list(questions.FAMILIES)
    grouped = (
        questions.CHOICE_FAMILIES + questions.NOUL_FAMILIES + questions.SCORE_FAMILIES
    )
    assert set(grouped) == set(questions.FAMILIES)
    assert len(grouped) == len(questions.FAMILIES)
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
    assert families["issue"].answer == labels.issue
    assert families["urgency"].answer == labels.urgency
    assert families["rule"].answer == labels.rule
    assert families["queue"].answer == labels.queue
    assert families["priority"].answer == labels.priority
    assert families["needs_human"].answer is labels.needs_human


def test_text_borne_answers_follow_the_tags() -> None:
    """输入：两个只差文本承载两维的组合。输出：无。

    预期行为：`issue` / `urgency` 两族的金答案就是注入取值本身——它们由正文承载，
    判据在标签里，不在文本里（标签先行，文本后写）。
    """
    other = TagSet(
        issue="feature", urgency="deadline", tier="free", compliance="none", area="payments"
    )
    left, right = by_family(TAGS), by_family(other)
    assert left["issue"].answer == "data_loss" and right["issue"].answer == "feature"
    assert left["urgency"].answer == "blocked" and right["urgency"].answer == "deadline"


def test_priority_depends_only_on_urgency_and_tier() -> None:
    """输入：两个只差「合规标记」与「产品域」的组合。输出：无。

    预期行为：优先级只由（紧急度 × 客户等级）决定——改合规标记或产品域都不动它。
    这条钉住「哪个标签由哪几个特征算出」的分工。
    """
    other = TagSet(
        issue="data_loss", urgency="blocked", tier="free", compliance="gdpr", area="identity"
    )
    left, right = by_family(TAGS), by_family(other)
    assert left["priority"].answer == right["priority"].answer
    assert left["priority"].answer == features.PRIORITY_TABLE[("blocked", "free")]
    assert left["queue"].answer != right["queue"].answer  # 队列那一族确实变了


def test_families_for_returns_the_full_table() -> None:
    """输入：任意特征。输出：无。预期行为：六个题族对任何组合都问得出来。"""
    for tags in features.build_cases(2048):
        assert questions.families_for(tags) == questions.FAMILIES
