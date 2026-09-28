"""检索结果判读的题族测试：题型、选项、金答案与打乱规则。"""

from __future__ import annotations

from workflows.search import locales, questions
from workflows.search.features import TagSet

TAGS = TagSet(
    query_kind="navigational",
    topicality="on_topic",
    spam="clean",
    authority="unknown",
    freshness="stale",
)


def test_families_are_fixed_and_typed() -> None:
    """输入：一组特征。输出：无。

    预期行为：六个题族恒出；choice / noul / score 三类的划分与题面 `type` 对得上。
    """
    built = questions.build(TAGS, locales.load("zh"), 0)
    assert [item.family for item in built] == list(questions.FAMILIES)
    by_family = {item.family: item for item in built}
    for family in questions.CHOICE_FAMILIES:
        assert by_family[family].question["type"] == "choice"
    for family in questions.NOUL_FAMILIES:
        assert by_family[family].question["type"] == "noul"
    for family in questions.SCORE_FAMILIES:
        assert by_family[family].question["type"] == "score"


def test_answers_match_the_labels() -> None:
    """输入：一组特征。输出：无。预期行为：每族金答案等于特征层算出来的那一格。"""
    labels = TAGS.label()
    by_family = {item.family: item for item in questions.build(TAGS, locales.load("zh"), 0)}
    assert by_family["query_kind"].answer == labels.query_kind
    assert by_family["topicality"].answer == labels.topicality
    assert by_family["spam"].answer is (labels.spam == "spam")
    assert by_family["rule"].answer == labels.rule
    assert by_family["verdict"].answer == labels.verdict
    assert by_family["trust"].answer == labels.trust


def test_choice_options_are_only_the_allowed_ones() -> None:
    """输入：一组特征。输出：无。预期行为：每道 choice 的选项名就是该族允许的取值。"""
    by_family = {item.family: item for item in questions.build(TAGS, locales.load("zh"), 0)}
    for family in questions.CHOICE_FAMILIES:
        criteria = by_family[family].question["criteria"]
        assert list(criteria) == list(questions.features.CHOICE_OPTIONS[family])


def test_only_the_score_family_is_unshuffled() -> None:
    """输入：一组特征。输出：无。预期行为：有序阶梯不许打乱，其余 choice 都允许打乱。"""
    by_family = {item.family: item for item in questions.build(TAGS, locales.load("zh"), 0)}
    for family in questions.FAMILIES:
        if family in questions.SCORE_FAMILIES or family in questions.NOUL_FAMILIES:
            assert by_family[family].shuffle is False
        else:
            assert by_family[family].shuffle is True


def test_phrasings_rotate() -> None:
    """输入：三个问法编号。输出：无。预期行为：同一族的题面在三个编号上都能取到。"""
    texts = {
        questions.build(TAGS, locales.load("zh"), phrasing)[0].question["instructions"]
        for phrasing in (0, 1, 2)
    }
    assert len(texts) == 3