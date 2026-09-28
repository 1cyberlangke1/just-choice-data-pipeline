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


def test_question_shape_matches_the_contract() -> None:
    """输入：一组特征。输出：无。预期行为：每道题只有契约允许的三个字段，且题面非空。"""
    for item in questions.build(TAGS, locales.load("zh"), 0):
        assert set(item.question) == {"type", "instructions", "criteria"}
        assert str(item.question["instructions"]).strip()


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


def test_noul_and_score_criteria_shapes() -> None:
    """输入：一组特征。输出：无。

    预期行为：noul 的选项是 `true` / `false`；score 的选项是有序阶梯（档数 == `PRIORITY_LEVELS`）。
    """
    loc = locales.load("zh")
    families = by_family()
    assert list(families["pin_overrode"].question["criteria"]) == ["true", "false"]
    assert list(families["priority"].question["criteria"]) == list(loc["fam"]["priority"]["levels"])
    assert len(families["priority"].question["criteria"]) == features.PRIORITY_LEVELS


def test_only_the_score_and_noul_families_are_unshuffled() -> None:
    """输入：一组特征。输出：无。

    预期行为：有序阶梯与布尔题不打乱（打乱破坏语义），四个 choice 题族都允许打乱
    （防「答案总在第 n 个」的位置捷径）。
    """
    families = by_family()
    for family in questions.FAMILIES:
        if family in questions.SCORE_FAMILIES or family in questions.NOUL_FAMILIES:
            assert families[family].shuffle is False
        else:
            assert families[family].shuffle is True


def test_phrasings_rotate() -> None:
    """输入：三个问法编号。输出：无。预期行为：同一族的题面在三个编号上都能取到。"""
    for family in questions.FAMILIES:
        texts = {
            next(
                item.question["instructions"]
                for item in questions.build(TAGS, locales.load("zh"), phrasing)
                if item.family == family
            )
            for phrasing in (0, 1, 2)
        }
        assert len(texts) == 3, family


def test_families_for_returns_the_full_table() -> None:
    """输入：任意特征。输出：无。预期行为：六个题族对任何组合都问得出来。"""
    for tags in features.build_cases(1024):
        assert questions.families_for(tags) == questions.FAMILIES