"""改动引入性判定的题族测试：题型、选项、金答案与打乱规则。"""

from __future__ import annotations

from workflows.findtriage import features, locales, questions
from workflows.findtriage.features import RecordCase

CASE = RecordCase(cite="touched", evidence="in_row", fix="concrete", size=5, decoy=0)
PLAN = features.plan_for(CASE, 0)
LABELS = features.label_for(PLAN)


def built() -> dict[str, object]:
    """输入：无。输出：{题族名: 题目}。"""
    return {item.family: item for item in questions.build(PLAN, locales.load("zh"), 0)}


def test_families_are_fixed_and_typed() -> None:
    """输入：一个计划。输出：无。

    预期行为：七个题族恒出；choice / noul 两类的划分与题面 `type` 对得上。
    """
    by_family = built()
    assert [item.family for item in questions.build(PLAN, locales.load("zh"), 0)] == list(
        questions.FAMILIES
    )
    for family in questions.CHOICE_FAMILIES:
        assert by_family[family].question["type"] == "choice"
    for family in questions.NOUL_FAMILIES:
        assert by_family[family].question["type"] == "noul"


def test_answers_match_the_labels() -> None:
    """输入：一个计划。输出：无。预期行为：每族金答案等于特征层算出来的那一格。"""
    by_family = built()
    assert by_family["cite"].answer == LABELS.cite_key
    assert by_family["introduced"].answer == LABELS.introduced
    assert by_family["supported"].answer == LABELS.supported
    assert by_family["actionable"].answer is LABELS.actionable
    assert by_family["triage"].answer == LABELS.triage
    assert by_family["owner"].answer == LABELS.owner
    assert by_family["decoy"].answer is LABELS.decoy


def test_owner_options_are_the_four_queues() -> None:
    """输入：一个计划。输出：无。

    预期行为：`owner` 的四个选项名与 `features.OWNERS` 完全一致、描述两两不同——
    四个去处少一个，那一档的答案就落不到选项上（组装器会当场报错，这里提前钉住）。
    """
    criteria = built()["owner"].question["criteria"]
    assert list(criteria) == list(features.OWNERS)
    assert len(set(criteria.values())) == len(features.OWNERS)
    assert all(str(text).strip() for text in criteria.values())


def test_cite_options_track_the_table_length() -> None:
    """输入：表长 5 的计划。输出：无。

    预期行为：`cite` 的选项名是 `s1`…`s5` 加一个 `none`，顺序与表序一致——
    答案「第几条」就是在这个顺序里的序号。
    """
    criteria = built()["cite"].question["criteria"]
    assert list(criteria) == ["s1", "s2", "s3", "s4", "s5", features.NO_CITE]
    assert str(criteria["s1"]).strip()
    assert str(criteria[features.NO_CITE]).strip()


def test_cite_options_shrink_with_the_table() -> None:
    """输入：表长 3 的计划。输出：无。预期行为：选项跟着表长走，不会多出第 4 条。"""
    small = features.plan_for(
        RecordCase(cite="missing", evidence="none", fix="vague", size=3, decoy=1), 0
    )
    options = questions.cite_options(small.size, locales.load("zh"))
    assert list(options) == ["s1", "s2", "s3", features.NO_CITE]


def test_fixed_option_families_are_the_declared_values() -> None:
    """输入：一个计划。输出：无。预期行为：三族固定选项的选项名就是特征层的取值表。"""
    by_family = built()
    assert list(by_family["introduced"].question["criteria"]) == list(features.INTRODUCED)
    assert list(by_family["supported"].question["criteria"]) == list(features.SUPPORTED)
    assert list(by_family["triage"].question["criteria"]) == list(features.TRIAGES)


def test_every_choice_family_is_shuffled_and_noul_is_not() -> None:
    """输入：一个计划。输出：无。预期行为：没有有序阶梯，所以 choice 全打乱、noul 不打乱。"""
    by_family = built()
    for family in questions.FAMILIES:
        if family in questions.NOUL_FAMILIES:
            assert by_family[family].shuffle is False
        else:
            assert by_family[family].shuffle is True


def test_phrasings_rotate() -> None:
    """输入：三个问法编号。输出：无。预期行为：同一族的题面在三个编号上都能取到。"""
    texts = {
        questions.build(PLAN, locales.load("zh"), phrasing)[0].question["instructions"]
        for phrasing in (0, 1, 2)
    }
    assert len(texts) == 3


def test_families_for_is_constant() -> None:
    """输入：任意一组特征。输出：无。预期行为：本 workflow 没有条件题族，七族恒出。"""
    assert questions.families_for(CASE) == questions.FAMILIES
