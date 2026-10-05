"""浏览器 / GUI 操作的题族测试：题型、选项、金答案与问法轮换。"""

from __future__ import annotations

from jc.workflow import LabeledQuestion

from workflows.browser import features, locales, questions
from workflows.browser.features import TagSet

TAGS = TagSet(intent="switch", state="disabled", sensitive="yes", decoy="same_role", screen="list")


def by_family(tags: TagSet = TAGS) -> dict[str, LabeledQuestion]:
    """输入：一组特征。输出：{题族名: 题目}。"""
    return {item.family: item for item in questions.build(tags, locales.load("zh"), 0)}


def test_every_family_has_the_declared_type() -> None:
    """输入：一组特征。输出：无。预期行为：三族 choice、三族 noul、一族 score。"""
    families = by_family()
    assert set(families) == set(questions.FAMILIES)
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
    assert families["intent"].answer == labels.intent
    assert families["target"].answer == labels.target
    assert families["action"].answer == labels.action
    assert families["done"].answer is labels.done
    assert families["blocked"].answer is labels.blocked
    assert families["confirm"].answer is labels.confirm
    assert families["risk"].answer == labels.risk


def test_intent_answer_follows_the_tags_not_the_table() -> None:
    """输入：两个只差意图的组合。输出：无。

    预期行为：`intent` 那一族的金答案就是注入取值本身——它由那句话承载，
    判据在标签里，不在文本里（标签先行，文本后写）。
    """
    left = by_family(TagSet(intent="fill", state="ready", sensitive="no", decoy="none", screen="form"))
    right = by_family(TagSet(intent="open", state="ready", sensitive="no", decoy="none", screen="form"))
    assert left["intent"].answer == "fill"
    assert right["intent"].answer == "open"


def test_target_options_are_this_screen_plus_none() -> None:
    """输入：一组特征。输出：无。

    预期行为：`target` 那一族的选项名就是这一屏的元素名（按显示顺序）再加一个「不操作」，
    描述只给行号——考生得去读元素表才知道那一行是什么角色、什么状态。
    """
    loc = locales.load("zh")
    options = questions.target_options(TAGS, loc)
    rows = features.rows_for(TAGS)
    assert list(options) == [row.label for row in rows] + [features.NONE_TARGET]
    assert len(options) == len(set(options))
    template = str(loc["fam"]["target"]["row_option"])
    for index, row in enumerate(rows, start=1):
        assert options[row.label] == template.format(index=index)
    assert options[features.NONE_TARGET] == str(loc["fam"]["target"]["none"])


def test_target_answer_is_always_one_of_the_options() -> None:
    """输入：全空间组合。输出：无。

    预期行为：每一条的 `target` 金答案都必须落在它自己的选项表里——否则组装器
    在 `criteria.index(答案)` 那一步就会炸，或者更糟：悄悄给出错的编号。
    """
    loc = locales.load("zh")
    for case in features.build_cases(1024):
        options = questions.target_options(case, loc)
        assert case.label().target in options


def test_choice_options_are_only_the_allowed_ones() -> None:
    """输入：一组特征。输出：无。预期行为：意图 / 动作两族的选项名就是该族允许的取值。"""
    families = by_family()
    for family in ("intent", "action"):
        criteria = families[family].question["criteria"]
        assert list(criteria) == list(features.CHOICE_OPTIONS[family])


def test_noul_and_score_criteria_shapes() -> None:
    """输入：一组特征。输出：无。

    预期行为：noul 的选项是 `true` / `false`；score 的选项是有序阶梯（档数 == `RISK_LEVELS`）。
    """
    loc = locales.load("zh")
    families = by_family()
    for family in questions.NOUL_FAMILIES:
        assert list(families[family].question["criteria"]) == ["true", "false"]
    assert list(families["risk"].question["criteria"]) == list(loc["fam"]["risk"]["levels"])
    assert len(families["risk"].question["criteria"]) == features.RISK_LEVELS


def test_question_order_is_fixed() -> None:
    """输入：一组特征。输出：无。预期行为：出题顺序恒等于 `FAMILIES`（题号才稳定）。"""
    assert [item.family for item in questions.build(TAGS, locales.load("zh"), 0)] == list(
        questions.FAMILIES
    )
