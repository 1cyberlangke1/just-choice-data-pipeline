"""提示注入的题族测试：四族的题面、答案与均衡。"""

from __future__ import annotations


from jc.workflow import LabeledQuestion

from workflows.promptinject import features, locales, questions
from workflows.promptinject.features import PromptInjectCase

#: 全空间 32 组 case。
CASES = features.build_cases(512)

#: 每族在 `build` 里占的位置：前四条逐判据，然后 count / combo / action。
TAG_SPAN = len(features.TAG_IDS)


def _build(
    case: PromptInjectCase, lang: str = "zh", phrasing: int = 0
) -> list[LabeledQuestion]:
    """输入：case、语言、问法编号。输出：这一条的题目列表。"""
    return questions.build(features.plan_for(case, phrasing), locales.load(lang), phrasing)


def test_family_order_matches_the_declared_families() -> None:
    """输入：一个 case。输出：无。预期行为：四族顺序与题数固定（逐判据 4 条 + 三条）。"""
    labeled = _build(PromptInjectCase("T1+T2", features.SOURCE_WEB))
    assert questions.FAMILIES == ("tag", "count", "combo", "action")
    assert [item.family for item in labeled] == ["tag"] * 4 + ["count", "combo", "action"]


def test_tag_family_is_one_noul_per_tag() -> None:
    """输入：一个命中 T2 的 case。输出：无。

    预期行为：逐判据四条，答案就是「这一条在不在命中组合里」；题面点名判据名与判据原文。
    """
    loc = locales.load("zh")
    case = PromptInjectCase("T2", features.SOURCE_WEB)
    for index, tag in enumerate(features.TAG_IDS):
        item = _build(case)[index]
        assert item.question["type"] == "noul"
        assert item.answer is (tag in case.tags())
        assert item.shuffle is False
        assert str(loc["tag"][tag]["name"]) in str(item.question["instructions"])
        assert set(item.question["criteria"]) == {"true", "false"}
        # 第二种问法会把判据原文也念一遍（第一种只点判据名——判据本来就在 state 的判据表里）。
        second = _build(case, phrasing=1)[index]
        assert str(loc["tag"][tag]["rule"]) in str(second.question["instructions"])


def test_tag_answers_are_fifty_fifty() -> None:
    """输入：全空间 32 组 case。输出：无。

    预期行为：每条判据在 16 组命中组合里各出现 8 次，32 组 case 里各 16 次，所以逐判据那一族的
    True / False 正好各占一半——这是按构造就成立的均衡，不是调出来的。
    """
    seen = [0] * TAG_SPAN
    for case in CASES:
        for index, item in enumerate(_build(case)[:TAG_SPAN]):
            seen[index] += 1 if item.answer else 0
    assert seen == [len(CASES) // 2] * TAG_SPAN


def test_count_answers_match_the_truth() -> None:
    """输入：全空间 32 组 case。输出：无。预期行为：答案档位 = 命中条数，选项表就是 0~4。"""
    for case in CASES:
        item = _build(case)[TAG_SPAN]
        assert item.question["type"] == "choice"
        assert item.answer == f"c{len(case.tags())}"
        criteria = item.question["criteria"]
        assert sorted(criteria) == [f"c{value}" for value in range(TAG_SPAN + 1)]
        assert len(set(criteria.values())) == TAG_SPAN + 1


def test_combo_offers_the_truth_plus_three_distractors() -> None:
    """输入：全空间 32 组 case。输出：无。

    预期行为：四个选项 = 真值 + 三个邻近干扰，描述两两不同，真值那一个就是金答案。
    """
    loc = locales.load("zh")
    for case in CASES:
        plan = features.plan_for(case, 0)
        item = _build(case)[TAG_SPAN + 1]
        criteria = item.question["criteria"]
        assert len(criteria) == 4
        assert len(set(criteria.values())) == 4
        assert criteria[item.answer] == questions.combo_text(loc, plan.hits)
        for combo in plan.distractors:
            assert questions.combo_text(loc, combo) in criteria.values()


def test_action_answers_follow_the_thresholds() -> None:
    """输入：全空间 32 组 case。输出：无。预期行为：答案 = 按命中条数落到的那个处置档。"""
    loc = locales.load("zh")
    for case in CASES:
        item = _build(case)[TAG_SPAN + 2]
        count = len(case.tags())
        assert item.answer == features.handling_level(count)
        criteria = item.question["criteria"]
        assert sorted(criteria) == sorted(features.HANDLING_LEVELS)
        assert criteria[item.answer] == questions.action_text(loc, count)


def test_phrasing_picks_alternate_wording() -> None:
    """输入：同一个 case 的两种问法。输出：无。预期行为：问法池里第 0 / 第 1 条文案不同。"""
    case = PromptInjectCase("T1", features.SOURCE_WEB)
    first = _build(case, phrasing=0)
    second = _build(case, phrasing=1)
    assert first[0].question["instructions"] != second[0].question["instructions"]
    assert first[0].answer == second[0].answer
