"""题族测试：四族题的形状、答案与自足性。"""

from __future__ import annotations

import pytest

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from workflows.chatgate import features, locales, questions
from workflows.chatgate.features import ChatCase

#: 一个普通 case：命中两条，同事私聊，该发第一条。
CASE = ChatCase(degree=2, scene=features.SCENE_COLLEAGUE, slot=0)


def _items(case: ChatCase, loc: dict[str, JsonValue], phrasing: int = 0) -> list[LabeledQuestion]:
    """输入：case、语言字典、问法编号。输出：该 case 的全部题目。"""
    return questions.build(features.plan_for(case, phrasing), loc, phrasing)


def _options(item: LabeledQuestion) -> dict[str, JsonValue]:
    """输入：一道题。输出：它的选项表（choice 才有；noul 的判据表也在这里）。"""
    return item.question["criteria"]


def test_tag_family_is_one_noul_per_tag() -> None:
    """输入：无。输出：无。预期行为：四条 noul，一条判据一道，答案与真值一致。"""
    plan = features.plan_for(CASE, 0)
    out = questions.build(plan, locales.load("zh"), 0)
    tags = [item for item in out if item.family == "tag"]
    assert len(tags) == len(features.TAG_IDS)
    for tag, item in zip(features.TAG_IDS, tags, strict=True):
        assert item.question["type"] == "noul"
        assert item.answer is (tag in plan.hits)
        assert set(_options(item)) == {"true", "false"}
        assert _options(item)["true"] != _options(item)["false"]


def test_tag_answers_stay_near_fifty_fifty() -> None:
    """输入：全空间 45 组 case。输出：无。

    预期行为：每条判据的 True / False 大致各半——`degree` 均匀时，每条判据被选中的概率正好是
    一半，但 45 是奇数、判据是 4 条，所以只能**近似**均匀。这里钉两条：每条落在 40%~60%
    （18~27 次），且四条加起来正好等于全部命中条数之和 90（这条是恒等式，不受近似影响）。
    """
    loc = locales.load("zh")
    totals = dict.fromkeys(features.TAG_IDS, 0)
    for case in features.build_cases(512):
        for tag, item in zip(features.TAG_IDS, _items(case, loc), strict=False):
            if item.answer is True:
                totals[tag] += 1
    assert all(18 <= count <= 27 for count in totals.values()), totals
    assert sum(totals.values()) == 90


def test_count_answers_match_the_truth() -> None:
    """输入：全空间 45 组 case。输出：无。预期行为：`count` 的答案是命中条数那一档。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in _items(case, loc) if q.family == "count")
        assert item.answer == f"c{len(plan.hits)}"
        assert _options(item)[str(item.answer)] == questions.count_text(loc, len(plan.hits))
        assert len(_options(item)) == len(features.TAG_IDS) + 1


def test_route_answers_follow_the_thresholds() -> None:
    """输入：全空间 45 组 case。输出：无。预期行为：`route` 的答案是阈值算出来的那一档。"""
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in _items(case, loc) if q.family == "route")
        assert item.answer == features.handling_level(len(plan.hits))
        assert _options(item)[str(item.answer)] == questions.route_text(loc, len(plan.hits))
        assert set(_options(item)) == set(features.HANDLING_LEVELS)


def test_reply_answers_point_at_the_clean_candidate() -> None:
    """输入：全空间 45 组 case。输出：无。

    预期行为：`reply` 的答案是「该发的那条」的位置，选项是三条候选的位置；选项文案与
    `reply_text` 同源（审计拿它比对 state 里的候选编号）。
    """
    loc = locales.load("zh")
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        item = next(q for q in _items(case, loc) if q.family == "reply")
        assert item.answer == f"r{plan.reply.clean_answer}"
        assert set(_options(item)) == {f"r{index}" for index in features.CANDIDATE_SLOTS}
        for index in features.CANDIDATE_SLOTS:
            assert _options(item)[f"r{index}"] == questions.reply_text(loc, index)


def test_reply_options_are_positions_not_candidate_bodies() -> None:
    """输入：一个 case × 两种语言。输出：无。

    预期行为：`reply` 的选项**只指位置**（第 N 条候选），不含候选正文——正文在渲染完之前不
    存在（`ask` 拿不到 state），选项必须由计划推得出来。正文靠 state 里按同一编号印出来，
    两者合起来才自足。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        item = next(q for q in _items(CASE, loc) if q.family == "reply")
        for index in features.CANDIDATE_SLOTS:
            assert str(index + 1) in str(_options(item)[f"r{index}"])


def test_choice_options_never_duplicate() -> None:
    """输入：全空间 45 组 case × 两种语言。输出：无。

    预期行为：三族 choice 的选项描述两两不同——两句话一样，考生没法分辨，是构造 bug。
    """
    for lang in locales.available():
        loc = locales.load(lang)
        for case in features.build_cases(512):
            for item in _items(case, loc):
                if item.question["type"] != "choice":
                    continue
                values = [str(value) for value in _options(item).values()]
                assert len(set(values)) == len(values), f"{lang}/{case.describe()}/{item.family}"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_tag_question_is_self_contained(lang: str) -> None:
    """输入：一种语言 × 第二种问法。输出：无。

    预期行为：题面里带上要问的那条判据的名字与原文——只读 state 的考生要能自己按字面判。
    （用第二种问法：第一种是短问法，只点名判据，不重述判据原文。）
    """
    loc = locales.load(lang)
    for tag, item in zip(features.TAG_IDS, _items(CASE, loc, 1), strict=False):
        entry = loc["tag"][tag]
        instructions = str(item.question["instructions"])
        assert str(entry["name"]) in instructions
        assert str(entry["rule"]) in instructions


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_choice_question_points_at_the_state_tables(lang: str) -> None:
    """输入：一种语言 × 第二种问法。输出：无。

    预期行为：三族 choice 的题面都非空、答案那一条的描述确实在选项表里——选项表就是取值表，
    考生照着 state 里的表勾就行，不用外部知识（范式 §9.8）。
    """
    loc = locales.load(lang)
    for item in _items(CASE, loc, 1):
        if item.question["type"] != "choice":
            continue
        assert str(item.question["instructions"]).strip()
        assert str(item.answer) in _options(item)
