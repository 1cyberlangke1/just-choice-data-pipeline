"""题族的单元测试：四个题族的出题规则、选项、以及「均衡靠构造」这条设计。"""

from __future__ import annotations

from collections import Counter

import pytest

from tests.workflows.browser.conftest import _make_case, _make_spec
from workflows.browser import locales, questions, spec as sp


def _build(**overrides: object) -> list[object]:
    """输入：case 覆盖项。输出：这道题的题目列表（中文、第一套句面）。"""
    spec = _make_spec(**overrides)
    return questions.build(spec.rows, spec.task, spec.object, locales.load("zh"), spec.phrasing)


def test_families_for_follows_the_completion_rule() -> None:
    """输入：没做完 / 做完两种 case。输出：无。

    预期行为：做完时只出完成度题；没做完时四族齐全。这条必须与 `build` 一一对应。
    """
    for done, want in (
        (False, ("action", "done", "risk", "target")),
        (True, ("done",)),
    ):
        spec = _make_spec(done=done)
        assert questions.families_for(spec.rows, spec.task, spec.object) == want
        assert tuple(sorted(item.family for item in _build(done=done))) == want


def test_done_question_is_always_asked() -> None:
    """输入：两种完成档。输出：无。预期行为：`done` 那一族总是出，答案就是真值。"""
    for done in (False, True):
        items = _build(done=done)
        assert items[0].family == "done"
        assert items[0].question["type"] == "noul"
        assert items[0].answer is done
        assert set(items[0].question["criteria"]) == {"true", "false"}


def test_target_answer_points_at_the_operable_row() -> None:
    """输入：全部 480 个 case。输出：无。

    预期行为：定位题的答案就是目标槽的位置，而且那个位置的选项描述里写着目标那一行的三格。
    """
    loc = locales.load("zh")
    for case in sp.build_cases(sp.ScreenConfig()):
        spec = sp.sample_spec(case, "train", 0, 0, None)
        if spec.done:
            continue
        items = questions.build(spec.rows, spec.task, spec.object, loc, 0)
        target = next(item for item in items if item.family == "target")
        index = int(str(target.answer)[1:])
        row = spec.rows[index - 1]
        description = target.question["criteria"][str(target.answer)]
        assert loc["object"][row.object] in description
        assert loc["state"][row.state] in description
        assert set(target.question["criteria"]) == {f"p{n}" for n in range(1, 6)}


def test_action_answer_follows_the_task() -> None:
    """输入：全部 480 个 case。输出：无。预期行为：动作题的答案就是任务意图表里的那一档。"""
    loc = locales.load("zh")
    for case in sp.build_cases(sp.ScreenConfig()):
        spec = sp.sample_spec(case, "train", 0, 0, None)
        if spec.done:
            continue
        items = questions.build(spec.rows, spec.task, spec.object, loc, 0)
        action = next(item for item in items if item.family == "action")
        assert action.answer == sp.action_of(case.task)
        assert set(action.question["criteria"]) == set(sp.ACTIONS)


def test_risk_answer_follows_the_sensitive_list() -> None:
    """输入：全部 480 个 case。输出：无。预期行为：风险题答案 = 对象是不是命中敏感清单。"""
    loc = locales.load("zh")
    for case in sp.build_cases(sp.ScreenConfig()):
        spec = sp.sample_spec(case, "train", 0, 0, None)
        if spec.done:
            continue
        items = questions.build(spec.rows, spec.task, spec.object, loc, 0)
        risk = next(item for item in items if item.family == "risk")
        assert risk.answer is sp.is_sensitive(spec.object)
        assert risk.answer is (case.form == "sensitive")


def test_answer_distributions_are_built_in() -> None:
    """输入：全开网格。输出：无。

    预期行为：四个题族的答案分布**全靠构造**——完成度 50/50、风险 50/50、动作四档各 1/4、
    定位五个位置各 1/5。均衡闸不靠事后过滤，这条是它的依据。
    """
    loc = locales.load("zh")
    counts: dict[str, Counter[object]] = {family: Counter() for family in questions.FAMILIES}
    for case in sp.build_cases(sp.ScreenConfig()):
        spec = sp.sample_spec(case, "train", 0, 0, None)
        for item in questions.build(spec.rows, spec.task, spec.object, loc, 0):
            counts[item.family][item.answer] += 1
    assert counts["done"] == Counter({True: 240, False: 240})
    assert counts["risk"] == Counter({True: 120, False: 120})
    assert counts["action"] == Counter({name: 60 for name in sp.ACTIONS})
    assert counts["target"] == Counter({f"p{n}": 48 for n in range(1, 6)})


def test_phrasing_only_changes_the_wording() -> None:
    """输入：同一 case 的两套句面。输出：无。

    预期行为：题号与答案逐题一致，只有题面文字不同——句面因此可以整档留出做 OOD。
    """
    loc = locales.load("zh")
    spec = _make_spec()
    first = questions.build(spec.rows, spec.task, spec.object, loc, 0)
    second = questions.build(spec.rows, spec.task, spec.object, loc, 1)
    assert [(item.family, item.answer) for item in first] == [
        (item.family, item.answer) for item in second
    ]
    assert first[0].question["instructions"] != second[0].question["instructions"]


def test_instructions_carry_the_whole_rulebook() -> None:
    """输入：四道题。输出：无。

    预期行为：判据整段写在题面里（范式 §9.8）——只读 state 与题面就该能算出来，
    所以每条题面里都得有那份 `policy`。
    """
    loc = locales.load("zh")
    for item in _build():
        assert loc["policy"]["text"] in item.question["instructions"]


def test_fill_reports_leftover_placeholders() -> None:
    """输入：一个少填槽的模板。输出：无。预期行为：报人话（不然发出去的是读不懂的题）。"""
    assert questions._fill("第 {n} 个", n="1") == "第 1 个"
    with pytest.raises(ValueError, match="占位符"):
        questions._fill("第 {n} 个：{role}", n="1")


def test_choice_rejects_duplicate_descriptions_and_unknown_answers() -> None:
    """输入：描述重复的选项 / 答案不在选项里。输出：无。预期行为：都报人话。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("target", "题面", {"p1": "一样", "p2": "一样"}, "p1")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("target", "题面", {"p1": "甲"}, "p2")


def test_build_rejects_a_wrong_number_of_rows() -> None:
    """输入：四行元素。输出：无。预期行为：报人话（少一行说明元素表没拼完）。"""
    spec = _make_spec()
    with pytest.raises(ValueError, match="5 行"):
        questions.build(spec.rows[:4], spec.task, spec.object, locales.load("zh"), 0)


def test_build_rejects_a_contradiction_between_the_two_rules() -> None:
    """输入：同一个对象挂两行（一行已是当前、一行可用）。输出：无。

    预期行为：报人话——这时「完成判据」与「可用判据」互相打架，静默出一套自相矛盾的题
    比直接报错危险得多。
    """
    spec = _make_spec()
    row = sp.Element(spec.rows[0].role, spec.object, "ready")
    broken = (sp.Element(spec.rows[0].role, spec.object, "current"), row, *spec.rows[2:])
    with pytest.raises(ValueError, match="打架"):
        questions.build(broken, spec.task, spec.object, locales.load("zh"), 0)


def test_unknown_task_fails_loudly() -> None:
    """输入：一个认不出的任务意图。输出：无。预期行为：查表报 KeyError，不静默兜底。"""
    spec = _make_spec()
    with pytest.raises(KeyError, match="task"):
        questions.build(spec.rows, "nope", spec.object, locales.load("zh"), 0)
