"""题族的测试：出题规则、答案推导、答案分布靠构造、判据写进题面。"""

from __future__ import annotations

from collections import Counter

import pytest

from tests.workflows.skillpick.conftest import _make_spec
from workflows.skillpick import locales, questions, spec as sp

LOC = locales.load("zh")


def _build(**overrides: object) -> tuple[sp.SkillSpec, list[object]]:
    """输入：case 覆盖项。输出：(规格, 题目列表)。"""
    spec = _make_spec(**overrides)
    return spec, questions.build(spec.case.tag, spec.case.grant, spec.inuse_tag, spec.rows, LOC, 0)


def test_families_for_matches_build() -> None:
    """输入：全开网格。输出：无。

    预期行为：`families_for` 点出来的题族与 `build` 真出的题族一一对应——审计靠它算
    「应有题族」，两处不一致就会误报缺题。
    """
    for case in sp.build_cases(sp.SkillConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        built = questions.build(
            spec.case.tag, spec.case.grant, spec.inuse_tag, spec.rows, LOC, 0
        )
        assert set(questions.families_for(spec.case.tag, spec.rows)) == {
            item.family for item in built
        }


def test_fits_is_always_asked_and_answers_the_match() -> None:
    """输入：全开网格。输出：无。

    预期行为：`fits` 一屏一道、总是出，答案就是「表里有没有完全一致的适用场景」。
    """
    for case in sp.build_cases(sp.SkillConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        items = questions.build(
            spec.case.tag, spec.case.grant, spec.inuse_tag, spec.rows, LOC, 0
        )
        fits = [item for item in items if item.family == "fits"]
        assert len(fits) == 1
        assert fits[0].answer is (sp.entry_of(case.tag) is not None)
        assert fits[0].question["type"] == "noul"
        assert set(fits[0].question["criteria"]) == {"true", "false"}


def test_pick_points_at_the_matching_slot() -> None:
    """输入：一个 case。输出：无。

    预期行为：`pick` 的答案是匹配到的那一行（选项名 `s<位置>`），选项描述里带上那一行的
    三格与位置号，所以六个选项天然互不相同。
    """
    spec, items = _build()
    pick = [item for item in items if item.family == "pick"]
    assert len(pick) == 1
    assert pick[0].answer == f"s{spec.target_slot}"
    assert set(pick[0].question["criteria"]) == {f"s{n}" for n in range(1, 7)}
    assert pick[0].shuffle is False


def test_scope_reads_the_scope_cell_of_that_row() -> None:
    """输入：一个 case。输出：无。

    预期行为：`scope` 的答案是**技能表里那一格的键**，选项描述是三档范围的说明；
    它不是从条目键反查表算出来的（所以「渲染把权限格写歪」会被判别器抓住）。
    """
    spec, items = _build()
    scope = [item for item in items if item.family == "scope"]
    assert len(scope) == 1
    assert scope[0].answer == spec.rows[spec.target_slot - 1].scope
    assert set(scope[0].question["criteria"]) == set(sp.SCOPES)
    assert scope[0].shuffle is False


def test_allow_and_conflict_are_noul_with_the_derived_truth() -> None:
    """输入：一个 case。输出：无。

    预期行为：`allow` 比权限档位、`conflict` 比资源格，两者都只吃技能表里的那几格；
    题面里的判据要带上真 / 假两行文案。
    """
    spec, items = _build(tag="ship_release", grant="write")
    allow = next(item for item in items if item.family == "allow")
    conflict = next(item for item in items if item.family == "conflict")
    assert allow.answer is spec.allowed is False
    assert conflict.answer is spec.conflicts
    for item in (allow, conflict):
        assert item.question["type"] == "noul"
        assert set(item.question["criteria"]) == {"true", "false"}
        assert item.shuffle is False


def test_only_fits_is_asked_when_nothing_matches() -> None:
    """输入：一个干扰标签。输出：无。

    预期行为：没有对得上的技能时「选哪个 / 什么权限 / 给不给 / 撞不撞」都无从谈起，
    只出 `fits` 一道。
    """
    spec, items = _build(tag="check_data_src")
    assert [item.family for item in items] == ["fits"]
    assert items[0].answer is False


def test_slot_options_are_pairwise_distinct() -> None:
    """输入：全开网格。输出：无。

    预期行为：位置选项的描述两两不同（位置号保证）——两行三格一样时也不会撞车。
    """
    for case in sp.build_cases(sp.SkillConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        options = questions._slot_options(spec.rows, LOC)
        assert len(set(options.values())) == len(sp.CATALOG)


def test_answer_distributions_are_built_in() -> None:
    """输入：全开网格。输出：无。

    预期行为：五个题族的答案分布**全靠构造**——有没有能用的 50/50、选哪个六个位置各 1/6、
    什么权限三档各 1/3、给不给用 50/50（只读技能两种授权都给、可写技能只有可写授权给、
    管理级两种都不给）、撞不撞 50/50（上下邻居里恰好一个同资源）。均衡闸不靠事后过滤，
    这条是它的依据。
    """
    total = len(sp.build_cases(sp.SkillConfig()))
    counts: dict[str, Counter[object]] = {family: Counter() for family in questions.FAMILIES}
    for case in sp.build_cases(sp.SkillConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        for item in questions.build(
            spec.case.tag, spec.case.grant, spec.inuse_tag, spec.rows, LOC, 0
        ):
            counts[item.family][item.answer] += 1
    assert counts["fits"] == Counter({True: total // 2, False: total // 2})
    assert counts["pick"] == Counter({f"s{n}": total // 12 for n in range(1, 7)})
    assert counts["scope"] == Counter({scope: total // 6 for scope in sp.SCOPES})
    assert counts["allow"] == Counter({True: total // 4, False: total // 4})
    assert counts["conflict"] == Counter({True: total // 4, False: total // 4})


def test_phrasing_only_changes_the_wording() -> None:
    """输入：同一 case 的两套句面。输出：无。

    预期行为：题号与答案逐题一致，只有题面文字不同——句面因此可以整档留出做 OOD。
    """
    spec = _make_spec()
    first = questions.build(spec.case.tag, spec.case.grant, spec.inuse_tag, spec.rows, LOC, 0)
    second = questions.build(spec.case.tag, spec.case.grant, spec.inuse_tag, spec.rows, LOC, 1)
    assert [(item.family, item.answer) for item in first] == [
        (item.family, item.answer) for item in second
    ]
    assert first[0].question["instructions"] != second[0].question["instructions"]


def test_instructions_carry_the_whole_rulebook() -> None:
    """输入：五道题。输出：无。

    预期行为：判据整段写在题面里（范式 §9.8）——只读 state 与题面就该能算出来，
    所以每条题面里都得有那份 `policy`。
    """
    for item in _build()[1]:
        assert LOC["policy"]["text"] in item.question["instructions"]


def test_fill_reports_leftover_placeholders() -> None:
    """输入：一个少填槽的模板。输出：无。预期行为：报人话（不然发出去的是读不懂的题）。"""
    assert questions._fill("第 {n} 个", n="1") == "第 1 个"
    with pytest.raises(ValueError, match="占位符"):
        questions._fill("第 {n} 个：{tag}", n="1")


def test_choice_rejects_duplicate_descriptions_and_unknown_answers() -> None:
    """输入：描述重复的选项 / 答案不在选项里。输出：无。预期行为：都报人话。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("pick", "题面", {"s1": "一样", "s2": "一样"}, "s1")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("pick", "题面", {"s1": "甲"}, "s2")


def test_build_rejects_a_wrong_number_of_rows() -> None:
    """输入：五行技能。输出：无。预期行为：报人话（少一行说明技能表没拼完）。"""
    spec = _make_spec()
    with pytest.raises(ValueError, match="6 行"):
        questions.build(spec.case.tag, spec.case.grant, spec.inuse_tag, spec.rows[:5], LOC, 0)


def test_index_of_reports_a_missing_row() -> None:
    """输入：一个不在表里的已占用标签。输出：无。

    预期行为：报人话——静默取第 0 行会让冲突判据跟着掷骰子。
    """
    spec = _make_spec()
    with pytest.raises(ValueError, match="已占用"):
        questions._index_of(spec.rows, "check_data_src", "已占用")
