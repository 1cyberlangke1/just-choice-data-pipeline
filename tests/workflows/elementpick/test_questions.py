"""题族的测试：出题规则、答案推导、答案分布靠构造、判据写进题面。"""

from __future__ import annotations

from collections import Counter

import pytest

from tests.workflows.elementpick.conftest import _make_spec
from workflows.elementpick import locales, questions, spec as sp

LOC = locales.load("zh")


def _build(**overrides: object) -> tuple[sp.ElementSpec, list[object]]:
    """输入：case 覆盖项。输出：(规格, 题目列表)。"""
    spec = _make_spec(**overrides)
    return spec, questions.build(spec.rows, spec.case.label, LOC, 0)


def test_families_for_matches_build() -> None:
    """输入：全开网格。输出：无。

    预期行为：`families_for` 点出来的题族与 `build` 真出的题族一一对应——审计靠它算
    「应有题族」，两处不一致就会误报缺题。
    """
    for case in sp.build_cases(sp.ElementConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        built = questions.build(spec.rows, spec.case.label, LOC, 0)
        assert set(questions.families_for(spec.rows, spec.case.label)) == {
            item.family for item in built
        }


def test_found_and_shadow_are_always_asked() -> None:
    """输入：全开网格。输出：无。

    预期行为：`found` 与 `shadow` 一屏各一道、总是出，答案就是「有没有落点」
    「有没有近似元素」。
    """
    for case in sp.build_cases(sp.ElementConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        items = questions.build(spec.rows, spec.case.label, LOC, 0)
        found = [item for item in items if item.family == "found"]
        shadow = [item for item in items if item.family == "shadow"]
        assert len(found) == 1 and len(shadow) == 1
        assert found[0].answer is (case.shape == "hit")
        assert shadow[0].answer is case.shadow
        for item in (found[0], shadow[0]):
            assert item.question["type"] == "noul"
            assert set(item.question["criteria"]) == {"true", "false"}


def test_target_points_at_the_landing_row() -> None:
    """输入：一个 case。输出：无。

    预期行为：`target` 的答案是落点那一行（选项名 `s<位置>`），选项描述里带上那一行的
    四格与位置号，所以六个选项天然互不相同。
    """
    spec, items = _build(extras=2, shadow=True)
    target = [item for item in items if item.family == "target"]
    assert len(target) == 1
    assert target[0].answer == f"s{spec.target_slot}"
    assert set(target[0].question["criteria"]) == {f"s{n}" for n in range(1, 7)}
    assert target[0].shuffle is False


def test_role_reads_the_class_of_the_landing_row() -> None:
    """输入：全开网格里的每个 `hit` case。输出：无。

    预期行为：`role` 的答案是**清单里那一行角色格**算出来的类，不是从 case 反查的
    （所以「渲染把角色格写歪」会被判别器抓住）。
    """
    for case in sp.build_cases(sp.ElementConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        if spec.target_slot is None:
            continue
        items = questions.build(spec.rows, spec.case.label, LOC, 0)
        role = next(item for item in items if item.family == "role")
        landing = spec.rows[spec.target_slot - 1]
        assert role.answer == sp.role_class_of(landing.role)
        assert set(role.question["criteria"]) == set(sp.ROLE_CLASSES)
        assert role.shuffle is False


def test_decoy_counts_the_blocked_same_name_rows() -> None:
    """输入：全开网格里的每个 `hit` case。输出：无。

    预期行为：`decoy` 是 score 题，答案是「被挡住的同名元素条数」，档位数组与 0~2 一一对应。
    """
    for case in sp.build_cases(sp.ElementConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        if spec.target_slot is None:
            continue
        items = questions.build(spec.rows, spec.case.label, LOC, 0)
        decoy = next(item for item in items if item.family == "decoy")
        assert decoy.question["type"] == "score"
        assert len(decoy.question["criteria"]) == len(sp.EXTRA_COUNTS)
        assert decoy.answer == spec.blocked == case.extras
        assert decoy.shuffle is False


def test_only_found_and_shadow_are_asked_when_there_is_no_landing() -> None:
    """输入：一个 `miss` case。输出：无。

    预期行为：没有落点时「第几个 / 哪一类 / 挡住几条」都无从谈起，只出两族。
    """
    spec, items = _build(shape="miss")
    assert [item.family for item in items] == ["found", "shadow"]
    assert items[0].answer is False


def test_slot_options_are_pairwise_distinct() -> None:
    """输入：全开网格。输出：无。

    预期行为：位置选项的描述两两不同（位置号保证）——六行四格一模一样时也不会撞车。
    """
    for case in sp.build_cases(sp.ElementConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        options = questions._slot_options(spec.rows, LOC)
        assert len(set(options.values())) == sp.SIZE


def test_answer_distributions_are_built_in() -> None:
    """输入：全开网格。输出：无。

    预期行为：五个题族的答案分布**全靠构造**——有没有落点 50/50、有没有近似 50/50、
    落点六个位置各 1/6、角色三档各 1/3、挡住的条数三档各 1/3。均衡闸不靠事后过滤，
    这条是它的依据。
    """
    cases = sp.build_cases(sp.ElementConfig())
    total = len(cases)
    counts: dict[str, Counter[object]] = {family: Counter() for family in questions.FAMILIES}
    for case in cases:
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        for item in questions.build(spec.rows, spec.case.label, LOC, 0):
            counts[item.family][item.answer] += 1
    landing = total // 2
    assert counts["found"] == Counter({True: landing, False: landing})
    assert counts["shadow"] == Counter({True: landing, False: landing})
    assert counts["target"] == Counter({f"s{n}": landing // sp.SIZE for n in range(1, 7)})
    assert counts["role"] == Counter({name: landing // 3 for name in sp.ROLE_CLASSES})
    assert counts["decoy"] == Counter({count: landing // 3 for count in sp.EXTRA_COUNTS})


def test_phrasing_only_changes_the_wording() -> None:
    """输入：同一 case 的两套句面。输出：无。

    预期行为：题号与答案逐题一致，只有题面文字不同——句面因此可以整档留出做 OOD。
    """
    spec = _make_spec()
    first = questions.build(spec.rows, spec.case.label, LOC, 0)
    second = questions.build(spec.rows, spec.case.label, LOC, 1)
    assert [(item.family, item.answer) for item in first] == [
        (item.family, item.answer) for item in second
    ]
    assert first[0].question["instructions"] != second[0].question["instructions"]


def test_instructions_carry_the_whole_rulebook() -> None:
    """输入：五道题。输出：无。

    预期行为：判据整段写在题面里（范式 §9.8）——只读 state 与题面就该能算出来，
    所以每条题面里都得有那份 `policy`。
    """
    for item in _build(extras=1, shadow=True)[1]:
        assert LOC["policy"]["text"] in item.question["instructions"]


def test_fill_reports_leftover_placeholders() -> None:
    """输入：一个少填槽的模板。输出：无。预期行为：报人话（不然发出去的是读不懂的题）。"""
    assert questions._fill("第 {n} 个", n="1") == "第 1 个"
    with pytest.raises(ValueError, match="占位符"):
        questions._fill("第 {n} 个：{role}", n="1")


def test_choice_rejects_duplicate_descriptions_and_unknown_answers() -> None:
    """输入：描述重复的选项 / 答案不在选项里。输出：无。预期行为：都报人话。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("target", "题面", {"s1": "一样", "s2": "一样"}, "s1")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("target", "题面", {"s1": "甲"}, "s2")


def test_build_rejects_a_wrong_number_of_rows() -> None:
    """输入：五行元素。输出：无。预期行为：报人话（少一行说明清单没拼完）。"""
    spec = _make_spec()
    with pytest.raises(ValueError, match="6 行"):
        questions.build(spec.rows[:5], spec.case.label, LOC, 0)
