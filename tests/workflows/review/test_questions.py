"""题族的测试：出题规则、答案推导、答案分布靠构造、判据写进题面。"""

from __future__ import annotations

from collections import Counter

import pytest

from tests.workflows.review.conftest import _make_spec
from workflows.review import locales, questions, spec as sp

LOC = locales.load("zh")


def _build(**overrides: object) -> tuple[sp.ChangeSpec, list[object]]:
    """输入：case 覆盖项。输出：(规格, 题目列表)。"""
    spec = _make_spec(**overrides)
    return spec, questions.build(spec.rows, LOC, 0)


def test_families_for_matches_build() -> None:
    """输入：全开网格。输出：无。

    预期行为：`families_for` 点出来的题族与 `build` 真出的题族一一对应——
    审计靠它算「应有题族」，两处不一致就会误报缺题。
    """
    for case in sp.build_cases(sp.ReviewConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        built = questions.build(spec.rows, LOC, 0)
        assert set(questions.families_for(spec.rows)) == {item.family for item in built}


def test_flag_questions_are_six_and_follow_the_raise_table() -> None:
    """输入：全开网格。输出：无。预期行为：逐处 6 道，答案 = 这一处该不该提。"""
    for case in sp.build_cases(sp.ReviewConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        flags = [item for item in questions.build(spec.rows, LOC, 0) if item.family == "flag"]
        assert len(flags) == len(sp.SLOTS)
        assert [item.answer for item in flags] == [sp.raises_(row) for row in spec.rows]
        assert all(item.question["type"] == "noul" for item in flags)


def test_severity_and_gate_cover_only_the_raised_rows() -> None:
    """输入：一个 case。输出：无。

    预期行为：`severity` 与 `gate` 各出 3 道、正好对上该提的三处；答案分别是
    严重度档位下标（轻 0 / 中 1 / 重 2）与门控键。
    """
    spec, items = _build()
    severities = [item for item in items if item.family == "severity"]
    gates = [item for item in items if item.family == "gate"]
    assert len(severities) == len(gates) == len(spec.raised) == 3
    assert [item.answer for item in severities] == [
        sp.severity_of(spec.rows[index - 1]) - 1 for index in spec.raised
    ]
    assert [item.answer for item in gates] == [
        sp.gate_of(spec.rows[index - 1]) for index in spec.raised
    ]
    for item in severities:
        assert item.question["type"] == "score"
        assert item.question["criteria"] == list(LOC["fam"]["severity"]["levels"])
    for item in gates:
        assert item.question["type"] == "choice"
        assert set(item.question["criteria"]) == set(sp.GATES)


def test_top_question_points_at_the_worst_position() -> None:
    """输入：一个 case。输出：无。预期行为：`top` 出一道，答案是严重度最高的那一处。"""
    spec, items = _build()
    top = [item for item in items if item.family == "top"]
    assert len(top) == 1
    assert top[0].answer == f"p{spec.top_position}"
    assert set(top[0].question["criteria"]) == {
        f"p{n}" for n in range(1, len(sp.SLOTS) + 1)
    }


def test_position_options_are_pairwise_distinct() -> None:
    """输入：全开网格。输出：无。

    预期行为：位置选项的描述两两不同（处号保证）——两处改动抬头三格一样时也不会撞车。
    """
    for case in sp.build_cases(sp.ReviewConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        options = questions._position_options(spec.rows, LOC)
        assert len(set(options.values())) == len(sp.SLOTS)


def test_answer_distributions_are_built_in() -> None:
    """输入：全开网格。输出：无。

    预期行为：四个题族的答案分布**全靠构造**——该不该提 50/50、严重度三档各 1/3、
    门控三档各 1/3、最严重的位置六个位置各 1/6。均衡闸不靠事后过滤，这条是它的依据。
    """
    total = len(sp.build_cases(sp.ReviewConfig()))
    counts: dict[str, Counter[object]] = {family: Counter() for family in questions.FAMILIES}
    for case in sp.build_cases(sp.ReviewConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        for item in questions.build(spec.rows, LOC, 0):
            counts[item.family][item.answer] += 1
    assert counts["flag"] == Counter({True: 3 * total, False: 3 * total})
    assert counts["severity"] == Counter({band: total for band in (0, 1, 2)})
    assert counts["gate"] == Counter({name: total for name in sp.GATES})
    assert counts["top"] == Counter({f"p{n}": total // 6 for n in range(1, 7)})


def test_phrasing_only_changes_the_wording() -> None:
    """输入：同一 case 的两套句面。输出：无。

    预期行为：题号与答案逐题一致，只有题面文字不同——句面因此可以整档留出做 OOD。
    """
    spec = _make_spec()
    first = questions.build(spec.rows, LOC, 0)
    second = questions.build(spec.rows, LOC, 1)
    assert [(item.family, item.answer) for item in first] == [
        (item.family, item.answer) for item in second
    ]
    assert first[0].question["instructions"] != second[0].question["instructions"]


def test_instructions_carry_the_whole_rulebook() -> None:
    """输入：十三道题。输出：无。

    预期行为：判据整段写在题面里（范式 §9.8）——只读 state 与题面就该能算出来，
    所以每条题面里都得有那份 `policy`。
    """
    for item in _build()[1]:
        assert LOC["policy"]["text"] in item.question["instructions"]


def test_fill_reports_leftover_placeholders() -> None:
    """输入：一个少填槽的模板。输出：无。预期行为：报人话（不然发出去的是读不懂的题）。"""
    assert questions._fill("第 {n} 处", n="1") == "第 1 处"
    with pytest.raises(ValueError, match="占位符"):
        questions._fill("第 {n} 处：{file}", n="1")


def test_choice_rejects_duplicate_descriptions_and_unknown_answers() -> None:
    """输入：描述重复的选项 / 答案不在选项里。输出：无。预期行为：都报人话。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("gate", "题面", {"pass": "一样", "block": "一样"}, "pass")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("gate", "题面", {"pass": "甲"}, "block")


def test_build_rejects_a_wrong_number_of_rows() -> None:
    """输入：五处改动。输出：无。预期行为：报人话（少一处说明改动单没拼完）。"""
    spec = _make_spec()
    with pytest.raises(ValueError, match="6 处"):
        questions.build(spec.rows[:5], LOC, 0)


def test_noul_and_score_questions_are_not_shuffled() -> None:
    """输入：十三道题。输出：无。

    预期行为：noul 的判据是两行文案、score 的档位有自然顺序，都不许打乱——
    打乱会让答案下标失去意义。
    """
    for item in _build()[1]:
        if item.family in {"flag", "severity"}:
            assert item.shuffle is False
    flag = next(item for item in _build()[1] if item.family == "flag")
    assert set(flag.question["criteria"]) == {"true", "false"}
