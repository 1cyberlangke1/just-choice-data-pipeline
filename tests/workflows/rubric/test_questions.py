"""题族的测试：出题规则、答案推导、答案分布靠构造、判据写进题面。"""

from __future__ import annotations

from collections import Counter

import pytest

from tests.workflows.rubric.conftest import _make_spec
from workflows.rubric import locales, questions, spec as sp

LOC = locales.load("zh")


def _build(**overrides: object) -> tuple[sp.RubricSpec, list[object]]:
    """输入：case 覆盖项。输出：(规格, 题目列表)。"""
    spec = _make_spec(**overrides)
    return spec, questions.build(spec.rows, LOC, 0)


def test_families_for_matches_build() -> None:
    """输入：全开网格。输出：无。

    预期行为：`families_for` 点出来的题族与 `build` 真出的题族一一对应——审计靠它算
    「应有题族」，两处不一致就会误报缺题（`top` 是最容易写歪的那个）。
    """
    for case in sp.build_cases(sp.RubricConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        built = questions.build(spec.rows, LOC, 0)
        assert set(questions.families_for(spec.rows)) == {item.family for item in built}


def test_met_is_asked_for_every_row() -> None:
    """输入：全开网格。输出：无。

    预期行为：`met` 一屏六道、条号与行序一致，答案就是「观测 ≥ 要求」。
    """
    for case in sp.build_cases(sp.RubricConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        items = questions.build(spec.rows, LOC, 0)
        met = [item for item in items if item.family == "met"]
        assert len(met) == len(sp.CRITERIA)
        for row, item in zip(spec.rows, met, strict=True):
            assert item.answer is sp.met_of(row.need, row.count)
            assert item.question["type"] == "noul"
            assert set(item.question["criteria"]) == {"true", "false"}
            assert item.shuffle is False


def test_met_asks_about_the_row_at_that_number() -> None:
    """输入：一份规格。输出：无。

    预期行为：第 n 道 `met` 问的就是**第 n 行**（不是判据表里的第 n 条）——排布换了之后
    行序也换，这条靠问句里的条号与行序对齐来保证。
    """
    spec = _make_spec(sheet="s0", layout="l4")
    items = [item for item in questions.build(spec.rows, LOC, 0) if item.family == "met"]
    for index, (row, item) in enumerate(zip(spec.rows, items, strict=True), start=1):
        assert f"第 {index} 条判据" in item.question["instructions"]
        assert item.answer is sp.met_of(row.need, row.count)


def test_band_reads_the_total_of_the_met_rows() -> None:
    """输入：全开网格。输出：无。

    预期行为：`band` 是 score 题、等级数组就是三档，答案下标与 `BANDS` 一一对应。
    """
    for case in sp.build_cases(sp.RubricConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        band = [item for item in questions.build(spec.rows, LOC, 0) if item.family == "band"]
        assert len(band) == 1
        assert band[0].question["type"] == "score"
        assert len(band[0].question["criteria"]) == len(sp.BANDS)
        assert band[0].answer == sp.BANDS.index(spec.band)
        assert band[0].shuffle is False


def test_gate_reads_the_disposition_table() -> None:
    """输入：全开网格。输出：无。

    预期行为：`gate` 是 choice 3、答案是处置键（选项名），选项描述就是三个处置。
    """
    for case in sp.build_cases(sp.RubricConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        gate = [item for item in questions.build(spec.rows, LOC, 0) if item.family == "gate"]
        assert len(gate) == 1
        assert set(gate[0].question["criteria"]) == set(sp.GATES)
        assert gate[0].answer == spec.gate
        assert gate[0].shuffle is False


def test_top_is_asked_only_when_the_worst_is_unique() -> None:
    """输入：并列的一张判据表 / 不并列的一张。输出：无。

    预期行为：并列时一条 `top` 都不出（「最严重的是哪一条」没有唯一解），不并列时出且
    答案是那个位置；选项名 `p<位置>` 与 `spec.top_slot` 对得上。
    """
    for sheet in sp.SHEETS:
        for layout in sp.LAYOUTS:
            spec = sp.sample_spec(sp.RubricCase(sheet, layout, "tight", 0), "train", 0, 0, None)
            items = questions.build(spec.rows, LOC, 0)
            top = [item for item in items if item.family == "top"]
            if spec.top_slot is None:
                assert top == []
                continue
            assert len(top) == 1
            assert top[0].answer == f"p{spec.top_slot}"
            assert set(top[0].question["criteria"]) == {f"p{n}" for n in range(1, 7)}
            assert top[0].shuffle is False


def test_a_tied_sheet_asks_three_families_only() -> None:
    """输入：`s2`（最严重的那一条永远并列）。输出：无。

    预期行为：只出 `met` × 6 + `band` + `gate` 八道，`top` 一道都不出。
    """
    spec, items = _build(sheet="s2")
    assert [item.family for item in items] == ["met"] * 6 + ["band", "gate"]
    assert spec.top_slot is None
    assert questions.families_for(spec.rows) == ("band", "gate", "met")


def test_slot_options_are_pairwise_distinct() -> None:
    """输入：全开网格。输出：无。

    预期行为：条号选项的描述两两不同（条号保证）——两行四格一样时也不会撞车。
    """
    for case in sp.build_cases(sp.RubricConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        options = questions._slot_options(spec.rows, LOC)
        assert len(set(options.values())) == len(sp.CRITERIA)


def test_answer_distributions_are_built_in() -> None:
    """输入：全开网格（每张表 × 六种排布，松紧与句面固定）。输出：无。

    预期行为：四条族的答案分布**构造出来就是均匀的**——`met` 每条判据达标九次（合计
    50/50）、三档各六张、三个处置各六张、`top` 的位置由拉丁方铺满。
    """
    met = Counter()
    band = Counter()
    gate = Counter()
    top = Counter()
    for case in sp.build_cases(sp.RubricConfig(slacks=("tight",), phrasings=(0,))):
        spec = sp.sample_spec(case, "train", 0, 0, None)
        for item in questions.build(spec.rows, LOC, 0):
            if item.family == "met":
                met[item.answer] += 1
            elif item.family == "band":
                band[item.answer] += 1
            elif item.family == "gate":
                gate[item.answer] += 1
            else:
                top[item.answer] += 1
    assert met[True] == met[False] == 18 * 6 * 3
    assert set(band.values()) == {18 * 6 // 3}
    assert set(gate.values()) == {18 * 6 // 3}
    assert set(top.values()) == {12 * 6 // 6}


def test_phrasing_only_changes_the_wording() -> None:
    """输入：同一张评分单、两套句面。输出：无。

    预期行为：题族、答案、选项**完全一样**，只有问句措辞不同——句面是留出维，但留出它
    不会把任何一族的分布压歪。
    """
    first = questions.build(_make_spec(sheet="s0", layout="l2").rows, LOC, 0)
    second = questions.build(_make_spec(sheet="s0", layout="l2").rows, LOC, 1)
    assert [item.family for item in first] == [item.family for item in second]
    assert [item.answer for item in first] == [item.answer for item in second]
    assert [item.question["criteria"] for item in first] == [
        item.question["criteria"] for item in second
    ]
    assert any(
        left.question["instructions"] != right.question["instructions"]
        for left, right in zip(first, second, strict=True)
    )


def test_instructions_carry_the_whole_rulebook() -> None:
    """输入：一份规格出的全部题目。输出：无。

    预期行为：判据原文整段拼在每道题的问句后面（范式 §9.8）——只读 state 与题面的读者
    必须能自己算出来。
    """
    spec, items = _build()
    for item in items:
        instructions = item.question["instructions"]
        policy = str(LOC["policy"]["text"])
        assert instructions.endswith(policy)
        assert instructions != policy  # 判据前面真的带了问句
        first = str(LOC["fam"][item.family]["q"][0])
        if "{n}" not in first:  # 带条号的问句由 `test_met_asks_about_the_row_at_that_number` 锁
            assert instructions.startswith(first)


def test_fill_reports_leftover_placeholders() -> None:
    """输入：一个没填完的模板。输出：无。预期行为：报人话（题面少填一个槽比报错危险）。"""
    with pytest.raises(ValueError, match="占位符"):
        questions._fill("第 {n} 条判据达标了吗？", other="x")


def test_choice_rejects_duplicate_descriptions_and_unknown_answers() -> None:
    """输入：两个描述一样的选项 / 答案不在选项里。输出：无。预期行为：都报人话。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("gate", "题面", {"a": "同一个", "b": "同一个"}, "a")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("gate", "题面", {"a": "甲", "b": "乙"}, "c")


def test_build_rejects_a_wrong_number_of_rows() -> None:
    """输入：五行判据。输出：无。预期行为：报人话（少一行说明构造漏了一行）。"""
    rows = _make_spec().rows[:5]
    with pytest.raises(ValueError, match="评分单应该是"):
        questions.build(rows, LOC, 0)
