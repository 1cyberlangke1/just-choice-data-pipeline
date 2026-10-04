"""题族的测试：出题规则、答案推导、答案分布靠构造、判据写进题面。"""

from __future__ import annotations

from collections import Counter

import pytest

from tests.workflows.bench.conftest import _make_spec
from workflows.bench import locales, questions
from workflows.bench import spec as sp

LOC = locales.load("zh")
#: 一屏记录在固定句面下的题量（第 1 个候选不是最好时多一道 `gap`）。
BASE_COUNT = sp.TOTAL + 3


def _build(**overrides: object) -> tuple[sp.BenchSpec, list[object]]:
    """输入：case 覆盖项。输出：(规格, 题目列表)。"""
    spec = _make_spec(**overrides)
    return spec, questions.build(spec.rows, spec, LOC, 0)


def test_families_for_matches_build() -> None:
    """输入：全开网格。输出：无。

    预期行为：`families_for` 点出来的题族与 `build` 真出的题族一一对应——审计靠它算
    「应有题族」，两处不一致就会误报缺题（`gap` 是最容易写歪的那一个）。
    """
    for case in sp.build_cases(sp.BenchConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        built = questions.build(spec.rows, spec, LOC, 0)
        assert set(questions.families_for(spec.rows, spec)) == {item.family for item in built}


def test_hit_is_asked_for_every_item() -> None:
    """输入：全开网格。输出：无。

    预期行为：`hit` 一屏五道、题号与行序一致，答案就是「第 1 列那一格的字母与参考字母相同」。
    """
    for case in sp.build_cases(sp.BenchConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        items = questions.build(spec.rows, spec, LOC, 0)
        hits = [item for item in items if item.family == "hit"]
        assert len(hits) == sp.TOTAL
        for index, (row, item) in enumerate(zip(spec.rows, hits, strict=True), start=1):
            assert item.answer is sp.correct_of_cell(row, 1)
            assert item.question["type"] == "noul"
            assert set(item.question["criteria"]) == {"true", "false"}
            assert item.shuffle is False
            assert f"第 {index} 题" in item.question["instructions"]


def test_count_and_pass_read_the_first_column() -> None:
    """输入：全开网格。输出：无。

    预期行为：`count` 是 score 题、等级数组就是六档、答案是第 1 列的答对条数；`pass` 是
    noul、答案是「条数不低于门槛」。
    """
    for case in sp.build_cases(sp.BenchConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        items = {item.family: item for item in questions.build(spec.rows, spec, LOC, 0)}
        count = items["count"]
        assert count.question["type"] == "score"
        assert len(count.question["criteria"]) == len(sp.COUNTS)
        assert count.answer == sp.count_of_column(spec.rows, 1)
        assert count.shuffle is False
        assert items["pass"].answer is sp.pass_of_column(spec.rows, 1, spec.bar)
        assert items["pass"].question["type"] == "noul"
        assert "{n}" not in items["pass"].question["instructions"]


def test_best_is_a_choice_over_the_columns() -> None:
    """输入：全开网格。输出：无。

    预期行为：`best` 是 choice 3、答案是列号选项名，而且选项描述里带着那一列的候选名。
    """
    for case in sp.build_cases(sp.BenchConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        items = {item.family: item for item in questions.build(spec.rows, spec, LOC, 0)}
        best = items["best"]
        assert best.question["type"] == "choice"
        assert set(best.question["criteria"]) == {f"c{n}" for n in range(1, 4)}
        assert best.answer == f"c{sp.best_column_of_rows(spec.rows)}"
        assert best.shuffle is False
        for column, cell in enumerate(spec.rows[0].cells, start=1):
            assert str(LOC["cand"][cell.cand]) in str(best.question["criteria"][f"c{column}"])


def test_gap_only_shows_up_when_the_first_column_is_behind() -> None:
    """输入：全开网格。输出：无。

    预期行为：`gap` 是 score 2 档、答案是「最好的比第 1 列多几道」的档位；**只在第 1 列
    不是最好时出**——第 1 列已经领先时没有「落后」可言。
    """
    shown = 0
    for case in sp.build_cases(sp.BenchConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        items = questions.build(spec.rows, spec, LOC, 0)
        families = [item.family for item in items]
        gap = [item for item in items if item.family == "gap"]
        if sp.gap_level_of_rows(spec.rows) is None:
            assert gap == []
            assert families == ["hit"] * sp.TOTAL + ["count", "pass", "best"]
            assert len(items) == BASE_COUNT
            continue
        shown += 1
        assert len(gap) == 1
        assert gap[0].question["type"] == "score"
        assert len(gap[0].question["criteria"]) == len(sp.GAP_VALUES)
        assert gap[0].answer == sp.gap_level_of_rows(spec.rows)
        assert gap[0].shuffle is False
    assert shown > 0


def test_phrasing_only_changes_the_wording() -> None:
    """输入：同一屏记录、两套句面。输出：无。

    预期行为：题族、答案、选项**完全一样**，只有问句措辞不同——句面是留出维，但留出它
    不会把任何一族的分布压歪。
    """
    spec = _make_spec(layout="l2")
    first = questions.build(spec.rows, spec, LOC, 0)
    second = questions.build(spec.rows, spec, LOC, 1)
    assert [item.family for item in first] == [item.family for item in second]
    assert [item.answer for item in first] == [item.answer for item in second]
    assert [item.question["criteria"] for item in first] == [
        item.question["criteria"] for item in second
    ]
    assert any(
        left.question["instructions"] != right.question["instructions"]
        for left, right in zip(first, second, strict=True)
    )


def test_answer_distributions_are_built_in() -> None:
    """输入：全开网格（三十张成绩表 × 六种列序，句面固定）。输出：无。

    预期行为：五族的答案分布**构造出来就是均匀的**——逐题答对一半、答对条数六档各占六分之一、
    达标一半、最好三档各占三分之一、差距两档各占一半。
    """
    hit: Counter[bool] = Counter()
    count: Counter[int] = Counter()
    passes: Counter[bool] = Counter()
    best: Counter[str] = Counter()
    gap: Counter[int] = Counter()
    for table in sp.TABLES:
        for layout in sp.LAYOUTS:
            spec = sp.sample_spec(sp.BenchCase(table, layout, 0), "train", 0, 0, None)
            for item in questions.build(spec.rows, spec, LOC, 0):
                if item.family == "hit":
                    hit[item.answer] += 1
                elif item.family == "count":
                    count[item.answer] += 1
                elif item.family == "pass":
                    passes[item.answer] += 1
                elif item.family == "best":
                    best[item.answer] += 1
                else:
                    gap[item.answer] += 1
    cells = len(sp.TABLES) * len(sp.LAYOUTS)
    assert hit[True] == hit[False] == cells * sp.TOTAL // 2
    assert set(count.values()) == {cells // len(sp.COUNTS)} and set(count) == set(sp.COUNTS)
    assert passes[True] == passes[False] == cells // 2
    assert set(best.values()) == {cells // len(sp.CANDIDATES)}
    assert set(gap.values()) == {cells * 2 // len(sp.CANDIDATES) // len(sp.GAP_VALUES)}
    assert set(gap) == set(range(len(sp.GAP_VALUES)))


def test_cand_options_are_pairwise_distinct() -> None:
    """输入：全开网格。输出：无。

    预期行为：候选选项的描述两两不同（列号保证）——三个候选名不同时也不会撞车。
    """
    for case in sp.build_cases(sp.BenchConfig()):
        spec = sp.sample_spec(case, "train", 0, case.phrasing, None)
        options = questions._cand_options(spec.rows, LOC)
        assert len(set(options.values())) == len(sp.CANDIDATES)
        assert set(options) == {f"c{n}" for n in range(1, 4)}


def test_fill_reports_leftover_placeholders() -> None:
    """输入：一个没填完的模板。输出：无。预期行为：报人话（题面少填一个槽比报错危险）。"""
    with pytest.raises(ValueError, match="占位符"):
        questions._fill("第 {n} 题，第 1 个候选答对了吗？", other="x")


def test_choice_rejects_duplicate_descriptions_and_unknown_answers() -> None:
    """输入：两个描述一样的选项 / 答案不在选项里。输出：无。预期行为：都报人话。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("best", "题面", {"a": "同一个", "b": "同一个"}, "a")
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("best", "题面", {"a": "第 1 个", "b": "第 2 个"}, "c")


def test_build_rejects_a_wrong_number_of_rows() -> None:
    """输入：四行记录。输出：无。预期行为：报人话（少一行说明构造漏了一行）。"""
    spec = _make_spec()
    with pytest.raises(ValueError, match="记录应该是"):
        questions.build(spec.rows[:4], spec, LOC, 0)
