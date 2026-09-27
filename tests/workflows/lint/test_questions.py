"""题族：出哪些题、选项与答案、以及条件题族（`spot`）。"""

from __future__ import annotations

from typing import Any

import pytest

from workflows.lint import locales, questions, render
from workflows.lint.spec import SIZE, LintCase, LintSpec, sample_spec

ZH = locales.load("zh")
EN = locales.load("en")


def build(rule: str, genre: str, hits: int, phrasing: int = 0, lang: str = "zh") -> list[Any]:
    """输入：四个维度 + 语言。输出：题目列表（走生产路径的渲染与出题）。"""
    loc = locales.load(lang)
    spec = sample_spec(LintCase(rule, genre, hits, phrasing), "train", 0)
    return questions.build(
        spec.plan.rule, render.render_lines(spec.plan, loc), spec.plan.matched(), loc, phrasing
    )


def test_families_for_marks_spot_as_conditional() -> None:
    """输入：各档命中行数。输出：无。

    预期行为：`spot` 只在恰好一行命中时算「应有题族」——多行命中时「哪一行命中」没有
    唯一解，出这种题就是坏数据。
    """
    assert questions.families_for(0) == ("count", "gate", "line")
    assert questions.families_for(1) == ("count", "gate", "line", "spot")
    assert questions.families_for(5) == ("count", "gate", "line")
    assert set(questions.FAMILIES) == {"count", "gate", "line", "spot"}


@pytest.mark.parametrize("hits", [0, 1, 3, 8])
def test_build_emits_the_expected_families(hits: int) -> None:
    """输入：各档命中行数。输出：无。

    预期行为：`line` 恒有 8 道（逐行），`count` / `gate` 各一道，`spot` 只在恰好一行命中时
    多出一道；题族集合与 `families_for` 完全一致。
    """
    built = build("todo", "code", hits)
    families = [item.family for item in built]
    assert families.count("line") == SIZE
    assert families.count("count") == 1
    assert families.count("gate") == 1
    assert families.count("spot") == (1 if hits == 1 else 0)
    assert set(families) == set(questions.families_for(hits))


def test_line_answers_follow_the_plan() -> None:
    """输入：命中 3 行的一条样本。输出：无。

    预期行为：逐行题的答案就是计划里的命中标记，且 True 的个数等于命中行数。
    """
    spec = sample_spec(LintCase("path", "code", 3, 0), "train", 0)
    built = questions.build(
        spec.plan.rule,
        render.render_lines(spec.plan, ZH),
        spec.plan.matched(),
        ZH,
        0,
    )
    lines = [item for item in built if item.family == "line"]
    assert [item.answer for item in lines] == list(spec.plan.matched())
    assert sum(1 for item in lines if item.answer) == 3
    assert [item.question["type"] for item in lines] == ["noul"] * SIZE


def test_line_instructions_carry_the_index_and_the_criteria() -> None:
    """输入：一条样本。输出：无。

    预期行为：第 k 道逐行题的题面里有行号 k，判据两行说清「命中 / 没命中」——考生只读
    state 与题面就能答（范式 §9.8）。
    """
    built = build("secret", "code", 1)
    lines = [item for item in built if item.family == "line"]
    assert "第 1 行" in lines[0].question["instructions"]
    assert "第 8 行" in lines[7].question["instructions"]
    criteria = lines[0].question["criteria"]
    assert set(criteria) == {"true", "false"}
    assert "第 1 行" in criteria["true"] and "第 1 行" in criteria["false"]


@pytest.mark.parametrize("hits", list(range(SIZE + 1)))
def test_count_question_has_all_nine_options(hits: int) -> None:
    """输入：0~8 全部命中档。输出：无。

    预期行为：选项恒为 9 个（c0~c8），答案是 c<命中行数>；描述互不相同。
    """
    built = [item for item in build("debug", "log", hits) if item.family == "count"]
    assert len(built) == 1
    item = built[0]
    options = item.question["criteria"]
    assert list(options) == [f"c{value}" for value in range(SIZE + 1)]
    assert len(set(options.values())) == SIZE + 1
    assert item.answer == f"c{hits}"


def test_gate_question_is_a_three_way_band() -> None:
    """输入：0 / 3 / 8 三档命中行数。输出：无。预期行为：分别落在放行 / 人工复核 / 拦下。"""
    answers = [item for hits in (0, 3, 8) for item in build("todo", "code", hits) if item.family == "gate"]
    assert [item.answer for item in answers] == ["pass", "review", "block"]
    options = answers[0].question["criteria"]
    assert list(options) == ["pass", "review", "block"]
    # 判据（处置办法）必须写在题面里，不能只留在生成器代码里
    assert "0~2" in answers[0].question["instructions"]
    assert "3~5" in answers[0].question["instructions"]


def test_spot_question_options_are_the_lines() -> None:
    """输入：恰好一行命中。输出：无。

    预期行为：选项 = 片段里的 8 行（描述写成「第 n 行：原文」，两两不同），答案是那一行的
    选项名，且描述里的原文就是命中行。
    """
    spec = sample_spec(LintCase("secret", "code", 1, 0), "train", 0)
    lines = render.render_lines(spec.plan, ZH)
    built = [
        item
        for item in questions.build("secret", lines, spec.plan.matched(), ZH, 0)
        if item.family == "spot"
    ]
    assert len(built) == 1
    item = built[0]
    options = item.question["criteria"]
    assert len(options) == SIZE
    assert len(set(options.values())) == SIZE
    index = spec.plan.matched().index(True)
    assert item.answer == f"L{index + 1}"
    assert lines[index] in options[item.answer]


def test_option_descriptions_are_never_empty() -> None:
    """输入：全部规则 × 全部体裁。输出：无。

    预期行为：每一道 choice 题的选项描述都是非空字符串——空描述是构造 bug，考生读不出
    选项之间有什么区别。
    """
    for rule in ZH["rule"]:
        for genre in ("code", "log", "doc"):
            for item in build(rule, genre, 1):
                if item.question["type"] != "choice":
                    continue
                assert all(str(text).strip() for text in item.question["criteria"].values())


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_build_works_in_both_locales(lang: str) -> None:
    """输入：两种语言。输出：无。预期行为：题数与答案一致（语言只换措辞，不换答案）。"""
    zh = build("path", "doc", 2, 1, "zh")
    en = build("path", "doc", 2, 1, "en")
    assert [item.answer for item in zh] == [item.answer for item in en]
    assert [item.question["type"] for item in zh] == [item.question["type"] for item in en]
    assert [item.family for item in zh] == [item.family for item in en]


def test_build_rejects_a_wrong_sized_snippet() -> None:
    """输入：行数不对的片段。输出：无。预期行为：报错（不静默按现有行数出题）。"""
    with pytest.raises(ValueError, match="必须是 8 行"):
        questions.build("secret", ("x",), (True,), ZH, 0)


def test_build_rejects_an_unknown_rule() -> None:
    """输入：语言文件里没有的规则名。输出：无。预期行为：报 KeyError。"""
    with pytest.raises(KeyError):
        questions.build("nope", ("x",) * SIZE, (False,) * SIZE, ZH, 0)


def test_choice_rejects_duplicate_descriptions() -> None:
    """输入：两个描述一样的选项。输出：无。预期行为：报错（考生没法分辨）。"""
    with pytest.raises(ValueError, match="选项描述重复"):
        questions._choice("count", "题面", {"a": "一样", "b": "一样"}, "a")


def test_choice_rejects_an_answer_outside_the_options() -> None:
    """输入：答案不在选项里。输出：无。预期行为：报错。"""
    with pytest.raises(ValueError, match="不在选项里"):
        questions._choice("count", "题面", {"a": "甲", "b": "乙"}, "c")


def test_noul_question_never_shuffles() -> None:
    """输入：一道 noul 题。输出：无。预期行为：`shuffle=False`（noul 的判据是有序的 true/false）。"""
    built = [item for item in build("todo", "code", 1) if item.family == "line"]
    assert all(item.shuffle is False for item in built)


def test_questions_carry_no_unfilled_placeholder() -> None:
    """输入：全部规则 × 全部体裁 × 两套句面。输出：无。

    预期行为：题面里不留 `{xxx}`——少填一个槽会变成一道读不懂的题，而答案照样发出去。
    """
    for rule in ZH["rule"]:
        for genre in ("code", "log", "doc"):
            for phrasing in (0, 1):
                for item in build(rule, genre, 1, phrasing):
                    text = str(item.question["instructions"])
                    assert "{" not in text and "}" not in text


def test_spec_type_is_the_one_judges_expect() -> None:
    """输入：`sample_spec` 的产物。输出：无。预期行为：类型就是 `LintSpec`（判别器按它判）。"""
    assert isinstance(sample_spec(LintCase("todo", "code", 1, 0), "train", 0), LintSpec)


def test_fill_rejects_a_leftover_placeholder() -> None:
    """输入：少给一个槽的模板。输出：无。

    预期行为：直接报错——题面少填一个槽会变成一道读不懂的题，而答案照样发出去，
    静默放过比报错危险得多。
    """
    with pytest.raises(ValueError, match="还剩占位符"):
        questions._fill("第 {n} 行还有 {extra}", n="1")
