"""语言文件测试：结构不变量（抬头、条款表、处置档位、占位符）逐条钉住。"""

from __future__ import annotations

import pytest

from workflows.guardrail import features, judges, locales, questions, render


def test_both_languages_exist() -> None:
    """输入：无。输出：无。预期行为：zh / en 都在（少一个就是漏分发）。"""
    assert locales.available() == ["en", "zh"]


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_marks_are_non_empty_and_pairwise_non_prefix(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：三个抬头两两不互为前缀——`【政策条款】` 与 `【政策条款附注】` 那种关系会让解析先
    命中短的那个，整段 state 当场错位。
    """
    loc = locales.load(lang)
    marks = [str(loc["state"][key]) for key in ("clause_mark", "rule_mark", "content_mark")]
    assert all(marks)
    assert len(set(marks)) == len(marks)
    for left in marks:
        for right in marks:
            if left != right:
                assert not right.startswith(left), f"{left!r} 是 {right!r} 的前缀"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_clause_table_lines_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：条款表的键与 `CLAUSE_IDS` 一一对应，每条四个字段都非空、条款名与表行两两不同，
    而且表行里**编号排在分隔符前面**（解析就是按这个切回编号的）。
    """
    loc = locales.load(lang)
    table = loc["clause"]
    assert isinstance(table, dict)
    assert sorted(table) == sorted(features.CLAUSE_IDS)
    for clause in features.CLAUSE_IDS:
        entry = render.clause_copy(loc, clause)
        for key in ("name", "rule", "example", "avoid"):
            assert str(entry[key]).strip(), f"{lang}.{clause}.{key}"
    names = [str(table[clause]["name"]) for clause in features.CLAUSE_IDS]
    assert len(set(names)) == len(names)
    sep = str(loc["state"]["clause_sep"])
    lines = [render.clause_line(loc, clause) for clause in features.CLAUSE_IDS]
    assert len(set(lines)) == len(lines)
    for clause, line in zip(features.CLAUSE_IDS, lines, strict=True):
        assert line.partition(sep)[0] == clause


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_examples_hit_exactly_their_own_clause(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：每条的示例片段只命中它自己那一条——示例同时踩中两条，提示词就会把模型往错的方向带。
    """
    loc = locales.load(lang)
    for clause in features.CLAUSE_IDS:
        example = str(render.clause_copy(loc, clause)["example"])
        assert features.hit_set(example) == (clause,), f"{lang}.{clause}: {example!r}"


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_action_and_threshold_tables_line_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。预期行为：两张表都覆盖四个档位，动作文案两两不同。"""
    loc = locales.load(lang)
    options = loc["fam"]["action"]["opt"]
    assert isinstance(options, dict)
    assert sorted(options) == sorted(features.ACTION_LEVELS)
    assert len({str(value) for value in options.values()}) == len(options)
    thresholds = loc["state"]["threshold"]
    assert isinstance(thresholds, dict)
    assert sorted(thresholds) == sorted(features.ACTION_LEVELS)
    assert len({str(value) for value in thresholds.values()}) == len(thresholds)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_every_family_has_questions_and_the_right_shape(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：四族的问法池非空；逐条款那一族有 true / false 判据；count 与 combo 的选项表齐全。
    """
    loc = locales.load(lang)
    for family in questions.FAMILIES:
        spec = loc["fam"][family]
        assert isinstance(spec["q"], list) and spec["q"], f"{family} 没有问法"
    assert str(loc["fam"]["clause"]["criteria_true"]).strip()
    assert str(loc["fam"]["clause"]["criteria_false"]).strip()
    assert sorted(loc["fam"]["count"]["opt"]) == [
        f"c{value}" for value in range(len(features.CLAUSE_IDS) + 1)
    ]
    assert str(loc["fam"]["combo"]["none"]).strip()
    assert str(loc["fam"]["combo"]["joiner"])


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompts_and_questions_leave_no_placeholder(lang: str) -> None:
    """输入：一种语言 × 全空间 32 组 case。输出：无。

    预期行为：提示词、反解提示词、题面与选项渲染完之后**一个花括号都不剩**——漏填会在 `format`
    上抛错，而多写的花括号会原样漏给模型，让它以为那是要填的槽。
    """
    loc = locales.load(lang)
    for case in features.build_cases(512):
        prompt = render.build_prompt(loc, case)
        assert "{" not in prompt and "}" not in prompt, case.describe()
        for item in questions.build(features.plan_for(case, 0), loc, 0):
            instructions = str(item.question["instructions"])
            assert "{" not in instructions and "}" not in instructions
    trace = judges.build_trace_prompt("[state]", loc)
    assert "{" not in trace and "}" not in trace


def test_pick_wraps_around() -> None:
    """输入：越界的问法编号。输出：无。预期行为：回绕（不抛异常）。"""
    assert locales.pick(["a", "b"], 3) == "b"
