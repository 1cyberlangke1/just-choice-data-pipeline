"""语言文件测试：结构不变量（抬头、条款表、处置档位、占位符）逐条钉住。"""

from __future__ import annotations

import pytest

from workflows.guardrail import features, judges, locales, questions, render


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_clause_table_lines_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：条款表的键与 `CLAUSE_IDS` 一一对应，每条四个字段都非空（`examples` 是**非空列表**，
    每项也非空）、条款名与表行两两不同，而且表行里**编号排在分隔符前面**（解析就是按这个切回编号的）。
    """
    loc = locales.load(lang)
    table = loc["clause"]
    assert isinstance(table, dict)
    assert sorted(table) == sorted(features.CLAUSE_IDS)
    for clause in features.CLAUSE_IDS:
        entry = render.clause_copy(loc, clause)
        for key in ("name", "rule", "avoid"):
            assert str(entry[key]).strip(), f"{lang}.{clause}.{key}"
        examples = entry["examples"]
        assert isinstance(examples, list) and examples, f"{lang}.{clause}.examples"
        assert all(str(value).strip() for value in examples), f"{lang}.{clause}.examples"
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

    预期行为：**池里每一个**示例片段都只命中它自己那一条——示例同时踩中两条，提示词就会把模型
    往错的方向带；池子里的值是要逐个发出去的，只验第一个等于没验。
    """
    loc = locales.load(lang)
    for clause in features.CLAUSE_IDS:
        raw = render.clause_copy(loc, clause)["examples"]
        assert isinstance(raw, list)
        for value in raw:
            example = str(value)
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
