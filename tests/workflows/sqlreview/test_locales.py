"""语言文件测试：结构不变量（抬头、判据表、处置档位、变更场景、占位符）逐条钉住。"""

from __future__ import annotations

import re

import pytest

from workflows.sqlreview import features, locales, render


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_workload_table_lines_up_with_features(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：变更场景表的键与 `WORKLOADS` 一一对应，描述非空且两两不同（三档描述写得一样，
    留出维就名存实亡了）。
    """
    loc = locales.load(lang)
    table = loc["workload"]
    assert isinstance(table, dict)
    assert sorted(table) == sorted(features.WORKLOADS)
    descs = [render.workload_copy(loc, workload) for workload in features.WORKLOADS]
    assert all(desc.strip() for desc in descs)
    assert len(set(descs)) == len(descs)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_format_spec_offers_no_copyable_statement(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：格式规格里**不许出现填好的成品**——给一条完整的语句、或者给一个带数字的
    `rows=`，模型照抄下去正好是「无条件写」或「大表全扫」（范式 §9.52）。规格只印键名与
    `<...>` 占位符。
    """
    loc = locales.load(lang)
    for line in render.format_lines(loc):
        assert not re.search(r"\brows=\d", line), line
        assert ";" not in line, line


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_prompt_pins_the_note_language(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：顶层有 `language_name`（评审备注的语言），提示词里得有一句把它钉死的约束——不写
    这一句，模型会照它自己的先验换语言（范式 §10.96 那条尺子的由来）。
    """
    loc = locales.load(lang)
    assert str(loc["language_name"]).strip()
    assert "{language}" in str(loc["prompt"]["lang_rule"])
    case = features.SqlReviewCase("Q1", features.WORKLOAD_OLTP)
    prompt = render.build_prompt(loc, case)
    assert str(loc["language_name"]) in prompt


@pytest.mark.parametrize(
    ("lang", "needle"),
    [("zh", "没有被真的执行"), ("en", "never actually executed")],
)
def test_prompt_says_the_change_is_text_only(lang: str, needle: str) -> None:
    """输入：一种语言、该语言里那句「只写不执行」的说法。输出：无。

    预期行为：提示词里**明说这一族只要文本**——语句与结构变更只写在记录里，不许连数据库、不许
    跑任何命令。判据认的也只是字面，跑不跑都不改变标签；写清楚是为了让模型不去「验证」它自己
    编出来的 SQL（主人明确要求过：不许实际执行合成出来的命令）。
    """
    loc = locales.load(lang)
    case = features.SqlReviewCase("Q3", features.WORKLOAD_MIGRATION)
    assert needle in render.build_prompt(loc, case)
