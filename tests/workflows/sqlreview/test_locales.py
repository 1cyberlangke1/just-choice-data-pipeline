"""语言文件测试：结构不变量（抬头、判据表、处置档位、变更场景、占位符）逐条钉住。"""

from __future__ import annotations

import re

import pytest

from workflows.sqlreview import features, locales, render


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_rule_and_avoid_copy_never_carries_braces(lang: str) -> None:
    """输入：一种语言。输出：无。

    预期行为：判据原文与反面文案里**一个花括号都不许有**——这两段会被 `build_prompt` 拼进
    提示词，而花括号是填槽标记（`test_prompts_and_questions_leave_no_placeholder` 会扫
    「一个都不剩」）。`subqa` 上踩过一次：判据原文里写了那个花括号指令的具体形状，提示词的
    填槽自检当场把它当成没填的槽报出来。要举例就写「一对花括号里夹着反斜杠开头的短代码」这种
    说法，别把花括号本身写进去。
    """
    loc = locales.load(lang)
    for tag in features.TAG_IDS:
        entry = render.tag_copy(loc, tag)
        for key in ("rule", "avoid"):
            text = str(entry[key])
            assert "{" not in text and "}" not in text, f"{lang}.{tag}.{key}: {text!r}"


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
