"""边界与错误路径：坏旋钮、坏名字、坏语言都报人话。"""

from __future__ import annotations

from typing import Any

import pytest

from jc import workflow as workflow_mod
from workflows.guardrail import judges, locales, parse, render
from workflows.guardrail import spec as sp

WORKFLOW = "guardrail"


def test_unknown_split_is_reported() -> None:
    """输入：不存在的切分名。输出：无。预期行为：KeyError 并列出可用切分。"""
    with pytest.raises(KeyError, match="未知切分"):
        workflow_mod.load(WORKFLOW).config_for("nope")


def test_split_names_are_the_documented_ones() -> None:
    """输入：无。输出：无。预期行为：三个同分布切分 + test_ood。"""
    assert workflow_mod.load(WORKFLOW).split_names() == ("train", "val", "test_iid", "test_ood")


def test_unknown_judge_is_reported() -> None:
    """输入：不存在的判别器名。输出：无。预期行为：KeyError 并列出可用的。"""
    with pytest.raises(KeyError, match="未知判别器"):
        workflow_mod.load(WORKFLOW).make_judges(("nope",), "zh")


def test_bind_backend_is_refused() -> None:
    """输入：给纯构造式 workflow 绑后端。输出：无。预期行为：AssertionError（接线错了）。"""
    with pytest.raises(AssertionError, match="纯构造式"):
        workflow_mod.load(WORKFLOW).bind_backend(object())


def test_workflow_declares_no_backend_need() -> None:
    """输入：无。输出：无。预期行为：`needs` 为空（零 token），三种题型与两个判别器齐备。"""
    workflow = workflow_mod.load(WORKFLOW)
    assert workflow.needs == frozenset()
    assert workflow.kinds == frozenset({"choice", "noul", "score"})
    assert workflow.judges == ("roundtrip", "label")


def test_unknown_language_lists_available() -> None:
    """输入：不存在的语言。输出：无。预期行为：FileNotFoundError 并列出可用语言。"""
    with pytest.raises(FileNotFoundError, match="zh"):
        locales.load("ja")


def test_pick_wraps_around() -> None:
    """输入：越界的问法编号。输出：无。预期行为：回绕（问法措辞不是判据）。"""
    assert locales.pick(["a", "b"], 5) == "b"


def test_dimension_titles_cover_all_seven_dimensions() -> None:
    """输入：无。输出：无。预期行为：七个维度都有中文标题（报告里才不会漏维度）。"""
    workflow = workflow_mod.load(WORKFLOW)
    case = workflow.build_cases(workflow.config_for("train"))[0]
    assert set(workflow.dimension_titles()) == set(workflow.stats_dimensions(case))


def test_reparse_returns_a_parsed_message(make_spec: Any) -> None:
    """输入：渲染后的 state。输出：无。预期行为：`reparse` 用合并标签表反解成功。"""
    spec = make_spec()
    loc = locales.load("zh")
    message = workflow_mod.load(WORKFLOW).reparse(render.build(spec, loc))
    assert isinstance(message, parse.ParsedMessage)
    assert message.body == render.body_lines(spec, loc)


def test_missing_body_is_caught_by_the_roundtrip_judge(make_spec: Any) -> None:
    """输入：把正文整段删掉。输出：无。

    预期行为：结构判别判不通过并说明反解失败——渲染漏正文不能悄悄过。
    """
    loc = locales.load("zh")
    spec = make_spec()
    head = render.build(spec, loc).split(loc["meta"]["body_label"])[0]
    sample = {"state": head, "questions": {}, "answers": {}}
    verdict = judges.RoundtripJudge().judge(sample, spec)
    assert not verdict.ok
    assert "反解失败" in verdict.reason


def test_config_rejects_unknown_subset_naming_the_field() -> None:
    """输入：不存在的条款子集。输出：无。预期行为：ValueError 点名 `violations`。"""
    with pytest.raises(ValueError, match="violations"):
        sp.GuardConfig(violations=(("harass", "nope"),))
