"""改动引入性判定的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

from pathlib import Path

import pytest

from jc import pipeline
from jc.backend.types import TransportError
from workflows.findtriage import features, workflow as ft_workflow

from tests.workflows._scaffold import run_config
from tests.workflows.findtriage.conftest import FakeBackend

CASE_LIMIT = 8


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train"
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return run_config(
        tmp_path,
        run_id,
        backend=backend,
        workflow="findtriage",
        split=split,
        seed=7,
        override=ft_workflow.FindTriageConfig(split=split, case_limit=CASE_LIMIT),
    )


@pytest.mark.parametrize(
    "switch",
    ["trace_lies_cite", "trace_lies_symbol", "trace_lies_actionable"],
)
def test_trace_judge_rejects_a_misread(
    tmp_path: Path, fake_backend: type[FakeBackend], switch: str
) -> None:
    """输入：反解说反话的假后端。输出：RunResult。

    预期行为：这些样本被拒，拒绝码是 `E_JUDGE_TRACE`——判别器抓的是模型真会犯的错。
    """
    result = pipeline.run(make_config(tmp_path, f"s3-{switch}", fake_backend(**{switch: True})))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


@pytest.mark.parametrize("switch", ["wrong_line", "no_backtick"])
def test_render_rejects_a_draft_that_breaks_the_shape(
    tmp_path: Path, fake_backend: type[FakeBackend], switch: str
) -> None:
    """输入：写歪了那两行的假后端。输出：RunResult。

    预期行为：渲染期就拒掉，拒绝码是 `E_ASSEMBLE`（生成阶段的统一拒绝码）——这两行拼进
    state 就洗不掉了，而且审计再也认不出标签。
    """
    result = pipeline.run(make_config(tmp_path, f"s4-{switch}", fake_backend(**{switch: True})))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


@pytest.mark.parametrize("switch", ["vague_with_backtick", "concrete_without_backtick"])
def test_render_rejects_a_fix_that_breaks_the_backtick_rule(
    tmp_path: Path, fake_backend: type[FakeBackend], switch: str
) -> None:
    """输入：修法那一行写歪了反引号规则的假后端。输出：RunResult。

    预期行为：**落在这一格上的**样本被渲染期拒掉（拒绝码 `E_ASSEMBLE`）。这一格由 `fix`
    维度决定，小 case 集合未必两种取值都踩到，所以只断言「拒过、且拒的都是这一码」，
    不断言「全拒」——那会让这条测试随 case 集合的偶然组成而红。
    """
    result = pipeline.run(make_config(tmp_path, f"s4b-{switch}", fake_backend(**{switch: True})))
    assert result.rejected, "这一格一次都没踩到，说明 case 集合没覆盖到 fix 的两个取值"
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


@pytest.mark.parametrize("switch", ["multi_line_finding", "fence_finding", "empty_fix"])
def test_hygiene_rejects_a_malformed_line(
    tmp_path: Path, fake_backend: type[FakeBackend], switch: str
) -> None:
    """输入：把某一行写成多行 / 带围栏 / 空白的假后端。输出：RunResult。

    预期行为：渲染期先拦下（多行与围栏在 `check_draft` 就拒），拒绝码是 `E_ASSEMBLE`；
    这条同时钉住「发现必须是一行」这个形状。
    """
    result = pipeline.run(make_config(tmp_path, f"s5-{switch}", fake_backend(**{switch: True})))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_trace_failure_is_a_skip_not_a_reject(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解时抛端点故障的假后端。输出：RunResult。

    预期行为：数据照样收下（端点抖动不是数据质量问题），但判别器记成「没判成」。
    """
    result = pipeline.run(
        make_config(tmp_path, "s6", fake_backend(trace_raises=TransportError("端点炸了")))
    )
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    assert result.stats["judges"]["trace"]["skipped"] == CASE_LIMIT


def test_ood_split_uses_the_held_out_table_size(fake_backend: type[FakeBackend]) -> None:
    """输入：无（零 token）。输出：无。

    预期行为：`test_ood` 侧的记录条数全落在留出取值上，而同分布侧一条都没有——
    这是「值留出」在本格式里的落地形态。
    """
    workflow = ft_workflow.FindTriageWorkflow()
    ood = {case.size for case in workflow.build_cases(workflow.config_for("test_ood"))}
    assert ood == set(ft_workflow.OOD_SIZES)
    for split in ("train", "val", "test_iid"):
        sizes = {case.size for case in workflow.build_cases(workflow.config_for(split))}
        assert not sizes & set(ft_workflow.OOD_SIZES), split


def test_ood_split_can_reach_every_triage_bucket(fake_backend: type[FakeBackend]) -> None:
    """输入：无（零 token）。输出：无。

    预期行为：留出表长**不杀任何一档**——三档处置在 OOD 侧都到得了。留出别的维度会
    把某一档冻死（见 `workflow.OOD_SIZES` 的注释），这条就是那句话的机器可查形态。
    """
    del fake_backend
    workflow = ft_workflow.FindTriageWorkflow()
    buckets = {
        features.label_for(features.plan_for(case, 0)).triage
        for case in workflow.build_cases(workflow.config_for("test_ood"))
    }
    assert buckets == set(features.TRIAGES)
