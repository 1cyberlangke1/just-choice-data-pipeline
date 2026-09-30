"""来源选择的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from jc import pipeline, workflow as workflow_mod
from workflows.sourceselect import features, workflow as ss_workflow

from tests.workflows.sourceselect.conftest import FakeBackend

CASE_LIMIT = 8


def make_config(
    tmp_path: Path, run_id: str, backend: object, *, split: str = "train"
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return pipeline.RunConfig(
        workflow="sourceselect",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=ss_workflow.SourceSelectConfig(split=split, case_limit=CASE_LIMIT),
        backend=backend,
    )


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：假后端（从不撒谎）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、三个判别器都记到通过；产物能过审计，
    而且四族可重算的答案全部与产物里的候选表自洽。
    """
    result = pipeline.run(make_config(tmp_path, "s1", fake_backend()))
    assert result.accepted == CASE_LIMIT
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"hygiene", "frame", "trace"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0
    report, missing = workflow_mod.load("sourceselect").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == CASE_LIMIT
    for key in ("frame_bad", "rules_bad", "pick_bad", "enough_bad", "cover_bad", "near_miss_bad"):
        assert report["train"][key] == 0, key


def test_trace_judge_rejects_a_misread_need(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解说反话的假后端。输出：RunResult。

    预期行为：这些样本被拒，拒绝码是 `E_JUDGE_TRACE`——判别器抓的是模型真会犯的错。
    """
    result = pipeline.run(make_config(tmp_path, "s3", fake_backend(trace_lies_need=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


def test_render_rejects_table_talk(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：查询里谈候选表的假后端。输出：RunResult。

    预期行为：渲染期就拒掉，拒绝码是 `E_ASSEMBLE`（生成阶段的统一拒绝码）——
    这类文本拼进 state 就洗不掉了。
    """
    result = pipeline.run(make_config(tmp_path, "s4", fake_backend(table_query=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_render_retries_then_succeeds(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：查询写成多行的假后端。输出：RunResult。预期行为：重试耗尽后仍被拒，不静默放行。"""
    result = pipeline.run(make_config(tmp_path, "s5", fake_backend(multi_line_query=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case 列表。输出：无。

    预期行为：同分布三个切分互不相交、`test_ood` 与它们不相交，四者并起来等于全空间——
    一个组合都不丢、也不重复。
    """
    wf = ss_workflow.SourceSelectWorkflow()
    parts: dict[str, set[tuple[tuple[str, object], ...]]] = {}
    for split in wf.split_names():
        cases = wf.build_cases(ss_workflow.SourceSelectConfig(split=split))
        parts[split] = {tuple(case.as_dict().items()) for case in cases}
    assert parts["train"] & parts["val"] == set()
    assert parts["train"] & parts["test_iid"] == set()
    assert parts["val"] & parts["test_iid"] == set()
    for split in ("train", "val", "test_iid"):
        assert parts[split] & parts["test_ood"] == set()
    union = set().union(*parts.values())
    assert len(union) == sum(len(part) for part in parts.values())
    assert union == {tuple(case.as_dict().items()) for case in features.build_cases(1024)}


def test_id_side_has_no_dead_options() -> None:
    """输入：同分布侧的 case 列表。输出：无。

    预期行为：`test_ood` 留出的取值在同分布侧一次都不出现，而且**每一族每一档都还到得了**
    ——留出一个取值连带杀死别的族的好几档，就是死选项。
    """
    cases = [
        case
        for split in ("train", "val", "test_iid")
        for case in ss_workflow.cases_for(ss_workflow.SourceSelectConfig(split=split))
    ]
    assert cases
    for case in cases:
        assert case.size not in ss_workflow.OOD_SIZES
    labels = [
        features.label_for(features.plan_for(case, phrasing))
        for case in cases
        for phrasing in (0, 1, 2)
    ]
    assert {item.need for item in labels} == set(features.NEEDS)
    assert {item.pick for item in labels} == {None, 1, 2, 3, 4, 5}
    assert {item.enough for item in labels} == {True, False}
    assert {item.cover for item in labels} == {0, 1, 2}
    assert {item.near_miss for item in labels} == {True, False}


def test_ood_side_only_carries_the_held_out_size() -> None:
    """输入：`test_ood` 的 case 列表。输出：无。预期行为：每一条都带留出的那个表长。"""
    cases = ss_workflow.cases_for(ss_workflow.SourceSelectConfig(split="test_ood"))
    assert cases
    for case in cases:
        assert case.size in ss_workflow.OOD_SIZES


def test_bad_knobs_are_rejected() -> None:
    """输入：坏旋钮。输出：无。预期行为：构造期就报人话。"""
    with pytest.raises(ValueError):
        ss_workflow.SourceSelectConfig(split="train", case_limit=-1)
    with pytest.raises(ValueError):
        ss_workflow.SourceSelectConfig(split="train", phrasing_pool=())
    with pytest.raises(ValueError):
        ss_workflow.SourceSelectConfig(split="train", cover_limit=0)
    with pytest.raises(KeyError):
        ss_workflow.SourceSelectWorkflow().config_for("nope")


def test_render_without_backend_raises() -> None:
    """输入：没注入后端的 workflow。输出：无。

    预期行为：直接报错，绝不悄悄降级成模板——那会让「文本由模型写」这条前提失效。
    """
    wf = ss_workflow.SourceSelectWorkflow()
    config = ss_workflow.SourceSelectConfig(split="train")
    plan = wf.plan_sample(random.Random(0), config, features.build_cases(1024)[0], 0)
    spec = wf.sample(random.Random(0), plan)
    with pytest.raises(RuntimeError):
        wf.render(spec, random.Random(0), "zh")


def test_unknown_judge_name_raises() -> None:
    """输入：不存在的判别器名。输出：无。预期行为：直接报错。"""
    with pytest.raises(KeyError):
        ss_workflow.SourceSelectWorkflow().make_judges(("nope",), "zh")
