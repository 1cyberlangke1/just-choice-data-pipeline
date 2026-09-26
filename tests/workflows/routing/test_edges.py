"""边界与接线：坏切分、坏产物、单值配置、注册表与协议。"""

from __future__ import annotations

from pathlib import Path

import pytest

from jc import jsonio, pipeline
from jc import workflow as workflow_mod
from tests.workflows.routing.conftest import _make_case, tiny_config
from workflows.routing import judges, locales, parse, render, spec as sp
from workflows.routing import workflow as routing_workflow


def test_unknown_split_speaks_human() -> None:
    """输入：不存在的切分名。输出：无。预期行为：报人话并列出可用切分。"""
    workflow = routing_workflow.RoutingWorkflow()
    with pytest.raises(KeyError, match="未知切分"):
        workflow.config_for("test_whatever")
    assert workflow.split_names() == ("train", "val", "test_iid", "test_ood")


def test_config_for_returns_a_distinct_config_per_split() -> None:
    """输入：三个同分布切分。输出：无。

    预期行为：切分名真的被替换进配置里——`ID_CONFIG` 是个单例，直接返回它会让
    train/val/test_iid 三次都按同一档切片（切出三份一模一样的数据）。
    """
    workflow = routing_workflow.RoutingWorkflow()
    splits = [workflow.config_for(split).split for split in ("train", "val", "test_iid")]
    assert splits == ["train", "val", "test_iid"]
    assert workflow.config_for("test_ood").ood_only is True


def test_registry_finds_the_workflow() -> None:
    """输入：无。输出：无。预期行为：自注册生效，且 `needs` 为空（纯构造式）。"""
    assert "routing" in workflow_mod.discover()
    loaded = workflow_mod.load("routing")
    assert loaded.name == "routing"
    assert loaded.needs == frozenset()
    assert loaded.kinds == {"choice", "noul", "score"}
    assert loaded.available_locales() == ["en", "zh"]


def test_bind_backend_is_a_hard_error() -> None:
    """输入：给构造式 workflow 绑一个后端。输出：无。

    预期行为：直接报错——收到后端说明接线错了（这条 workflow 一行模型都不用调）。
    """
    with pytest.raises(AssertionError, match="纯构造式"):
        routing_workflow.RoutingWorkflow().bind_backend(object())


def test_make_judges_rejects_unknown_names() -> None:
    """输入：不存在的判别器名。输出：无。预期行为：报人话并列出可用名字。"""
    with pytest.raises(KeyError, match="未知判别器"):
        routing_workflow.RoutingWorkflow().make_judges(("nope",), "zh")
    judges_out = routing_workflow.RoutingWorkflow().make_judges(("roundtrip", "label"), "en")
    assert [item.name for item in judges_out] == ["roundtrip", "label"]


def test_single_value_configs_still_construct() -> None:
    """输入：每一维都只留一个取值的配置。输出：无。

    预期行为：照常枚举出唯一的 case 并出题——单值配置是「边界旋钮」，
    该报错的是空表，不是单值。
    """
    config = sp.RoutingConfig(
        areas=("payments",),
        issues=("bug",),
        tiers=("pro",),
        impacts=("team",),
        urgencies=("blocked",),
        compliances=("none",),
        waits=("medium",),
        phrasings=(0,),
    )
    cases = sp.build_cases(config)
    assert len(cases) == 1
    spec = sp.sample_spec(None, cases[0], "train", 0, 0)
    state = render.build(spec, locales.load("zh"))
    assert parse.parse_ticket(state, parse.label_index(locales.load("zh"))).ticket_id


def test_audit_reports_missing_products(tmp_path: Path) -> None:
    """输入：一个空目录。输出：无。预期行为：报「产物不存在」而不是崩掉。"""
    report, missing = routing_workflow.RoutingWorkflow().audit(tmp_path, ("train",), "zh")
    assert report["train"]["samples"] == 0
    assert any("产物不存在" in item for item in missing)


def test_audit_catches_a_tampered_answer(tmp_path: Path) -> None:
    """输入：一次跑之后手动改坏一条答案。输出：无。

    预期行为：审计从**产物本身**重算，改坏的那条被记成 `label_mismatch`——
    这条是「老产物事后也能查」的保证。
    """
    config = pipeline.RunConfig(
        workflow="routing",
        split="train",
        locales=("zh",),
        run_seed=17,
        per_case=1,
        concurrency=1,
        out_root=tmp_path,
        run_id="edge-audit",
        config_override=tiny_config("train"),
    )
    result = pipeline.run(config)
    path = result.run_dir / "train.jsonl"
    rows = [row for _, row in jsonio.iter_jsonl(path)]
    rows[0]["answers"]["needs_human"]["noul"] = not rows[0]["answers"]["needs_human"]["noul"]
    jsonio.write_jsonl(path, rows)
    report, missing = routing_workflow.RoutingWorkflow().audit(result.run_dir, ("train",), "zh")
    assert report["train"]["label_mismatch"] >= 1
    assert any("重算不一致" in item for item in missing)


def test_audit_catches_an_unknown_family(tmp_path: Path) -> None:
    """输入：一次跑之后往一条样本里塞一道陌生题。输出：无。

    预期行为：记成「未知题族」——产物里混进没人验过的题必须当场露出来。
    """
    config = pipeline.RunConfig(
        workflow="routing",
        split="train",
        locales=("zh",),
        run_seed=17,
        per_case=1,
        concurrency=1,
        out_root=tmp_path,
        run_id="edge-unknown",
        config_override=tiny_config("train"),
    )
    result = pipeline.run(config)
    path = result.run_dir / "train.jsonl"
    rows = [row for _, row in jsonio.iter_jsonl(path)]
    rows[0]["questions"]["mystery"] = {"type": "noul", "instructions": "?"}
    rows[0]["answers"]["mystery"] = {"type": "noul", "noul": True}
    jsonio.write_jsonl(path, rows)
    report, missing = routing_workflow.RoutingWorkflow().audit(result.run_dir, ("train",), "zh")
    assert report["train"]["unknown_families"]["mystery"] == 1
    assert any("未知题族" in item for item in missing)


def test_judges_refuse_a_state_that_cannot_be_parsed() -> None:
    """输入：一段乱七八糟的文本。输出：无。

    预期行为：两个判别器都判不通过并给出人话理由，而不是抛异常把整批跑挂掉。
    （结构判别器没有真值时会主动跳过，所以这里给它一份真值规格。）
    """
    sample = {"state": "这里不是工单", "questions": {}, "answers": {}}
    spec = sp.sample_spec(None, _make_case(), "train", 0, 0)
    for judge in (judges.RoundtripJudge(), judges.LabelJudge()):
        verdict = judge.judge(sample, spec)
        assert not verdict.ok
        assert verdict.reason
