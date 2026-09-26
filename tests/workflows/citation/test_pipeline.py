"""端到端测试：小规模跑通整条管线、可复现、四切分不重叠、审计与接线。"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from jc import cli, jsonio, pipeline
from jc import workflow as workflow_mod
from tests.workflows.citation.conftest import tiny_config
from workflows.citation import judges as judges_mod
from workflows.citation import locales, questions, render
from workflows.citation import spec as sp
from workflows.citation import workflow as citation_workflow


def make_config(
    tmp_path: Path,
    run_id: str,
    *,
    split: str = "train",
    concurrency: int = 1,
    override: Any = None,
) -> pipeline.RunConfig:
    """输入：临时目录、run 名、切分、并发度、可选的采样参数。输出：指向 tmp 的运行配置。"""
    return pipeline.RunConfig(
        workflow="citation",
        split=split,
        locales=("zh",),
        run_seed=17,
        per_case=1,
        concurrency=concurrency,
        out_root=tmp_path,
        run_id=run_id,
        config_override=override if override is not None else tiny_config(split),
    )


def test_end_to_end_all_judges_pass(tmp_path: Path) -> None:
    """输入：小配置（零 token，不需要后端）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、两个判别器全通过；产物过审计。
    """
    config = make_config(tmp_path, "c1")
    result = pipeline.run(config)
    assert result.accepted > 0
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"roundtrip", "label"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0 and bucket["ok"] > 0
    report, missing = workflow_mod.load("citation").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == result.accepted
    assert report["train"]["index_bad"] == 0
    assert report["train"]["label_mismatch"] == 0
    assert set(report["train"]["types"]) == {"choice", "noul", "score"}


def test_two_runs_are_byte_identical(tmp_path: Path) -> None:
    """输入：同种子、不同并发度跑两次。输出：两份产物。

    预期行为：逐字节一致——整份文档由 case 决定，并发不改变内容。
    """
    serial = pipeline.run(make_config(tmp_path, "c2a", concurrency=1))
    parallel = pipeline.run(make_config(tmp_path, "c2b", concurrency=4))
    assert (serial.run_dir / "train.jsonl").read_bytes() == (
        parallel.run_dir / "train.jsonl"
    ).read_bytes()


def test_manifest_and_stats_are_written(tmp_path: Path) -> None:
    """输入：一次运行。输出：无。预期行为：manifest 与统计都在，且统计含维度分布。"""
    result = pipeline.run(make_config(tmp_path, "c3"))
    manifest = jsonio.loads(
        (result.run_dir / "manifest-train.json").read_text(encoding="utf-8")
    )
    assert manifest["workflow"] == "citation"
    assert manifest["split"] == "train"
    dimensions = result.stats["dimensions"]
    assert set(dimensions) == {
        "topic",
        "status",
        "quote_form",
        "sections",
        "noise",
        "claim_style",
        "where",
        "doc_style",
    }
    assert len(result.stats["families"]) == len(questions.FAMILIES)


def test_ood_split_only_uses_left_out_values(tmp_path: Path) -> None:
    """输入：值留出那一档。输出：RunResult。预期行为：留出的取值**都真的出现了**。"""
    result = pipeline.run(
        make_config(tmp_path, "c4", split="test_ood", override=citation_workflow.OOD_CONFIG)
    )
    dimensions = result.stats["dimensions"]
    assert "rollout" in dimensions["topic"]
    assert "mislocated" in dimensions["status"]
    assert "reworded" in dimensions["quote_form"]
    assert "off_topic" in dimensions["noise"]
    assert 0 in dimensions["claim_style"] or "0" in dimensions["claim_style"]
    report, missing = workflow_mod.load("citation").audit(result.run_dir, ("test_ood",), "zh")
    assert missing == []
    assert report["test_ood"]["samples"] == result.accepted


def test_splits_do_not_share_a_case() -> None:
    """输入：四个切分的 case 列表。输出：无。

    预期行为：三个同分布档两两不共享组合（按组合分桶）；值留出档每一格都含没见过的取值。
    """
    workflow = citation_workflow.CitationWorkflow()
    combos = {
        split: {tuple(case.as_dict().items()) for case in workflow.build_cases(workflow.config_for(split))}
        for split in ("train", "val", "test_iid")
    }
    names = list(combos)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            assert combos[names[left]] & combos[names[right]] == set()
    ood = workflow.build_cases(workflow.config_for("test_ood"))
    assert ood
    assert all(case.unseen_dims() for case in ood)
    assert not ({tuple(case.as_dict().items()) for case in ood} & combos["train"])


def test_audit_reports_a_missing_artifact(tmp_path: Path) -> None:
    """输入：空目录。输出：(报告, 缺失项)。预期行为：明说「产物不存在」，不当成 0 条样本。"""
    report, missing = workflow_mod.load("citation").audit(tmp_path, ("train",), "zh")
    assert report["train"]["samples"] == 0
    assert any("产物不存在" in item for item in missing)


def _sample_row(case: sp.CitationCase, seed: int = 1) -> tuple[dict[str, Any], sp.CitationSpec]:
    """输入：case、种子。输出：(契约样本行, 真值规格)。预期行为：走真的构造与组装。"""
    loc = locales.load("zh")
    rng = random.Random(seed)
    spec = sp.sample_spec(rng, case, "train", 0, 0)
    labeled = questions.build(spec, loc, spec.phrasing, rng)
    from jc import assemble

    row = assemble.assemble_sample(
        "citation-train-0000000", render.build(spec, loc), labeled, rng
    )
    return row, spec


def test_audit_catches_a_bad_artifact(tmp_path: Path) -> None:
    """输入：一条「答案对不上 + 编号越界 + 缺题族 + 多出未知题族」的产物行。输出：缺失项。

    预期行为：四件事都要报出来，缺掉的题族与没覆盖到的主题也要点名。
    """
    row, _spec = _sample_row(sp.CitationCase("auth", "verified", "verbatim", 3, "plain", 1, "first", "bracketed"))
    options = list(row["questions"]["quote_source"]["criteria"])
    index = row["answers"]["quote_source"]["choice"]
    row["answers"]["quote_source"]["choice"] = (index + 1) % len(options)
    row["answers"]["verdict"]["choice"] = 99
    del row["questions"]["relation"]
    del row["answers"]["relation"]
    row["questions"]["made_up"] = {"type": "choice", "instructions": "x", "criteria": {"a": "1", "b": "2"}}
    row["answers"]["made_up"] = {"type": "choice", "choice": 0}
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    _, missing = workflow_mod.load("citation").audit(tmp_path, ("train",), "zh")
    assert any("答案与文本重算不一致" in item for item in missing)
    assert any("答案编号越界" in item for item in missing)
    assert any("未知题族" in item for item in missing)
    assert any("题族 relation 一条都没出" in item for item in missing)
    assert any("主题 billing 一条都没出" in item for item in missing)


def test_audit_passes_a_self_consistent_artifact(tmp_path: Path) -> None:
    """输入：一条自己组装出来的、自洽的产物行。输出：(报告, 缺失项)。

    预期行为：没有「答案对不上 / 编号越界 / 未知题族」这类缺失项——覆盖面由更大的批次保证，
    这里只查这条行本身。
    """
    case = sp.CitationCase("auth", "mislocated", "verbatim", 3, "plain", 1, "middle", "dotted")
    row, _spec = _sample_row(case, seed=5)
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = workflow_mod.load("citation").audit(tmp_path, ("train",), "zh")
    assert not [item for item in missing if "重算" in item or "越界" in item or "未知题族" in item]
    assert report["train"]["samples"] == 1
    assert report["train"]["sources"]["other_section"] == 1


def test_expected_values_follows_the_split_config() -> None:
    """输入：训练档与值留出档。输出：两档各自的期望取值集合。

    预期行为：训练档**不**含 `rollout` 主题（值留出），也不含「别的小节」这一种出处
    （那是 `mislocated` 才有）——写死全量词表会让审计天天误报。
    """
    workflow = citation_workflow.CitationWorkflow()
    train = workflow.expected_values("train")
    ood = workflow.expected_values("test_ood")
    assert "rollout" not in train["topics"] and "rollout" in ood["topics"]
    assert "other_section" not in train["sources"] and "other_section" in ood["sources"]
    assert train["verdicts"] == {"verified", "contradicted", "unsupported", "fabricated"}
    assert train["strengths"] == {0, 2, 3}
    assert ood["strengths"] == {0, 1, 2, 3}


def test_judges_are_the_declared_defaults() -> None:
    """输入：无。输出：workflow 声明。预期行为：题型三种都产、判别器是两个规则判别器。"""
    workflow = citation_workflow.CitationWorkflow()
    assert workflow.kinds == frozenset({"choice", "noul", "score"})
    assert workflow.judges == ("roundtrip", "label")
    assert workflow.needs == frozenset()
    assert workflow.name == "citation" and workflow.version == citation_workflow.WORKFLOW_VERSION
    assert isinstance(judges_mod.RoundtripJudge(), object)
    with pytest.raises(AssertionError, match="不该被绑定后端"):
        workflow.bind_backend(object())


def test_cli_audit_prints_the_citation_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """输入：一次真跑 + `jc audit`。输出：退出码与 stdout。

    预期行为：退出码 0，报告里出现 citation 自己的计数键——通用层的审计打印必须按
    产物报的键走。
    """
    result = pipeline.run(make_config(tmp_path, "c6"))
    code = cli.main(
        ["audit", "--workflow", "citation", "--dir", str(result.run_dir), "--split", "train"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "审计通过" in out
    assert "label_mismatch" in out and "verdicts" in out
