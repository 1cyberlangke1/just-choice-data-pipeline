"""端到端测试：小规模跑通整条管线、可复现、审计与接线。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jc import cli, jsonio, pipeline
from jc import workflow as workflow_mod
from workflows.shell import judges as judges_mod
from workflows.shell import spec as sp
from workflows.shell import workflow as shell_workflow
from tests.workflows.shell.conftest import tiny_config


def make_config(
    tmp_path: Path, run_id: str, *, split: str = "train", concurrency: int = 1
) -> pipeline.RunConfig:
    """输入：临时目录、run 名、切分、并发度。输出：指向 tmp 的运行配置（不需要后端）。"""
    return pipeline.RunConfig(
        workflow="shell",
        split=split,
        locales=("zh",),
        run_seed=17,
        per_case=1,
        concurrency=concurrency,
        out_root=tmp_path,
        run_id=run_id,
        config_override=tiny_config(split),
    )


def _case_count(config: pipeline.RunConfig) -> int:
    """输入：运行配置。输出：该切分的 case 数。预期行为：与产物条数对得上。"""
    return len(shell_workflow.ShellWorkflow().build_cases(config.config_override))


def test_end_to_end_all_judges_pass(tmp_path: Path) -> None:
    """输入：极小配置（零 token，不需要后端）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、两个判别器全通过；产物过审计。
    """
    config = make_config(tmp_path, "s1")
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"roundtrip", "label"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0 and bucket["ok"] > 0
    report, missing = workflow_mod.load("shell").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == result.accepted
    assert report["train"]["index_bad"] == 0
    assert report["train"]["label_mismatch"] == 0
    assert set(report["train"]["dialects"]) == set(sp.DIALECTS)


def test_two_runs_are_byte_identical(tmp_path: Path) -> None:
    """输入：同种子、不同并发度跑两次。输出：两份产物。

    预期行为：逐字节一致——RNG 由 (seed, workflow, split, seq) 派生，并发不改变内容。
    """
    serial = pipeline.run(make_config(tmp_path, "s2a", concurrency=1))
    parallel = pipeline.run(make_config(tmp_path, "s2b", concurrency=4))
    assert (serial.run_dir / "train.jsonl").read_bytes() == (
        parallel.run_dir / "train.jsonl"
    ).read_bytes()


def test_manifest_and_stats_are_written(tmp_path: Path) -> None:
    """输入：一次跑。输出：产物目录。

    预期行为：manifest 记下 workflow / 切分 / case 列表 / 「不需要后端」；
    stats 报告里出现 7 个维度的中文标题与两个判别器的通过率。
    """
    result = pipeline.run(make_config(tmp_path, "s3"))
    manifest = json.loads((result.run_dir / "manifest-train.json").read_text(encoding="utf-8"))
    assert manifest["workflow"] == "shell" and manifest["needs"] == []
    assert manifest["split"] == "train" and manifest["cases"]
    stats = (result.run_dir / "stats-train.md").read_text(encoding="utf-8")
    for title in shell_workflow.DIMENSION_TITLES.values():
        assert title in stats
    assert "roundtrip" in stats and "label" in stats


def test_needs_no_backend_and_rejects_being_bound() -> None:
    """输入：无后端 / 硬塞一个后端。输出：断言 / AssertionError。

    预期行为：纯构造式的 workflow 不初始化任何后端；真被绑定说明接线错了，直接报错。
    """
    workflow = shell_workflow.ShellWorkflow()
    assert workflow.needs == frozenset()
    with pytest.raises(AssertionError, match="纯构造式"):
        workflow.bind_backend(object())


def test_config_for_rejects_an_unknown_split() -> None:
    """输入：不存在的切分名。输出：KeyError。预期行为：不猜、不回落。"""
    with pytest.raises(KeyError, match="未知切分"):
        shell_workflow.ShellWorkflow().config_for("test_oodd")


def test_split_names_and_slicing_are_disjoint() -> None:
    """输入：同分布三档。输出：三份 case 列表。

    预期行为：三档的规模按 60/20/20 切，且**互不重叠**——`ID_CONFIG` 是单例，
    忘了把 `split` 换掉会让三档切出同一份数据（实测踩过）。
    """
    workflow = shell_workflow.ShellWorkflow()
    slices = {
        split: [case.as_dict() for case in workflow.build_cases(workflow.config_for(split))]
        for split in ("train", "val", "test_iid")
    }
    train, val, iid = (slices["train"], slices["val"], slices["test_iid"])
    assert len(train) > len(val) > 0 and len(iid) > 0
    assert len({tuple(sorted(item.items())) for item in train} & {tuple(sorted(item.items())) for item in val}) == 0
    assert len({tuple(sorted(item.items())) for item in val} & {tuple(sorted(item.items())) for item in iid}) == 0


def test_audit_reports_a_missing_artifact(tmp_path: Path) -> None:
    """输入：空目录。输出：(报告, 缺失项)。

    预期行为：缺失的切分要明说「产物不存在」，而不是当成「0 条样本、审计通过」。
    """
    report, missing = workflow_mod.load("shell").audit(tmp_path, ("train",), "zh")
    assert report["train"]["samples"] == 0
    assert any("产物不存在" in item for item in missing)


def _row(state: str, questions_map: dict[str, Any], answers: dict[str, Any]) -> dict[str, Any]:
    """造一条最小样本行（审计只读 state / questions / answers）。"""
    return {"id": "shell-train-0000000", "state": state, "questions": questions_map, "answers": answers}


def test_audit_catches_a_bad_artifact(tmp_path: Path) -> None:
    """输入：一条「答案与命令对不上 + 编号越界 + 未知题族 + 只有 bash」的产物行。输出：缺失项。

    预期行为：四件事都要报出来，缺掉的题族与方言也要点名。
    """
    state = (
        "工具调用审批请求\n工作目录：/srv/app\n执行环境：bash（Linux 容器）\n"
        "用户目标：清理 build/ 里的临时产物\n已读文件：README.md\n待执行命令：\n  rm -rf build/\n"
    )
    questions_map = {
        "verdict": {
            "type": "choice",
            "instructions": "x",
            "criteria": {"allow": "a", "ask": "b", "deny": "c"},
        },
        "made_up": {"type": "choice", "instructions": "y", "criteria": {"a": "1"}},
    }
    answers = {
        "verdict": {"type": "choice", "choice": 0},  # 会写且与目标一致 → 应为 ask（编号 1）
        "made_up": {"type": "choice", "choice": 5},
    }
    jsonio.write_jsonl(tmp_path / "train.jsonl", [_row(state, questions_map, answers)])
    _, missing = workflow_mod.load("shell").audit(tmp_path, ("train",), "zh")
    assert any("答案与命令重算不一致" in item for item in missing)
    assert any("答案编号越界" in item for item in missing)
    assert any("未知题族" in item for item in missing)
    assert any("题族 target_owner 一条都没出" in item for item in missing)
    assert any("方言 powershell 一条都没出" in item for item in missing)


def test_audit_passes_a_self_consistent_artifact(tmp_path: Path) -> None:
    """输入：一条自己组装出来的、自洽的产物行。输出：(报告, 缺失项)。

    预期行为：没有「答案对不上 / 编号越界 / 未知题族」这类缺失项
    （题族齐全与两种方言由更大的批次保证，这里只查这条行本身）。
    """
    import random

    from jc import assemble
    from workflows.shell import locales, questions, render

    spec = sp.sample_spec(
        random.Random(11),
        sp.ShellCase("bash", "remove_tree", "build_dir", "build_dir", "plain", "imperative", 1),
        "train",
        0,
        0,
    )
    rng = random.Random(11)
    labeled = questions.build(spec, locales.load("zh"), spec.phrasing, rng)
    row = assemble.assemble_sample("shell-train-0000000", render.build(spec, locales.load("zh")), labeled, rng)
    jsonio.write_jsonl(tmp_path / "train.jsonl", [row])
    report, missing = workflow_mod.load("shell").audit(tmp_path, ("train",), "zh")
    assert not [item for item in missing if "重算" in item or "越界" in item or "未知题族" in item]
    assert report["train"]["samples"] == 1
    assert report["train"]["families"]["verdict"] == 1


def test_judges_are_the_declared_defaults() -> None:
    """输入：无。输出：workflow 声明。预期行为：题型三种都产、判别器是两个规则判别器。"""
    workflow = shell_workflow.ShellWorkflow()
    assert workflow.kinds == frozenset({"choice", "noul", "score"})
    assert workflow.judges == ("roundtrip", "label")
    assert workflow.name == "shell" and workflow.version == shell_workflow.WORKFLOW_VERSION
    assert isinstance(judges_mod.RoundtripJudge(), object)


def test_cli_audit_prints_the_shell_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """输入：一次真跑 + `jc audit`。输出：退出码与 stdout。

    预期行为：退出码 0，报告里出现 shell 自己的计数键——通用层的审计打印必须按
    产物报的键走；写死某个 workflow 的键名时，别的 workflow 一跑审计就 KeyError（实测踩过）。
    """
    result = pipeline.run(make_config(tmp_path, "s6"))
    code = cli.main(["audit", "--workflow", "shell", "--dir", str(result.run_dir), "--split", "train"])
    assert code == 0
    out = capsys.readouterr().out
    assert "审计通过" in out
    assert "label_mismatch" in out and "dialects" in out
