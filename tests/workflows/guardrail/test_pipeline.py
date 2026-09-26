"""端到端测试：小规模跑通整条管线、可复现、审计与接线。"""

from __future__ import annotations

from pathlib import Path

from jc import jsonio, pipeline
from jc import workflow as workflow_mod
from tests.workflows.guardrail.conftest import tiny_config
from workflows.guardrail import workflow as guardrail_workflow


def make_config(
    tmp_path: Path, run_id: str, *, split: str = "train", concurrency: int = 1
) -> pipeline.RunConfig:
    """输入：临时目录、run 名、切分、并发度。输出：指向 tmp 的运行配置（不需要后端）。"""
    return pipeline.RunConfig(
        workflow="guardrail",
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
    return len(guardrail_workflow.GuardrailWorkflow().build_cases(config.config_override))


def test_end_to_end_all_judges_pass(tmp_path: Path) -> None:
    """输入：极小配置（零 token，不需要后端）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、两个判别器全通过；产物过审计。
    """
    config = make_config(tmp_path, "g1")
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"roundtrip", "label"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0 and bucket["ok"] > 0
    report, missing = workflow_mod.load("guardrail").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == result.accepted
    assert report["train"]["index_bad"] == 0
    assert report["train"]["label_mismatch"] == 0
    assert set(report["train"]["types"]) == {"choice", "noul", "score"}


def test_two_runs_are_byte_identical(tmp_path: Path) -> None:
    """输入：同种子、不同并发度跑两次。输出：两份产物。

    预期行为：逐字节一致——RNG 由 (seed, workflow, split, seq) 派生，并发不改变内容。
    """
    serial = pipeline.run(make_config(tmp_path, "g2a", concurrency=1))
    parallel = pipeline.run(make_config(tmp_path, "g2b", concurrency=4))
    assert (serial.run_dir / "train.jsonl").read_bytes() == (
        parallel.run_dir / "train.jsonl"
    ).read_bytes()


def test_manifest_and_stats_are_written(tmp_path: Path) -> None:
    """输入：一次跑。输出：产物目录。

    预期行为：manifest 记下 workflow / 切分；统计报告里出现七个维度的中文标题与
    两个判别器的通过率。
    """
    result = pipeline.run(make_config(tmp_path, "g3"))
    manifest = jsonio.loads((result.run_dir / "manifest-train.json").read_text("utf-8"))
    assert manifest["workflow"] == "guardrail"
    assert manifest["split"] == "train"
    stats = (result.run_dir / "stats-train.md").read_text("utf-8")
    for title in guardrail_workflow.DIMENSION_TITLES.values():
        assert title in stats
    assert "roundtrip" in stats and "label" in stats


def test_balance_report_covers_every_family_and_has_no_constant(tmp_path: Path) -> None:
    """输入：一次跑。输出：均衡报告。

    预期行为：11 个题族都出现在报告里，且**没有哪一族退化成常量**——
    常量答案意味着这一族在教「永远选同一个」，是坏数据而不是难题。
    """
    result = pipeline.run(make_config(tmp_path, "g4"))
    keys = {item["key"] for item in result.stats["balance"]}
    for family in ("choice:action", "choice:top_clause", "score:severity"):
        assert family in keys
    for clause in guardrail_workflow.spec_mod.CLAUSES:
        assert f"noul:violates_{clause}" in keys
    assert "noul:minor_exposure" in keys and "noul:repeat_offender" in keys
    degenerate = [item for item in result.stats["balance"] if item["max_dev"] > 0.5]
    assert not degenerate, f"有题族退化成常量：{degenerate}"


def test_ood_split_runs(tmp_path: Path) -> None:
    """输入：`test_ood` 切分的极小配置。输出：RunResult。

    预期行为：能跑通、零拒绝，产物条数与 case 数对得上。
    """
    config = make_config(tmp_path, "g5", split="test_ood")
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "test_ood.jsonl")]
    assert len(rows) == result.accepted


def test_three_id_splits_are_disjoint(tmp_path: Path) -> None:
    """输入：train / val / test_iid 三次跑。输出：三份产物。

    预期行为：三份产物的 state 集合两两不相交——切分按素材哈希分桶，同一屏内容不会
    同时出现在两档里（`config_for` 忘了替换 `split` 时这条会红）。
    """
    seen: list[set[str]] = []
    for split in ("train", "val", "test_iid"):
        result = pipeline.run(make_config(tmp_path, f"g6-{split}", split=split))
        rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / f"{split}.jsonl")]
        assert rows
        seen.append({row["state"] for row in rows})
    assert not seen[0] & seen[1]
    assert not seen[0] & seen[2]
    assert not seen[1] & seen[2]


def test_id_config_produces_no_unseen_values() -> None:
    """输入：三个同分布切分的配置。输出：无。

    预期行为：它们产出的 case 里**一个留出取值都没有**——否则 test_ood 与同分布侧会重叠。
    """
    workflow = workflow_mod.load("guardrail")
    for split in ("train", "val", "test_iid"):
        cases = workflow.build_cases(workflow.config_for(split))
        assert cases and not any(case.unseen_dims() for case in cases)


def test_ood_config_produces_only_cases_with_unseen_values() -> None:
    """输入：`test_ood` 配置。输出：无。预期行为：每条都至少含一个留出取值。"""
    workflow = workflow_mod.load("guardrail")
    cases = workflow.build_cases(workflow.config_for("test_ood"))
    assert cases and all(case.unseen_dims() for case in cases)
