"""端到端测试：小规模跑通整条管线、可复现、审计与接线。"""

from __future__ import annotations

from pathlib import Path

from jc import jsonio, pipeline
from jc import workflow as workflow_mod
from tests.workflows.siftrank.conftest import tiny_config
from workflows.siftrank import locales
from workflows.siftrank import workflow as siftrank_workflow


def make_config(
    tmp_path: Path, run_id: str, *, split: str = "train", concurrency: int = 1
) -> pipeline.RunConfig:
    """输入：临时目录、run 名、切分、并发度。输出：指向 tmp 的运行配置（不需要后端）。"""
    return pipeline.RunConfig(
        workflow="siftrank",
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
    return len(siftrank_workflow.SiftRankWorkflow().build_cases(config.config_override))


def test_end_to_end_all_judges_pass(tmp_path: Path) -> None:
    """输入：极小配置（零 token，不需要后端）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、两个判别器全通过；产物过审计。
    """
    config = make_config(tmp_path, "c1")
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"roundtrip", "label"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0 and bucket["ok"] > 0
    report, missing = workflow_mod.load("siftrank").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == result.accepted
    assert report["train"]["index_bad"] == 0
    assert report["train"]["label_mismatch"] == 0
    assert set(report["train"]["types"]) == {"choice", "noul", "score"}


def test_manifest_and_stats_are_written(tmp_path: Path) -> None:
    """输入：一次跑。输出：产物目录。

    预期行为：manifest 记下 workflow / 切分；统计报告里出现三个维度的中文标题与两个判别器。
    """
    result = pipeline.run(make_config(tmp_path, "c3"))
    manifest = jsonio.loads((result.run_dir / "manifest-train.json").read_text("utf-8"))
    assert manifest["workflow"] == "siftrank"
    assert manifest["split"] == "train"
    stats = (result.run_dir / "stats-train.md").read_text("utf-8")
    for title in siftrank_workflow.DIMENSION_TITLES.values():
        assert title in stats
    assert "roundtrip" in stats and "label" in stats


def test_balance_report_covers_every_family_and_has_no_constant(tmp_path: Path) -> None:
    """输入：一次跑。输出：均衡报告。

    预期行为：四个题族都出现在报告里，且**没有哪一族退化成常量**——常量答案意味着这一族
    在教「永远选同一个」，是坏数据而不是难题。
    """
    result = pipeline.run(make_config(tmp_path, "c4"))
    keys = {item["key"] for item in result.stats["balance"]}
    for family in (
        "choice:agg_top",
        "choice:pair_order",
        "score:cut_size",
        "noul:stable",
    ):
        assert family in keys
    degenerate = [item for item in result.stats["balance"] if item["max_dev"] > 0.2]
    assert not degenerate, f"有题族退化成常量或严重偏斜：{degenerate}"


def test_ood_split_runs(tmp_path: Path) -> None:
    """输入：`test_ood` 的真实配置（不是小规格）。输出：RunResult。

    预期行为：能跑通、零拒绝，产物条数与 case 数对得上，且每条都带留出那一档提示——
    小规格配置不走 `ood_only`，拿它跑 `test_ood` 只是换个切分名，测不到值留出这条路径。
    """
    workflow = siftrank_workflow.SiftRankWorkflow()
    config = pipeline.RunConfig(
        workflow="siftrank",
        split="test_ood",
        locales=("zh",),
        run_seed=17,
        per_case=1,
        concurrency=1,
        out_root=tmp_path,
        run_id="c5",
        config_override=workflow.config_for("test_ood"),
    )
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    held_out = locales.load("zh")["flavor"][siftrank_workflow.spec_mod.FLAVORS[-1]]
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "test_ood.jsonl")]
    assert len(rows) == result.accepted
    assert all(str(held_out) in str(row["state"]) for row in rows)


def test_id_config_produces_no_unseen_values() -> None:
    """输入：三个同分布切分的配置。输出：无。

    预期行为：它们产出的 case 里**一个留出取值都没有**——否则 test_ood 与同分布侧会重叠。
    """
    workflow = workflow_mod.load("siftrank")
    for split in ("train", "val", "test_iid"):
        cases = workflow.build_cases(workflow.config_for(split))
        assert cases and not any(case.unseen_dims() for case in cases)


def test_ood_config_produces_only_cases_with_unseen_values() -> None:
    """输入：`test_ood` 配置。输出：无。预期行为：每条都至少含一个留出取值。"""
    workflow = workflow_mod.load("siftrank")
    cases = workflow.build_cases(workflow.config_for("test_ood"))
    assert cases and all(case.unseen_dims() for case in cases)


def test_ood_holds_out_exactly_the_last_flavor() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：同分布侧的提示取值 = 训练侧见过的那几档，`test_ood` 侧**只有**留出那一档——
    这是「值留出」在数据上的形态，只写注释不写断言的话，改一次取值表就能把它悄悄变回同分布。
    """
    workflow = workflow_mod.load("siftrank")
    id_flavors = {
        str(workflow.stats_dimensions(case)["flavor"])
        for split in ("train", "val", "test_iid")
        for case in workflow.build_cases(workflow.config_for(split))
    }
    ood_flavors = {
        str(workflow.stats_dimensions(case)["flavor"])
        for case in workflow.build_cases(workflow.config_for("test_ood"))
    }
    assert id_flavors == set(siftrank_workflow.spec_mod.ID_FLAVORS)
    assert ood_flavors == {siftrank_workflow.spec_mod.FLAVORS[-1]}
    assert ood_flavors - id_flavors


def test_split_sizes_add_up_to_the_training_side_matrix() -> None:
    """输入：四个切分的配置。输出：无。

    预期行为：三个同分布切分都非空（切片按**位置**而不是桶号过滤，否则哈希方差会让 val
    只剩个位数），且加起来正好等于训练侧取值表的全枚举：6 形状 × 3 映射 × 3 提示。
    """
    workflow = workflow_mod.load("siftrank")
    sizes = {
        split: len(workflow.build_cases(workflow.config_for(split)))
        for split in workflow.split_names()
    }
    assert min(sizes.values()) > 0
    assert sum(sizes[split] for split in ("train", "val", "test_iid")) == 6 * 3 * 3


def test_request_id_ignores_the_text_only_dim_so_the_holdout_shares_it() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：请求号由**素材**指纹派生（不含提示），所以同一份请求在 `launch` / `audit`
    两种提示下是**同一个请求号**——这是设计：值留出换的是「怎么读」，不是「问什么」。
    同分布三档合起来的请求号集合与 `test_ood` 侧完全相等。
    """
    workflow = workflow_mod.load("siftrank")
    id_ids = {
        siftrank_workflow.spec_mod.request_id_of(case)
        for split in ("train", "val", "test_iid")
        for case in workflow.build_cases(workflow.config_for(split))
    }
    ood_ids = {
        siftrank_workflow.spec_mod.request_id_of(case)
        for case in workflow.build_cases(workflow.config_for("test_ood"))
    }
    assert id_ids == ood_ids
    assert len(id_ids) == 6 * 3
