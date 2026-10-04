"""端到端测试：小规模跑通整条管线、可复现、审计与接线。"""

from __future__ import annotations

from pathlib import Path

from jc import jsonio, pipeline
from jc import workflow as workflow_mod
from tests.workflows._scaffold import run_config
from tests.workflows.cachereuse.conftest import tiny_config
from workflows.cachereuse import locales
from workflows.cachereuse import workflow as cachereuse_workflow


def make_config(
    tmp_path: Path, run_id: str, *, split: str = "train", concurrency: int = 1
) -> pipeline.RunConfig:
    """输入：临时目录、run 名、切分、并发度。输出：指向 tmp 的运行配置（不需要后端）。"""
    return run_config(
        tmp_path,
        run_id,
        workflow="cachereuse",
        split=split,
        seed=17,
        concurrency=concurrency,
        override=tiny_config(split),
    )


def _case_count(config: pipeline.RunConfig) -> int:
    """输入：运行配置。输出：该切分的 case 数。预期行为：与产物条数对得上。"""
    return len(cachereuse_workflow.CacheReuseWorkflow().build_cases(config.config_override))


def test_ood_split_runs(tmp_path: Path) -> None:
    """输入：`test_ood` 的真实配置（不是小规格）。输出：RunResult。

    预期行为：能跑通、零拒绝，产物条数与 case 数对得上，且每条都带留出那一档来源——
    小规格配置不走 `ood_only`，拿它跑 `test_ood` 只是换个切分名，测不到值留出这条路径。
    """
    workflow = cachereuse_workflow.CacheReuseWorkflow()
    config = pipeline.RunConfig(
        workflow="cachereuse",
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
    kiosk = locales.load("zh")["flavor"][cachereuse_workflow.spec_mod.FLAVORS[-1]]
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "test_ood.jsonl")]
    assert len(rows) == result.accepted
    assert all(kiosk in str(row["state"]) for row in rows)


def test_ood_holds_out_exactly_the_last_flavor() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：同分布侧的来源取值 = 训练侧见过的那几档，`test_ood` 侧**只有**留出那一档——
    这是「值留出」在数据上的形态，只写注释不写断言的话，改一次取值表就能把它悄悄变回同分布。
    """
    workflow = workflow_mod.load("cachereuse")
    id_flavors = {
        str(workflow.stats_dimensions(case)["flavor"])
        for split in ("train", "val", "test_iid")
        for case in workflow.build_cases(workflow.config_for(split))
    }
    ood_flavors = {
        str(workflow.stats_dimensions(case)["flavor"])
        for case in workflow.build_cases(workflow.config_for("test_ood"))
    }
    assert id_flavors == set(cachereuse_workflow.spec_mod.ID_FLAVORS)
    assert ood_flavors == {cachereuse_workflow.spec_mod.FLAVORS[-1]}
    assert ood_flavors - id_flavors


def test_split_sizes_add_up_to_the_training_side_matrix() -> None:
    """输入：四个切分的配置。输出：无。

    预期行为：三个同分布切分都非空（切片按**位置**而不是桶号过滤，否则哈希方差会让 val
    只剩个位数），且加起来正好等于训练侧取值表的全枚举：3 步数 × 4 档 × 2 分叉原因
    × 2 写法 × 2 来源。
    """
    workflow = workflow_mod.load("cachereuse")
    sizes = {
        split: len(workflow.build_cases(workflow.config_for(split)))
        for split in workflow.split_names()
    }
    assert min(sizes.values()) > 0
    assert sum(sizes[split] for split in ("train", "val", "test_iid")) == 3 * 4 * 2 * 2 * 2


def test_query_id_ignores_the_text_only_dims_so_the_holdout_shares_it() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：请求号由**素材**指纹派生（不含写法与来源），所以同一条请求在 `web` / `kiosk`
    两种来源下是**同一个请求号**——这是设计：值留出换的是「怎么读」，不是「问什么」。
    同分布三档合起来的请求号集合与 `test_ood` 侧完全相等。
    """
    workflow = workflow_mod.load("cachereuse")
    id_ids = {
        cachereuse_workflow.spec_mod.query_id_of(case)
        for split in ("train", "val", "test_iid")
        for case in workflow.build_cases(workflow.config_for(split))
    }
    ood_ids = {
        cachereuse_workflow.spec_mod.query_id_of(case)
        for case in workflow.build_cases(workflow.config_for("test_ood"))
    }
    assert id_ids == ood_ids
    assert len(id_ids) == 3 * 4 * 2
