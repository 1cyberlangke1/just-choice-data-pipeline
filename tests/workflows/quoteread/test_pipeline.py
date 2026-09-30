"""端到端测试：小规模跑通整条管线、可复现、审计与接线。"""

from __future__ import annotations

from pathlib import Path

from jc import jsonio, pipeline
from jc import workflow as workflow_mod
from tests.workflows.quoteread.conftest import tiny_config
from workflows.quoteread import features as ft
from workflows.quoteread import locales
from workflows.quoteread import spec as sp
from workflows.quoteread import workflow as quoteread_workflow
from workflows.quoteread.workflow import QuoteReadWorkflow


def make_config(
    tmp_path: Path, run_id: str, *, split: str = "train", concurrency: int = 1
) -> pipeline.RunConfig:
    """输入：临时目录、run 名、切分、并发度。输出：指向 tmp 的运行配置（不需要后端）。"""
    return pipeline.RunConfig(
        workflow="quoteread",
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
    return len(QuoteReadWorkflow().build_cases(config.config_override))


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
    report, missing = workflow_mod.load("quoteread").audit(result.run_dir, ("train",), "zh")
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
    assert manifest["workflow"] == "quoteread"
    assert manifest["split"] == "train"
    stats = (result.run_dir / "stats-train.md").read_text("utf-8")
    for title in quoteread_workflow.DIMENSION_TITLES.values():
        assert title in stats
    assert "roundtrip" in stats and "label" in stats


def test_balance_report_covers_every_family_and_has_no_constant(tmp_path: Path) -> None:
    """输入：一次跑。输出：均衡报告。

    预期行为：四个题族都出现在报告里，且**没有哪一族退化成常量**——常量答案意味着这一族
    在教「永远选同一个」，是坏数据而不是难题。
    """
    result = pipeline.run(make_config(tmp_path, "c4"))
    keys = {item["key"] for item in result.stats["balance"]}
    for family in ("score:tier", "choice:leader", "noul:alert", "noul:divergent"):
        assert family in keys
    degenerate = [item for item in result.stats["balance"] if item["max_dev"] > 0.2]
    assert not degenerate, f"有题族退化成常量或严重偏斜：{degenerate}"


def test_ood_split_runs(tmp_path: Path) -> None:
    """输入：`test_ood` 的真实配置（不是小规格）。输出：RunResult。

    预期行为：能跑通、零拒绝，产物条数与 case 数对得上，且每条都带留出标的名——小规格配置
    不走 `ood_dims`，拿它跑 `test_ood` 只是换个切分名，测不到维度留出这条路径。
    """
    workflow = QuoteReadWorkflow()
    config = pipeline.RunConfig(
        workflow="quoteread",
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
    holdout = locales.load("zh")["name"][ft.OOD_NAME_IDS[0]]
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "test_ood.jsonl")]
    assert len(rows) == result.accepted
    assert all(str(holdout) in str(row["state"]) for row in rows)


def test_id_splits_only_use_the_id_name_set() -> None:
    """输入：三个同分布切分的配置。输出：无。

    预期行为：它们产出的 case 里**一个留出名都没有**——否则 test_ood 与同分布侧会重叠。
    """
    workflow = QuoteReadWorkflow()
    for split in ("train", "val", "test_iid"):
        cases = workflow.build_cases(workflow.config_for(split))
        assert cases
        assert {case.names for case in cases} <= set(ft.DIMS["names"]), split


def test_ood_uses_only_the_holdout_name_set() -> None:
    """输入：`test_ood` 配置。输出：无。预期行为：每一条都用留出那组标的名。"""
    workflow = QuoteReadWorkflow()
    cases = workflow.build_cases(workflow.config_for("test_ood"))
    assert cases
    assert {case.names for case in cases} == set(ft.OOD_DIMS["names"])


def test_case_limit_keeps_full_value_and_unit_coverage() -> None:
    """输入：`case_limit=10` 的同分布配置。输出：无。

    预期行为：小批量仍然覆盖该切分里出现过的**全部读数模式与单位模式**——纯截断会把某条
    读数模式整片丢掉，`tier` 的某些档位就取不到了（均衡闸与审计会当场报出来）。
    """
    workflow = QuoteReadWorkflow()
    full = workflow.build_cases(sp.QuoteConfig(split="train"))
    picked = workflow.build_cases(sp.QuoteConfig(split="train", case_limit=10))
    assert len(picked) == 10
    assert set(picked) <= set(full)
    assert {case.values for case in picked} == {case.values for case in full}
    assert {case.units for case in picked} == {case.units for case in full}


def test_cover_pick_prefers_unseen_values() -> None:
    """输入：一条「同读数模式、单位模式循环」的假列表。输出：无。

    预期行为：挑前三个时先补没见过的维度取值，所以三个单位模式各不相同——直接截断会拿到
    三条一样的单位模式，那正是「小批量必然缺一档」的坑。
    """
    cases = [
        ft.QuoteCase(values=0, units=index % len(ft.UNIT_PATTERNS), names=0) for index in range(20)
    ]
    picked = quoteread_workflow._cover_pick(cases, 3)
    assert picked[0] is cases[0]
    assert len({case.units for case in picked}) == 3
