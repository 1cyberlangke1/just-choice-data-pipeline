"""端到端测试：小规模跑通整条管线、可复现、审计与接线。"""

from __future__ import annotations

from pathlib import Path

import pytest

from jc import jsonio, pipeline
from jc import workflow as workflow_mod
from tests.workflows.planstep.conftest import tiny_config
from workflows.planstep import features as ft
from workflows.planstep import parse, spec as sp
from workflows.planstep import workflow as planstep_workflow
from workflows.planstep.workflow import PlanStepWorkflow


def make_config(
    tmp_path: Path, run_id: str, *, split: str = "train", concurrency: int = 1
) -> pipeline.RunConfig:
    """输入：临时目录、run 名、切分、并发度。输出：指向 tmp 的运行配置（不需要后端）。"""
    return pipeline.RunConfig(
        workflow="planstep",
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
    return len(PlanStepWorkflow().build_cases(config.config_override))


def test_end_to_end_all_judges_pass(tmp_path: Path) -> None:
    """输入：极小配置（零 token，不需要后端）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、两个判别器全通过；产物过审计。
    """
    config = make_config(tmp_path, "p1")
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"roundtrip", "label"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0 and bucket["ok"] > 0
    report, missing = workflow_mod.load("planstep").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == result.accepted
    assert report["train"]["index_bad"] == 0
    assert report["train"]["label_mismatch"] == 0
    assert set(report["train"]["types"]) == {"choice", "noul", "score"}


def test_manifest_and_stats_are_written(tmp_path: Path) -> None:
    """输入：一次跑。输出：产物目录。

    预期行为：manifest 记下 workflow / 切分；统计报告里出现两个维度的中文标题与两个判别器。
    """
    result = pipeline.run(make_config(tmp_path, "p3"))
    manifest = jsonio.loads((result.run_dir / "manifest-train.json").read_text("utf-8"))
    assert manifest["workflow"] == "planstep"
    assert manifest["split"] == "train"
    stats = (result.run_dir / "stats-train.md").read_text("utf-8")
    for title in planstep_workflow.DIMENSION_TITLES.values():
        assert title in stats
    assert "roundtrip" in stats and "label" in stats


def test_balance_report_covers_every_family_and_has_no_constant(tmp_path: Path) -> None:
    """输入：一次跑。输出：均衡报告。

    预期行为：四个题族都出现在报告里，且**没有哪一族退化成常量**——常量答案意味着这一族
    在教「永远选同一个」，是坏数据而不是难题。
    """
    result = pipeline.run(make_config(tmp_path, "p4"))
    keys = {item["key"] for item in result.stats["balance"]}
    for family in ("score:remaining", "choice:next", "score:rounds", "noul:unlocks"):
        assert family in keys
    degenerate = [item for item in result.stats["balance"] if item["max_dev"] > 0.2]
    assert not degenerate, f"有题族退化成常量或严重偏斜：{degenerate}"


def test_ood_split_runs(tmp_path: Path) -> None:
    """输入：`test_ood` 的真实配置（不是小规格）。输出：RunResult。

    预期行为：能跑通、零拒绝，产物条数与 case 数对得上，且**每一条的依赖图都是留出形状**
    ——小规格配置不走 `ood_dims`，拿它跑 `test_ood` 只是换个切分名，测不到值留出这条路径。
    """
    workflow = PlanStepWorkflow()
    config = pipeline.RunConfig(
        workflow="planstep",
        split="test_ood",
        locales=("zh",),
        run_seed=17,
        per_case=1,
        concurrency=1,
        out_root=tmp_path,
        run_id="p5",
        config_override=workflow.config_for("test_ood"),
    )
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    tables = parse.merged_tables()
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "test_ood.jsonl")]
    assert len(rows) == result.accepted
    holdout = {ft.deps_of(graph) for graph in ft.OOD_GRAPHS}
    for row in rows:
        assert parse.parse_state(row["state"], tables).deps in holdout


def test_three_id_splits_are_disjoint(tmp_path: Path) -> None:
    """输入：train / val / test_iid 三次跑。输出：三份产物。

    预期行为：三份产物的 state 集合两两不相交——切分按 case 哈希分桶，同一张计划不会同时
    出现在两档里（`config_for` 忘了替换 `split` 时这条会红）。
    """
    seen: list[set[str]] = []
    for split in ("train", "val", "test_iid"):
        result = pipeline.run(make_config(tmp_path, f"p6-{split}", split=split))
        rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / f"{split}.jsonl")]
        assert rows
        seen.append({str(row["state"]) for row in rows})
    assert not seen[0] & seen[1]
    assert not seen[0] & seen[2]
    assert not seen[1] & seen[2]


def test_id_splits_only_use_the_id_graphs() -> None:
    """输入：三个同分布切分的配置。输出：无。

    预期行为：它们产出的 case 里**一个留出形状都没有**——否则 test_ood 与同分布侧会重叠。
    """
    workflow = PlanStepWorkflow()
    for split in ("train", "val", "test_iid"):
        cases = workflow.build_cases(workflow.config_for(split))
        assert cases
        assert {case.graph for case in cases} <= set(ft.ID_GRAPHS), split


def test_ood_uses_only_the_holdout_graphs() -> None:
    """输入：`test_ood` 配置。输出：无。预期行为：每一条都用留出图形状。"""
    workflow = PlanStepWorkflow()
    cases = workflow.build_cases(workflow.config_for("test_ood"))
    assert cases
    assert {case.graph for case in cases} == set(ft.OOD_GRAPHS)


def test_case_limit_keeps_full_graph_coverage() -> None:
    """输入：`case_limit=10` 的同分布配置。输出：无。

    预期行为：小批量仍然覆盖全部十张图——纯截断会把某张图整片丢掉，`next` 与 `remaining`
    的均匀性就没了（均衡闸与审计会当场报出来）。
    """
    workflow = PlanStepWorkflow()
    full = workflow.build_cases(sp.PlanConfig(split="train"))
    picked = workflow.build_cases(sp.PlanConfig(split="train", case_limit=10))
    assert len(picked) == 10
    assert set(picked) <= set(full)
    assert {case.graph for case in picked} == set(ft.ID_GRAPHS)


def test_cover_pick_prefers_unseen_values() -> None:
    """输入：一条「同图、已完成项数循环」的假列表。输出：无。

    预期行为：挑前三个时先补没见过的维度取值，所以三个已完成项数各不相同——直接截断会
    拿到三条一样的，那正是「小批量必然缺一档」的坑。
    """
    cases = [ft.PlanCase(graph=0, done=index % 6) for index in range(20)]
    picked = planstep_workflow._cover_pick(cases, 3)
    assert picked[0] is cases[0]
    assert len({case.done for case in picked}) == 3


def test_config_for_switches_the_split() -> None:
    """输入：三个同分布切分名。输出：无。

    预期行为：`ID_CONFIG` 是单例，直接返回它会让三档切出同一份数据——`config_for` 必须
    把 `split` 换成调用方要的那一档。
    """
    workflow = PlanStepWorkflow()
    assert workflow.split_names() == ("train", "val", "test_iid", "test_ood")
    assert workflow.config_for("val").split == "val"
    assert workflow.config_for("test_ood") == sp.PlanConfig(split="test_ood")


def test_config_for_rejects_an_unknown_split() -> None:
    """输入：不存在的切分名。输出：无。预期行为：报错并列出可用切分。"""
    with pytest.raises(KeyError, match="未知切分"):
        PlanStepWorkflow().config_for("nope")


def test_workflow_is_construct_only() -> None:
    """输入：无。输出：无。

    预期行为：`needs` 为空（零 token），题型覆盖 choice / noul / score；被绑后端说明接线错了。
    """
    workflow = PlanStepWorkflow()
    assert workflow.needs == frozenset()
    assert workflow.kinds == {"choice", "noul", "score"}
    assert workflow.judges == ("roundtrip", "label")
    assert workflow.available_locales() == ["en", "zh"]
    assert workflow.language_texts({}) == {}


def test_binding_a_backend_is_an_error() -> None:
    """输入：给构造式 workflow 绑后端。输出：无。预期行为：当场报错，不静默吞掉。"""
    with pytest.raises(AssertionError, match="纯构造式"):
        PlanStepWorkflow().bind_backend(object())


def test_make_judges_rejects_an_unknown_name() -> None:
    """输入：不存在的判别器名。输出：无。预期行为：报错并列出可用的两个。"""
    with pytest.raises(KeyError, match="未知判别器"):
        PlanStepWorkflow().make_judges(("nope",), "zh")


def test_reparse_uses_the_merged_tables(tmp_path: Path) -> None:
    """输入：两种语言渲染的同一个 case。输出：无。预期行为：`reparse` 不带语言也能反解。"""
    workflow = PlanStepWorkflow()
    case = ft.PlanCase(graph=9, done=2)
    for lang in ("zh", "en"):
        spec = sp.sample_spec(None, case, "train", 0, 0)
        parsed = workflow.reparse(workflow.render(spec, None, lang))
        assert parsed.deps == case.deps
        assert parsed.done_set == case.done_set


def test_case_label_and_dimensions() -> None:
    """输入：一个 case。输出：无。预期行为：可读描述与两个维度的取值都对得上。"""
    workflow = PlanStepWorkflow()
    case = ft.PlanCase(graph=7, done=3)
    assert workflow.case_label(case) == "G7-D3"
    assert workflow.stats_dimensions(case) == {"graph": 7, "done": 3}
    assert set(workflow.dimension_titles()) == {"graph", "done"}
