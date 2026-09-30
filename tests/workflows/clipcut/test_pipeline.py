"""端到端测试：小规模跑通整条管线、可复现、审计与接线。"""

from __future__ import annotations

from pathlib import Path

from jc import jsonio, pipeline
from jc import workflow as workflow_mod
from tests.workflows.clipcut.conftest import tiny_config
from workflows.clipcut import features as ft
from workflows.clipcut import locales
from workflows.clipcut import spec as sp
from workflows.clipcut import workflow as clipcut_workflow
from workflows.clipcut.workflow import ClipCutWorkflow


def make_config(
    tmp_path: Path, run_id: str, *, split: str = "train", concurrency: int = 1
) -> pipeline.RunConfig:
    """输入：临时目录、run 名、切分、并发度。输出：指向 tmp 的运行配置（不需要后端）。"""
    return pipeline.RunConfig(
        workflow="clipcut",
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
    return len(ClipCutWorkflow().build_cases(config.config_override))


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
    report, missing = workflow_mod.load("clipcut").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == result.accepted
    assert report["train"]["index_bad"] == 0
    assert report["train"]["label_mismatch"] == 0
    assert set(report["train"]["types"]) == {"choice", "noul", "score"}


def test_manifest_and_stats_are_written(tmp_path: Path) -> None:
    """输入：一次跑。输出：产物目录。

    预期行为：manifest 记下 workflow / 切分；统计报告里出现两个维度的中文标题与两个判别器。
    """
    result = pipeline.run(make_config(tmp_path, "c3"))
    manifest = jsonio.loads((result.run_dir / "manifest-train.json").read_text("utf-8"))
    assert manifest["workflow"] == "clipcut"
    assert manifest["split"] == "train"
    stats = (result.run_dir / "stats-train.md").read_text("utf-8")
    for title in clipcut_workflow.DIMENSION_TITLES.values():
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
        "score:keep_count",
        "choice:first_kept",
        "noul:blocked_any",
        "noul:dup_any",
    ):
        assert family in keys
    degenerate = [item for item in result.stats["balance"] if item["max_dev"] > 0.2]
    assert not degenerate, f"有题族退化成常量或严重偏斜：{degenerate}"


def test_ood_split_runs(tmp_path: Path) -> None:
    """输入：`test_ood` 的真实配置（不是小规格）。输出：RunResult。

    预期行为：能跑通、零拒绝，产物条数与 case 数对得上，且每条都带留出主题——小规格配置
    不走 `ood_dims`，拿它跑 `test_ood` 只是换个切分名，测不到值留出这条路径。
    """
    workflow = ClipCutWorkflow()
    config = pipeline.RunConfig(
        workflow="clipcut",
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
    holdout = locales.load("zh")["topic"][ft.OOD_TOPIC_ID]
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "test_ood.jsonl")]
    assert len(rows) == result.accepted
    assert all(str(holdout) in str(row["state"]) for row in rows)


def test_three_id_splits_are_disjoint(tmp_path: Path) -> None:
    """输入：train / val / test_iid 三次跑。输出：三份产物。

    预期行为：三份产物的 state 集合两两不相交——切分按 case 哈希分桶，同一屏内容不会同时
    出现在两档里（`config_for` 忘了替换 `split` 时这条会红）。
    """
    seen: list[set[str]] = []
    for split in ("train", "val", "test_iid"):
        result = pipeline.run(make_config(tmp_path, f"c6-{split}", split=split))
        rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / f"{split}.jsonl")]
        assert rows
        seen.append({str(row["state"]) for row in rows})
    assert not seen[0] & seen[1]
    assert not seen[0] & seen[2]
    assert not seen[1] & seen[2]


def test_id_splits_only_use_the_id_topic_patterns() -> None:
    """输入：三个同分布切分的配置。输出：无。

    预期行为：它们产出的 case 里**一个留出主题序列都没有**——否则 test_ood 与同分布侧会重叠。
    """
    workflow = ClipCutWorkflow()
    for split in ("train", "val", "test_iid"):
        cases = workflow.build_cases(workflow.config_for(split))
        assert cases
        assert {case.topics for case in cases} <= set(ft.ID_TOPIC_PATTERNS), split


def test_ood_uses_only_the_holdout_topic_pattern() -> None:
    """输入：`test_ood` 配置。输出：无。预期行为：每一条都用留出那条主题序列。"""
    workflow = ClipCutWorkflow()
    cases = workflow.build_cases(workflow.config_for("test_ood"))
    assert cases
    assert {case.topics for case in cases} == set(ft.OOD_TOPIC_PATTERNS)


def test_case_limit_keeps_full_topic_coverage() -> None:
    """输入：`case_limit=10` 的同分布配置。输出：无。

    预期行为：小批量仍然覆盖全部四条主题序列——纯截断会把某条主题序列整片丢掉，
    `first_kept` 的某些档位就取不到了（均衡闸与审计会当场报出来）。
    """
    workflow = ClipCutWorkflow()
    full = workflow.build_cases(sp.ClipConfig(split="train"))
    picked = workflow.build_cases(sp.ClipConfig(split="train", case_limit=10))
    assert len(picked) == 10
    assert set(picked) <= set(full)
    assert {case.topics for case in picked} == set(ft.ID_TOPIC_PATTERNS)


def test_cover_pick_prefers_unseen_values() -> None:
    """输入：一条「同主题、高亮序列循环」的假列表。输出：无。

    预期行为：挑前三个时先补没见过的维度取值，所以三个高亮序列各不相同——直接截断会拿到
    三条一样的高亮序列，那正是「小批量必然缺一档」的坑。
    """
    cases = [ft.ClipCase(topics=0, flags=index % len(ft.FLAG_PATTERNS)) for index in range(20)]
    picked = clipcut_workflow._cover_pick(cases, 3)
    assert picked[0] is cases[0]
    assert len({case.flags for case in picked}) == 3
