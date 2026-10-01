"""端到端测试：小规模跑通整条管线、可复现、审计与接线。"""

from __future__ import annotations

from pathlib import Path

import pytest

from jc import jsonio, pipeline
from jc import workflow as workflow_mod
from tests.workflows.sentiment.conftest import tiny_config
from workflows.sentiment import features as ft
from workflows.sentiment import parse, spec as sp
from workflows.sentiment import workflow as sentiment_workflow
from workflows.sentiment.workflow import SentimentWorkflow


def make_config(
    tmp_path: Path, run_id: str, *, split: str = "train", concurrency: int = 1
) -> pipeline.RunConfig:
    """输入：临时目录、run 名、切分、并发度。输出：指向 tmp 的运行配置（不需要后端）。"""
    return pipeline.RunConfig(
        workflow="sentiment",
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
    return len(SentimentWorkflow().build_cases(config.config_override))


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
    report, missing = workflow_mod.load("sentiment").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == result.accepted
    assert report["train"]["index_bad"] == 0
    assert report["train"]["label_mismatch"] == 0
    assert set(report["train"]["types"]) == {"choice", "noul", "score"}


def test_manifest_and_stats_are_written(tmp_path: Path) -> None:
    """输入：一次跑。输出：产物目录。

    预期行为：manifest 记下 workflow / 切分；统计报告里出现两个维度的中文标题与两个判别器。
    """
    result = pipeline.run(make_config(tmp_path, "s3"))
    manifest = jsonio.loads((result.run_dir / "manifest-train.json").read_text("utf-8"))
    assert manifest["workflow"] == "sentiment"
    assert manifest["split"] == "train"
    stats = (result.run_dir / "stats-train.md").read_text("utf-8")
    for title in sentiment_workflow.DIMENSION_TITLES.values():
        assert title in stats
    assert "roundtrip" in stats and "label" in stats


def test_balance_report_covers_every_family_and_has_no_constant(tmp_path: Path) -> None:
    """输入：一次跑。输出：均衡报告。

    预期行为：四个题族都出现在报告里，且**没有哪一族退化成常量或严重偏斜**——常量答案意味着
    这一族在教「永远选同一个」，是坏数据而不是难题。
    """
    result = pipeline.run(make_config(tmp_path, "s4"))
    keys = {item["key"] for item in result.stats["balance"]}
    for family in ("choice:tone", "choice:pivot", "score:drop", "noul:rebound"):
        assert family in keys
    degenerate = [item for item in result.stats["balance"] if item["max_dev"] > 0.2]
    assert not degenerate, f"有题族退化成常量或严重偏斜：{degenerate}"


def test_ood_split_runs(tmp_path: Path) -> None:
    """输入：`test_ood` 的真实配置（不是小规格）。输出：RunResult。

    预期行为：能跑通、零拒绝，产物条数与 case 数对得上，且**每一条的分值序列都是留出形状**
    ——小规格配置不走 `ood_dims`，拿它跑 `test_ood` 只是换个切分名，测不到值留出这条路径。
    """
    workflow = SentimentWorkflow()
    config = pipeline.RunConfig(
        workflow="sentiment",
        split="test_ood",
        locales=("zh",),
        run_seed=17,
        per_case=1,
        concurrency=1,
        out_root=tmp_path,
        run_id="s5",
        config_override=workflow.config_for("test_ood"),
    )
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    tables = parse.merged_tables()
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "test_ood.jsonl")]
    assert len(rows) == result.accepted
    holdout = {ft.arc_of(arc) for arc in ft.OOD_ARCS}
    for row in rows:
        assert parse.parse_state(row["state"], tables).levels in holdout


def test_three_id_splits_are_disjoint(tmp_path: Path) -> None:
    """输入：train / val / test_iid 三次跑。输出：三份产物。

    预期行为：三份产物的 state 集合两两不相交——切分按 case 哈希分桶，同一段对话不会同时
    出现在两档里（`config_for` 忘了替换 `split` 时这条会红）。
    """
    seen: list[set[str]] = []
    for split in ("train", "val", "test_iid"):
        result = pipeline.run(make_config(tmp_path, f"s6-{split}", split=split))
        rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / f"{split}.jsonl")]
        assert rows
        seen.append({str(row["state"]) for row in rows})
    assert not seen[0] & seen[1]
    assert not seen[0] & seen[2]
    assert not seen[1] & seen[2]


def test_id_splits_only_use_the_id_arcs() -> None:
    """输入：三个同分布切分的配置。输出：无。

    预期行为：它们产出的 case 里**一条留出弧线都没有**——否则 test_ood 与同分布侧会重叠。
    """
    workflow = SentimentWorkflow()
    for split in ("train", "val", "test_iid"):
        cases = workflow.build_cases(workflow.config_for(split))
        assert cases
        assert {case.arc for case in cases} <= set(ft.ID_ARCS), split


def test_ood_uses_only_the_holdout_arcs() -> None:
    """输入：`test_ood` 配置。输出：无。预期行为：每一条都用留出弧线。"""
    workflow = SentimentWorkflow()
    cases = workflow.build_cases(workflow.config_for("test_ood"))
    assert cases
    assert {case.arc for case in cases} == set(ft.OOD_ARCS)


def test_case_limit_keeps_full_arc_coverage() -> None:
    """输入：`case_limit=15` 的同分布配置。输出：无。

    预期行为：小批量仍然覆盖全部十五条弧线——纯截断会把某些弧线整片丢掉，四个题族的某些档位
    就取不到了（均衡闸与审计会当场报出来）。
    """
    workflow = SentimentWorkflow()
    full = workflow.build_cases(sp.SentimentConfig(split="train"))
    picked = workflow.build_cases(sp.SentimentConfig(split="train", case_limit=15))
    assert len(picked) == 15
    assert set(picked) <= set(full)
    assert {case.arc for case in picked} == set(ft.ID_ARCS)


def test_cover_pick_prefers_unseen_values() -> None:
    """输入：一条「同弧线、场景循环」的假列表。输出：无。

    预期行为：挑前三个时先补没见过的维度取值，所以三个场景各不相同——直接截断会拿到三条
    一样的，那正是「小批量必然缺一档」的坑。
    """
    cases = [ft.SentimentCase(arc=0, topic=ft.TOPICS[index % len(ft.TOPICS)]) for index in range(20)]
    picked = sentiment_workflow._cover_pick(cases, 3)
    assert picked[0] is cases[0]
    assert len({case.topic for case in picked}) == 3


def test_config_for_switches_the_split() -> None:
    """输入：三个同分布切分名。输出：无。

    预期行为：`ID_CONFIG` 是单例，直接返回它会让三档切出同一份数据——`config_for` 必须
    把 `split` 换成调用方要的那一档。
    """
    workflow = SentimentWorkflow()
    assert workflow.split_names() == ("train", "val", "test_iid", "test_ood")
    assert workflow.config_for("val").split == "val"
    assert workflow.config_for("test_ood") == sp.SentimentConfig(split="test_ood")


def test_config_for_rejects_an_unknown_split() -> None:
    """输入：不存在的切分名。输出：无。预期行为：报错并列出可用切分。"""
    with pytest.raises(KeyError, match="未知切分"):
        SentimentWorkflow().config_for("nope")


def test_workflow_is_construct_only() -> None:
    """输入：无。输出：无。

    预期行为：`needs` 为空（零 token），题型覆盖 choice / noul / score；语言只有 zh / en。
    """
    workflow = SentimentWorkflow()
    assert workflow.needs == frozenset()
    assert workflow.kinds == {"choice", "noul", "score"}
    assert workflow.judges == ("roundtrip", "label")
    assert workflow.available_locales() == ["en", "zh"]
    assert workflow.language_texts({}) == {}


def test_binding_a_backend_is_an_error() -> None:
    """输入：给构造式 workflow 绑后端。输出：无。预期行为：当场报错，不静默吞掉。"""
    with pytest.raises(AssertionError, match="纯构造式"):
        SentimentWorkflow().bind_backend(object())


def test_make_judges_rejects_an_unknown_name() -> None:
    """输入：不存在的判别器名。输出：无。预期行为：报错并列出可用的两个。"""
    with pytest.raises(KeyError, match="未知判别器"):
        SentimentWorkflow().make_judges(("nope",), "zh")


def test_reparse_uses_the_merged_tables() -> None:
    """输入：两种语言渲染的同一个 case。输出：无。预期行为：`reparse` 不带语言也能反解。"""
    workflow = SentimentWorkflow()
    case = ft.SentimentCase(arc=13, topic="neighbors")
    for lang in ("zh", "en"):
        spec = sp.sample_spec(None, case, "train", 0, 0)
        parsed = workflow.reparse(workflow.render(spec, None, lang))
        assert parsed.levels == case.levels


def test_case_label_and_dimensions() -> None:
    """输入：一个 case。输出：无。预期行为：可读描述与两个维度的取值都对得上。"""
    workflow = SentimentWorkflow()
    case = ft.SentimentCase(arc=7, topic="review")
    assert workflow.case_label(case) == "A7-review"
    assert workflow.stats_dimensions(case) == {"arc": 7, "topic": "review"}
    assert set(workflow.dimension_titles()) == {"arc", "topic"}
