"""端到端测试：小规模跑通整条管线、可复现、审计与切分不重叠。

零 token：函数调用是纯构造式（`needs` 为空），所以这里不碰任何后端。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from jc import jsonio, pipeline
from jc import workflow as workflow_mod
from jc.workflow import family_of
from tests.workflows.functions.conftest import tiny_config
from workflows.functions import spec as spec_mod
from workflows.functions import workflow as fn_workflow

#: 几个题族在均衡报告里的键（`类型:题族`）——每个工具各挑一个代表。
BALANCE_KEYS = (
    "choice:tool",
    "choice:arg_window",
    "choice:arg_benchmark",
    "noul:stated_style",
    "noul:stated_include_volume",
    "noul:member_NVDA",
    "noul:stated_direction",
)


def make_config(
    tmp_path: Path, run_id: str, *, split: str = "train", concurrency: int = 1
) -> pipeline.RunConfig:
    """输入：临时目录、run 名、切分、并发度。输出：指向 tmp 的运行配置（不需要后端）。"""
    return pipeline.RunConfig(
        workflow="functions",
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
    return len(fn_workflow.FunctionsWorkflow().build_cases(config.config_override))


def test_end_to_end_all_judges_pass(tmp_path: Path) -> None:
    """输入：极小配置（零 token，不需要后端）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、两个判别器全通过；产物过审计。
    """
    config = make_config(tmp_path, "fn1")
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"roundtrip", "label"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0 and bucket["ok"] > 0
    report, missing = workflow_mod.load("functions").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == result.accepted
    assert report["train"]["index_bad"] == 0
    assert report["train"]["label_mismatch"] == 0
    assert set(report["train"]["types"]) == {"choice", "noul"}
    assert set(report["train"]["tools"]) == set(spec_mod.TOOL_NAMES)


def test_two_runs_are_byte_identical(tmp_path: Path) -> None:
    """输入：同种子、不同并发度跑两次。输出：两份产物。预期行为：逐字节一致。"""
    serial = pipeline.run(make_config(tmp_path, "fn2a", concurrency=1))
    parallel = pipeline.run(make_config(tmp_path, "fn2b", concurrency=4))
    assert (serial.run_dir / "train.jsonl").read_bytes() == (
        parallel.run_dir / "train.jsonl"
    ).read_bytes()


def test_manifest_and_stats_are_written(tmp_path: Path) -> None:
    """输入：一次跑。输出：产物目录。

    预期行为：manifest 记下 workflow / 切分；统计报告里出现五个维度的中文标题与两个判别器。
    """
    result = pipeline.run(make_config(tmp_path, "fn3"))
    manifest = jsonio.loads((result.run_dir / "manifest-train.json").read_text("utf-8"))
    assert manifest["workflow"] == "functions"
    assert manifest["split"] == "train"
    stats = (result.run_dir / "stats-train.md").read_text("utf-8")
    for title in fn_workflow.DIMENSION_TITLES.values():
        assert title in stats
    assert "roundtrip" in stats and "label" in stats


def test_no_family_falls_back_to_a_constant_answer(tmp_path: Path) -> None:
    """输入：一次 train 跑。输出：无。

    预期行为：代表题族都出现在均衡报告里，且每一族在产物里的答案**不止一种**——
    一族只剩一个答案就是在教「永远选同一个」，是坏数据而不是难题（范式 §5.7）。
    """
    result = pipeline.run(make_config(tmp_path, "fn4"))
    keys = {item["key"] for item in result.stats["balance"]}
    for key in BALANCE_KEYS:
        assert key in keys, f"均衡报告里没有 {key}"
    seen: dict[str, set[Any]] = defaultdict(set)
    for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl"):
        for qid, question in row["questions"].items():
            answer = row["answers"][qid]
            value = answer["choice"] if question["type"] == "choice" else answer["noul"]
            seen[family_of(qid)].add(value)
    for family, values in seen.items():
        assert len(values) > 1, f"{family} 的答案退化成常量：{values}"


def test_ood_split_runs(tmp_path: Path) -> None:
    """输入：`test_ood` 切分的极小配置。输出：RunResult。

    预期行为：能跑通、零拒绝，产物条数与 case 数对得上。
    """
    config = make_config(tmp_path, "fn5", split="test_ood")
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "test_ood.jsonl")]
    assert len(rows) == result.accepted


def test_three_id_splits_are_disjoint(tmp_path: Path) -> None:
    """输入：train / val / test_iid 三次跑。输出：三份产物。

    预期行为：三份产物的 state 集合两两不相交——切分按**素材**哈希分桶，同一份请求
    不会同时出现在两档里（`config_for` 忘了替换 `split` 时这条会红）。
    """
    seen: list[set[str]] = []
    for split in ("train", "val", "test_iid"):
        result = pipeline.run(make_config(tmp_path, f"fn6-{split}", split=split))
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
    workflow = workflow_mod.load("functions")
    for split in ("train", "val", "test_iid"):
        cases = workflow.build_cases(workflow.config_for(split))
        assert cases and not any(case.unseen_dims() for case in cases)


def test_ood_config_produces_only_cases_with_unseen_values() -> None:
    """输入：`test_ood` 配置。输出：无。预期行为：每条都至少含一个留出取值。"""
    workflow = workflow_mod.load("functions")
    cases = workflow.build_cases(workflow.config_for("test_ood"))
    assert cases and all(case.unseen_dims() for case in cases)


def test_ood_materials_never_overlap_the_id_splits() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：OOD 侧的素材指纹与同分布侧完全不相交——值留出与素材留出叠加之后
    两侧按构造不重叠。
    """
    workflow = workflow_mod.load("functions")
    id_materials = {
        spec_mod.material_of(case)
        for split in ("train", "val", "test_iid")
        for case in workflow.build_cases(workflow.config_for(split))
    }
    ood_materials = {
        spec_mod.material_of(case)
        for case in workflow.build_cases(workflow.config_for("test_ood"))
    }
    assert ood_materials and not id_materials & ood_materials


def test_split_sizes_add_up_to_the_training_side_matrix() -> None:
    """输入：四个切分的配置。输出：无。

    预期行为：三个同分布切分都非空（切片按**位置**而不是桶号过滤，否则哈希方差会让 val
    只剩个位数），且加起来正好等于训练侧取值表的全枚举——`build_cases(ID_CONFIG)` 本身
    已经切到 train 那一档，所以全枚举要直接问 `spec.build_cases`。
    """
    workflow = workflow_mod.load("functions")
    sizes = {
        split: len(workflow.build_cases(workflow.config_for(split)))
        for split in workflow.split_names()
    }
    assert min(sizes.values()) > 0
    matrix = len(spec_mod.build_cases(fn_workflow.ID_CONFIG))
    assert sum(sizes[split] for split in ("train", "val", "test_iid")) == matrix


def test_audit_reports_a_missing_split(tmp_path: Path) -> None:
    """输入：一个不存在的切分。输出：无。预期行为：审计如实记一笔「产物不存在」。"""
    result = pipeline.run(make_config(tmp_path, "fn7"))
    report, missing = workflow_mod.load("functions").audit(result.run_dir, ("val",), "zh")
    assert any("产物不存在" in item for item in missing)
    assert report["val"]["samples"] == 0
