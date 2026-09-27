"""端到端测试：小规模跑通整条管线、可复现、审计与切分不重叠、均衡靠构造。"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from jc import jsonio, pipeline
from jc import workflow as workflow_mod
from tests.workflows.review.conftest import tiny_config
from workflows.review import spec as spec_mod
from workflows.review.workflow import ReviewWorkflow

#: 四个题族在均衡报告里的键（`类型:题族`）。
BALANCE_KEYS = ("noul:flag", "score:severity", "choice:gate", "choice:top")


def make_config(
    tmp_path: Path, run_id: str, *, split: str = "train", concurrency: int = 1
) -> pipeline.RunConfig:
    """输入：临时目录、run 名、切分、并发度。输出：指向 tmp 的运行配置（不需要后端）。"""
    return pipeline.RunConfig(
        workflow="review",
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
    return len(ReviewWorkflow().build_cases(config.config_override))


def test_end_to_end_all_judges_pass(tmp_path: Path) -> None:
    """输入：极小配置（零 token，不需要后端）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、两个判别器全通过；产物过审计。
    """
    config = make_config(tmp_path, "rv1")
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"roundtrip", "label"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0 and bucket["ok"] > 0
    report, missing = workflow_mod.load("review").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == result.accepted
    assert report["train"]["index_bad"] == 0
    assert report["train"]["label_mismatch"] == 0
    assert set(report["train"]["types"]) == {"choice", "noul", "score"}
    assert set(report["train"]["families"]) == {"flag", "severity", "gate", "top"}


def test_two_runs_are_byte_identical(tmp_path: Path) -> None:
    """输入：同种子、不同并发度跑两次。输出：两份产物。预期行为：逐字节一致。"""
    serial = pipeline.run(make_config(tmp_path, "rv2a", concurrency=1))
    parallel = pipeline.run(make_config(tmp_path, "rv2b", concurrency=4))
    assert (serial.run_dir / "train.jsonl").read_bytes() == (
        parallel.run_dir / "train.jsonl"
    ).read_bytes()


def test_every_case_yields_exactly_one_sample(tmp_path: Path) -> None:
    """输入：一次 train 跑。输出：无。预期行为：样本数 = case 数（per_case=1）。"""
    config = make_config(tmp_path, "rv3")
    result = pipeline.run(config)
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl")]
    assert len(rows) == _case_count(config)
    for row in rows:
        assert row["id"].startswith("review-train-")
        assert row["questions"] and row["answers"]
        assert set(row["questions"]) == set(row["answers"])


def test_ood_split_only_carries_unseen_values(tmp_path: Path) -> None:
    """输入：一次 test_ood 跑。输出：无。预期行为：零拒绝、审计零问题。"""
    config = make_config(tmp_path, "rv4", split="test_ood")
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    report, missing = workflow_mod.load("review").audit(result.run_dir, ("test_ood",), "zh")
    assert missing == [] and report["test_ood"]["samples"] == result.accepted


def test_three_id_splits_are_disjoint(tmp_path: Path) -> None:
    """输入：train / val / test_iid 三次跑。输出：三份产物。

    预期行为：三份产物的 state 集合两两不相交——切分按**素材**哈希分桶，同一屏改动
    不会同时出现在两档里（`config_for` 忘了替换 `split` 时这条会红）。
    """
    seen: list[set[str]] = []
    for split in ("train", "val", "test_iid"):
        result = pipeline.run(make_config(tmp_path, f"rv5-{split}", split=split))
        rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / f"{split}.jsonl")]
        assert rows
        seen.append({row["state"] for row in rows})
    assert not seen[0] & seen[1]
    assert not seen[0] & seen[2]
    assert not seen[1] & seen[2]


def test_states_are_distinct_within_a_split(tmp_path: Path) -> None:
    """输入：一次 train 跑。输出：无。

    预期行为：同一档里的 state 两两不同——「case 维度不同 ⇒ 文本不同」是构造不变量，
    破了它就会出现两条一模一样的样本。
    """
    result = pipeline.run(make_config(tmp_path, "rv6"))
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl")]
    states = [row["state"] for row in rows]
    assert len(set(states)) == len(states)


def test_answer_balance_is_built_in(tmp_path: Path) -> None:
    """输入：一次 train 跑。输出：无。

    预期行为：四个题族的答案分布全在容差内——该不该提 50/50、严重度与门控各三档均分、
    最严重的位置六处均分。均衡是**构造**出来的，不靠事后过滤。
    """
    result = pipeline.run(make_config(tmp_path, "rv7"))
    offenders = [item for item in result.stats["balance"] if not item["ok"]]
    assert offenders == [], offenders
    assert {item["key"] for item in result.stats["balance"]} == set(BALANCE_KEYS)


def test_id_config_produces_no_unseen_values() -> None:
    """输入：三个同分布切分的配置。输出：无。

    预期行为：它们产出的 case 里**一个留出取值都没有**——否则 test_ood 与同分布侧会重叠。
    """
    workflow = workflow_mod.load("review")
    for split in ("train", "val", "test_iid"):
        cases = workflow.build_cases(workflow.config_for(split))
        assert cases and not any(case.unseen_dims() for case in cases)


def test_ood_config_produces_only_cases_with_unseen_values() -> None:
    """输入：`test_ood` 配置。输出：无。预期行为：每条都至少含一个留出取值。"""
    workflow = workflow_mod.load("review")
    cases = workflow.build_cases(workflow.config_for("test_ood"))
    assert cases and all(case.unseen_dims() for case in cases)


def test_ood_materials_never_overlap_the_id_splits() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：OOD 侧的素材指纹与同分布侧完全不相交——值留出与素材留出叠加之后
    两侧按构造不重叠。
    """
    workflow = workflow_mod.load("review")
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


def test_english_locale_runs_end_to_end(tmp_path: Path) -> None:
    """输入：英文一次跑。输出：RunResult。预期行为：零拒绝、审计零问题。"""
    config = dataclasses.replace(make_config(tmp_path, "rv8"), locales=("en",))
    result = pipeline.run(config)
    assert result.accepted == _case_count(config)
    report, missing = workflow_mod.load("review").audit(result.run_dir, ("train",), "en")
    assert missing == [] and report["train"]["samples"] == result.accepted
