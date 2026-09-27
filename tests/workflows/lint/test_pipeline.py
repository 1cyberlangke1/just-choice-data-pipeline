"""端到端测试：小规模跑通整条管线、可复现、审计与切分不重叠。

零 token：行级规则检查是纯构造式（`needs` 为空），所以这里不碰任何后端。
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import Any

from jc import jsonio, pipeline
from jc import workflow as workflow_mod
from jc.workflow import family_of
from tests.workflows.lint.conftest import tiny_config
from workflows.lint import spec as spec_mod
from workflows.lint import workflow as lint_workflow

#: 四个题族在均衡报告里的键（`类型:题族`）。
BALANCE_KEYS = ("noul:line", "choice:count", "choice:gate", "choice:spot")


def make_config(
    tmp_path: Path, run_id: str, *, split: str = "train", concurrency: int = 1
) -> pipeline.RunConfig:
    """输入：临时目录、run 名、切分、并发度。输出：指向 tmp 的运行配置（不需要后端）。"""
    return pipeline.RunConfig(
        workflow="lint",
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
    return len(lint_workflow.LintWorkflow().build_cases(config.config_override))


def test_end_to_end_all_judges_pass(tmp_path: Path) -> None:
    """输入：极小配置（零 token，不需要后端）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、两个判别器全通过；产物过审计。
    """
    config = make_config(tmp_path, "ln1")
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"roundtrip", "label"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0 and bucket["ok"] > 0
    report, missing = workflow_mod.load("lint").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == result.accepted
    assert report["train"]["index_bad"] == 0
    assert report["train"]["label_mismatch"] == 0
    assert set(report["train"]["types"]) == {"choice", "noul"}
    # tiny_config 四张表全开（含留出的 `long`），所以这一档的规则就是整张规则表。
    assert set(report["train"]["rules"]) == set(spec_mod.RULES)


def test_two_runs_are_byte_identical(tmp_path: Path) -> None:
    """输入：同种子、不同并发度跑两次。输出：两份产物。预期行为：逐字节一致。"""
    serial = pipeline.run(make_config(tmp_path, "ln2a", concurrency=1))
    parallel = pipeline.run(make_config(tmp_path, "ln2b", concurrency=4))
    assert (serial.run_dir / "train.jsonl").read_bytes() == (
        parallel.run_dir / "train.jsonl"
    ).read_bytes()


def test_manifest_and_stats_are_written(tmp_path: Path) -> None:
    """输入：一次跑。输出：产物目录。

    预期行为：manifest 记下 workflow / 切分；统计报告里出现四个维度的中文标题与两个判别器。
    """
    result = pipeline.run(make_config(tmp_path, "ln3"))
    manifest = jsonio.loads((result.run_dir / "manifest-train.json").read_text("utf-8"))
    assert manifest["workflow"] == "lint"
    assert manifest["split"] == "train"
    stats = (result.run_dir / "stats-train.md").read_text("utf-8")
    for title in lint_workflow.DIMENSION_TITLES.values():
        assert title in stats
    assert "roundtrip" in stats and "label" in stats


def test_no_family_falls_back_to_a_constant_answer(tmp_path: Path) -> None:
    """输入：一次 train 跑。输出：无。

    预期行为：四个题族都出现在均衡报告里，且每一族在产物里的答案**不止一种**——
    一族只剩一个答案就是在教「永远选同一个」，是坏数据而不是难题（范式 §5.7）。
    """
    result = pipeline.run(make_config(tmp_path, "ln4"))
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


def test_spot_only_appears_when_exactly_one_line_matches(tmp_path: Path) -> None:
    """输入：一次 train 跑。输出：无。

    预期行为：`spot` 题只出现在「恰好一行命中」的样本里——多行命中时「哪一行命中」没有
    唯一解，出了就是坏数据。
    """
    result = pipeline.run(make_config(tmp_path, "ln5"))
    for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl"):
        families = {family_of(qid) for qid in row["questions"]}
        hits = sum(1 for qid in row["questions"] if family_of(qid) == "line"
                   and row["answers"][qid]["noul"])
        assert ("spot" in families) == (hits == 1), f"命中 {hits} 行却{'有' if 'spot' in families else '没有'} spot 题"


def test_ood_split_runs(tmp_path: Path) -> None:
    """输入：`test_ood` 切分的极小配置。输出：RunResult。

    预期行为：能跑通、零拒绝，产物条数与 case 数对得上，且每种规则都出现过（含留出的
    `long`）。
    """
    config = make_config(tmp_path, "ln6", split="test_ood")
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "test_ood.jsonl")]
    assert len(rows) == result.accepted
    report, missing = workflow_mod.load("lint").audit(result.run_dir, ("test_ood",), "zh")
    assert missing == []
    assert set(report["test_ood"]["rules"]) == set(spec_mod.RULES)


def test_three_id_splits_are_disjoint(tmp_path: Path) -> None:
    """输入：train / val / test_iid 三次跑。输出：三份产物。

    预期行为：三份产物的 state 集合两两不相交——切分按**素材**哈希分桶，同一份片段
    不会同时出现在两档里（`config_for` 忘了替换 `split` 时这条会红）。
    """
    seen: list[set[str]] = []
    for split in ("train", "val", "test_iid"):
        result = pipeline.run(make_config(tmp_path, f"ln7-{split}", split=split))
        rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / f"{split}.jsonl")]
        assert rows
        seen.append({row["state"] for row in rows})
    assert not seen[0] & seen[1]
    assert not seen[0] & seen[2]
    assert not seen[1] & seen[2]


def test_states_are_distinct_within_a_split(tmp_path: Path) -> None:
    """输入：一次 train 跑。输出：无。

    预期行为：同一档里的 state 两两不同——「case 维度不同 ⇒ 文本不同」是构造不变量，
    破了它就会出现两条一模一样的样本（同素材同题不同答的温床）。
    """
    result = pipeline.run(make_config(tmp_path, "ln8"))
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl")]
    states = [row["state"] for row in rows]
    assert len(set(states)) == len(states)


def test_id_config_produces_no_unseen_values() -> None:
    """输入：三个同分布切分的配置。输出：无。

    预期行为：它们产出的 case 里**一个留出取值都没有**——否则 test_ood 与同分布侧会重叠。
    """
    workflow = workflow_mod.load("lint")
    for split in ("train", "val", "test_iid"):
        cases = workflow.build_cases(workflow.config_for(split))
        assert cases and not any(case.unseen_dims() for case in cases)


def test_ood_config_produces_only_cases_with_unseen_values() -> None:
    """输入：`test_ood` 配置。输出：无。预期行为：每条都至少含一个留出取值。"""
    workflow = workflow_mod.load("lint")
    cases = workflow.build_cases(workflow.config_for("test_ood"))
    assert cases and all(case.unseen_dims() for case in cases)


def test_ood_materials_never_overlap_the_id_splits() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：OOD 侧的素材指纹与同分布侧完全不相交——值留出与素材留出叠加之后
    两侧按构造不重叠。
    """
    workflow = workflow_mod.load("lint")
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
    只剩个位数），且加起来正好等于训练侧取值表的全枚举。
    """
    workflow = workflow_mod.load("lint")
    sizes = {
        split: len(workflow.build_cases(workflow.config_for(split)))
        for split in workflow.split_names()
    }
    assert min(sizes.values()) > 0
    matrix = len(spec_mod.build_cases(lint_workflow.ID_CONFIG))
    assert sum(sizes[split] for split in ("train", "val", "test_iid")) == matrix


def test_english_locale_runs_end_to_end(tmp_path: Path) -> None:
    """输入：英文一次跑。输出：RunResult。预期行为：零拒绝、审计零问题。"""
    config = make_config(tmp_path, "ln9")
    config = dataclasses.replace(config, locales=("en",))
    result = pipeline.run(config)
    assert result.accepted == _case_count(config)
    report, missing = workflow_mod.load("lint").audit(result.run_dir, ("train",), "en")
    assert missing == [] and report["train"]["samples"] == result.accepted
