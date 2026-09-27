"""端到端测试：小规模跑通整条管线、可复现、审计与切分不重叠、均衡靠构造。"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from pathlib import Path
from typing import Any

from jc import jsonio, pipeline
from jc import workflow as workflow_mod
from jc.workflow import family_of
from tests.workflows.modelroute.conftest import tiny_config
from workflows.modelroute import judges, locales, parse
from workflows.modelroute import spec as spec_mod
from workflows.modelroute.workflow import ModelRouteWorkflow

#: 六个题族在均衡报告里的键（`类型:题族`）。
BALANCE_KEYS = (
    "choice:effort",
    "choice:next",
    "choice:tier",
    "noul:cache",
    "noul:degrade",
    "noul:is_first",
)


def make_config(
    tmp_path: Path, run_id: str, *, split: str = "train", concurrency: int = 1
) -> pipeline.RunConfig:
    """输入：临时目录、run 名、切分、并发度。输出：指向 tmp 的运行配置（不需要后端）。"""
    return pipeline.RunConfig(
        workflow="modelroute",
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
    return len(ModelRouteWorkflow().build_cases(config.config_override))


def test_end_to_end_all_judges_pass(tmp_path: Path) -> None:
    """输入：极小配置（零 token，不需要后端）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、两个判别器全通过；产物过审计。
    """
    config = make_config(tmp_path, "mr1")
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"roundtrip", "label"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0 and bucket["ok"] > 0
    report, missing = workflow_mod.load("modelroute").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == result.accepted
    assert report["train"]["index_bad"] == 0
    assert report["train"]["label_mismatch"] == 0
    assert set(report["train"]["types"]) == {"choice", "noul"}
    # tiny_config 七张表全开（含留出的 `reason` 与第二套句面），所以档位就是整张档位表。
    assert set(report["train"]["tiers"]) == set(spec_mod.TIERS)


def test_two_runs_are_byte_identical(tmp_path: Path) -> None:
    """输入：同种子、不同并发度跑两次。输出：两份产物。预期行为：逐字节一致。"""
    serial = pipeline.run(make_config(tmp_path, "mr2a", concurrency=1))
    parallel = pipeline.run(make_config(tmp_path, "mr2b", concurrency=4))
    assert (serial.run_dir / "train.jsonl").read_bytes() == (
        parallel.run_dir / "train.jsonl"
    ).read_bytes()


def test_manifest_and_stats_are_written(tmp_path: Path) -> None:
    """输入：一次跑。输出：产物目录。

    预期行为：manifest 记下 workflow / 切分；统计报告里出现七个维度的中文标题与两个判别器。
    """
    result = pipeline.run(make_config(tmp_path, "mr3"))
    manifest = jsonio.loads((result.run_dir / "manifest-train.json").read_text("utf-8"))
    assert manifest["workflow"] == "modelroute"
    assert manifest["split"] == "train"
    stats = (result.run_dir / "stats-train.md").read_text("utf-8")
    for title in ModelRouteWorkflow().dimension_titles().values():
        assert title in stats
    assert "roundtrip" in stats and "label" in stats


def test_no_family_falls_back_to_a_constant_answer(tmp_path: Path) -> None:
    """输入：一次 train 跑。输出：无。

    预期行为：六个题族都出现在均衡报告里，且每一族在产物里的答案**不止一种**——
    一族只剩一个答案就是在教「永远选同一个」，是坏数据而不是难题（范式 §5.7）。
    """
    result = pipeline.run(make_config(tmp_path, "mr4"))
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


def test_next_only_appears_when_there_is_a_chain(tmp_path: Path) -> None:
    """输入：一次 train 跑。输出：无。

    预期行为：`next` 题只出现在「首选档之上还有档」的样本里——`max` 之上没有档，
    出了就是坏数据。这条是**两阶段依赖**，也是这个 workflow 与纯查表最不一样的地方。
    """
    result = pipeline.run(make_config(tmp_path, "mr5"))
    loc = locales.load("zh")
    for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl"):
        request = parse.parse_request(row["state"], parse.label_index(loc))
        task = judges.task_of(request.body, loc)
        families = {family_of(qid) for qid in row["questions"]}
        assert ("next" in families) is spec_mod.has_next(task)
        if "next" in families:
            qid = next(key for key in row["questions"] if family_of(key) == "next")
            options = list(row["questions"][qid]["criteria"])
            assert options[row["answers"][qid]["choice"]] == spec_mod.next_tier_of(task)


def test_ood_split_runs(tmp_path: Path) -> None:
    """输入：`test_ood` 切分的极小配置。输出：RunResult。

    预期行为：能跑通、零拒绝，产物条数与 case 数对得上，且四个任务都出现过（含留出的
    `reason`）。
    """
    config = make_config(tmp_path, "mr6", split="test_ood")
    result = pipeline.run(config)
    assert result.accepted == _case_count(config) > 0
    assert result.rejected == []
    report, missing = workflow_mod.load("modelroute").audit(result.run_dir, ("test_ood",), "zh")
    assert missing == []
    assert set(report["test_ood"]["tiers"]) == set(spec_mod.TIERS)


def test_three_id_splits_are_disjoint(tmp_path: Path) -> None:
    """输入：train / val / test_iid 三次跑。输出：三份产物。

    预期行为：三份产物的 state 集合两两不相交——切分按**素材**哈希分桶，同一条请求
    不会同时出现在两档里（`config_for` 忘了替换 `split` 时这条会红）。
    """
    seen: list[set[str]] = []
    for split in ("train", "val", "test_iid"):
        result = pipeline.run(make_config(tmp_path, f"mr7-{split}", split=split))
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
    result = pipeline.run(make_config(tmp_path, "mr8"))
    rows = [row for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl")]
    states = [row["state"] for row in rows]
    assert len(set(states)) == len(states)


def test_id_config_produces_no_unseen_values() -> None:
    """输入：三个同分布切分的配置。输出：无。

    预期行为：它们产出的 case 里**一个留出取值都没有**——否则 test_ood 与同分布侧会重叠。
    """
    workflow = workflow_mod.load("modelroute")
    for split in ("train", "val", "test_iid"):
        cases = workflow.build_cases(workflow.config_for(split))
        assert cases and not any(case.unseen_dims() for case in cases)


def test_ood_config_produces_only_cases_with_unseen_values() -> None:
    """输入：`test_ood` 配置。输出：无。预期行为：每条都至少含一个留出取值。"""
    workflow = workflow_mod.load("modelroute")
    cases = workflow.build_cases(workflow.config_for("test_ood"))
    assert cases and all(case.unseen_dims() for case in cases)


def test_ood_materials_never_overlap_the_id_splits() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：OOD 侧的素材指纹与同分布侧完全不相交——值留出与素材留出叠加之后
    两侧按构造不重叠。
    """
    workflow = workflow_mod.load("modelroute")
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
    workflow = workflow_mod.load("modelroute")
    sizes = {
        split: len(workflow.build_cases(workflow.config_for(split)))
        for split in workflow.split_names()
    }
    assert min(sizes.values()) > 0
    assert sum(sizes[split] for split in ("train", "val", "test_iid")) == len(
        spec_mod.build_cases(workflow.config_for("train"))
    )


def test_answer_distributions_stay_within_the_balance_gate(tmp_path: Path) -> None:
    """输入：**生产配置**的 train 切分（不套 `config_override`）。输出：无。

    预期行为：六个题族的答案分布都落在均衡闸内——这是「均衡靠构造」这条设计的验收：
    `tier` 四档、`next` / `effort` 各三档、三个 noul 各 0.5，全都不靠事后过滤。
    """
    config = dataclasses.replace(
        make_config(tmp_path, "mr10"), config_override=None, per_case=1
    )
    result = pipeline.run(config)
    offenders = [item for item in result.stats["balance"] if not item["ok"]]
    assert offenders == [], offenders
    assert {item["key"] for item in result.stats["balance"]} == set(BALANCE_KEYS)


def test_english_locale_runs_end_to_end(tmp_path: Path) -> None:
    """输入：英文一次跑。输出：RunResult。预期行为：零拒绝、审计零问题。"""
    config = make_config(tmp_path, "mr9")
    config = dataclasses.replace(config, locales=("en",))
    result = pipeline.run(config)
    assert result.accepted == _case_count(config)
    report, missing = workflow_mod.load("modelroute").audit(result.run_dir, ("train",), "en")
    assert missing == [] and report["train"]["samples"] == result.accepted
