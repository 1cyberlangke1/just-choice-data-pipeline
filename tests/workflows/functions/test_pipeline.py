"""端到端测试：小规模跑通整条管线、可复现、审计与切分不重叠。

零 token：函数调用是纯构造式（`needs` 为空），所以这里不碰任何后端。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


from jc import jsonio, pipeline
from jc import workflow as workflow_mod
from jc.workflow import family_of
from tests.workflows._scaffold import run_config
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
    return run_config(
        tmp_path,
        run_id,
        workflow="functions",
        split=split,
        seed=17,
        concurrency=concurrency,
        override=tiny_config(split),
    )


def _case_count(config: pipeline.RunConfig) -> int:
    """输入：运行配置。输出：该切分的 case 数。预期行为：与产物条数对得上。"""
    return len(fn_workflow.FunctionsWorkflow().build_cases(config.config_override))


def test_no_family_falls_back_to_a_constant_answer(tmp_path: Path) -> None:
    """输入：一次 train 跑。输出：无。

    预期行为：代表题族都出现在均衡报告里，且每一族在产物里的答案**不止一种**——
    一族只剩一个答案就是在教「永远选同一个」，是坏数据而不是难题（范式 §5.7）。
    """
    result = pipeline.run(make_config(tmp_path, "fn4"))
    keys = {item["key"] for item in result.stats["balance"]}
    for key in BALANCE_KEYS:
        assert key in keys, f"均衡报告里没有 {key}"
    seen: dict[str, set[object]] = defaultdict(set)
    for _, row in jsonio.iter_jsonl(result.run_dir / "train.jsonl"):
        for qid, question in row["questions"].items():
            answer = row["answers"][qid]
            value = answer["choice"] if question["type"] == "choice" else answer["noul"]
            seen[family_of(qid)].add(value)
    for family, values in seen.items():
        assert len(values) > 1, f"{family} 的答案退化成常量：{values}"


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
