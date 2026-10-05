"""内容审核的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path


from jc import pipeline
from jc.jsontypes import JsonValue

from tests.workflows._scaffold import run_config
from tests.workflows.guardrail.conftest import FakeBackend, text_of
from workflows.guardrail import features, locales, render, workflow as guardrail_workflow
from workflows.guardrail.features import GuardrailCase
from workflows.guardrail.spec import GuardrailSpec
from workflows.guardrail.workflow import GuardrailConfig, GuardrailWorkflow

#: 小批量试跑的条数上限。train 切片一共 9 组（16 组的 60%），所以实际跑满的是 9 条——
#: 断言一律拿 `cases_of("train")` 的规模去比，不写死数字。
CASE_LIMIT = 10
ID_SPLITS = ("train", "val", "test_iid")
ALL_SPLITS = ("train", "val", "test_iid", "test_ood")


def make_config(
    tmp_path: Path,
    run_id: str,
    backend: object,
    *,
    split: str = "train",
    limit: int = CASE_LIMIT,
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return run_config(
        tmp_path,
        run_id,
        backend=backend,
        workflow="guardrail",
        split=split,
        seed=7,
        override=GuardrailConfig(split=split, case_limit=limit),
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[GuardrailCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return guardrail_workflow.cases_for(GuardrailConfig(split=split, case_limit=limit))


def test_dropping_a_fragment_only_spares_the_cases_without_one(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：少写一条该命中的片段。输出：RunResult。

    预期行为：本来就没有命中片段的 case 照样合格，其余全被 `E_ASSEMBLE` 拦下——一个都不放过。
    """
    unaffected = sum(1 for case in cases_of("train") if not case.clauses())
    result = pipeline.run(make_config(tmp_path, "g3-drop", fake_backend(drop_fragment=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_adding_a_spare_fragment_only_spares_the_full_hits(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：多写一条不该命中的片段。输出：RunResult。

    预期行为：四条全中的 case 没有「多余的一条」可加，其余全被拦下。
    """
    unaffected = sum(
        1 for case in cases_of("train") if len(case.clauses()) == len(features.CLAUSE_IDS)
    )
    result = pipeline.run(make_config(tmp_path, "g3-extra", fake_backend(extra_fragment=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case 列表。输出：无。

    预期行为：四份互不相交，合起来正好是全空间 32 组（同分布侧 16 组全是 `decoy=none`，
    留出侧 16 组全是 `decoy=dashed`）。
    """
    keys: dict[tuple[tuple[str, JsonValue], ...], str] = {}
    for split in ALL_SPLITS:
        cases = cases_of(split, limit=0)
        assert cases, split
        for case in cases:
            key = tuple(case.as_dict().items())
            assert key not in keys, f"{case.describe()} 同时落在 {keys[key]} 与 {split}"
            keys[key] = split
    assert len(keys) == 32
    assert {dict(key)["decoy"] for key, split in keys.items() if split == "test_ood"} == {
        features.DECOY_DASHED
    }
    assert {dict(key)["decoy"] for key, split in keys.items() if split != "test_ood"} == {
        features.DECOY_NONE
    }


def test_every_count_and_action_level_is_reachable() -> None:
    """输入：同分布三个切分。输出：无。

    预期行为：命中条数 0~4 与四个处置档位在同分布侧全都取得到——不然那几族在训练侧会缺档，
    均衡闸与审计都会当场报出来。
    """
    counts: set[int] = set()
    actions: set[str] = set()
    for split in ID_SPLITS:
        for case in cases_of(split, limit=0):
            labels = features.label_for(features.plan_for(case, 0))
            counts.add(labels.count)
            actions.add(labels.action)
    assert counts == set(range(len(features.CLAUSE_IDS) + 1))
    assert actions == set(features.ACTION_LEVELS)


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：审核计划与切分、进程无关。"""
    case = GuardrailCase("P2+P4", features.DECOY_DASHED)
    config = GuardrailConfig(split="train", phrasing_pool=(1,))
    workflow = GuardrailWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, GuardrailSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = GuardrailWorkflow()
    case = GuardrailCase("P3", features.DECOY_DASHED)
    state = render.to_state(render.ContentDraft(text=text_of(case)), locales.load("zh"))
    parsed = workflow.reparse(state)
    assert parsed.text == text_of(case)
    assert parsed.clauses == features.CLAUSE_IDS
    assert workflow.case_label(case) == "P3-dashed"
    assert workflow.stats_dimensions(case) == {"hits": "P3", "decoy": "dashed"}
    assert workflow.dimension_titles()["hits"]
