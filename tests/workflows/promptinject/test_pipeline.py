"""提示注入的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path


from jc import pipeline
from jc.jsontypes import JsonValue

from tests.workflows._scaffold import run_config
from tests.workflows.promptinject.conftest import FakeBackend, text_of
from workflows.promptinject import features, locales, render, workflow as promptinject_workflow
from workflows.promptinject.features import PromptInjectCase
from workflows.promptinject.spec import PromptInjectSpec
from workflows.promptinject.workflow import PromptInjectConfig, PromptInjectWorkflow

#: 小批量试跑的条数上限。train 切片一共 9 组（16 组的 60%），所以实际跑满的是 9 条——
#: 断言一律拿 `cases_of("train")` 的规模去比，不写死数字。
CASE_LIMIT = 10
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
        workflow="promptinject",
        split=split,
        seed=7,
        override=PromptInjectConfig(split=split, case_limit=limit),
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[PromptInjectCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return promptinject_workflow.cases_for(PromptInjectConfig(split=split, case_limit=limit))


def test_ood_split_only_uses_the_held_out_source(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：留出侧跑一遍。输出：RunResult。预期行为：零拒绝，产物里全是 `file` 那一档。"""
    cases = cases_of("test_ood", limit=0)
    assert cases and {case.source for case in cases} == {features.SOURCE_FILE}
    result = pipeline.run(
        make_config(tmp_path, "p3-ood", fake_backend(), split="test_ood", limit=0)
    )
    assert result.rejected == []
    assert result.accepted == len(cases)


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case 列表。输出：无。

    预期行为：四份互不相交，合起来正好是全空间 48 组（16 组命中 × 3 种素材来源；同分布侧只出现
    `web` / `tool`，留出侧 16 组全是 `file`）。
    """
    keys: dict[tuple[tuple[str, JsonValue], ...], str] = {}
    for split in ALL_SPLITS:
        cases = cases_of(split, limit=0)
        assert cases, split
        for case in cases:
            key = tuple(case.as_dict().items())
            assert key not in keys, f"{case.describe()} 同时落在 {keys[key]} 与 {split}"
            keys[key] = split
    assert len(keys) == 48
    assert {dict(key)["source"] for key, split in keys.items() if split == "test_ood"} == {
        features.SOURCE_FILE
    }
    assert {dict(key)["source"] for key, split in keys.items() if split != "test_ood"} == set(
        features.SOURCES
    ) - {features.SOURCE_FILE}


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：注入计划与切分、进程无关。"""
    case = PromptInjectCase("T2+T4", features.SOURCE_TOOL)
    config = PromptInjectConfig(split="train", phrasing_pool=(1,))
    workflow = PromptInjectWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, PromptInjectSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = PromptInjectWorkflow()
    case = PromptInjectCase("T3", features.SOURCE_FILE)
    state = render.to_state(render.ContentDraft(text=text_of(case)), locales.load("zh"))
    parsed = workflow.reparse(state)
    assert parsed.text == text_of(case)
    assert parsed.tags == features.TAG_IDS
    assert workflow.case_label(case) == "T3-file"
    assert workflow.stats_dimensions(case) == {"hits": "T3", "source": "file"}
    assert workflow.dimension_titles()["hits"]
