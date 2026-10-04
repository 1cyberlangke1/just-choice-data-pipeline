"""日志行分诊的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path


from jc import pipeline

from tests.workflows._scaffold import run_config
from tests.workflows.logtriage.conftest import text_of
from workflows.logtriage import features, locales, render, workflow as logtriage_workflow
from workflows.logtriage.features import LogTriageCase
from workflows.logtriage.spec import LogTriageSpec
from workflows.logtriage.workflow import LogTriageConfig, LogTriageWorkflow

#: 小批量试跑的条数上限。train 切片一共 19 组，`_cover_pick` 会从里面挑 10 组。
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
        workflow="logtriage",
        split=split,
        seed=7,
        override=LogTriageConfig(split=split, case_limit=limit),
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[LogTriageCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return logtriage_workflow.cases_for(LogTriageConfig(split=split, case_limit=limit))


def test_ood_split_only_uses_the_held_out_source() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：`test_ood` 全是 `access`，同分布三个切分一个 `access` 都没有——这就是值留出。
    """
    assert {case.source for case in cases_of("test_ood", limit=0)} == set(
        logtriage_workflow.OOD_SOURCES
    )
    for split in ID_SPLITS:
        sources = {case.source for case in cases_of(split, limit=0)}
        assert not (sources & set(logtriage_workflow.OOD_SOURCES)), split


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：四个切分两两不重、并起来正好是全空间 48 组（同分布侧剔掉留出取值之后按桶分）。
    """
    seen: list[tuple[str, str]] = []
    for split in ALL_SPLITS:
        seen.extend((case.hits, case.source) for case in cases_of(split, limit=0))
    assert len(seen) == len(set(seen)) == 48
    assert set(seen) == {(case.hits, case.source) for case in features.build_cases(512)}


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：入库计划与切分、进程无关。"""
    case = LogTriageCase("N2+N4", features.SOURCE_ACCESS)
    config = LogTriageConfig(split="train", phrasing_pool=(1,))
    workflow = LogTriageWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, LogTriageSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = LogTriageWorkflow()
    case = LogTriageCase("N3", features.SOURCE_ACCESS)
    state = render.to_state(render.ContentDraft(text=text_of(case)), locales.load("zh"))
    parsed = workflow.reparse(state)
    assert parsed.text == text_of(case)
    assert parsed.tags == features.TAG_IDS
    assert workflow.case_label(case) == "N3-access"
    assert workflow.stats_dimensions(case) == {"hits": "N3", "source": "access"}
    assert workflow.dimension_titles()["hits"]

