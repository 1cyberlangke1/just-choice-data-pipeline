"""端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path


from jc import pipeline

from tests.workflows._scaffold import run_config
from tests.workflows.tracescore.conftest import text_of
from workflows.tracescore import features, locales, render, workflow as tracescore_workflow
from workflows.tracescore.features import TraceScoreCase
from workflows.tracescore.spec import TraceScoreSpec
from workflows.tracescore.workflow import TraceScoreConfig, TraceScoreWorkflow

#: 小批量试跑的条数上限。train 切片一共 19 组，`_cover_pick` 会从里面挑 10 组。
#: 断言一律拿 `cases_of("train")` 的规模去比，不写死数字。
CASE_LIMIT = 10
ID_SPLITS = ("train", "val", "test_iid")


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
        workflow="tracescore",
        split=split,
        seed=7,
        override=TraceScoreConfig(split=split, case_limit=limit),
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[TraceScoreCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return tracescore_workflow.cases_for(TraceScoreConfig(split=split, case_limit=limit))


def test_ood_split_only_uses_the_held_out_source() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：`test_ood` 全是 `chat`，同分布三个切分一个 `chat` 都没有——这就是值留出。
    """
    assert {case.source for case in cases_of("test_ood", limit=0)} == set(
        tracescore_workflow.OOD_SOURCES
    )
    for split in ID_SPLITS:
        sources = {case.source for case in cases_of(split, limit=0)}
        assert not (sources & set(tracescore_workflow.OOD_SOURCES)), split


def test_every_count_and_route_level_is_reachable() -> None:
    """输入：同分布三个切分。输出：无。

    预期行为：命中条数 0~4 与三个处置档位在同分布侧全都取得到——不然那几族在训练侧会缺档，
    均衡闸与审计都会当场报出来。
    """
    counts: set[int] = set()
    routes: set[str] = set()
    for split in ID_SPLITS:
        for case in cases_of(split, limit=0):
            labels = features.label_for(features.plan_for(case, 0))
            counts.add(labels.count)
            routes.add(labels.route)
    assert counts == set(range(len(features.TAG_IDS) + 1))
    assert routes == set(features.ROUTE_LEVELS)


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：入库计划与切分、进程无关。"""
    case = TraceScoreCase("S2+S4", features.SOURCE_CHAT)
    config = TraceScoreConfig(split="train", phrasing_pool=(1,))
    workflow = TraceScoreWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, TraceScoreSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = TraceScoreWorkflow()
    case = TraceScoreCase("S3", features.SOURCE_CHAT)
    state = render.to_state(render.ContentDraft(text=text_of(case)), locales.load("zh"))
    parsed = workflow.reparse(state)
    assert parsed.text == text_of(case)
    assert parsed.tags == features.TAG_IDS
    assert workflow.case_label(case) == "S3-chat"
    assert workflow.stats_dimensions(case) == {"hits": "S3", "source": "chat"}
    assert workflow.dimension_titles()["hits"]

