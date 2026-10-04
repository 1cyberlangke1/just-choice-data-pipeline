"""根因定位的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path


from jc import pipeline

from tests.workflows._scaffold import run_config
from tests.workflows.causechain.conftest import FakeBackend, logs_of, state_of
from workflows.causechain import features, locales, render, workflow as causechain_workflow
from workflows.causechain.features import CauseCase
from workflows.causechain.spec import CauseSpec
from workflows.causechain.workflow import CauseChainWorkflow, CauseConfig

#: 小批量试跑的条数上限。train 切片一共 30 组，`_cover_pick` 会从里面挑 10 组。
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
        workflow="causechain",
        split=split,
        seed=7,
        override=CauseConfig(split=split, case_limit=limit),
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[CauseCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return causechain_workflow.cases_for(CauseConfig(split=split, case_limit=limit))


def test_a_lying_trace_rejects_everything(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解换一个服务。输出：RunResult。

    预期行为：全被 `E_JUDGE_TRACE` 拦下——「反解说反话」与「日志写歪」是两条独立的闸。
    """
    result = pipeline.run(make_config(tmp_path, "p3-trace", fake_backend(trace_lies=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


def test_ood_split_only_uses_the_held_out_graph() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：`test_ood` 全是 G4，同分布三个切分一个 G4 都没有——这就是值留出。
    """
    assert {case.graph for case in cases_of("test_ood", limit=0)} == set(
        causechain_workflow.OOD_GRAPHS
    )
    for split in ID_SPLITS:
        graphs = {case.graph for case in cases_of(split, limit=0)}
        assert not (graphs & set(causechain_workflow.OOD_GRAPHS)), split


def test_case_splits_are_disjoint_and_respect_the_per_pair_cap() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：四个切分两两不重；每个 (图, 根因) 组合在**全部切分加起来**里留的条数正好是
    「可行探针数」与 `CASES_PER_PAIR` 里小的那个——可行探针少的那几组（根因是入口服务）
    只有 2 条，其余都留满 3 条。这一步是为了让 `root` 族的答案分布不被拓扑带偏。
    """
    seen: list[tuple[str, str, str]] = []
    for split in ALL_SPLITS:
        seen.extend((case.graph, case.root, case.probe) for case in cases_of(split, limit=0))
    assert len(seen) == len(set(seen))
    pairs: dict[tuple[str, str], int] = {}
    for graph, root, _ in seen:
        pairs[(graph, root)] = pairs.get((graph, root), 0) + 1
    feasible: dict[tuple[str, str], int] = {}
    for case in features.build_cases(512):
        key = (case.graph, case.root)
        feasible[key] = feasible.get(key, 0) + 1
    assert set(pairs) == set(feasible)
    assert pairs == {
        key: min(causechain_workflow.CASES_PER_PAIR, count) for key, count in feasible.items()
    }


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：入库计划与切分、进程无关。"""
    case = CauseCase("G2", "S5", "rno")
    config = CauseConfig(split="train", phrasing_pool=(1,))
    workflow = CauseChainWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, CauseSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = CauseChainWorkflow()
    case = CauseCase("G3", "S4", "h2")
    plan = features.plan_for(case, 0)
    logs = logs_of(plan)
    parsed = workflow.reparse(state_of(plan, logs))
    assert parsed.text is not None
    assert render.parse_logs(parsed.text, locales.load("zh")) == logs
    assert workflow.case_label(case) == "G3-S4-h2"
    assert workflow.stats_dimensions(case) == {"graph": "G3", "root": "S4", "probe": "h2"}
    assert workflow.dimension_titles()["probe"]

