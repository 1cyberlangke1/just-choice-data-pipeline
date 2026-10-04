"""调试器下一步动作的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path


from jc import language, pipeline

from tests.workflows._scaffold import run_config
from tests.workflows.debugstep.conftest import BASE_LINES, FakeBackend, render_lines, session_of
from workflows.debugstep import features, locales, render, workflow as debugstep_workflow
from workflows.debugstep.features import DebugStepCase
from workflows.debugstep.spec import DebugStepSpec
from workflows.debugstep.workflow import DebugStepConfig, DebugStepWorkflow

#: 小批量试跑的条数上限。train 切片一共 32 组里的六成，`_cover_pick` 会从里面挑 10 组。
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
        workflow="debugstep",
        split=split,
        seed=7,
        override=DebugStepConfig(split=split, case_limit=limit),
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[DebugStepCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return debugstep_workflow.cases_for(DebugStepConfig(split=split, case_limit=limit))


def test_dropping_a_line_only_spares_the_cases_without_one(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：少写一条该命中的形状。输出：RunResult。

    预期行为：只有「本来就一条都不命中」的那些 case 能活下来（没有形状可少写）。
    """
    unaffected = sum(1 for case in cases_of("train") if not case.tags())
    result = pipeline.run(make_config(tmp_path, "p3-drop", fake_backend(drop_line=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_adding_a_spare_line_only_spares_the_full_hits(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：多写一条不该命中的形状。输出：RunResult。

    预期行为：只有「四条全命中」的 case 无处可加，其余全被 `E_ASSEMBLE` 拦下。
    """
    unaffected = sum(1 for case in cases_of("train") if len(case.tags()) == len(features.TAG_IDS))
    result = pipeline.run(make_config(tmp_path, "p3-extra", fake_backend(extra_line=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_ood_split_only_uses_the_held_out_runtime() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：`test_ood` 全是 `go`，同分布三个切分一个 `go` 都没有——这就是值留出。
    """
    assert {case.runtime for case in cases_of("test_ood", limit=0)} == set(
        debugstep_workflow.OOD_RUNTIMES
    )
    for split in ID_SPLITS:
        runtimes = {case.runtime for case in cases_of(split, limit=0)}
        assert not (runtimes & set(debugstep_workflow.OOD_RUNTIMES)), split


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：四个切分两两不重、并起来正好是全空间 48 组（同分布侧剔掉留出取值之后按桶分）。
    """
    seen: list[tuple[str, str]] = []
    for split in ALL_SPLITS:
        seen.extend((case.hits, case.runtime) for case in cases_of(split, limit=0))
    assert len(seen) == len(set(seen)) == 48
    assert set(seen) == {(case.hits, case.runtime) for case in features.build_cases(512)}


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：入库计划与切分、进程无关。"""
    case = DebugStepCase("D2+D4", features.RUNTIME_NODE)
    config = DebugStepConfig(split="train", phrasing_pool=(1,))
    workflow = DebugStepWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, DebugStepSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = DebugStepWorkflow()
    case = DebugStepCase("D3", features.RUNTIME_NODE)
    state = render.to_state(
        render.ContentDraft(session=session_of(case.tags())), locales.load("zh")
    )
    parsed = workflow.reparse(state)
    assert parsed.text == session_of(case.tags())
    assert parsed.tags == features.TAG_IDS
    assert workflow.case_label(case) == "D3-node"
    assert workflow.stats_dimensions(case) == {"hits": "D3", "runtime": "node"}
    assert workflow.dimension_titles()["hits"]


def test_language_texts_reports_only_the_note_line() -> None:
    """输入：一屏只有 `Note:` 是中文的会话（键名与标识符全是 ASCII）。输出：无。

    预期行为：`language_texts` 只报**说明那一行**，不报整屏会话——键名、栈帧与标识符不是自然
    语言正文，交给语言判据本身就是错的（`jc.language` 模块文档那条「只喂自然语言正文」的
    规矩，范式 §9.50）。
    """
    session = session_of(("D1", "D3"))
    row = {"state": render.to_state(render.ContentDraft(session=session), locales.load("zh"))}
    texts = DebugStepWorkflow().language_texts(row)
    assert list(texts) == ["会话说明"]
    assert texts["会话说明"] == "手动暂停，先看看当前这一帧里的取值喵"
    assert language.sample_problem(texts, ("zh",)) is None
    # 判据 2026-10-02 起 zh 侧按**词**比（见 jc.language 的阈值表），所以「整屏拿去比会被拦」
    # 已经不是不能报整屏的理由了。它真正拦下的是**一条中文都没有**的正文：
    assert language.sample_problem({"一屏会话": "Stopped: TypeError\nLocals: count=3"}, ("zh",)) is not None


def test_language_texts_is_empty_when_there_is_no_note() -> None:
    """输入：一屏没有 `Note:` 那一行的会话。输出：无。

    预期行为：报空表——那把语言尺子对空表不发表意见，正是「没有自然语言正文」该有的形状。
    """
    session = render_lines({key: value for key, value in BASE_LINES.items() if key != "note"})
    row = {"state": render.to_state(render.ContentDraft(session=session), locales.load("zh"))}
    assert DebugStepWorkflow().language_texts(row) == {}

