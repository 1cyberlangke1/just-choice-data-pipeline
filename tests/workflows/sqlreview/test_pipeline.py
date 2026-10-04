"""数据库变更评审的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path


from jc import language, pipeline

from tests.workflows._scaffold import run_config
from tests.workflows.sqlreview.conftest import BASE_LINES, packet_of, render_lines
from workflows.sqlreview import features, locales, render, workflow as sqlreview_workflow
from workflows.sqlreview.features import SqlReviewCase
from workflows.sqlreview.spec import SqlReviewSpec
from workflows.sqlreview.workflow import SqlReviewConfig, SqlReviewWorkflow

#: 小批量试跑的条数上限。train 切片一共 19 组里的前十来个，`_cover_pick` 会从里面挑 10 组。
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
        workflow="sqlreview",
        split=split,
        seed=7,
        override=SqlReviewConfig(split=split, case_limit=limit),
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[SqlReviewCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return sqlreview_workflow.cases_for(SqlReviewConfig(split=split, case_limit=limit))


def test_ood_split_only_uses_the_held_out_workload() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：`test_ood` 全是 `migration`，同分布三个切分一个 `migration` 都没有——这就是值留出。
    """
    assert {case.workload for case in cases_of("test_ood", limit=0)} == set(
        sqlreview_workflow.OOD_WORKLOADS
    )
    for split in ID_SPLITS:
        workloads = {case.workload for case in cases_of(split, limit=0)}
        assert not (workloads & set(sqlreview_workflow.OOD_WORKLOADS)), split


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：四个切分两两不重、并起来正好是全空间 48 组（同分布侧剔掉留出取值之后按桶分）。
    """
    seen: list[tuple[str, str]] = []
    for split in ALL_SPLITS:
        seen.extend((case.hits, case.workload) for case in cases_of(split, limit=0))
    assert len(seen) == len(set(seen)) == 48
    assert set(seen) == {(case.hits, case.workload) for case in features.build_cases(512)}


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：入库计划与切分、进程无关。"""
    case = SqlReviewCase("Q2+Q4", features.WORKLOAD_ANALYTICS)
    config = SqlReviewConfig(split="train", phrasing_pool=(1,))
    workflow = SqlReviewWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, SqlReviewSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = SqlReviewWorkflow()
    case = SqlReviewCase("Q3", features.WORKLOAD_ANALYTICS)
    state = render.to_state(
        render.ContentDraft(packet=packet_of(case.tags())), locales.load("zh")
    )
    parsed = workflow.reparse(state)
    assert parsed.text == packet_of(case.tags())
    assert parsed.tags == features.TAG_IDS
    assert workflow.case_label(case) == "Q3-analytics"
    assert workflow.stats_dimensions(case) == {"hits": "Q3", "workload": "analytics"}
    assert workflow.dimension_titles()["hits"]


def test_language_texts_reports_only_the_natural_language_field() -> None:
    """输入：一条备注是中文的记录（键名、SQL 与执行计划全是 ASCII）。输出：无。

    预期行为：`language_texts` 只报**一段自然语言正文**——评审备注。键名、SQL 关键字、执行计划
    与结构变更都是格式，把它们一起报上去只会把汉字占比稀释掉（范式 §9.50）。
    """
    packet = packet_of(("Q1", "Q2"))
    row = {"state": render.to_state(render.ContentDraft(packet=packet), locales.load("zh"))}
    texts = SqlReviewWorkflow().language_texts(row)
    assert list(texts) == ["评审备注"]
    assert texts["评审备注"] == BASE_LINES["note"].removeprefix("note: ")
    assert language.sample_problem(texts, ("zh",)) is None


def test_language_texts_catches_a_note_written_in_the_wrong_language() -> None:
    """输入：模型把 `note:` 那一行的备注写成了英文。输出：无。

    预期行为：报上去的那一段能被语言尺子抓住——这正是「只报自然语言正文」换来的东西：报整屏
    记录时，SQL 与执行计划会把汉字占比稀释到看不出来。
    """
    packet = render_lines(
        {
            **BASE_LINES,
            "note": "note: the backfill script for this column still needs a decision from upstream",
        }
    )
    row = {"state": render.to_state(render.ContentDraft(packet=packet), locales.load("zh"))}
    texts = SqlReviewWorkflow().language_texts(row)
    assert language.sample_problem(texts, ("zh",)) is not None


def test_language_texts_is_empty_when_there_is_no_note() -> None:
    """输入：一条没有 `note:` 那一行的记录。输出：无。

    预期行为：报空表——那把语言尺子对空表不发表意见，正是「没有自然语言正文」该有的形状。
    """
    packet = render_lines(
        {key: value for key, value in BASE_LINES.items() if key != "note"}
    )
    row = {"state": render.to_state(render.ContentDraft(packet=packet), locales.load("zh"))}
    assert SqlReviewWorkflow().language_texts(row) == {}

