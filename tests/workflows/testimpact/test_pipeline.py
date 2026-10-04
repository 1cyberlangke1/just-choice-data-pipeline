"""测试影响选择的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path


from jc import pipeline

from tests.workflows._scaffold import run_config
from tests.workflows.testimpact.conftest import TITLE_ZH, FakeBackend, diff_for, faithful
from workflows.testimpact import features, render, workflow as testimpact_workflow
from workflows.testimpact.features import ImpactCase
from workflows.testimpact.spec import ImpactSpec
from workflows.testimpact.workflow import ImpactConfig, ImpactWorkflow

#: 小批量试跑的条数上限。train 切片一共 28 组，`_cover_pick` 会从里面挑 10 组。
#: 断言一律拿 `cases_of("train")` 的规模去比，不写死数字。
CASE_LIMIT = 10
ID_SPLITS = ("train", "val", "test_iid")
ALL_SPLITS = (*ID_SPLITS, "test_ood")


#: 一个普通 case：改到两条路径（接口层 + 界面层），性质是修缺陷。
CASE = ImpactCase(touch="P1+P3", flavor="fix")


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
        workflow="testimpact",
        split=split,
        seed=7,
        override=ImpactConfig(split=split, case_limit=limit),
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[ImpactCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return testimpact_workflow.cases_for(ImpactConfig(split=split, case_limit=limit))


def test_a_lying_trace_is_caught_on_every_case(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解漏报最后一个文件。输出：RunResult。

    预期行为：**一条都留不下**——枚举里每一组触及组合都非空（空组合不是合法 case），所以每一
    条都有可漏报的文件，全被 `E_JUDGE_TRACE` 拦下。
    """
    assert all(case.touched() for case in cases_of("train"))
    result = pipeline.run(make_config(tmp_path, "p3-trace", fake_backend(trace_lies=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


def test_dropping_a_file_only_spares_the_single_path_cases(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：少写一个该碰的文件。输出：RunResult。

    预期行为：只有「本来就只碰一个文件」的那些 case 能活下来（少写这一步没处生效），其余全被
    `E_ASSEMBLE` 拦下。
    """
    unaffected = sum(1 for case in cases_of("train") if len(case.touched()) == 1)
    result = pipeline.run(make_config(tmp_path, "p3-drop", fake_backend(drop_file=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_adding_a_spare_file_is_caught_on_every_case(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：多写一个不该碰的文件。输出：RunResult。

    预期行为：**一条都留不下**——枚举里最大的触及组合只有四条路径（盲区 `P4` 从不与别的路径
    同现），总有文件可多加，全被 `E_ASSEMBLE` 拦下。
    """
    assert all(len(case.touched()) < len(features.PATH_IDS) for case in cases_of("train"))
    result = pipeline.run(make_config(tmp_path, "p3-extra", fake_backend(extra_file=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_ood_split_only_uses_the_held_out_flavor() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：`test_ood` 全是 `revert`，同分布三个切分一个 `revert` 都没有——这就是值留出。
    """
    assert {case.flavor for case in cases_of("test_ood", limit=0)} == set(
        testimpact_workflow.OOD_FLAVORS
    )
    for split in ID_SPLITS:
        flavors = {case.flavor for case in cases_of(split, limit=0)}
        assert not (flavors & set(testimpact_workflow.OOD_FLAVORS)), split


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case。输出：无。

    预期行为：四个切分两两不重、并起来正好是全空间 64 组（同分布侧剔掉留出取值之后按桶分）。
    """
    seen: list[tuple[str, str]] = []
    for split in ALL_SPLITS:
        seen.extend((case.touch, case.flavor) for case in cases_of(split, limit=0))
    assert len(seen) == len(set(seen)) == 64
    assert set(seen) == {(case.touch, case.flavor) for case in features.build_cases(512)}


def test_every_level_and_count_is_reachable() -> None:
    """输入：同分布三个切分。输出：无。

    预期行为：需要 0~3 套与四个跑法档位在同分布侧全都取得到——不然那几族在训练侧会缺档，
    均衡闸与审计都会当场报出来。
    """
    counts: set[int] = set()
    levels: set[str] = set()
    for split in ID_SPLITS:
        for case in cases_of(split, limit=0):
            labels = features.label_for(features.plan_for(case, 0))
            counts.add(len(labels.need))
            levels.add(labels.level)
    assert counts == set(range(len(features.SUITES) + 1))
    assert levels == set(features.LEVELS)


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：入库计划与切分、进程无关。"""
    case = ImpactCase(touch="P1+P2", flavor="fix")
    config = ImpactConfig(split="train", phrasing_pool=(1,))
    workflow = ImpactWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, ImpactSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = ImpactWorkflow()
    _spec, sample = faithful(CASE)
    parsed = workflow.reparse(sample["state"])
    assert parsed.title == TITLE_ZH
    assert parsed.touched == CASE.touched()
    assert workflow.case_label(CASE) == "P1+P3-fix"
    assert workflow.stats_dimensions(CASE) == {"touch": "P1+P3", "flavor": "fix"}
    assert workflow.dimension_titles()["touch"]


def test_language_texts_report_only_the_model_written_parts() -> None:
    """输入：一条忠实样本。输出：无。

    预期行为：只报模型写的那一行提交标题——覆盖表、档位表与改动都是代码写的，报进去会把
    「文案与声明语言一致」这条尺子变成空转。
    """
    _spec, sample = faithful(CASE)
    texts = ImpactWorkflow().language_texts(sample)
    assert set(texts) == {"提交标题"}
    assert texts["提交标题"] == TITLE_ZH


def test_state_round_trips_through_the_parser() -> None:
    """输入：一份渲染好的 state。输出：无。预期行为：解析回来的正文与渲染前的草稿逐字相同。"""
    _spec, sample = faithful(CASE)
    parsed = render.parse_state(str(sample["state"]))
    assert parsed.title == TITLE_ZH
    assert parsed.diff == diff_for(CASE)
    assert parsed.touched == CASE.touched()
    assert render.detect_locale(str(sample["state"])) == "zh"

