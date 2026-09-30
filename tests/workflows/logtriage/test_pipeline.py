"""日志行分诊的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from jc import pipeline, workflow as workflow_mod

from tests.workflows.logtriage.conftest import FakeBackend, spec_for, text_of
from workflows.logtriage import features, locales, render, workflow as logtriage_workflow
from workflows.logtriage.features import LogTriageCase
from workflows.logtriage.spec import LogTriageSpec, spec_to_dict
from workflows.logtriage.workflow import LogTriageConfig, LogTriageWorkflow

#: 小批量试跑的条数上限。train 切片一共 19 组，`_cover_pick` 会从里面挑 10 组。
#: 断言一律拿 `cases_of("train")` 的规模去比，不写死数字。
CASE_LIMIT = 10
ID_SPLITS = ("train", "val", "test_iid")
ALL_SPLITS = ("train", "val", "test_iid", "test_ood")

#: 审计报告里那五个「不自洽」计数器。
BAD_KEYS = ("content_bad", "flag_bad", "count_bad", "combo_bad", "route_bad")


def make_config(
    tmp_path: Path,
    run_id: str,
    backend: object,
    *,
    split: str = "train",
    limit: int = CASE_LIMIT,
) -> pipeline.RunConfig:
    """构造一个指向 tmp 目录、用小 case 集合试跑的运行配置。"""
    return pipeline.RunConfig(
        workflow="logtriage",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=LogTriageConfig(split=split, case_limit=limit),
        backend=backend,
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[LogTriageCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return logtriage_workflow.cases_for(LogTriageConfig(split=split, case_limit=limit))


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：假后端（从不撒谎）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、三个判别器都记到通过；产物能过审计，四族题都真的
    出得来。
    """
    result = pipeline.run(make_config(tmp_path, "p1", fake_backend()))
    expected = len(cases_of("train"))
    assert result.accepted == expected
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"hygiene", "frame", "trace"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0
    report, missing = workflow_mod.load("logtriage").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == expected
    for family in ("flag", "count", "combo", "route"):
        assert report["train"]["families"][family] > 0, family
    for key in BAD_KEYS:
        assert report["train"][key] == 0, key


@pytest.mark.parametrize(
    ("switch", "code"),
    [
        ("too_short", "E_ASSEMBLE"),
        ("fence", "E_ASSEMBLE"),
    ],
)
def test_always_broken_drafts_are_rejected(
    tmp_path: Path, fake_backend: type[FakeBackend], switch: str, code: str
) -> None:
    """输入：每一组 case 都会踩到的「写歪」。输出：RunResult。

    预期行为：一条都留不下，拒绝码就是该有的那一个——写歪的日志是渲染期的 `E_ASSEMBLE`
    （重试耗尽后仍被拒，不静默放行）。
    """
    result = pipeline.run(make_config(tmp_path, f"p3-{switch}", fake_backend(**{switch: True})))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {code}


def test_a_lying_trace_only_spares_the_empty_hits(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解漏报最后一条。输出：RunResult。

    预期行为：命中组合为空的那一条照样合格（没有可漏报的），其余全被 `E_JUDGE_TRACE` 拦下——
    「反解说反话」与「日志写歪」是两条独立的闸，各自拦各自那一类。
    """
    unaffected = sum(1 for case in cases_of("train") if not case.tags())
    result = pipeline.run(make_config(tmp_path, "p3-trace", fake_backend(trace_lies=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


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


def test_render_without_a_backend_is_an_error() -> None:
    """输入：没绑后端的 workflow。输出：无。预期行为：抛 RuntimeError，不静默降级成模板。"""
    workflow = LogTriageWorkflow()
    spec = spec_for(LogTriageCase("N1", features.SOURCE_APP))
    with pytest.raises(RuntimeError, match="bind_backend"):
        workflow.render(spec, random.Random(0), "zh")


def test_trace_judge_without_a_backend_is_an_error() -> None:
    """输入：没绑后端却要 trace 判别器。输出：无。预期行为：抛 RuntimeError（接线错了）。"""
    workflow = LogTriageWorkflow()
    with pytest.raises(RuntimeError, match="trace"):
        workflow.make_judges(("trace",), "zh")


def test_unknown_split_and_judge_and_locale_report_available() -> None:
    """输入：不存在的切分 / 判别器 / 语言。输出：无。预期行为：都报人话。"""
    workflow = LogTriageWorkflow()
    with pytest.raises(KeyError, match="未知切分"):
        workflow.config_for("nope")
    with pytest.raises(KeyError, match="未知判别器"):
        workflow.make_judges(("nope",), "zh")
    with pytest.raises(FileNotFoundError):
        workflow.audit(Path("dataset"), ("train",), "nope")


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
    assert spec_to_dict(first)["hits"] == ["N2", "N4"]


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


def test_locales_are_reported() -> None:
    """输入：无。输出：无。预期行为：支持的语言与语言文件目录一致。"""
    workflow = LogTriageWorkflow()
    assert workflow.available_locales() == locales.available() == ["en", "zh"]


def test_declared_contract_fields() -> None:
    """输入：workflow 的声明项。输出：无。预期行为：题型、后端需求与判别器名单都对得上。"""
    workflow = LogTriageWorkflow()
    assert workflow.name == "logtriage"
    assert workflow.kinds == frozenset({"choice", "noul"})
    assert workflow.needs == frozenset({"text"})
    assert workflow.judges == ("hygiene", "frame", "trace")


def test_config_knobs_are_validated() -> None:
    """输入：坏旋钮。输出：无。预期行为：构造期就报人话，不拖到采样阶段。"""
    with pytest.raises(ValueError, match="phrasing_pool"):
        LogTriageConfig(split="train", phrasing_pool=())
    with pytest.raises(ValueError, match="case_limit"):
        LogTriageConfig(split="train", case_limit=-1)
