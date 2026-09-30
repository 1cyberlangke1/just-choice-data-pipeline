"""测试影响选择的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from jc import pipeline, workflow as workflow_mod

from tests.workflows.testimpact.conftest import (
    TITLE_ZH,
    FakeBackend,
    diff_for,
    faithful,
    spec_for,
)
from workflows.testimpact import features, locales, render, workflow as testimpact_workflow
from workflows.testimpact.features import ImpactCase
from workflows.testimpact.spec import ImpactSpec, spec_to_dict
from workflows.testimpact.workflow import ImpactConfig, ImpactWorkflow

#: 小批量试跑的条数上限。train 切片一共 28 组，`_cover_pick` 会从里面挑 10 组。
#: 断言一律拿 `cases_of("train")` 的规模去比，不写死数字。
CASE_LIMIT = 10
ID_SPLITS = ("train", "val", "test_iid")
ALL_SPLITS = (*ID_SPLITS, "test_ood")

#: 审计报告里那五个「不自洽」计数器。
BAD_KEYS = ("table_bad", "flag_bad", "count_bad", "which_bad", "route_bad")

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
    return pipeline.RunConfig(
        workflow="testimpact",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=ImpactConfig(split=split, case_limit=limit),
        backend=backend,
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[ImpactCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return testimpact_workflow.cases_for(ImpactConfig(split=split, case_limit=limit))


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：假后端（从不撒谎）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、三个判别器都记到通过；产物能过审计，四族题都真的
    出得来，五个计数器一个都不动。
    """
    result = pipeline.run(make_config(tmp_path, "p1", fake_backend()))
    expected = len(cases_of("train"))
    assert result.accepted == expected
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"hygiene", "frame", "trace"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0
    report, missing = workflow_mod.load("testimpact").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == expected
    for family in ("flag", "count", "which", "route"):
        assert report["train"]["families"][family] > 0, family
    for key in BAD_KEYS:
        assert report["train"][key] == 0, key


@pytest.mark.parametrize(
    ("switch", "code"),
    [
        ("prose", "E_ASSEMBLE"),
        ("fence", "E_ASSEMBLE"),
        ("short_title", "E_ASSEMBLE"),
    ],
)
def test_always_broken_drafts_are_rejected(
    tmp_path: Path, fake_backend: type[FakeBackend], switch: str, code: str
) -> None:
    """输入：每一组 case 都会踩到的「写歪」。输出：RunResult。

    预期行为：一条都留不下，拒绝码就是该有那一个——写成散文、拿围栏包住、提交标题太短，都是
    渲染期的 `E_ASSEMBLE`（重试耗尽后仍被拒，不静默放行）。
    """
    result = pipeline.run(make_config(tmp_path, f"p3-{switch}", fake_backend(**{switch: True})))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {code}


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


def test_render_without_a_backend_is_an_error() -> None:
    """输入：没绑后端的 workflow。输出：无。预期行为：抛 RuntimeError，不静默降级成模板。"""
    workflow = ImpactWorkflow()
    with pytest.raises(RuntimeError, match="bind_backend"):
        workflow.render(spec_for(CASE), random.Random(0), "zh")


def test_trace_judge_without_a_backend_is_an_error() -> None:
    """输入：没绑后端却要 trace 判别器。输出：无。预期行为：抛 RuntimeError（接线错了）。"""
    workflow = ImpactWorkflow()
    with pytest.raises(RuntimeError, match="trace"):
        workflow.make_judges(("trace",), "zh")


def test_unknown_split_and_judge_and_locale_report_available() -> None:
    """输入：不存在的切分 / 判别器 / 语言。输出：无。预期行为：都报人话。"""
    workflow = ImpactWorkflow()
    with pytest.raises(KeyError, match="未知切分"):
        workflow.config_for("nope")
    with pytest.raises(KeyError, match="未知判别器"):
        workflow.make_judges(("nope",), "zh")
    with pytest.raises(FileNotFoundError):
        workflow.audit(Path("dataset"), ("train",), "nope")


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
    assert spec_to_dict(first)["touch"] == list(case.touched())
    assert spec_to_dict(first)["need"] == list(case.need())


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


def test_locales_are_reported() -> None:
    """输入：无。输出：无。预期行为：支持的语言与语言文件目录一致。"""
    workflow = ImpactWorkflow()
    assert workflow.available_locales() == locales.available() == ["en", "zh"]


def test_declared_contract_fields() -> None:
    """输入：workflow 的声明项。输出：无。预期行为：题型、后端需求与判别器名单都对得上。"""
    workflow = ImpactWorkflow()
    assert workflow.name == "testimpact"
    assert workflow.kinds == frozenset({"choice", "noul"})
    assert workflow.needs == frozenset({"text"})
    assert workflow.judges == ("hygiene", "frame", "trace")
    assert workflow.split_names() == ("train", "val", "test_iid", "test_ood")


def test_config_knobs_are_validated() -> None:
    """输入：坏旋钮。输出：无。预期行为：构造期就报人话，不拖到采样阶段。"""
    with pytest.raises(ValueError, match="phrasing_pool"):
        ImpactConfig(split="train", phrasing_pool=())
    with pytest.raises(ValueError, match="case_limit"):
        ImpactConfig(split="train", case_limit=-1)
