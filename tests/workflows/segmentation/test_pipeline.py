"""端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from jc import pipeline, workflow as workflow_mod
from jc.backend.types import AllProvidersDown, TransportError

from tests.workflows.segmentation.conftest import FakeBackend, pages_of, spec_for
from workflows.segmentation import features, locales, render, workflow as segmentation_workflow
from workflows.segmentation.features import SegmentationCase
from workflows.segmentation.spec import SegmentationSpec, spec_to_dict
from workflows.segmentation.workflow import SegmentationConfig, SegmentationWorkflow

#: 小批量试跑的条数上限。train 切片一共几十组，`_cover_pick` 会从里面挑 10 组。
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
        workflow="segmentation",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=SegmentationConfig(split=split, case_limit=limit),
        backend=backend,
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[SegmentationCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return segmentation_workflow.cases_for(SegmentationConfig(split=split, case_limit=limit))


def test_end_to_end_all_judges_pass(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：假后端（从不撒谎）。输出：RunResult。

    预期行为：每个 case 出一条样本、零拒绝、三个判别器都记到通过；产物能过审计，四族题都真的
    出得来。
    """
    result = pipeline.run(make_config(tmp_path, "p1", fake_backend()))
    expected = len(cases_of("train"))
    assert result.accepted == expected
    assert result.rejected == []
    assert set(result.stats["judges"]) == {"hygiene", "frame", "read"}
    for bucket in result.stats["judges"].values():
        assert bucket["fail"] == 0
        assert bucket["ok"] > 0
    report, missing = workflow_mod.load("segmentation").audit(result.run_dir, ("train",), "zh")
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
        ("drop_marker", "E_ASSEMBLE"),
        ("extra_marker", "E_ASSEMBLE"),
        ("no_trace", "E_ASSEMBLE"),
        ("wrong_pages", "E_ASSEMBLE"),
    ],
)
def test_always_broken_drafts_are_rejected(
    tmp_path: Path, fake_backend: type[FakeBackend], switch: str, code: str
) -> None:
    """输入：每一组 case 都会踩到的「写歪」。输出：RunResult。

    预期行为：一条都留不下，拒绝码就是该有的那一个——写歪的页面是渲染期的 `E_ASSEMBLE`
    （重试耗尽后仍被拒，不静默放行）。
    """
    result = pipeline.run(make_config(tmp_path, f"p3-{switch}", fake_backend(**{switch: True})))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {code}


def test_a_lying_read_rejects_everything(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解漏报最后一类。输出：RunResult。

    预期行为：每一包都至少有一类，所以一条都过不了，全部被 `E_JUDGE_READ` 拦下——
    「反解说反话」与「页面写歪」是两条独立的闸，各自拦各自那一类。
    """
    result = pipeline.run(make_config(tmp_path, "p3-read", fake_backend(read_lies=True)))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_READ"}


def test_a_broken_read_endpoint_only_marks_skipped(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解端点抛异常。输出：RunResult。

    预期行为：样本照样留下（页面本身没问题），`read` 那一格记的是跳过而不是不通过——
    把端点抖动记成数据质量问题会让整批看起来「全是坏数据」。
    """
    result = pipeline.run(
        make_config(
            tmp_path,
            "p3-endpoint",
            fake_backend(read_raises=TransportError("端点抖了")),
        )
    )
    assert result.accepted == len(cases_of("train"))
    assert result.stats["judges"]["read"]["skipped"] == result.accepted
    assert result.stats["judges"]["read"]["fail"] == 0


def test_every_provider_down_stops_the_whole_batch(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解时「所有 provider 都挂了」。输出：`RunAborted`。

    预期行为：`AllProvidersDown` 是**整批停下**的唯一条件（PLAN 3.1.4），它带着
    `fatal_for_run` 穿过判别器，不会被当成「这条样本不通过」吞掉；管线用异常把这件事
    告诉调用方，而不是返回一个「看上去正常、其实全是被拒」的结果。
    """
    with pytest.raises(pipeline.RunAborted) as excinfo:
        pipeline.run(
            make_config(
                tmp_path,
                "p3-down",
                fake_backend(read_raises=AllProvidersDown("全挂了")),
            )
        )
    assert excinfo.value.seq == 0


def test_small_batch_covers_every_route_and_count(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：10 条的小批量。输出：无。

    预期行为：三个切分档位与三个类数档位在小批量里**全都取得到**——`_cover_pick` 覆盖的正是
    这两个维度，不然 `route` / `count` 的某一档在小批量里一条都出不来，均衡闸标红。
    """
    cases = cases_of("train")
    routes = {features.plan_for(case, 0).route for case in cases}
    counts = {features.plan_for(case, 0).count for case in cases}
    assert routes == set(features.ROUTE_LEVELS)
    assert counts == set(range(1, features.MAX_PACKET_CLASSES + 1))


def test_every_level_appears_in_the_id_splits() -> None:
    """输入：同分布三个切分的全部 case。输出：无。

    预期行为：类数 1~3 与三个切分档位在同分布侧全都取得到——不然那几族在训练侧会缺档，均衡闸
    与审计都会当场报出来。**这是「先验分布再定档位」那条纪律的落地**（范式 §9.70）。
    """
    counts: set[int] = set()
    routes: set[str] = set()
    for split in ID_SPLITS:
        for case in cases_of(split, limit=0):
            plan = features.plan_for(case, 0)
            counts.add(plan.count)
            routes.add(plan.route)
    assert counts == set(range(1, features.MAX_PACKET_CLASSES + 1))
    assert routes == set(features.ROUTE_LEVELS)


def test_no_level_dominates_the_id_space() -> None:
    """输入：同分布两个来源取值的全部 case。输出：无。

    预期行为：**没有任何一档占比超过 50%**，且每一档都至少有 10% 的份额——「某一档把别的档挤掉」
    正是这一族在设计期验分布时要挡掉的东西（多类混合那一版占 90%，当场被换掉）。
    """
    cases = [
        case
        for split in ID_SPLITS
        for case in cases_of(split, limit=0)
    ]
    assert cases
    for key, levels in (
        ("route", features.ROUTE_LEVELS),
        ("count", tuple(str(value) for value in range(1, features.MAX_PACKET_CLASSES + 1))),
    ):
        shares = [
            sum(1 for case in cases if str(getattr(features.plan_for(case, 0), key)) == level)
            / len(cases)
            for level in levels
        ]
        assert max(shares) <= 0.5, (key, shares)
        assert min(shares) >= 0.1, (key, shares)


def test_every_split_is_non_empty(tmp_path: Path, fake_backend: type[FakeBackend]) -> None:
    """输入：四个切分各跑一遍。输出：RunResult。预期行为：都出得来样本，零拒绝。"""
    for split in ALL_SPLITS:
        result = pipeline.run(
            make_config(tmp_path, f"p5-{split}", fake_backend(), split=split, limit=0)
        )
        assert result.accepted == len(cases_of(split, limit=0)) > 0, split
        assert result.rejected == [], split


def test_render_without_a_backend_is_an_error() -> None:
    """输入：没绑后端的 workflow。输出：无。预期行为：抛 RuntimeError，不静默降级成模板。"""
    workflow = SegmentationWorkflow()
    spec = spec_for(SegmentationCase("form", "2", features.SOURCE_SCAN))
    with pytest.raises(RuntimeError, match="bind_backend"):
        workflow.render(spec, random.Random(0), "zh")


def test_read_judge_without_a_backend_is_an_error() -> None:
    """输入：没绑后端却要 read 判别器。输出：无。预期行为：抛 RuntimeError（接线错了）。"""
    workflow = SegmentationWorkflow()
    with pytest.raises(RuntimeError, match="read"):
        workflow.make_judges(("read",), "zh")


def test_unknown_split_and_judge_and_locale_report_available() -> None:
    """输入：不存在的切分 / 判别器 / 语言。输出：无。预期行为：都报人话。"""
    workflow = SegmentationWorkflow()
    with pytest.raises(KeyError, match="未知切分"):
        workflow.config_for("nope")
    with pytest.raises(KeyError, match="未知判别器"):
        workflow.make_judges(("nope",), "zh")
    with pytest.raises(FileNotFoundError):
        workflow.audit(Path("dataset"), ("train",), "nope")


def test_plan_is_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：入库计划与切分、进程无关。"""
    case = SegmentationCase("receipt+contract", "4", features.SOURCE_PHOTO)
    config = SegmentationConfig(split="train", phrasing_pool=(1,))
    workflow = SegmentationWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, SegmentationSpec)
    assert first.plan == second.plan
    assert first.plan == features.plan_for(case, 1)
    assert spec_to_dict(first)["shape"] == ["receipt", "receipt", "contract", "contract"]
    assert spec_to_dict(first)["docs"] == 4
    assert spec_to_dict(first)["route"] == "many"


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = SegmentationWorkflow()
    case = SegmentationCase("form+receipt", "3", features.SOURCE_FAX)
    spec = spec_for(case)
    state = render.to_state(
        render.PacketDraft(pages=pages_of(spec.plan, case.source)), locales.load("zh")
    )
    parsed = workflow.reparse(state)
    assert parsed.pages == tuple(pages_of(spec.plan, case.source))
    assert parsed.classes == features.CLASS_IDS
    assert workflow.case_label(case) == "form+receipt-3页-fax"
    assert workflow.stats_dimensions(case) == {
        "classes": "form+receipt",
        "pages": "3",
        "source": "fax",
    }
    assert workflow.dimension_titles()["classes"]


def test_locales_are_reported() -> None:
    """输入：无。输出：无。预期行为：支持的语言与语言文件目录一致。"""
    workflow = SegmentationWorkflow()
    assert workflow.available_locales() == locales.available() == ["en", "zh"]


def test_declared_contract_fields() -> None:
    """输入：workflow 的声明项。输出：无。预期行为：题型、后端需求与判别器名单都对得上。"""
    workflow = SegmentationWorkflow()
    assert workflow.name == "segmentation"
    assert workflow.kinds == frozenset({"choice", "noul"})
    assert workflow.needs == frozenset({"text"})
    assert workflow.judges == ("hygiene", "frame", "read")


def test_config_knobs_are_validated() -> None:
    """输入：坏旋钮。输出：无。预期行为：构造期就报人话，不拖到采样阶段。"""
    with pytest.raises(ValueError, match="phrasing_pool"):
        SegmentationConfig(split="train", phrasing_pool=())
    with pytest.raises(ValueError, match="case_limit"):
        SegmentationConfig(split="train", case_limit=-1)
    with pytest.raises(ValueError, match="cover_limit"):
        SegmentationConfig(split="train", cover_limit=0)
