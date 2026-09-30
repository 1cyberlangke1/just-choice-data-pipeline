"""报销收据归类的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from jc import pipeline, workflow as workflow_mod
from jc.backend.types import BackendError
from jc.jsontypes import JsonValue

from tests.workflows.receiptcat.conftest import FakeBackend, text_of
from workflows.receiptcat import features, locales, render
from workflows.receiptcat import workflow as receiptcat_workflow
from workflows.receiptcat.features import ReceiptCase
from workflows.receiptcat.spec import ReceiptCatSpec, spec_to_dict
from workflows.receiptcat.workflow import ReceiptCatConfig, ReceiptCatWorkflow

#: 小批量试跑的条数上限。train 切片一共 7 组（12 组的 60%），所以实际跑满的是 6 条——
#: 断言一律拿 `cases_of("train")` 的规模去比，不写死数字。
CASE_LIMIT = 6
ID_SPLITS = ("train", "val", "test_iid")
ALL_SPLITS = ("train", "val", "test_iid", "test_ood")

#: 审计报告里那六个「不自洽」计数器。
BAD_KEYS = ("frame_bad", "content_bad", "cat_bad", "over_bad", "tier_bad", "near_bad")

#: 假后端的「故意写歪」开关 → 期望的拒绝码。这些歪法在**同分布侧**每一组 case 上都成立
#: （诱饵那两个开关不是——它们只在留出侧有意义，单拎出去测）。
BROKEN_DRAFTS = (
    ("too_short", "E_ASSEMBLE"),
    ("fence", "E_ASSEMBLE"),
    ("name_leak", "E_ASSEMBLE"),
    ("wrong_category", "E_ASSEMBLE"),
    ("wrong_amount", "E_ASSEMBLE"),
    ("decoy_extra", "E_ASSEMBLE"),
)


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
        workflow="receiptcat",
        split=split,
        locales=("zh",),
        run_seed=7,
        per_case=1,
        out_root=tmp_path,
        run_id=run_id,
        config_override=ReceiptCatConfig(split=split, case_limit=limit),
        backend=backend,
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[ReceiptCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return receiptcat_workflow.cases_for(ReceiptCatConfig(split=split, case_limit=limit))


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
    report, missing = workflow_mod.load("receiptcat").audit(result.run_dir, ("train",), "zh")
    assert missing == []
    assert report["train"]["samples"] == expected
    for family in ("cat", "over", "tier", "near"):
        assert report["train"]["families"][family] > 0, family
    for key in BAD_KEYS:
        assert report["train"][key] == 0, key


@pytest.mark.parametrize(("switch", "code"), BROKEN_DRAFTS)
def test_always_broken_drafts_are_rejected(
    tmp_path: Path, fake_backend: type[FakeBackend], switch: str, code: str
) -> None:
    """输入：每一组 case 都会踩到的「写歪」。输出：RunResult。

    预期行为：一条都留不下，拒绝码就是 `E_ASSEMBLE`——写歪的收据行在渲染期被拦下（重试耗尽后
    仍被拒，不静默放行）。
    """
    result = pipeline.run(make_config(tmp_path, f"p3-{switch}", fake_backend(**{switch: True})))
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {code}


def test_missing_decoy_on_the_ood_split_is_rejected(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：留出侧（`decoy=near`）却不埋诱饵。输出：RunResult。预期行为：整批都被拦下。"""
    result = pipeline.run(
        make_config(tmp_path, "p3-decoy", fake_backend(decoy_missing=True), split="test_ood")
    )
    assert result.accepted == 0
    assert {item["code"] for item in result.rejected} == {"E_ASSEMBLE"}


def test_a_lying_reader_only_spares_the_cases_with_nothing_to_lie_about(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解时一律说成「第一类 + 第一档」。输出：RunResult。

    预期行为：真值**正好**是「第一类 + 第一档」的 case 没得可撒谎，照样合格；其余全被
    `E_JUDGE_TRACE` 拦下。

    为什么单拎出来：`trace_lies` 的「说反」是一个**定值**（C1 + 档位 0），它在真值恰好等于这个
    定值的 case 上是空操作——把它塞进「整批都会被拦下」那条断言里，就会依赖「train 切片里恰好
    没有 C1 未超限的 case」这种跟哈希桶有关的巧合。
    """

    def lie_is_a_no_op(case: ReceiptCase) -> bool:
        """输入：一个 case。输出：假后端的定值谎言是否恰好等于真值。"""
        labels = features.label_for(case)
        return (
            labels.category == features.CATEGORY_IDS[0]
            and features.TIERS.index(labels.tier) == 0
        )

    unaffected = sum(1 for case in cases_of("train") if lie_is_a_no_op(case))
    result = pipeline.run(make_config(tmp_path, "p3-lie", fake_backend(trace_lies=True)))
    assert result.accepted == unaffected
    assert {item["code"] for item in result.rejected} == {"E_JUDGE_TRACE"}


def test_a_broken_endpoint_is_not_a_data_problem(
    tmp_path: Path, fake_backend: type[FakeBackend]
) -> None:
    """输入：反解时端点故障（`BackendError`）。输出：RunResult。

    预期行为：一条都不拒——端点抖动只算「没判成」（`skipped`），写成「不通过」会把基础设施的
    问题记成合成数据的质量问题。
    """
    result = pipeline.run(
        make_config(tmp_path, "p3-endpoint", fake_backend(trace_raises=BackendError("端点炸了")))
    )
    assert result.accepted == len(cases_of("train"))
    assert result.rejected == []
    assert result.stats["judges"]["trace"]["fail"] == 0
    assert result.stats["judges"]["trace"]["skipped"] > 0


def test_case_splits_are_disjoint_and_cover_the_space() -> None:
    """输入：四个切分的 case 列表。输出：无。

    预期行为：四份互不相交，合起来正好是全空间 24 组（同分布侧 12 组全是 `decoy=none`，
    留出侧 12 组全是 `decoy=near`）。
    """
    keys: dict[tuple[tuple[str, JsonValue], ...], str] = {}
    for split in ALL_SPLITS:
        cases = cases_of(split, limit=0)
        assert cases, split
        for case in cases:
            key = tuple(case.as_dict().items())
            assert key not in keys, f"{case.describe()} 同时落在 {keys[key]} 与 {split}"
            keys[key] = split
    assert len(keys) == 24
    assert {dict(key)["decoy"] for key, split in keys.items() if split == "test_ood"} == {
        features.DECOY_NEAR
    }
    assert {dict(key)["decoy"] for key, split in keys.items() if split != "test_ood"} == {
        features.DECOY_NONE
    }


def test_every_category_band_and_tier_is_reachable() -> None:
    """输入：同分布三个切分。输出：无。

    预期行为：四个类别、三个金额档位与四个审核档位在同分布侧全都取得到——不然那几族在训练侧
    会缺档，均衡闸与审计都会当场报出来。
    """
    categories: set[str] = set()
    bands: set[str] = set()
    tiers: set[str] = set()
    for split in ID_SPLITS:
        for case in cases_of(split, limit=0):
            labels = features.label_for(case)
            categories.add(labels.category)
            bands.add(labels.band)
            tiers.add(labels.tier)
    assert categories == set(features.CATEGORY_IDS)
    assert bands == set(features.BANDS)
    assert tiers == set(features.TIERS)


def test_the_ood_side_is_all_decoys() -> None:
    """输入：留出切分。输出：无。预期行为：`near` 那一族的真值在留出侧恒为真。"""
    labels = [features.label_for(case) for case in cases_of("test_ood", limit=0)]
    assert labels
    assert all(item.near for item in labels)


def test_render_without_a_backend_is_an_error() -> None:
    """输入：没绑后端的 workflow。输出：无。预期行为：抛 RuntimeError，不静默降级成模板。"""
    workflow = ReceiptCatWorkflow()
    spec = ReceiptCatSpec(
        case=ReceiptCase("C1", "under", features.DECOY_NONE), phrasing=0, split="train", seq=1
    )
    with pytest.raises(RuntimeError, match="bind_backend"):
        workflow.render(spec, random.Random(0), "zh")


def test_trace_judge_without_a_backend_is_an_error() -> None:
    """输入：没绑后端却要 trace 判别器。输出：无。预期行为：抛 RuntimeError（接线错了）。"""
    workflow = ReceiptCatWorkflow()
    with pytest.raises(RuntimeError, match="trace"):
        workflow.make_judges(("trace",), "zh")


def test_unknown_split_and_judge_and_locale_report_available() -> None:
    """输入：不存在的切分 / 判别器 / 语言。输出：无。预期行为：都报人话。"""
    workflow = ReceiptCatWorkflow()
    with pytest.raises(KeyError, match="未知切分"):
        workflow.config_for("nope")
    with pytest.raises(KeyError, match="未知判别器"):
        workflow.make_judges(("nope",), "zh")
    with pytest.raises(FileNotFoundError):
        workflow.audit(Path("dataset"), ("train",), "nope")


def test_labels_are_reproducible_from_the_case_fingerprint() -> None:
    """输入：同一个 case 的两次采样。输出：无。预期行为：标签与问法只由 case 与计划决定。"""
    case = ReceiptCase("C2", "heavy", features.DECOY_NONE)
    config = ReceiptCatConfig(split="train", phrasing_pool=(1,))
    workflow = ReceiptCatWorkflow()
    first = workflow.sample(
        random.Random(0), workflow.plan_sample(random.Random(1), config, case, 0)
    )
    second = workflow.sample(
        random.Random(9), workflow.plan_sample(random.Random(2), config, case, 0)
    )
    assert isinstance(first, ReceiptCatSpec)
    assert first.labels == second.labels == features.label_for(case)
    assert spec_to_dict(first)["amount"] == features.AMOUNTS[("C2", "heavy")]


def test_reparse_and_case_label_are_wired() -> None:
    """输入：一段 state、一个 case。输出：无。预期行为：反解与可读标签都接得上。"""
    workflow = ReceiptCatWorkflow()
    case = ReceiptCase("C3", "mild", features.DECOY_NONE)
    state = render.to_state(render.ContentDraft(text=text_of(case)), locales.load("zh"))
    parsed = workflow.reparse(state)
    assert parsed.text == text_of(case)
    assert parsed.categories == features.CATEGORY_IDS
    assert workflow.case_label(case) == "C3-mild-none"
    assert workflow.stats_dimensions(case) == {
        "category": "C3",
        "band": "mild",
        "decoy": "none",
    }
    assert workflow.dimension_titles()["band"]


def test_locales_are_reported() -> None:
    """输入：无。输出：无。预期行为：支持的语言与语言文件目录一致。"""
    workflow = ReceiptCatWorkflow()
    assert workflow.available_locales() == locales.available() == ["en", "zh"]


def test_declared_contract_fields() -> None:
    """输入：workflow 的声明项。输出：无。预期行为：题型、后端需求与判别器名单都对得上。"""
    workflow = ReceiptCatWorkflow()
    assert workflow.name == "receiptcat"
    assert workflow.kinds == frozenset({"choice", "noul", "score"})
    assert workflow.needs == frozenset({"text"})
    assert workflow.judges == ("hygiene", "frame", "trace")
    assert workflow.split_names() == ALL_SPLITS


def test_language_texts_reports_the_receipt_line() -> None:
    """输入：一个忠实样本。输出：无。

    预期行为：只报模型写的那条收据行——三张表是代码写的，报出去会让「产物文本与声明语言一致」
    这条尺子去量代码写的表头。
    """
    workflow = ReceiptCatWorkflow()
    case = ReceiptCase("C1", "under", features.DECOY_NONE)
    sample: dict[str, JsonValue] = {
        "state": render.to_state(render.ContentDraft(text=text_of(case)), locales.load("zh"))
    }
    declared = workflow.language_texts(sample)
    assert list(declared) == ["收据行"]
    assert declared["收据行"] == text_of(case)
