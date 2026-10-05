"""报销收据归类的端到端测试：假后端跑通整条管线，并验证切分不变量与判别器拦错。"""

from __future__ import annotations

import random
from pathlib import Path

from jc import pipeline
from jc.backend.types import BackendError
from jc.jsontypes import JsonValue

from tests.workflows._scaffold import run_config
from tests.workflows.receiptcat.conftest import FakeBackend, text_of
from workflows.receiptcat import features, locales, render
from workflows.receiptcat import workflow as receiptcat_workflow
from workflows.receiptcat.features import ReceiptCase
from workflows.receiptcat.spec import ReceiptCatSpec
from workflows.receiptcat.workflow import ReceiptCatConfig, ReceiptCatWorkflow

#: 小批量试跑的条数上限。train 切片一共 7 组（12 组的 60%），所以实际跑满的是 6 条——
#: 断言一律拿 `cases_of("train")` 的规模去比，不写死数字。
CASE_LIMIT = 6
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
        workflow="receiptcat",
        split=split,
        seed=7,
        override=ReceiptCatConfig(split=split, case_limit=limit),
    )


def cases_of(split: str, limit: int = CASE_LIMIT) -> list[ReceiptCase]:
    """输入：切分名、条数。输出：该切分被挑中的 case（与管线看到的完全一致）。"""
    return receiptcat_workflow.cases_for(ReceiptCatConfig(split=split, case_limit=limit))


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
