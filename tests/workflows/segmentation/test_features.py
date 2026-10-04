"""文档包切分的特征层测试：种类表、判据、页序、切分口径、计划与分布。"""

from __future__ import annotations

from collections import Counter

import pytest

from tests.workflows.segmentation.conftest import MARKERS, TRACES, page_text
from workflows.segmentation import features


def test_class_ids_are_four_distinct_keys() -> None:
    """输入：无。输出：无。预期行为：四个种类编号两两不同，且顺序稳定。"""
    assert features.CLASS_IDS == ("form", "letter", "receipt", "contract")
    assert len(set(features.CLASS_IDS)) == len(features.CLASS_IDS)


def test_mergeable_classes_are_a_strict_subset() -> None:
    """输入：无。输出：无。

    预期行为：可并的类是种类表的真子集，而且**两边都非空**——只有可并的类就没有「一页一份」
    这一半规则，只有不可并的类就没有「整段并成一份」这一半，切分口径会退化。
    """
    assert features.MERGEABLE < set(features.CLASS_IDS)
    assert features.MERGEABLE
    assert set(features.CLASS_IDS) - features.MERGEABLE


def test_every_class_set_is_a_non_empty_subset_within_the_cap() -> None:
    """输入：无。输出：无。

    预期行为：14 个子集（大小 1~3）、没有重复、大小不越界；每一类在**子集层面**各出现 7 次
    （4 类里含某一类的子集恰好一半），这是 `flag` 那一族接近对半的构造原因。
    """
    assert len(features.CLASS_SETS) == 14
    assert len(set(features.CLASS_SETS)) == 14
    for value in features.CLASS_SETS:
        classes = features.parse_classes(value)
        assert 1 <= len(classes) <= features.MAX_PACKET_CLASSES
        assert features.join_classes(classes) == value
    for cls in features.CLASS_IDS:
        assert sum(1 for value in features.CLASS_SETS if cls in features.parse_classes(value)) == 7


def test_patterns_are_purely_literal() -> None:
    """输入：一段页面正文。输出：无。预期行为：命中按 `CLASS_IDS` 顺序、只看字面。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for text, expect in [
        ("表单编号：A-1001", ("form",)),
        ("这是申请表，请填报人签字", ("form",)),
        ("Form No. 42 attached", ("form",)),
        ("尊敬的先生：见字如面", ("letter",)),
        ("Dear Madam, thank you", ("letter",)),
        ("收据编号 R-2002，收款人李四", ("receipt",)),
        ("Paid in full, cashier Wang", ("receipt",)),
        ("合同编号 C-3003，甲方王五", ("contract",)),
        ("Signed on 2026-09-30 by Party B", ("contract",)),
        # 只认字面：换个说法、复合词都不算。
        ("这是一份材料，内容自述", ()),
        ("表 单 编 号：A-1001", ()),
        # 两类抬头混着写：算不清，读者也分不清。
        ("表单编号 A-1001，甲方王五", ("form", "contract")),
    ]:
        try:
            assert features.matches(text) == expect
        except AssertionError as exc:
            bad.append(f"{(text, expect)!r}: {exc}")
    assert not bad, "\n".join(bad)


@pytest.mark.parametrize("cls", features.CLASS_IDS)
def test_every_marker_is_recognised_and_isolated(cls: str) -> None:
    """输入：每一类的抬头片段。输出：无。

    预期行为：这一类的片段**只**命中它自己——判据之间不重叠，否则一页会同时属于两类，
    `class_of` 判不出来，渲染期会一直重试。
    """
    assert features.matches(MARKERS[cls]) == (cls,)
    assert features.class_of(MARKERS[cls]) == cls


def test_class_of_is_none_when_ambiguous_or_absent() -> None:
    """输入：没有抬头 / 有两类抬头的正文。输出：无。预期行为：两种都判不出来（None）。"""
    assert features.class_of("一页什么都没写的纸") is None
    assert features.class_of(f"{MARKERS['form']}{MARKERS['receipt']}") is None


@pytest.mark.parametrize("source", features.SOURCES)
def test_every_source_trace_is_recognised_and_isolated(source: str) -> None:
    """输入：每一种来源的痕迹。输出：无。

    预期行为：痕迹被认出来，而且**不会**命中任何一类的判据——痕迹把页面算成某一类的话，
    标签会被渲染细节带偏。
    """
    assert features.source_trace(TRACES[source], source)
    assert features.matches(TRACES[source]) == ()
    for other in features.SOURCES:
        if other != source:
            assert not features.source_trace(TRACES[source], other)


@pytest.mark.parametrize(
    ("classes", "pages", "expect"),
    [
        (("form",), 1, ("form",)),
        (("form",), 4, ("form", "form", "form", "form")),
        (("form", "receipt"), 3, ("form", "form", "receipt")),
        (("form", "receipt"), 4, ("form", "form", "receipt", "receipt")),
        (("form", "letter", "receipt"), 4, ("form", "form", "letter", "receipt")),
    ],
)
def test_shape_for_puts_every_class_on_a_page(
    classes: tuple[str, ...], pages: int, expect: tuple[str, ...]
) -> None:
    """输入：种类、页数。输出：无。

    预期行为：页数按类均分、余数给排在前的类；**每一类至少一页**，同类页在页序里连续
    （不然「相邻同类」这条规则就判不到东西了）。
    """
    shape = features.shape_for(classes, pages)
    assert shape == expect
    assert len(shape) == pages
    assert set(shape) == set(classes)


def test_docs_for_applies_the_splitting_rule() -> None:
    """输入：页序。输出：无。

    预期行为：跨类一定切、可并的类整段算一份、不可并的类一页一份。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for shape, docs in [
        ((), 0),
        (("form",), 1),
        (("form", "form"), 1),
        (("form", "form", "letter"), 2),
        (("receipt",), 1),
        (("receipt", "receipt"), 2),
        (("receipt", "receipt", "receipt"), 3),
        (("form", "receipt"), 2),
        (("form", "form", "receipt"), 2),
        (("form", "receipt", "receipt"), 3),
        (("receipt", "contract"), 2),
        (("receipt", "receipt", "contract", "contract"), 4),
        (("form", "letter", "receipt"), 3),
    ]:
        try:
            assert features.docs_for(shape) == docs
        except AssertionError as exc:
            bad.append(f"{(shape, docs)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_route_for_is_three_levels() -> None:
    """输入：份数。输出：无。预期行为：1 份 / 2 份 / 3 份及以上。

    本用例把表里每一行都跑一遍，失败的行攒成一份清单一起报，好一次看全。
    """
    bad: list[str] = []
    for docs, route in [(0, "one"), (1, "one"), (2, "two"), (3, "many"), (4, "many"), (6, "many")]:
        try:
            assert features.route_for(docs) == route
        except AssertionError as exc:
            bad.append(f"{(docs, route)!r}: {exc}")
    assert not bad, "\n".join(bad)


def test_plan_is_reproducible_and_distractors_swap_one_class() -> None:
    """输入：同一个 case。输出：无。

    预期行为：计划与进程无关；三个干扰组合两两不同、都不等于真值，且**与真值同大小、恰好差
    一处**（同大小就没法靠数数划掉，差一处才有区分度）。
    """
    case = features.SegmentationCase("form+receipt", "3", features.SOURCE_SCAN)
    plan = features.plan_for(case, 0)
    assert plan == features.plan_for(case, 0)
    assert plan.shape == ("form", "form", "receipt")
    assert plan.classes == ("form", "receipt")
    assert len(plan.distractors) == 3
    assert len(set(plan.distractors)) == 3
    for combo in plan.distractors:
        assert combo != plan.classes
        assert len(combo) == len(plan.classes)
        assert len(set(combo) ^ set(plan.classes)) == 2


def test_distractor_pool_is_never_too_small() -> None:
    """输入：全空间的每一个 case。输出：无。

    预期行为：每一组干扰都能抽满三个且互不相同——池子最小的两种情形（真值 1 类、真值 3 类）
    恰好各 3 个候选，抽三个不会撞车。
    """
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        assert len(plan.distractors) == 3
        assert len(set(plan.distractors)) == 3


def test_build_cases_covers_the_whole_space() -> None:
    """输入：无。输出：无。

    预期行为：全空间 126 组（14 子集 × 受约束的页数 × 3 来源）；每一组都满足「页数 ≥ 类数」；
    三种来源都被枚举到（留出维只砍取值，不该把别的取值一起砍掉）。
    """
    cases = features.build_cases(512)
    assert len(cases) == 126
    assert len({tuple(case.as_dict().items()) for case in cases}) == 126
    for case in cases:
        assert case.page_count >= len(case.class_ids())
        assert case.classes == features.join_classes(case.class_ids())
    assert {case.source for case in cases} == set(features.SOURCES)
    assert features.DIMS["source"] == features.SOURCES
    assert features.DIMS["pages"] == features.PAGE_COUNTS


def test_pages_allowed_only_restricts_the_classes_pair() -> None:
    """输入：取值对。输出：无。

    预期行为：只对 `classes` / `pages` 这一对生效（页数不能少于类数），别的组合一律放行；
    维度顺序反着传也要给出同一个结论。
    """
    assert features.pages_allowed("classes", "form+letter+receipt", "pages", "2") is False
    assert features.pages_allowed("classes", "form+letter+receipt", "pages", "3") is True
    assert features.pages_allowed("pages", "1", "classes", "form") is True
    assert features.pages_allowed("classes", "form", "source", "scan") is True
    assert features.pages_allowed("pages", "2", "source", "fax") is True


def test_labels_and_observed_labels_agree_on_faithful_text() -> None:
    """输入：全空间每一个 case 的忠实页面。输出：无。

    预期行为：构造侧（`label_for`）与审计侧（`observed_labels`）两条路走出同一个结论——
    对不上就说明渲染没兑现计划。
    """
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        pages = tuple(page_text(cls, case.source) for cls in plan.shape)
        assert features.observed_labels(pages) == features.label_for(plan)


def test_answer_distributions_are_not_degenerate() -> None:
    """输入：全空间 126 组 case。输出：无。

    预期行为：**先验分布再定档位**（范式 §9.70）——三档切分、三档类数、四条逐类判据都不许
    出现「某一档吃掉一半以上」的退化，也不许有档位一条都取不到。这一族最初按「单一文档 /
    同类相邻要切 / 多类混合」分档时，多类那一档占 90%，当场被换掉。
    """
    cases = features.build_cases(512)
    plans = [features.plan_for(case, 0) for case in cases]
    total = len(plans)
    for key, levels in (
        ("route", features.ROUTE_LEVELS),
        ("count", tuple(range(1, features.MAX_PACKET_CLASSES + 1))),
    ):
        counts = Counter(getattr(plan, key) for plan in plans)
        assert set(counts) == set(levels), (key, counts)
        shares = [counts[level] / total for level in levels]
        assert max(shares) <= 0.5, (key, shares)
        assert min(shares) >= 0.1, (key, shares)
    for cls in features.CLASS_IDS:
        share = sum(1 for plan in plans if cls in plan.classes) / total
        assert 0.3 <= share <= 0.7, (cls, share)
