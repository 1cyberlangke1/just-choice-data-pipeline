"""结构恢复的特征层测试：取值表、组合矩阵、值留出与块计划。"""

from __future__ import annotations

import pytest

from workflows.autoformat import features, locales, parse
from workflows.autoformat.features import (
    KIND_CALLOUT,
    KIND_CODE,
    KIND_HEADING,
    KIND_LIST_ITEM,
    KIND_PARAGRAPH,
    KIND_QUOTE,
)

#: 全空间 345 组：下面好几条测试都拿它当「都要过」的清单。
ALL_CASES = features.build_cases(features.DocConfig())


def test_shape_tables_are_keyed_by_shapes_and_never_empty() -> None:
    """输入：无。输出：无。预期行为：两张形状表都以 `SHAPES` 为键，每格的取值表都非空。"""
    assert set(features.SHAPE_VARIANTS) == set(features.SHAPES)
    assert set(features.SHAPE_CODE_FAMILIES) == set(features.SHAPES)
    for shape in features.SHAPES:
        assert features.SHAPE_VARIANTS[shape], shape
        assert features.SHAPE_CODE_FAMILIES[shape], shape


def test_block_kinds_and_callout_kinds_are_stable() -> None:
    """输入：无。输出：无。预期行为：六种块类型、三种提示框种类，顺序固定（题面选项按它排）。"""
    assert len(features.BLOCK_KINDS) == 6
    assert len(set(features.BLOCK_KINDS)) == 6
    assert features.BLOCK_KINDS[0] == KIND_HEADING
    assert features.CALLOUT_KINDS == ("note", "tip", "warning")
    assert set(features.CALLOUT_KINDS) <= set(features.ALL_VARIANTS)


def test_all_variants_and_families_are_deduped_and_cover_the_tables() -> None:
    """输入：无。输出：无。预期行为：两张「全部取值」表去重保序，且恰好是各形状的并集。"""
    assert len(set(features.ALL_VARIANTS)) == len(features.ALL_VARIANTS)
    assert len(set(features.ALL_CODE_FAMILIES)) == len(features.ALL_CODE_FAMILIES)
    assert set(features.ALL_VARIANTS) == {
        value for shape in features.SHAPES for value in features.SHAPE_VARIANTS[shape]
    }
    assert set(features.ALL_CODE_FAMILIES) == {
        value for shape in features.SHAPES for value in features.SHAPE_CODE_FAMILIES[shape]
    }


def test_character_budgets_are_consistent_with_the_reading_rules() -> None:
    """输入：无。输出：无。预期行为：字符预算与反解判据自洽（逐语言）。

    两条前提：行宽**明显大于**短行上限（折行后的非末行才不会被读成标题）；段落比两倍短行上限还长
    （才一定折成两行以上，「并段」那一族才有真话可问）。预算本身按语言写在 `[limit]` 里，
    这里把每一种语言都过一遍——中文的 6~24 字标题按字面搬到英文去，模型根本写不出来。
    """
    assert features.SHORT_LIMIT < min(features.WRAPS)
    for lang in locales.available():
        lim = parse.limits(locales.load(lang))
        for name in parse.LIMIT_KEYS:
            low, high = getattr(lim, name)
            assert 0 < low < high, (lang, name)
        assert lim.prose[0] > max(features.WRAPS), lang
        assert lim.heading[1] <= features.SHORT_LIMIT, lang
        assert lim.item[1] <= features.SHORT_LIMIT, lang
        assert lim.callout[0] > features.SHORT_LIMIT, lang
    assert features.MIN_CODE_LINES < features.MAX_CODE_LINES
    assert features.MIN_LIST_ITEMS >= 2
    assert features.MIN_LIST_ITEMS < features.MAX_LIST_ITEMS


def test_build_cases_is_the_full_cartesian_product() -> None:
    """输入：默认采样参数。输出：无。

    预期行为：规模 = 各形状允许的（变体 × 命令族）之和 × 主题 × 行宽；组合两两不同。
    """
    expect = (
        sum(
            len(features.SHAPE_VARIANTS[shape]) * len(features.SHAPE_CODE_FAMILIES[shape])
            for shape in features.SHAPES
        )
        * len(features.TOPICS)
        * len(features.WRAPS)
    )
    assert len(ALL_CASES) == expect
    assert len(set(ALL_CASES)) == len(ALL_CASES)
    assert len(ALL_CASES) == 345


def test_build_cases_skips_a_shape_whose_pool_is_empty() -> None:
    """输入：把变体表收成只有 `na`。输出：无。

    预期行为：带变体的三个形状整片消失——**不**静默改写成别的形状（那会让形状维度的覆盖报告撒谎）。
    """
    cases = features.build_cases(features.DocConfig(variants=("na",)))
    assert {case.shape for case in cases} == {"plain", "code", "quote"}
    assert all(case.variant == "na" for case in cases)


def test_ood_only_keeps_exactly_the_cases_with_an_unseen_value() -> None:
    """输入：`ood_only=True` 与默认。输出：无。

    预期行为：留出的那批每条都至少含一个训练侧没见过的取值，且与同分布侧**按构造不重叠**。
    """
    ood = features.build_cases(features.DocConfig(ood_only=True))
    id_side = [case for case in ALL_CASES if not case.unseen_dims()]
    assert ood
    assert all(case.unseen_dims() for case in ood)
    assert len(ood) + len(id_side) == len(ALL_CASES)
    assert not (set(ood) & set(id_side))


@pytest.mark.parametrize(
    ("case", "want"),
    [
        (features.DocCase("build", 64, "plain", "na", "na"), ()),
        (features.DocCase("incident", 64, "plain", "na", "na"), ("topic",)),
        (features.DocCase("build", 56, "plain", "na", "na"), ("wrap",)),
        (features.DocCase("build", 64, "callout", "tip", "na"), ("variant",)),
        (features.DocCase("build", 64, "code", "na", "container"), ("code_family",)),
        (
            features.DocCase("migration", 56, "mixed", "tip", "container"),
            ("topic", "wrap", "variant", "code_family"),
        ),
    ],
)
def test_unseen_dims_points_at_exactly_the_held_out_values(
    case: features.DocCase, want: tuple[str, ...]
) -> None:
    """输入：几种组合。输出：无。预期行为：四个留出取值各点名自己的维度，见过的取值一个都不点名。"""
    assert case.unseen_dims() == want


def test_material_of_is_stable_and_distinguishes_cases() -> None:
    """输入：全空间。输出：无。预期行为：素材指纹两两不同（否则两条样本会共用同一份块计划）。"""
    materials = [features.material_of(case) for case in ALL_CASES]
    assert len(set(materials)) == len(materials)
    assert materials[0] == features.material_of(ALL_CASES[0])


def test_build_plan_is_reproducible_with_a_fixed_skeleton() -> None:
    """输入：同一个 case。输出：无。

    预期行为：计划与进程、切分无关（种子由素材指纹派生）；骨架是「标题 + 正文段 …… 收尾段」。
    """
    case = features.DocCase("build", 64, "mixed", "note", "vcs")
    plan = features.build_plan(case)
    assert plan == features.build_plan(case)
    assert plan.case == case
    assert plan.kinds()[:2] == (KIND_HEADING, KIND_PARAGRAPH)
    assert plan.kinds()[-1] == KIND_PARAGRAPH


@pytest.mark.parametrize(
    ("shape", "needs", "forbids"),
    [
        ("plain", (), (KIND_LIST_ITEM, KIND_CODE, KIND_QUOTE, KIND_CALLOUT)),
        ("list", (KIND_LIST_ITEM,), (KIND_CODE, KIND_QUOTE, KIND_CALLOUT)),
        ("code", (KIND_CODE,), (KIND_LIST_ITEM, KIND_QUOTE, KIND_CALLOUT)),
        ("quote", (KIND_QUOTE,), (KIND_LIST_ITEM, KIND_CODE, KIND_CALLOUT)),
        ("callout", (KIND_CALLOUT,), (KIND_LIST_ITEM, KIND_CODE, KIND_QUOTE)),
        ("mixed", (KIND_LIST_ITEM, KIND_CODE, KIND_QUOTE, KIND_CALLOUT), ()),
    ],
)
def test_each_shape_produces_exactly_its_own_blocks(
    shape: str, needs: tuple[str, ...], forbids: tuple[str, ...]
) -> None:
    """输入：每种形状 × 它名下的全部组合。输出：无。

    预期行为：该形状该有的块一定在、不该有的块一定不在；列表项数落在预算区间里。
    """
    cases = [case for case in ALL_CASES if case.shape == shape]
    assert cases
    for case in cases:
        kinds = features.build_plan(case).kinds()
        for kind in needs:
            assert kind in kinds, case.describe()
        for kind in forbids:
            assert kind not in kinds, case.describe()
        items = kinds.count(KIND_LIST_ITEM)
        if KIND_LIST_ITEM in needs:
            assert features.MIN_LIST_ITEMS <= items <= features.MAX_LIST_ITEMS, case.describe()
        else:
            assert items == 0, case.describe()


def test_list_items_share_one_list_and_the_flag_follows_the_variant() -> None:
    """输入：`list` 形状的两种变体。输出：无。

    预期行为：同一个列表的项 `list_id` 相同（渲染靠它决定「项与项之间不空行」），有序性由
    `variant` 直接决定——这一维才留得出来。
    """
    cases = [case for case in ALL_CASES if case.shape == "list"]
    assert {case.variant for case in cases} == {"ordered", "bullet"}
    for case in cases:
        slots = [slot for slot in features.build_plan(case).slots if slot.kind == KIND_LIST_ITEM]
        assert len({slot.list_id for slot in slots}) == 1, case.describe()
        assert all(slot.list_ordered == (case.variant == "ordered") for slot in slots)


def test_callout_kind_and_code_family_come_from_the_case() -> None:
    """输入：全空间。输出：无。

    预期行为：提示框种类就是 `variant`（只有带提示框的形状有）、命令族就是 `code_family`
    （只有带代码块的形状有），`code_families_in` 与它一致。
    """
    for case in ALL_CASES:
        plan = features.build_plan(case)
        if case.shape in ("callout", "mixed"):
            assert plan.callout_kinds() == (case.variant,), case.describe()
        else:
            assert plan.callout_kinds() == (), case.describe()
        families = features.code_families_in(plan)
        if case.shape in ("code", "mixed"):
            assert families == (case.code_family,), case.describe()
        else:
            assert families == (), case.describe()


def test_label_for_projects_the_plan_three_ways() -> None:
    """输入：全空间 345 组。输出：无。预期行为：金标签的三条都与计划的三个投影逐项相同。"""
    for case in ALL_CASES:
        plan = features.build_plan(case)
        labels = features.label_for(plan)
        assert labels.kinds == plan.kinds(), case.describe()
        assert labels.callout_kinds == plan.callout_kinds(), case.describe()
        assert labels.list_flags == plan.list_flags(), case.describe()
        assert set(labels.kinds) <= set(features.BLOCK_KINDS)


def test_case_describe_and_as_dict_agree() -> None:
    """输入：一个 case。输出：无。预期行为：可读描述与摊平的维度表指的是同一组取值。"""
    case = features.DocCase("release", 72, "mixed", "warning", "container")
    assert case.describe() == "release/72/mixed/warning/container"
    assert case.as_dict() == {
        "topic": "release",
        "wrap": 72,
        "shape": "mixed",
        "variant": "warning",
        "code_family": "container",
    }
