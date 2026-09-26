"""真值规格的不变量：条款均衡、无蕴含、严重度矩阵、处置规则、句序。"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Any

import pytest

from workflows.guardrail import spec as sp


def all_cases() -> list[sp.GuardCase]:
    """输入：无。输出：默认配置下的全部 case（零 token）。"""
    return sp.build_cases(sp.GuardConfig())


def test_every_clause_appears_five_times() -> None:
    """输入：违规子集表。输出：无。预期行为：六条条款各出现 5 次（共 11 项，占比 0.455）。"""
    counts = Counter(clause for subset in sp.VIOLATION_SETS for clause in subset)
    assert set(counts) == set(sp.CLAUSES)
    assert set(counts.values()) == {5}
    assert len(sp.VIOLATION_SETS) == 11


def test_no_clause_implies_another() -> None:
    """输入：违规子集表。输出：无。

    预期行为：没有任何一对条款总是同时出现——「有 A 就一定有 B」是送答案的捷径，
    这条断言把这张表钉住。
    """
    for left, right in itertools.permutations(sp.CLAUSES, 2):
        where = [subset for subset in sp.VIOLATION_SETS if left in subset]
        assert where and not all(right in subset for subset in where), (left, right)


def test_top_clause_covers_all_seven_values() -> None:
    """输入：全部 case。输出：无。预期行为：七个取值都出得来，且最大偏差 ≤ 0.06。"""
    counter = Counter(
        sp.top_clause_of(case.violations, sp.violation_order(case)) for case in all_cases()
    )
    assert set(counter) == set(sp.CLAUSES) | {"none"}
    total = sum(counter.values())
    assert max(abs(count / total - 1 / 7) for count in counter.values()) <= 0.08


def test_sentence_order_is_not_the_clause_table_order() -> None:
    """输入：11 张子集（至少两条条款的）。输出：无。

    预期行为：多数 case 的句序与条款表顺序不同——否则「并列取正文里最先出现的」
    会退化成「取表里靠前的」，`top_clause` 的均衡立刻垮掉。
    """
    pairs = []
    for subset in sp.VIOLATION_SETS:
        if len(subset) < 2:
            continue
        case = sp.GuardCase("public", "inbound", subset, (), "general", "clean", "plain")
        pairs.append((subset, sp.violation_order(case)))
    same = sum(
        1 for subset, order in pairs if order == tuple(c for c in sp.CLAUSES if c in subset)
    )
    assert same * 2 < len(pairs), f"{same}/{len(pairs)} 的句序与条款表顺序一致"


def test_severity_matrix_cells() -> None:
    """输入：严重度矩阵。输出：无。预期行为：九格与判据原文一致；无违规固定 0 档。"""
    assert sp.SEVERITY_TABLE == {
        ("low", "single"): 0,
        ("low", "group"): 1,
        ("low", "public"): 1,
        ("medium", "single"): 1,
        ("medium", "group"): 2,
        ("medium", "public"): 3,
        ("high", "single"): 2,
        ("high", "group"): 3,
        ("high", "public"): 4,
    }
    for reach in sp.REACHES:
        assert sp.severity_of((), reach) == 0


def test_severity_uses_the_highest_harm_level() -> None:
    """输入：混级条款。输出：无。预期行为：取最高级别查表，不做加权求和。"""
    assert sp.max_harm_of(("spam", "harass", "meds")) == "high"
    assert sp.severity_of(("spam", "harass", "meds"), "group") == 3
    assert sp.max_harm_of(()) is None
    assert sp.top_clause_of((), ()) == "none"


def test_action_rules_are_ordered() -> None:
    """输入：六条规则各自的例子。输出：无。

    预期行为：按顺序取第一个命中的——①（转专项）要压过 ②③④；⑤ 只在没命中拦截时给复核。
    """
    assert (
        sp.action_of(
            severity=4,
            direction="outbound",
            reach="public",
            audience="minors",
            history="repeat",
            violations=("meds",),
        )
        == "route"
    )
    assert (
        sp.action_of(
            severity=3,
            direction="outbound",
            reach="group",
            audience="general",
            history="clean",
            violations=("meds",),
        )
        == "block"
    )
    assert (
        sp.action_of(
            severity=3,
            direction="inbound",
            reach="public",
            audience="general",
            history="clean",
            violations=("harass",),
        )
        == "block"
    )
    assert (
        sp.action_of(
            severity=2,
            direction="inbound",
            reach="group",
            audience="general",
            history="repeat",
            violations=("harass",),
        )
        == "block"
    )
    assert (
        sp.action_of(
            severity=2,
            direction="inbound",
            reach="group",
            audience="general",
            history="clean",
            violations=("harass",),
        )
        == "review"
    )
    assert (
        sp.action_of(
            severity=1,
            direction="inbound",
            reach="group",
            audience="general",
            history="clean",
            violations=("spam",),
        )
        == "pass"
    )


def test_minor_exposure_and_repeat_offender_boundaries() -> None:
    """输入：两条分界线两侧的例子。输出：无。预期行为：口径与题面原文一致。"""
    assert sp.minor_exposure_of("minors", ("meds",)) is True
    assert sp.minor_exposure_of("minors", ("harass",)) is True  # 中危也算暴露
    assert sp.minor_exposure_of("minors", ("spam",)) is False  # 只有低危不算
    assert sp.minor_exposure_of("general", ("meds",)) is False
    assert sp.repeat_offender_of("repeat", 2) is True
    assert sp.repeat_offender_of("repeat", 1) is False
    assert sp.repeat_offender_of("warned", 4) is False


def test_violation_order_is_a_permutation_and_stable(make_case: Any) -> None:
    """输入：一个 case。输出：无。预期行为：句序是违规集合的排列，两次调用一致。"""
    case = make_case()
    order = sp.violation_order(case)
    assert sorted(order) == sorted(case.violations)
    assert order == sp.violation_order(case)
    slots = sp.slot_order(case)
    assert len(slots) == len(case.violations) + len(case.mentions) + sp.FILLER_SLOTS


def test_content_id_is_deterministic_and_split_free(make_case: Any) -> None:
    """输入：同一个 case 反复取内容号。输出：无。

    预期行为：确定性、带前缀；内容号由素材指纹派生，**不含切分名**——
    否则同一条内容在不同切分里会拿到不同号，等于把切分信息写进 state。
    """
    case = make_case()
    ids = {sp.content_id_of(case) for _ in range(3)}
    assert len(ids) == 1
    only = ids.pop()
    assert only.startswith(sp.CONTENT_PREFIX)
    assert not any(split in only for split in ("train", "val", "test"))


def test_material_fingerprint_covers_all_seven_dimensions(make_case: Any) -> None:
    """输入：逐维改一个取值的 case。输出：无。预期行为：任一维变化都改指纹。"""
    base = make_case()
    overrides = (
        {"reach": "group"},
        {"direction": "inbound"},
        {"violations": ("spam",)},
        {"mentions": ("meds",)},
        {"audience": "minors"},
        {"history": "repeat"},
        {"wording": "quoted"},
    )
    prints = {sp.material_of(make_case(**override)) for override in overrides}
    assert len(prints) == len(overrides)
    assert sp.material_of(base) not in prints


def test_label_of_normalizes_order() -> None:
    """输入：同一集合的两种写法。输出：无。预期行为：标签唯一（按条款表顺序归一）。"""
    assert sp.label_of(("spam", "harass")) == sp.label_of(("harass", "spam")) == "harass+spam"
    assert sp.label_of(()) == "none"


def test_config_rejects_unknown_values_and_empty_pools() -> None:
    """输入：坏旋钮。输出：无。预期行为：构造期报人话，点名是哪个字段。"""
    with pytest.raises(ValueError, match="reaches"):
        sp.GuardConfig(reaches=())
    with pytest.raises(ValueError, match="violations"):
        sp.GuardConfig(violations=(("nope",),))


def test_build_cases_enumerates_the_whole_grid() -> None:
    """输入：默认配置。输出：无。预期行为：全枚举（七维笛卡尔积），顺序稳定。"""
    cases = all_cases()
    assert len(cases) == 11 * 3 * 2 * 4 * 2 * 3 * 3
    assert cases == all_cases()


def test_ood_only_keeps_cases_with_an_unseen_value() -> None:
    """输入：`ood_only` 配置与训练侧配置。输出：无。

    预期行为：`ood_only` 每条都至少含一个训练侧没见过的取值；训练侧配置下一条都没有。
    """
    ood = sp.build_cases(sp.GuardConfig(split="test_ood", ood_only=True))
    assert ood and all(case.unseen_dims() for case in ood)
    seen = sp.build_cases(
        sp.GuardConfig(
            mentions=tuple(sorted(sp.SEEN_MENTIONS)),
            histories=tuple(sorted(sp.SEEN_HISTORIES)),
            wordings=tuple(sorted(sp.SEEN_WORDINGS)),
        )
    )
    assert seen and not any(case.unseen_dims() for case in seen)
