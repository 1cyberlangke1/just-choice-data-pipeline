"""模型路由的特征层：维度表、候选目录、门槛与标签规则。"""

from __future__ import annotations

from collections import Counter
from itertools import product

import pytest

from jc.cover import cover_dims, missing_pairs
from workflows.modelroute import features
from workflows.modelroute.features import TagSet


def test_full_space_is_the_cartesian_product() -> None:
    """输入：无。输出：无。

    预期行为：把上限开到全空间以上时，枚举结果**恰好**是 6×3×2×2×3 = 216 组——本 workflow
    的维度之间没有禁用对，所以全空间就是笛卡尔积，这条把「没有禁用对」钉死。
    """
    cases = features.build_cases(1000)
    assert len(cases) == 216
    assert {tuple(case.as_dict().values()) for case in cases} == set(
        product(
            features.TASK_KINDS,
            features.SIZES,
            features.LATENCIES,
            features.BUDGETS,
            features.HEALTHS,
        )
    )


def test_pairwise_cover_still_covers_every_pair() -> None:
    """输入：无。输出：无。

    预期行为：把上限压到全空间以下时走成对覆盖，行数远少于 216，但任意两维的每个取值对
    至少出现一次——这是 `jc.cover` 在本 workflow 的维度表上的交叉验证。
    """
    cases = features.build_cases(48)
    assert 0 < len(cases) < 216
    assert not missing_pairs(features.DIMS, [case.as_dict() for case in cases])


def test_catalog_triples_are_pairwise_distinct() -> None:
    """输入：无。输出：无。

    预期行为：四行的 (能力, 价格, 时延) 三元组两两不同——不然「按能力最高、再按便宜、
    再按快」会退化成「看谁排在前面」，答案就不再是表上可复核的事实。
    """
    triples = [(cand.ability, cand.price, cand.latency) for cand in features.CATALOG]
    assert len(set(triples)) == len(triples)
    assert len({cand.name for cand in features.CATALOG}) == len(features.CATALOG)


def test_catalog_names_match_route_options() -> None:
    """输入：无。输出：无。预期行为：route 的选项就是目录里的模型名加 `none`，不多不少。"""
    assert set(features.CHOICE_OPTIONS["route"]) == {
        cand.name for cand in features.CATALOG
    } | {features.NO_CANDIDATE}


def test_head_row_window_exceeds_every_need() -> None:
    """输入：无。输出：无。

    预期行为：第一行的窗口比任何规模的窗口需求都大——这正是 `blocked` 那一族**不设
    「窗口」档**的理由；这条断言把那个前提钉住，将来改目录时会当场暴露。
    """
    head = features.CATALOG[0]
    assert head.window > max(features.WINDOW_NEED.values())
    assert "window" not in features.BLOCKED_REASONS


def test_needs_for_maps_every_dimension_value() -> None:
    """输入：无。输出：无。

    预期行为：每个维度的每个取值都在门槛表里有对应项——漏一个会在采样时抛 `KeyError`，
    那时已经烧了配额。
    """
    for kind in features.TASK_KINDS:
        assert features.ABILITY_NEED[kind] >= 1
    for size in features.SIZES:
        assert features.WINDOW_NEED[size] > 0
    for latency in features.LATENCIES:
        assert features.LATENCY_LIMIT[latency] >= 1
    for budget in features.BUDGETS:
        assert features.PRICE_LIMIT[budget] >= 1
    for health in features.HEALTHS:
        assert isinstance(features.DOWN_PROVIDERS[health], tuple)
    needs = TagSet("chat", "small", "batch", "normal", "all_ok").needs
    assert needs == features.Needs(window=8, ability=1, latency=3, price=3, down=())


def test_survivors_apply_all_four_gates() -> None:
    """输入：一组门槛。输出：无。

    预期行为：四道门槛是「与」关系——任何一道不过就不在可用集合里；目录顺序保持不变。
    """
    needs = TagSet("code_gen", "small", "batch", "tight", "all_ok").needs
    assert [cand.name for cand in features.survivors(needs)] == ["borealis-pro", "cinder-mini"]
    blocked = TagSet("math_reason", "huge", "interactive", "tight", "borealis_down").needs
    assert features.survivors(blocked) == []


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (("chat", "small", "batch", "normal", "all_ok"), "atlas-xl"),
        (("code_gen", "small", "batch", "tight", "all_ok"), "cinder-mini"),
        (("code_gen", "medium", "interactive", "normal", "all_ok"), "borealis-pro"),
        (("extraction", "small", "interactive", "normal", "borealis_down"), "atlas-lite"),
        (("code_gen", "huge", "batch", "normal", "atlas_down"), "borealis-pro"),
        (("math_reason", "huge", "interactive", "tight", "borealis_down"), "none"),
    ],
)
def test_route_rules(tags: tuple[str, ...], expected: str) -> None:
    """输入：一组特征与期望路由。输出：无。预期行为：选人规则与文档一致。"""
    assert TagSet(*tags).label().route == expected


@pytest.mark.parametrize(
    ("tags", "expected"),
    [
        (("chat", "small", "batch", "normal", "all_ok"), "usable"),
        (("code_gen", "small", "batch", "tight", "all_ok"), "budget"),
        (("chat", "small", "interactive", "normal", "all_ok"), "latency"),
        (("chat", "small", "batch", "normal", "atlas_down"), "provider_down"),
        # borealis 挂了不影响第一行：它的提供方是 atlas
        (("chat", "small", "batch", "normal", "borealis_down"), "usable"),
    ],
)
def test_blocked_rules(tags: tuple[str, ...], expected: str) -> None:
    """输入：一组特征与期望原因。输出：无。预期行为：第一行卡在哪一关，判据与文档一致。"""
    assert TagSet(*tags).label().blocked == expected


@pytest.mark.parametrize(
    ("tags", "cost", "slack"),
    [
        (("chat", "small", "batch", "normal", "all_ok"), 3, True),
        (("extraction", "small", "interactive", "normal", "borealis_down"), 1, True),
        (("chat", "medium", "interactive", "tight", "borealis_down"), 1, False),
        (("code_gen", "huge", "batch", "normal", "atlas_down"), 2, False),
        (("math_reason", "huge", "interactive", "tight", "borealis_down"), 0, False),
    ],
)
def test_cost_and_slack_rules(tags: tuple[str, ...], cost: int, slack: bool) -> None:
    """输入：一组特征与期望的价位 / 窗口富余。输出：无。

    预期行为：价位是选中模型的价格档（没有候选时 0 档）；富余是「窗口至少两倍于需求」
    （没有候选时不算富余）。
    """
    labels = TagSet(*tags).label()
    assert labels.cost == cost
    assert labels.slack is slack


def test_labels_cover_every_value() -> None:
    """输入：全空间。输出：无。

    预期行为：五个标签的每个取值都真的出现过——某个取值一次都不出现，说明目录或门槛表
    把它挤没了（均衡闸在真跑时才会发现，那时已经烧了配额）。
    """
    counts: dict[str, Counter[str]] = {
        name: Counter() for name in ("kind", "route", "blocked", "cost", "slack")
    }
    for case in features.build_cases(1000):
        labels = case.label()
        for name in counts:
            counts[name][str(getattr(labels, name))] += 1
    assert set(counts["kind"]) == set(features.TASK_KINDS)
    assert set(counts["route"]) == set(features.CHOICE_OPTIONS["route"])
    assert set(counts["blocked"]) == set(features.BLOCKED_REASONS)
    assert set(counts["cost"]) == {"0", "1", "2", "3"}
    assert set(counts["slack"]) == {"True", "False"}


def test_label_flip_axes_lists_one_change_neighbours() -> None:
    """输入：一组特征。输出：无。

    预期行为：邻居数是「每维其它取值个数」之和（5+2+1+1+2 = 11），且不含自己——
    最小对覆盖率就是拿它当参照算的。
    """
    case = TagSet("chat", "small", "batch", "normal", "all_ok")
    neighbours = features.label_flip_axes(case)
    assert len(neighbours) == 11
    assert case not in neighbours
    for neighbour in neighbours:
        changed = [
            key
            for key, value in neighbour.as_dict().items()
            if value != case.as_dict()[key]
        ]
        assert len(changed) == 1, neighbour.as_dict()


def test_build_cases_is_deterministic() -> None:
    """输入：无。输出：无。预期行为：同一入参两次调用给出同一份列表。"""
    first = features.build_cases(48)
    second = features.build_cases(48)
    assert [c.as_dict() for c in first] == [c.as_dict() for c in second]
    assert cover_dims(features.DIMS, max_full=48) == [
        case.as_dict() for case in first
    ]
