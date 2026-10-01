"""根因定位的特征层测试：依赖图、事件形状、两条标签路径与计划。"""

from __future__ import annotations

import pytest

from tests.workflows.causechain.conftest import FRAGMENTS, logs_of
from workflows.causechain import features
from workflows.causechain.features import CauseCase

#: 一个普通 case：链式图上根因在末端，五个探针全都可行。
CASE = CauseCase(graph="G1", root="S6", probe="h2")

#: 另一个 case：根因在中间，出错集合只有三个服务（改坏文本时用得上）。
PARTIAL = CauseCase(graph="G2", root="S5", probe="h1")


def test_dimensions_line_up_with_the_graphs_and_probes() -> None:
    """输入：无。输出：无。预期行为：三维取值表就是图名 × 服务 × 探针，顺序稳定。"""
    assert features.DIMS["graph"] == tuple(features.GRAPHS)
    assert features.DIMS["root"] == features.SERVICES
    assert features.DIMS["probe"] == features.PROBES
    assert features.SERVICES == ("S1", "S2", "S3", "S4", "S5", "S6")


def test_edges_are_consistent_with_callees_and_callers() -> None:
    """输入：无。输出：无。

    预期行为：四张图的每条边都同时出现在 `callees[上游]` 与 `callers[下游]` 里，两个方向的
    总边数相等，且三种角色在每张图上都取得到。
    """
    for name, edges in features.GRAPHS.items():
        out = features.callees(name)
        incoming = features.callers(name)
        assert set(out) == set(incoming) == set(features.SERVICES), name
        for upstream, downstream in edges:
            assert downstream in out[upstream], (name, upstream, downstream)
            assert upstream in incoming[downstream], (name, upstream, downstream)
        assert sum(len(peers) for peers in out.values()) == len(edges), name
        roles = {features.role_of(name, service) for service in features.SERVICES}
        assert roles == set(features.ROLES), name


def test_hops_and_ancestors_follow_the_call_direction() -> None:
    """输入：无。输出：无。

    预期行为：链式图 G1 上，末端根因的跳数就是「沿调用方向往上游数」；反方向的（根因的
    下游）不进表。错误只沿上游传播，所以 `errored_services` 就是根因加上它的全部调用方。
    """
    assert features.hops_to("G1", "S6") == {
        "S1": 5, "S2": 4, "S3": 3, "S4": 2, "S5": 1, "S6": 0,
    }
    assert features.ancestors("G1", "S6") == ("S1", "S2", "S3", "S4", "S5")
    assert features.ancestors("G1", "S1") == ()
    assert features.errored_services("G1", "S3") == ("S1", "S2", "S3")
    assert features.errored_services("G2", "S1") == ("S1",)
    assert features.role_of("G1", "S1") == features.ROLE_ENTRY
    assert features.role_of("G1", "S3") == features.ROLE_MIDDLE
    assert features.role_of("G1", "S6") == features.ROLE_LEAF


@pytest.mark.parametrize("hop", [0, 1, 2, 3, 7])
def test_hop_classes_bucket_the_distance(hop: int) -> None:
    """输入：跳数。输出：无。预期行为：0→zero、1→one、2 及以上→more（最后一档吃掉更大的数）。"""
    expect = features.HOP_ZERO if hop == 0 else features.HOP_ONE if hop == 1 else features.HOP_MORE
    assert features.hop_class(hop) == expect
    assert features.hop_class(-1) == features.HOP_ZERO


def test_patterns_are_mutually_exclusive_on_every_fragment() -> None:
    """输入：三段假片段。输出：无。

    预期行为：每段只踩中它自己那一种形状——三条判据互斥是全 workflow 的前提（事件类型既是
    金标签的一半，又是渲染期的兑现闸），有交叠的话「唯一命中」当场失效。
    """
    for event, fragment in FRAGMENTS.items():
        assert features.events_of(fragment) == (event,), fragment
        assert features.event_of(fragment) == event


def test_event_of_needs_exactly_one_hit() -> None:
    """输入：几种日志。输出：无。预期行为：一条都不中、或中了两条以上都给 None。"""
    assert features.event_of("订单 88213 已受理，等待人工复核") is None
    assert features.event_of("上游连接超时，随后写入失败") is None
    assert features.event_of("timeout while calling upstream") == features.EVENT_TIMEOUT
    assert features.event_of("traceback: KeyError") == features.EVENT_ERROR
    assert features.event_of("success") == features.EVENT_OK


def test_build_cases_keeps_only_feasible_probes() -> None:
    """输入：无。输出：无。

    预期行为：全空间 4×6×5 = 120 组里剔掉 20 组不可行的（根因是入口服务时问不出「1 跳」
    与「2 跳以上」，根因没有调用方时也问不出「非根因的出错服务」），剩下 100 组，顺序稳定。
    """
    cases = features.build_cases(512)
    assert len(cases) == 100
    assert cases == features.build_cases(512)
    assert all(features.probe_target(case, 0) is not None for case in cases)
    for case in cases:
        if case.probe in ("h1", "h2", "rno"):
            assert features.callers(case.graph)[case.root], case.describe()


def test_probe_counts_are_capped_by_the_topology() -> None:
    """输入：全空间 100 组 case。输出：无。

    预期行为：每个探针的可用次数由拓扑定死，这里是它的**上限**——`h1` / `rno` 要求根因有
    调用方（4 张图 × 5 个非入口服务 = 20），`h2` 还要求根因的上游有两层（只有 12 组满足），
    `ryes` 每组都可行（24）。挑 case 的那一步（`workflow._balanced_pick`）把这几个上限都用满，
    多出来的名额只能给到 `h0`，所以三档跳数的实际占比是 24 / 20 / 12 而不是等分。
    """
    counted: dict[str, int] = dict.fromkeys(features.PROBES, 0)
    for case in features.build_cases(512):
        counted[case.probe] += 1
    assert counted == {"h0": 24, "h1": 20, "h2": 12, "ryes": 24, "rno": 20}


def test_two_label_paths_agree_on_every_case() -> None:
    """输入：全空间 100 组 case。输出：无。

    预期行为：「构造侧从计划推」与「审计侧从文本重推」两条路走出来的根因、出错集合与跳数
    逐条相同——日志兑现了计划，两条路就不该打架。
    """
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        logs = logs_of(plan)
        assert features.event_of(logs[case.root]) in (
            features.EVENT_TIMEOUT,
            features.EVENT_ERROR,
        ), case.describe()
        seen = features.observed_labels(case.graph, logs)
        assert seen is not None, case.describe()
        labels = features.label_for(plan)
        assert seen.root == labels.root == case.root
        assert seen.errored == labels.errored
        assert seen.distances == labels.distances
        assert labels.probed in labels.errored


def test_observed_labels_refuses_text_that_breaks_the_graph() -> None:
    """输入：两种兑现不了依赖图的文本。输出：无。

    预期行为：多出一个出错服务后，两个服务同时「没有出错下游」、或出错集合与「根因 + 它的
    调用方」对不上，都给 None——审计据此记成「现场日志认不出来」，而不是硬凑一个根因出来。
    """
    logs = logs_of(features.plan_for(PARTIAL, 0))
    assert features.observed_labels(PARTIAL.graph, logs) is not None
    broken = dict(logs)
    broken["S3"] = FRAGMENTS[features.EVENT_ERROR]
    assert features.observed_labels(PARTIAL.graph, broken) is None
    all_ok = dict.fromkeys(features.SERVICES, FRAGMENTS[features.EVENT_OK])
    assert features.observed_labels(PARTIAL.graph, all_ok) is None


def test_plan_is_reproducible_and_covers_both_root_events() -> None:
    """输入：同一个 case 的两次取样。输出：无。

    预期行为：计划与进程无关；六个服务都有事件；**根因那条是超时还是报错由指纹定**，
    两种都要出现——否则「唯一的超时那条就是根因」会变成一条真的捷径。
    """
    plan = features.plan_for(CASE, 0)
    assert plan == features.plan_for(CASE, 0)
    assert set(plan.events) == set(features.SERVICES)
    assert set(plan.order) == set(features.SERVICES)
    assert plan.times() == {service: f"T{index + 1}" for index, service in enumerate(plan.order)}
    assert len(plan.distractors) == 3
    assert len(set(plan.distractors)) == 3
    assert all(combo != plan.errored for combo in plan.distractors)
    root_events = {
        features.plan_for(case, 0).events[case.root] for case in features.build_cases(512)
    }
    assert root_events == {features.EVENT_TIMEOUT, features.EVENT_ERROR}


def test_graph_of_roundtrips_and_rejects_unknown_edges() -> None:
    """输入：几张边表。输出：无。

    预期行为：按**集合**比——行序换了照样认得出来；不是四张图里任何一张就给 None。
    """
    for name, edges in features.GRAPHS.items():
        assert features.graph_of(edges) == name
        assert features.graph_of(tuple(reversed(edges))) == name
    assert features.graph_of(()) is None
    assert features.graph_of((("S1", "S6"),)) is None


def test_case_describe_and_fingerprint() -> None:
    """输入：一个 case。输出：无。预期行为：指纹与可读描述都稳定、键序固定。"""
    assert CASE.describe() == "G1-S6-h2"
    assert CASE.fingerprint() == "graph=G1|root=S6|probe=h2"
    assert CASE.as_dict() == {"graph": "G1", "root": "S6", "probe": "h2"}


def test_count_index_is_the_size_of_the_errored_set() -> None:
    """输入：全空间 100 组 case。输出：无。

    预期行为：下标 = 出错服务数 − 1，且永远落在 `0 .. COUNT_LEVELS-1` 里；两条标签路径
    算出来的影响面必须一致（构造侧 `label_for` 与审计侧 `observed_labels` 走的是两条路）。
    """
    for case in features.build_cases(512):
        plan = features.plan_for(case, 0)
        labels = features.label_for(plan)
        assert labels.count_index == len(labels.errored) - 1
        assert 0 <= labels.count_index < features.COUNT_LEVELS
        observed = features.observed_labels(plan.graph, logs_of(plan))
        assert observed is not None
        assert observed.count_index == labels.count_index


def test_count_levels_covers_every_possible_affected_size() -> None:
    """输入：全空间 100 组 case。输出：无。

    预期行为：档位数就是服务数（最少只有根因自己出错、最多整张图都出错），而且这 100 组里
    **每一档都真的取到过**——有一档取不到，那一族的选项就是死选项。
    """
    assert features.COUNT_LEVELS == len(features.SERVICES)
    seen = {
        features.count_index(features.errored_services(case.graph, case.root))
        for case in features.build_cases(512)
    }
    assert seen == set(range(features.COUNT_LEVELS))
