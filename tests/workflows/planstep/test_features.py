"""特征层：四条规则的纯函数、两条独立标签路径、case 空间与留出下标。"""

from __future__ import annotations

from collections import Counter
from typing import Final

import pytest

from workflows.planstep import features as ft

#: 同分布全空间的 case（10 张同分布图 × 6 个已完成项数）。
FULL: Final[tuple[ft.PlanCase, ...]] = tuple(
    ft.build_cases(512, dims={"graph": ft.ID_GRAPHS, "done": ft.DONE_COUNTS})
)
#: 留出全空间的 case（两张留出图 × 6 个已完成项数）。
OOD: Final[tuple[ft.PlanCase, ...]] = tuple(
    ft.build_cases(512, dims={"graph": ft.OOD_GRAPHS, "done": ft.DONE_COUNTS})
)


def test_case_space_covers_the_whole_matrix() -> None:
    """输入：无。输出：无。预期行为：全空间 = 图形状条数 × 已完成项数，一条不多不少。"""
    assert len(FULL) == len(ft.ID_GRAPHS) * len(ft.DONE_COUNTS) == 60
    assert len(OOD) == len(ft.OOD_GRAPHS) * len(ft.DONE_COUNTS) == 12


def test_default_dims_are_the_id_space() -> None:
    """输入：无。输出：无。预期行为：`DIMS` 就是同分布侧那 60 组，不给维度表时按它枚举。"""
    assert ft.DIMS == {"graph": ft.ID_GRAPHS, "done": ft.DONE_COUNTS}
    assert ft.build_cases(512) == list(FULL)


def test_every_graph_is_a_dag_with_five_tasks() -> None:
    """输入：全部依赖图。输出：无。

    预期行为：每张图正好 `TASK_COUNT` 项、前置编号不越界，且**推进一遍能把五项全部做完**
    （完成顺序是 0~4 的一个排列）——有环的图会让 `completion_order` 静默少项。
    """
    for graph in range(len(ft.GRAPHS)):
        deps = ft.deps_of(graph)
        assert len(deps) == ft.TASK_COUNT
        assert all(0 <= dep < ft.TASK_COUNT for row in deps for dep in row), graph
        assert sorted(ft.completion_order(deps)) == list(range(ft.TASK_COUNT)), graph


def test_deps_of_reports_a_bad_index() -> None:
    """输入：越界下标。输出：无。预期行为：当场报错，不静默回绕到别的图。"""
    with pytest.raises(ValueError, match="图形状下标"):
        ft.deps_of(len(ft.GRAPHS))


def test_ood_graphs_are_never_in_the_id_space() -> None:
    """输入：无。输出：无。预期行为：留出下标与同分布下标不相交，合起来正好是全部图。"""
    assert not set(ft.OOD_GRAPHS) & set(ft.ID_GRAPHS)
    assert set(ft.OOD_GRAPHS) | set(ft.ID_GRAPHS) == set(range(len(ft.GRAPHS)))


def test_depths_are_the_longest_prerequisite_chain() -> None:
    """输入：手写的链式图、菱形图与交叉图。输出：无。

    预期行为：层数 = 从最源头到这一项的最长前置链长度；有多条链时取**最长**的那条。
    """
    assert ft.depths_of(((), (0,), (1,), (2,), (3,))) == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    assert ft.depths_of(((), (0,), (0,), (1, 2), (3,))) == {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
    assert ft.depths_of(((), (), (0, 1), (1,), (2, 3))) == {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}


def test_ready_requires_every_prerequisite_done() -> None:
    """输入：手写图与已完成集合。输出：无。

    预期行为：规则 1——只要**有一个**前置没做完就不算可开工，其余前置做完了也没用。
    """
    deps = ((), (0,), (0,), (1, 2), (3,))
    assert ft.ready_of(deps, frozenset()) == (0,)
    assert ft.ready_of(deps, frozenset({0})) == (1, 2)
    assert ft.ready_of(deps, frozenset({0, 1})) == (2,)
    assert ft.ready_of(deps, frozenset({0, 1, 2})) == (3,)
    assert ft.ready_of(deps, frozenset(range(5))) == ()


def test_next_prefers_the_shallowest_then_the_smallest_number() -> None:
    """输入：手写图。输出：无。预期行为：规则 2——先比层数，同层再比编号。"""
    deps = ((), (0,), (0,), (1, 2), (3,))
    depths = ft.depths_of(deps)
    assert ft.next_of(deps, frozenset(), depths) == 0
    assert ft.next_of(deps, frozenset({0}), depths) == 1
    assert ft.next_of(deps, frozenset({0, 1}), depths) == 2
    assert ft.next_of(deps, frozenset(range(5)), depths) is None


def test_next_prefers_depth_over_a_smaller_number() -> None:
    """输入：一个「编号大的反而更浅」的图。输出：无。

    预期行为：层数是第一关键字——第 4 项（编号更大、层数 0）压过第 2 项（编号更小、层数 1）。
    """
    deps = ((), (2,), (0,), (1,), ())
    depths = ft.depths_of(deps)
    assert (depths[2], depths[4]) == (1, 0)
    assert ft.next_of(deps, frozenset({0}), depths) == 4


def test_rounds_batches_every_ready_task_together() -> None:
    """输入：链式图与扇出图。输出：无。

    预期行为：规则 3 + 4——一轮把**当时**所有可开工的项一起做完；链式图每轮只推一项，
    扇出图一轮能推好几项。
    """
    chain = ft.deps_of(0)
    prefix = ft.completion_order(chain)
    assert [ft.rounds_of(chain, frozenset(prefix[:k])) for k in range(6)] == [5, 4, 3, 2, 1, 0]
    fan = ft.deps_of(6)  # ((), (0,), (0,), (0,), (1, 2))：一轮推 0，一轮推 1/2/3，再一轮推 4
    assert ft.rounds_of(fan, frozenset()) == 3
    assert ft.rounds_of(fan, frozenset({0})) == 2
    assert ft.rounds_of(fan, frozenset(range(5))) == 0


def test_unlocks_only_counts_an_immediate_new_ready_task() -> None:
    """输入：菱形图与几种已完成集合。输出：无。

    预期行为：只算「做完下一步就**立刻**多一项可开工」——还差两个前置的项不算解锁。
    """
    deps = ((), (0,), (0,), (1, 2), (3,))
    depths = ft.depths_of(deps)
    assert ft.unlocks_of(deps, frozenset(), ft.next_of(deps, frozenset(), depths)) is True
    assert ft.unlocks_of(deps, frozenset({0}), ft.next_of(deps, frozenset({0}), depths)) is False
    assert ft.unlocks_of(deps, frozenset({0, 1}), ft.next_of(deps, frozenset({0, 1}), depths)) is True
    assert ft.unlocks_of(deps, frozenset(range(5)), None) is False


def test_every_family_takes_every_level() -> None:
    """输入：同分布全空间。输出：无。

    预期行为：四族各自把取值铺满——`remaining` / `rounds` 取满 0~5，`next` 取满六个选项，
    `unlocks` 两个布尔都出现。某一档取不到时那一族的等级表就有一个等级永远不是答案。
    """
    labels = [ft.label_for(case) for case in FULL]
    assert {item.remaining for item in labels} == set(range(ft.TASK_COUNT + 1))
    assert {item.rounds for item in labels} == set(range(ft.TASK_COUNT + 1))
    assert {item.next_key for item in labels} == {f"t{i}" for i in range(1, 6)} | {"none"}
    assert {item.unlocks for item in labels} == {True, False}


def test_remaining_and_next_are_exactly_uniform_on_the_id_space() -> None:
    """输入：同分布全空间。输出：无。

    预期行为：每张图的完成顺序都是 0~4 的全排列，所以「还剩几项」与「下一步是谁」在
    10 张图 × 6 档上**恰好各出现 10 次**——不是近似均匀，是精确均匀。
    """
    labels = [ft.label_for(case) for case in FULL]
    assert set(Counter(item.remaining for item in labels).values()) == {10}
    assert set(Counter(item.next_key for item in labels).values()) == {10}


def test_next_follows_the_completion_order() -> None:
    """输入：同分布全空间。输出：无。

    预期行为：已完成 k 项时，下一步就是完成顺序里的第 k 项；全做完时没有下一步。
    这条把 `next_of` 与 `completion_order` 两种写法绑在一起，改坏任何一个都会红。
    """
    for case in FULL:
        order = ft.completion_order(case.deps)
        want = "none" if case.done >= ft.TASK_COUNT else f"t{order[case.done] + 1}"
        assert ft.label_for(case).next_key == want, case.describe()


def test_done_set_is_the_prefix_of_the_completion_order() -> None:
    """输入：全空间 case。输出：无。预期行为：已完成集合 = 完成顺序的前 k 项。"""
    for case in FULL + OOD:
        assert case.done_set == frozenset(ft.completion_order(case.deps)[: case.done])


def test_the_two_label_paths_agree() -> None:
    """输入：全空间 case。输出：无。

    预期行为：`label_for`（注入特征）与 `observed_labels`（反解出来的表）给同一批标签
    ——构造侧与审计侧共用同一个函数，这条防的是将来有人把它拆成两份实现。
    """
    for case in FULL + OOD:
        assert ft.label_for(case) == ft.observed_labels(case.deps, case.done_set)


def test_fingerprint_and_describe_are_stable() -> None:
    """输入：同一个 case 造两次。输出：无。预期行为：指纹与描述都不随进程变化。"""
    assert ft.PlanCase(graph=7, done=3).fingerprint() == "graph=7|done=3"
    assert ft.PlanCase(graph=7, done=3).describe() == "G7-D3"
    assert ft.PlanCase(graph=7, done=3).as_dict() == {"graph": 7, "done": 3}
