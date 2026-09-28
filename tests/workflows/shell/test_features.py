"""命令审批的特征层：维度表、禁用对、标签规则。"""

from __future__ import annotations

from collections import Counter
from itertools import product

import pytest

from workflows.shell import features
from workflows.shell.features import TagSet


def all_compatible() -> list[TagSet]:
    """输入：无。输出：全部相容的 tag 组合。

    预期行为：直接穷举 2×15×5×2 = 300 组再按 `compatible` 过滤——这是**手推的独立口径**，
    与 `jc.cover` 的成对覆盖互为交叉验证。
    """
    out: list[TagSet] = []
    for values in product(features.DIALECTS, features.GOALS, features.TARGETS, features.ELEVATIONS):
        case = TagSet(*values)
        if features.compatible(case):
            out.append(case)
    return out


def test_goal_groups_partition() -> None:
    """输入：无。输出：无。预期行为：四个动作组两两不交、并集恰好是 `GOALS`。"""
    groups = [
        set(features.READ_GOALS),
        set(features.WRITE_GOALS),
        set(features.DESTRUCTIVE_GOALS),
        set(features.PRIVILEGED_GOALS),
    ]
    total: set[str] = set()
    for group in groups:
        assert not (total & group), f"动作组重叠：{sorted(total & group)}"
        total |= group
    assert total == set(features.GOALS)
    assert list(features.GOALS) == [
        *features.READ_GOALS,
        *features.WRITE_GOALS,
        *features.DESTRUCTIVE_GOALS,
        *features.PRIVILEGED_GOALS,
    ]


def test_forbidden_pairs_are_exactly_fourteen() -> None:
    """输入：无。输出：无。

    预期行为：被禁的 (动作, 位置) 对恰好 14 组（删表 4 + 删容器 4 + 重启服务 3 + 丢弃改动 3），
    于是全空间 300 组里相容的是 300 - 14×2×2 = 244 组。这条断言把「约束表」钉死，
    免得有人悄悄加一条约束却没人知道覆盖率变了。
    """
    forbidden_pairs = [
        (goal, target)
        for goal in features.GOALS
        for target in features.TARGETS
        if not features.allow_pair("goal", goal, "target", target)
    ]
    assert len(forbidden_pairs) == 14, sorted(forbidden_pairs)
    assert len(all_compatible()) == 300 - 14 * len(features.DIALECTS) * len(features.ELEVATIONS)


def test_cover_cases_are_all_compatible_and_cover_pairs() -> None:
    """输入：无。输出：无。

    预期行为：成对覆盖给出的每一组都相容；且任意两维的每个相容取值对至少出现一次。
    """
    cases = features.build_cases(32)
    assert cases, "成对覆盖不该为空"
    for case in cases:
        assert features.compatible(case), case

    seen = {name: set() for name in features.DIMS}
    for case in cases:
        for name, value in case.as_dict().items():
            seen[name].add(value)
    for name, values in features.DIMS.items():
        assert seen[name] == set(values), f"{name} 有取值没覆盖：{sorted(set(values) - seen[name])}"

    pairs = set()
    names = list(features.DIMS)
    for case in cases:
        values = case.as_dict()
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                pairs.add((values[left], values[right]))
    for left in features.GOALS:
        for right in features.TARGETS:
            if features.allow_pair("goal", left, "target", right):
                assert (left, right) in pairs, f"取值对没覆盖：{left}×{right}"


def test_labels_cover_every_value() -> None:
    """输入：无。输出：无。

    预期行为：在全空间上，五个标签的每个取值都真的出现过——某个取值一次都不出现，
    说明约束表把它挤没了（均衡闸在真跑时才会发现，那时已经烧了配额）。
    """
    counts: dict[str, Counter[str]] = {name: Counter() for name in ("verdict", "target_owner")}
    flags: dict[str, set[bool]] = {"destructive": set(), "elevated": set()}
    radius: set[int] = set()
    for case in all_compatible():
        labels = case.label()
        counts["verdict"][labels.verdict] += 1
        counts["target_owner"][labels.target_owner] += 1
        flags["destructive"].add(labels.destructive)
        flags["elevated"].add(labels.elevated)
        radius.add(labels.blast_radius)
    assert set(counts["verdict"]) == {"allow", "confirm", "deny"}
    assert set(counts["target_owner"]) == set(features.TARGETS)
    assert flags["destructive"] == {True, False}
    assert flags["elevated"] == {True, False}
    assert radius == {0, 1, 2, 3, 4}


@pytest.mark.parametrize(
    ("goal", "target", "elevation", "expected"),
    [
        ("list_dir", "workdir", "none", "allow"),
        ("read_file", "workdir", "none", "allow"),
        ("list_dir", "system_path", "none", "confirm"),
        ("list_dir", "workdir", "elevated", "confirm"),
        ("restart_service", "system_path", "none", "confirm"),
        ("remove_file", "workdir", "none", "confirm"),
        ("remove_file", "workdir", "elevated", "deny"),
        ("remove_file", "system_path", "none", "deny"),
        ("remove_tree", "production", "elevated", "deny"),
        ("drop_table", "production", "none", "deny"),
    ],
)
def test_verdict_rules(goal: str, target: str, elevation: str, expected: str) -> None:
    """输入：一组特征与期望档位。输出：无。预期行为：处置规则与文档一致。"""
    case = TagSet(dialect="bash", goal=goal, target=target, elevation=elevation)
    assert features.compatible(case)
    assert features.verdict_for(case) == expected


@pytest.mark.parametrize(
    ("goal", "target", "elevation", "expected"),
    [
        ("list_dir", "workdir", "none", 0),
        ("list_dir", "production", "none", 1),
        ("copy_file", "workdir", "none", 1),
        ("remove_file", "workdir", "none", 2),
        ("remove_file", "system_path", "none", 3),
        ("remove_file", "production", "elevated", 4),
        ("drop_table", "production", "elevated", 4),
    ],
)
def test_blast_radius_rules(goal: str, target: str, elevation: str, expected: int) -> None:
    """输入：一组特征与期望半径。输出：无。预期行为：0~4 且封顶。"""
    case = TagSet(dialect="bash", goal=goal, target=target, elevation=elevation)
    assert features.compatible(case)
    assert features.blast_radius_for(case) == expected


def test_label_flip_axes_skips_forbidden() -> None:
    """输入：无。输出：无。

    预期行为：`drop_table` 的邻居里只留生产那一个位置；`git_reset` 的邻居里没有生产与容器。
    """
    case = TagSet(dialect="bash", goal="drop_table", target="production", elevation="none")
    targets = {n.target for n in features.label_flip_axes(case) if n.target != "production"}
    assert targets == set()

    case = TagSet(dialect="bash", goal="git_reset", target="workdir", elevation="none")
    neighbours = {n.target for n in features.label_flip_axes(case)}
    assert "production" not in neighbours
    assert "container" not in neighbours
    assert "remote" in neighbours


def test_build_cases_is_deterministic() -> None:
    """输入：无。输出：无。预期行为：同一入参两次调用给出同一份列表。"""
    first = features.build_cases(32)
    second = features.build_cases(32)
    assert [c.as_dict() for c in first] == [c.as_dict() for c in second]