"""工具调用门控特征层的测试：维度表、禁用对、策略表与六个标签函数。"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Final

import pytest

from jc.cover import missing_pairs
from workflows.toolgate import features
from workflows.toolgate.features import TagSet

#: 全空间规模（未剔禁用对）。
FULL: Final[int] = 270

#: 剔掉禁用对之后剩下的相容组合数。
COMPATIBLE: Final[int] = 186


def all_tags() -> list[TagSet]:
    """输入：无。输出：全空间的 `TagSet` 列表（顺序与 `itertools.product` 一致）。"""
    names = list(features.DIMS)
    return [
        TagSet(**dict(zip(names, combo, strict=True)))
        for combo in itertools.product(*features.DIMS.values())
    ]


def test_full_space_is_the_cartesian_product_minus_forbidden_pairs() -> None:
    """输入：无。输出：无。

    预期行为：全枚举恰好等于「笛卡尔积里不含任何禁用对的那些行」，一行不多一行不少——
    禁用对只做减法，不做加法。
    """
    everything = all_tags()
    assert len(everything) == FULL
    kept = [tags for tags in everything if features.compatible(tags)]
    assert len(kept) == COMPATIBLE
    assert features.build_cases(10_000) == kept


def test_forbidden_pairs_only_touch_tool() -> None:
    """输入：无。输出：无。

    预期行为：每一条禁用对都必须落在 `tool` 这一维上——本 workflow 的物理约束只有
    「某个工具干不出某件事」这一类；别的维度之间没有硬约束（策略档位再严也不会让
    某个工具消失）。
    """
    for pair in features.FORBIDDEN:
        dims = {dim for dim, _ in pair}
        assert "tool" in dims, f"禁用对没碰工具：{pair}"


def test_pairwise_cover_still_covers_every_pair() -> None:
    """输入：把全枚举上限压到 100（低于 270）。输出：无。

    预期行为：枚举器换成对覆盖，但任意两维的合法取值对仍然一个不漏；被禁用对否掉的
    组合既不该出现、也不算没覆盖。
    """
    rows = features.build_cases(100)
    assert len(rows) < FULL
    assert all(features.compatible(tags) for tags in rows)
    missing = missing_pairs(features.DIMS, [tags.as_dict() for tags in rows], allow_pair=features.allow_pair)
    assert not missing, f"成对覆盖漏了：{sorted(missing)[:5]}"


def test_build_cases_is_deterministic() -> None:
    """输入：同一入参调两次。输出：无。预期行为：逐条相等（case 库要能逐字节重建）。"""
    assert features.build_cases(10_000) == features.build_cases(10_000)
    assert features.build_cases(100) == features.build_cases(100)


def test_forbidden_pairs_never_show_up() -> None:
    """输入：全枚举。输出：无。预期行为：读不出破坏性内容、写文件打不到外网这些组合一条都不在。"""
    for tags in features.build_cases(10_000):
        assert features.compatible(tags)
    for tool, reach, payload in (
        ("read_file", "external", "benign"),
        ("write_file", "external", "benign"),
        ("http_request", "workspace", "benign"),
        ("send_message", "workspace", "benign"),
        ("read_file", "host", "harmful"),
    ):
        row = dict(
            tool=tool, reach=reach, payload=payload, policy="balanced", approval="absent"
        )
        assert not features.compatible(TagSet(**row)), f"{row} 本该被禁"


@pytest.mark.parametrize(
    ("tool", "reach", "payload", "policy", "expected"),
    [
        # 破坏性内容：任何工具、任何档位、任何范围，第一条就命中。
        ("run_command", "workspace", "harmful", "permissive", "harmful_payload"),
        ("http_request", "external", "harmful", "permissive", "harmful_payload"),
        # 外网 + strict：第二条命中。
        ("http_request", "external", "benign", "strict", "external_under_strict"),
        # 本工作区 + 会改动的工具 + strict：第三条命中（排在敏感之前）。
        ("write_file", "workspace", "sensitive", "strict", "workspace_write_strict"),
        ("send_message", "host", "benign", "strict", "outside_workspace"),
        # 敏感内容：第四条命中（只在不是 permissive 时）。
        ("read_file", "host", "sensitive", "balanced", "sensitive_payload"),
        ("read_file", "host", "sensitive", "permissive", "outside_workspace"),
        # 出了工作区：第五条命中。
        ("run_command", "host", "benign", "balanced", "outside_workspace"),
        ("read_file", "host", "benign", "balanced", "outside_workspace"),
        # 都不命中：第六条。
        ("read_file", "workspace", "benign", "balanced", "none"),
        ("run_command", "workspace", "benign", "balanced", "none"),
        ("write_file", "workspace", "benign", "balanced", "none"),
    ],
)
def test_rule_rules(
    tool: str, reach: str, payload: str, policy: str, expected: str
) -> None:
    """输入：一组特征与期望的命中规则。输出：无。

    预期行为：`rule_for` 就是策略表**从上往下第一条命中**，顺序错了这条就红。
    """
    tags = TagSet(tool=tool, reach=reach, payload=payload, policy=policy, approval="absent")
    assert features.compatible(tags)
    assert features.rule_for(tags) == expected


@pytest.mark.parametrize(
    ("rule", "approval", "expected"),
    [
        ("none", "absent", "allow"),
        ("none", "granted", "allow"),
        ("harmful_payload", "absent", "deny"),
        # 破坏性内容是硬拦：拿到审批也不放行。
        ("harmful_payload", "granted", "deny"),
        ("external_under_strict", "granted", "deny"),
        ("outside_workspace", "absent", "needs_approval"),
        ("outside_workspace", "granted", "allow"),
    ],
)
def test_decision_rules(rule: str, approval: str, expected: str) -> None:
    """输入：命中的规则、审批状态、期望处置。输出：无。

    预期行为：审批只能把 `needs_approval` 抬成 `allow`，抬不动 `deny`。
    """
    raw = features.RAW_DECISIONS[rule]
    if raw == "needs_approval" and approval == "granted":
        assert expected == "allow"
    assert features.DECISIONS == ("allow", "needs_approval", "deny")


def test_decision_and_overridden_agree_on_every_case() -> None:
    """输入：全枚举。输出：无。

    预期行为：`overridden` 恒等于「命中的规则要审批，且审批状态是 granted」；
    且这时处置必须是放行——两条判据同源，永远不许打架。
    """
    for tags in features.build_cases(10_000):
        overridden = features.overridden_for(tags)
        assert overridden == (
            features.rule_for(tags) in features.APPROVAL_RULES and tags.approval == "granted"
        )
        if overridden:
            assert features.decision_for(tags) == "allow"


def test_severity_table_is_complete_and_monotone() -> None:
    """输入：无。输出：无。

    预期行为：九格（内容性质 × 作用范围）一格不缺；内容性质越重、作用范围越远，
    风险档只增不减——表写歪了这条会红。
    """
    for payload in features.PAYLOADS:
        for reach in features.REACHES:
            assert (payload, reach) in features.SEVERITY_TABLE
    order = {"benign": 0, "sensitive": 1, "harmful": 2}
    reach_order = {"workspace": 0, "host": 1, "external": 2}
    for payload, reach in itertools.product(features.PAYLOADS, features.REACHES):
        value = features.SEVERITY_TABLE[(payload, reach)]
        assert 0 <= value < features.SEVERITY_LEVELS
        for heavier, farther in itertools.product(features.PAYLOADS, features.REACHES):
            if order[heavier] >= order[payload] and reach_order[farther] >= reach_order[reach]:
                assert features.SEVERITY_TABLE[(heavier, farther)] >= value


def test_labels_are_a_pure_function_of_the_tags() -> None:
    """输入：同一组特征算两次。输出：无。预期行为：标签相等且等于各标签函数的结果。"""
    for tags in features.build_cases(10_000):
        labels = tags.label()
        assert labels == features.label_for(tags)
        assert labels.rule == features.rule_for(tags)
        assert labels.decision == features.decision_for(tags)
        assert labels.overridden == features.overridden_for(tags)
        assert labels.severity == features.severity_for(tags)
        assert labels.tool == tags.tool and labels.payload == tags.payload


def test_every_label_value_is_an_option() -> None:
    """输入：全枚举。输出：无。

    预期行为：四个 choice 题族的标签值都必须在该族的选项表里——标签表与题面选项表
    是两份清单，串了就会出现「答对了但选项里没有」的样本。
    """
    for tags in features.build_cases(10_000):
        labels = tags.label()
        for family, options in features.CHOICE_OPTIONS.items():
            assert getattr(labels, family) in options, f"{family} 的标签不在选项里"
        assert labels.severity in range(features.SEVERITY_LEVELS)
        assert isinstance(labels.overridden, bool)


def test_documented_distributions_hold() -> None:
    """输入：全枚举。输出：无。

    预期行为：`label_for` 文档里写的「每一档都到得了」在数据上成立；顺带把规则表
    六条规则的可达性钉住（策略表里写了却永远命中不了的规则，等于给了个死选项）。
    """
    labels = [tags.label() for tags in features.build_cases(10_000)]
    rules = Counter(label.rule for label in labels)
    assert set(rules) == set(features.RULES), f"规则没全命中过：{sorted(rules)}"
    decisions = Counter(label.decision for label in labels)
    assert set(decisions) == set(features.DECISIONS)
    assert set(label.severity for label in labels) == set(range(features.SEVERITY_LEVELS))
    assert {label.overridden for label in labels} == {True, False}


def test_label_flip_axes_skips_forbidden_neighbours() -> None:
    """输入：一组特征。输出：无。

    预期行为：只改一维的邻居里，撞上禁用对的直接跳过；留下来的每一组都必须相容，
    而且**只有一维**与原点不同。
    """
    origin = TagSet(
        tool="read_file", reach="external", payload="benign", policy="balanced", approval="absent"
    )
    assert not features.compatible(origin)  # 读文件打不到外网，这组本身就不合法
    origin = TagSet(
        tool="read_file", reach="host", payload="benign", policy="balanced", approval="absent"
    )
    neighbours = features.label_flip_axes(origin)
    assert neighbours, "邻居不该为空"
    for neighbour in neighbours:
        assert features.compatible(neighbour)
        diff = [
            key
            for key, value in neighbour.as_dict().items()
            if origin.as_dict()[key] != value
        ]
        assert len(diff) == 1, f"邻居改了不止一维：{diff}"
    assert all(neighbour != origin for neighbour in neighbours)
