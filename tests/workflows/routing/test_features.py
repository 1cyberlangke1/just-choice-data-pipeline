"""工单分诊的特征层测试：五维表、派单表、优先级表、附加规则与禁用对。"""

from __future__ import annotations

from collections import Counter

import pytest

from workflows.routing import features
from workflows.routing.features import TagSet



def all_tags() -> list[TagSet]:
    """输入：无。输出：全空间里所有相容的特征组合（432 组，走全枚举）。"""
    return features.build_cases(2048)


def test_full_space_drops_exactly_the_three_forbidden_pairs() -> None:
    """输入：维度表与全空间枚举。输出：无。

    预期行为：5×3×3×3×4 = 540 组；三对禁用对各砍掉 3×3×4 = 36 组
    （`故障 × 可以等`、`数据丢失 × 可以等`、`咨询 × 业务受阻`），正好剩 432 组。
    """
    total = 1
    for values in features.DIMS.values():
        total *= len(values)
    assert total == 540
    assert len(all_tags()) == 432


def test_every_rule_is_reachable() -> None:
    """输入：全空间组合。输出：无。预期行为：七条派单规则每一条都至少命中一次。"""
    hit = Counter(features.rule_for(tags) for tags in all_tags())
    for rule in features.QUEUE_RULES:
        assert hit[rule] > 0, f"规则 {rule} 一次都没命中"


def test_every_queue_is_reachable() -> None:
    """输入：全空间组合。输出：无。预期行为：六个队列都到得了。"""
    assert {features.queue_for(tags) for tags in all_tags()} == set(features.QUEUES)


def test_needs_human_reaches_both_values() -> None:
    """输入：全空间组合。输出：无。预期行为：真假都到得了。"""
    assert {features.needs_human_for(tags) for tags in all_tags()} == {True, False}


def test_queue_matches_the_raw_table() -> None:
    """输入：全空间组合。输出：无。预期行为：队列恒等于命中规则那一格。"""
    for tags in all_tags():
        assert features.queue_for(tags) == features.RAW_QUEUES[features.rule_for(tags)]


def test_queue_table_is_many_to_one() -> None:
    """输入：两张表。输出：无。

    预期行为：规则名集合与队列表一致；`rule → queue` 是**多对一**（故障与数据丢失都进
    事故响应组），否则 `rule` 与 `queue` 两族问的是同一件事。
    """
    assert set(features.RAW_QUEUES) == set(features.QUEUE_RULES)
    assert set(features.RAW_QUEUES.values()) == set(features.QUEUES)
    assert len(features.QUEUE_RULES) > len(features.QUEUES)
    assert len(features.QUEUE_RULES) == len(set(features.QUEUE_RULES))


def test_first_match_wins_in_table_order() -> None:
    """输入：构造出的组合。输出：无。

    预期行为：判定顺序就是表里的顺序——合规标记压过问题类型，问题类型压过兜底。
    """
    def rule(issue: str, compliance: str) -> str:
        return features.rule_for(
            TagSet(issue=issue, urgency="deadline", tier="pro", compliance=compliance, area="storage")
        )

    assert rule("outage", "security") == "security_report"
    assert rule("feature", "gdpr") == "privacy_request"
    assert rule("outage", "none") == "outage_response"
    assert rule("data_loss", "none") == "loss_response"
    assert rule("bug", "none") == "defect"
    assert rule("howto", "none") == "howto"
    assert rule("feature", "none") == "feature_request"
    # 安全报告只管故障 / 数据丢失 / 缺陷三类；需求那一条落不到它手里
    assert rule("feature", "security") == "feature_request"
    # 数据删除请求只管数据丢失与需求两类
    assert rule("bug", "gdpr") == "defect"


def test_the_forbidden_pairs_never_appear() -> None:
    """输入：全空间组合。输出：无。

    预期行为：三格都是「判据在这里没有唯一答案」，所以干脆不生成。
    """
    for tags in all_tags():
        assert not (tags.issue == "outage" and tags.urgency == "can_wait")
        assert not (tags.issue == "data_loss" and tags.urgency == "can_wait")
        assert not (tags.issue == "howto" and tags.urgency == "blocked")
        assert features.compatible(tags)


def test_compatible_rejects_the_forbidden_pairs_directly() -> None:
    """输入：三对被禁的取值。输出：无。预期行为：`compatible` 当场为假。"""
    for issue, urgency in (
        ("outage", "can_wait"),
        ("data_loss", "can_wait"),
        ("howto", "blocked"),
    ):
        tags = TagSet(
            issue=issue, urgency=urgency, tier="free", compliance="none", area="payments"
        )
        assert not features.compatible(tags)


def test_allow_pair_is_order_free() -> None:
    """输入：一个禁用对（正反两种写法）。输出：无。预期行为：换顺序也一样被否掉。"""
    assert not features.allow_pair("issue", "outage", "urgency", "can_wait")
    assert not features.allow_pair("urgency", "can_wait", "issue", "outage")
    assert features.allow_pair("issue", "outage", "urgency", "blocked")
    assert features.allow_pair("issue", "bug", "urgency", "can_wait")


def test_priority_table_covers_every_urgency_tier_pair() -> None:
    """输入：优先级表。输出：无。预期行为：三种紧急度 × 三种客户等级全都定义了。"""
    expected = {(urgency, tier) for urgency in features.URGENCIES for tier in features.TIERS}
    assert set(features.PRIORITY_TABLE) == expected


def test_priority_raises_on_an_undefined_cell() -> None:
    """输入：一个没定义的客户等级。输出：无。

    预期行为：缺格抛 KeyError，绝不静默兜底——兜底会把「没定义」伪装成「有定义」。
    """
    tags = TagSet(
        issue="bug", urgency="deadline", tier="bogus", compliance="none", area="payments"
    )
    with pytest.raises(KeyError):
        features.priority_for(tags)


def test_needs_human_needs_either_branch() -> None:
    """输入：构造出的组合。输出：无。

    预期行为：事故队列一律要人工；非事故队列只有「关键域 + 企业版」才要；
    非关键域的企业版与非关键域的其它等级都不要。
    """
    def needs(issue: str, area: str, tier: str) -> bool:
        return features.needs_human_for(
            TagSet(issue=issue, urgency="can_wait", tier=tier, compliance="none", area=area)
        )

    assert needs("bug", "storage", "free") is False  # 非事故队列、非关键域
    assert needs("bug", "payments", "enterprise") is True  # 关键域 + 企业版
    assert needs("bug", "payments", "pro") is False  # 关键域但不是企业版
    assert needs("bug", "storage", "enterprise") is False  # 企业版但非关键域
    assert needs("outage", "storage", "free") is True  # 事故队列，不看别的


def test_labels_match_the_pure_functions() -> None:
    """输入：全空间组合。输出：无。预期行为：`label_for` 只是几个纯函数的打包。"""
    for tags in all_tags():
        labels = tags.label()
        assert labels.issue == tags.issue
        assert labels.urgency == tags.urgency
        assert labels.rule == features.rule_for(tags)
        assert labels.queue == features.queue_for(tags)
        assert labels.priority == features.priority_for(tags)
        assert labels.needs_human == features.needs_human_for(tags)


def test_choice_options_are_a_subset_of_the_dimensions() -> None:
    """输入：choice 选项表。输出：无。预期行为：每题族的选项就是该维取值本身。"""
    assert features.CHOICE_OPTIONS["issue"] == features.ISSUES
    assert features.CHOICE_OPTIONS["urgency"] == features.URGENCIES
    assert features.CHOICE_OPTIONS["rule"] == features.QUEUE_RULES
    assert features.CHOICE_OPTIONS["queue"] == features.QUEUES


def test_tag_set_round_trips_through_dict() -> None:
    """输入：一个组合。输出：无。预期行为：`as_dict` 的键序与 `DIMS` 一致、可重建。"""
    tags = TagSet(
        issue="data_loss", urgency="deadline", tier="pro", compliance="gdpr", area="storage"
    )
    assert list(tags.as_dict()) == list(features.DIMS)
    assert TagSet(**tags.as_dict()) == tags
    assert tags.label().rule == "privacy_request"
    assert tags.label().queue == "privacy_ops"
    assert tags.label().priority == 3
