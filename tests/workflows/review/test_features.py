"""改动审查的特征层测试：五维表、审查规则表、严重度表、门控表与禁用对。"""

from __future__ import annotations

from collections import Counter

import pytest

from workflows.review import features, locales, render
from workflows.review.features import TagSet

from tests.workflows.review.conftest import body_for


def all_tags() -> list[TagSet]:
    """输入：无。输出：全空间里所有相容的特征组合（108 组，走全枚举）。"""
    return features.build_cases(2048)


def test_size_line_bounds_keep_the_floor_and_slack_the_ceiling() -> None:
    """输入：三个规模取值。输出：无。

    预期行为：下界照 `SIZE_LINES` 不放（正文短于下界才是真与清单打架），上界乘
    `SIZE_LINES_SLACK`——清单那一行是权威，正文比写作指引长几行不算坏数据。
    """
    assert features.size_line_bounds("small") == (1, 4 * features.SIZE_LINES_SLACK)
    assert features.size_line_bounds("large") == (6, 12 * features.SIZE_LINES_SLACK)
    for size, (low, high) in features.SIZE_LINES.items():
        assert features.size_line_bounds(size)[0] == low
        assert features.size_line_bounds(size)[1] >= high


def test_every_rule_is_reachable() -> None:
    """输入：全空间组合。输出：无。预期行为：四条规则每一条都至少命中一次。"""
    hit = Counter(features.rule_for(tags) for tags in all_tags())
    for rule in features.REVIEW_RULES:
        assert hit[rule] > 0, f"规则 {rule} 一次都没命中"


def test_severity_reaches_every_level() -> None:
    """输入：全空间组合。输出：无。预期行为：0~3 四档都到得了。"""
    assert {features.severity_for(tags) for tags in all_tags()} == set(
        range(features.SEVERITY_LEVELS)
    )


def test_gate_reaches_every_action() -> None:
    """输入：全空间组合。输出：无。预期行为：三种门控动作都到得了。"""
    assert {features.gate_for(tags) for tags in all_tags()} == set(features.GATES)


def test_first_match_wins_in_table_order() -> None:
    """输入：构造出的组合。输出：无。

    预期行为：判定顺序就是表里的顺序——凭据压过调试，调试压过规模，规模压过兜底。
    """
    def rule(secret: str, debug: str, file_kind: str, size: str) -> str:
        return features.rule_for(
            TagSet(secret=secret, debug=debug, file_kind=file_kind, change_kind="modified", size=size)
        )

    assert rule("yes", "yes", "source", "large") == "credential"
    assert rule("no", "yes", "source", "large") == "debug"
    assert rule("no", "no", "source", "large") == "size"
    assert rule("no", "no", "source", "small") == "none"
    # 凭据规则不适用文档：正文里真有凭据也轮不到它
    assert rule("yes", "no", "docs", "small") == "none"
    # 调试规则不适用测试文件
    assert rule("no", "yes", "test", "small") == "none"
    # 规模规则只适用源码
    assert rule("no", "no", "config", "large") == "none"


def test_severity_depends_on_both_rule_and_file_kind() -> None:
    """输入：构造出的组合。输出：无。

    预期行为：同一条规则落在不同文件类型上档位不同（测试夹具里的凭据只算「中」），
    所以 `severity` 不是 `rule` 的换皮；不适用的文件类型上压根走不到那一格。
    """
    def severity(secret: str, file_kind: str) -> int:
        return features.severity_for(
            TagSet(
                secret=secret, debug="no", file_kind=file_kind, change_kind="modified", size="small"
            )
        )

    assert severity("yes", "source") == 3
    assert severity("yes", "config") == 3
    assert severity("yes", "test") == 2
    assert severity("no", "source") == 0


def test_severity_table_covers_every_applicable_cell() -> None:
    """输入：严重度表。输出：无。

    预期行为：每条非兜底规则的每个适用文件类型都定义了档位；表里**不该**出现不适用的格子
    （那会让「缺格抛 KeyError」这条保护失效）。
    """
    expected = {
        (rule, kind)
        for rule in features.REVIEW_RULES
        if rule != "none"
        for kind in features.RULE_SCOPES[rule]
    }
    assert set(features.SEVERITY_TABLE) == expected
    for rule, kind in expected:
        assert 1 <= features.SEVERITY_TABLE[(rule, kind)] < features.SEVERITY_LEVELS


def test_severity_table_raises_on_an_undefined_cell() -> None:
    """输入：一个没定义的格子。输出：无。

    预期行为：缺格抛 KeyError，绝不静默兜底——兜底会把「没定义」伪装成「有定义」。
    """
    with pytest.raises(KeyError):
        features.SEVERITY_TABLE[("credential", "docs")]


def test_gate_table_covers_every_level_and_change_kind() -> None:
    """输入：门控表。输出：无。预期行为：四档严重度 × 三种变更类型全都定义了。"""
    expected = {
        (level, kind)
        for level in range(features.SEVERITY_LEVELS)
        for kind in features.CHANGE_KINDS
    }
    assert set(features.GATE_TABLE) == expected
    assert set(features.GATE_TABLE.values()) == set(features.GATES)


def test_rule_hits_rejects_the_fallback_name() -> None:
    """输入：兜底规则名。输出：无。预期行为：`rule_hits` 只认三条真规则，兜底直接报错。"""
    tags = all_tags()[0]
    with pytest.raises(KeyError):
        features.rule_hits("none", tags)
    with pytest.raises(KeyError):
        features.rule_hits("nope", tags)


def test_the_forbidden_pairs_never_appear() -> None:
    """输入：全空间组合。输出：无。

    预期行为：两格都是「判据在这里没有唯一答案」（删掉凭据 / 调试输出是好事），
    所以干脆不生成。
    """
    for tags in all_tags():
        if tags.change_kind == "deleted":
            assert tags.secret == "no"
            assert tags.debug == "no"
        assert features.compatible(tags)


def test_compatible_rejects_the_forbidden_pairs_directly() -> None:
    """输入：两对被禁的取值。输出：无。预期行为：`compatible` 当场为假。"""
    for secret, debug in (("yes", "no"), ("no", "yes")):
        tags = TagSet(
            secret=secret, debug=debug, file_kind="source", change_kind="deleted", size="small"
        )
        assert not features.compatible(tags)


def test_labels_match_the_pure_functions() -> None:
    """输入：全空间组合。输出：无。预期行为：`label_for` 只是几个纯函数的打包。"""
    for tags in all_tags():
        labels = tags.label()
        assert labels.secret == tags.secret
        assert labels.debug == tags.debug
        assert labels.rule == features.rule_for(tags)
        assert labels.severity == features.severity_for(tags)
        assert labels.gate == features.gate_for(tags)


def test_gate_follows_the_severity_then_change_kind_table() -> None:
    """输入：构造出的组合。输出：无。

    预期行为：重的三种变更类型都拦下；中的只在删除时拦下；轻的只在修改时请求修改。
    """
    def gate(secret: str, file_kind: str, change_kind: str, size: str) -> str:
        return features.gate_for(
            TagSet(
                secret=secret,
                debug="no",
                file_kind=file_kind,
                change_kind=change_kind,
                size=size,
            )
        )

    assert gate("yes", "source", "added", "small") == "block"
    assert gate("no", "config", "deleted", "large") == "allow"  # 规模规则不适用配置 → 不提
    assert gate("no", "source", "added", "large") == "allow"  # 轻 + 新增
    assert gate("no", "source", "modified", "large") == "request_changes"  # 轻 + 修改
    assert gate("no", "source", "deleted", "large") == "allow"  # 轻 + 删除
    assert gate("no", "source", "added", "small") == "allow"  # 不提 → 一律放行


def test_choice_options_are_a_subset_of_the_dimensions() -> None:
    """输入：choice 选项表。输出：无。预期行为：每题族的选项就是该维取值本身。"""
    assert features.CHOICE_OPTIONS["rule"] == features.REVIEW_RULES
    assert features.CHOICE_OPTIONS["gate"] == features.GATES


def test_tag_set_round_trips_through_dict() -> None:
    """输入：一个组合。输出：无。预期行为：`as_dict` 的键序与 `DIMS` 一致、可重建。"""
    tags = TagSet(
        secret="yes", debug="no", file_kind="config", change_kind="modified", size="large"
    )
    assert list(tags.as_dict()) == list(features.DIMS)
    assert TagSet(**tags.as_dict()) == tags
    assert tags.label().rule == "credential"
    assert tags.label().severity == 3
    assert tags.label().gate == "block"
