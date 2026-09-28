"""会话上下文压缩的特征层测试：五维表、处置表、优先级表与禁用对。"""

from __future__ import annotations

from collections import Counter

import pytest

from workflows.compact import features
from workflows.compact.features import TagSet

from tests.workflows.compact.conftest import body_for


def all_tags() -> list[TagSet]:
    """输入：无。输出：全空间里所有相容的特征组合（162 组，走全枚举）。"""
    return features.build_cases(1024)


def test_full_space_drops_exactly_the_two_forbidden_pairs() -> None:
    """输入：维度表与全空间枚举。输出：无。

    预期行为：5×2×3×2×3 = 180 组；两对禁用对各砍掉 3×2×3 = 18 组
    （`工具输出 × 持久事实`、`系统提示词 × 没有持久事实`），正好剩 144 组。
    """
    total = 1
    for values in features.DIMS.values():
        total *= len(values)
    assert total == 180
    assert len(all_tags()) == 144


def test_every_rule_is_reachable() -> None:
    """输入：全空间组合。输出：无。预期行为：八条规则每一条都至少命中一次。"""
    hit = Counter(features.rule_for(tags) for tags in all_tags())
    for rule in features.RULES:
        assert hit[rule] > 0, f"规则 {rule} 一次都没命中"


def test_every_disposition_is_reachable() -> None:
    """输入：全空间组合。输出：无。预期行为：四档处置都到得了。"""
    assert {features.disposition_for(tags) for tags in all_tags()} == set(
        features.DISPOSITIONS
    )


def test_priority_reaches_every_level() -> None:
    """输入：全空间组合。输出：无。预期行为：0~3 四档都到得了。"""
    assert {features.priority_for(tags) for tags in all_tags()} == set(
        range(features.PRIORITY_LEVELS)
    )


def test_disposition_matches_the_raw_table() -> None:
    """输入：全空间组合。输出：无。预期行为：处置恒等于命中规则那一格。"""
    for tags in all_tags():
        assert features.disposition_for(tags) == features.RAW_DISPOSITIONS[
            features.rule_for(tags)
        ]


def test_rule_table_order_and_raw_table_agree() -> None:
    """输入：两张表。输出：无。预期行为：规则名集合一致，处置词元都在取值表里。"""
    assert set(features.RAW_DISPOSITIONS) == set(features.RULES)
    assert set(features.RAW_DISPOSITIONS.values()) <= set(features.DISPOSITIONS)
    assert len(features.RULES) == len(set(features.RULES))


def test_first_match_wins_in_table_order() -> None:
    """输入：构造出的组合。输出：无。

    预期行为：判定顺序就是表里的顺序——钉住的一律 `pinned_keep`，
    哪怕它同时也是系统提示词、也承载事实、也又旧又缺预算。
    """
    tags = TagSet(kind="system", fact="fact", age="old", pins="pinned", budget="critical")
    assert features.rule_for(tags) == "pinned_keep"
    assert features.rule_for(TagSet(**{**tags.as_dict(), "pins": "unpinned"})) == "system_keep"
    assert (
        features.rule_for(
            TagSet(kind="tool_output", fact="chatter", age="old", pins="unpinned", budget="critical")
        )
        == "budget_tool"
    )
    assert (
        features.rule_for(
            TagSet(kind="note", fact="chatter", age="old", pins="unpinned", budget="critical")
        )
        == "critical_old"
    )
    assert (
        features.rule_for(
            TagSet(kind="note", fact="chatter", age="old", pins="unpinned", budget="tight")
        )
        == "tight_old"
    )
    assert (
        features.rule_for(
            TagSet(kind="note", fact="chatter", age="old", pins="unpinned", budget="ample")
        )
        == "old_truncate"
    )
    assert (
        features.rule_for(
            TagSet(kind="note", fact="chatter", age="current", pins="unpinned", budget="ample")
        )
        == "none"
    )


def test_the_forbidden_pair_never_appears() -> None:
    """输入：全空间组合。输出：无。

    预期行为：工具输出一律不承载持久事实；系统提示词一律承载持久事实——
    两格都是「判据在这里没有唯一答案」，所以干脆不生成。
    """
    for tags in all_tags():
        assert not (tags.kind == "tool_output" and tags.fact == "fact")
        assert not (tags.kind == "system" and tags.fact == "chatter")
        assert features.compatible(tags)


def test_compatible_rejects_the_forbidden_pair_directly() -> None:
    """输入：两对被禁的取值。输出：无。预期行为：`compatible` 当场为假。"""
    for kind, fact in (("tool_output", "fact"), ("system", "chatter")):
        tags = TagSet(kind=kind, fact=fact, age="current", pins="unpinned", budget="ample")
        assert not features.compatible(tags)


def test_allow_pair_is_order_free() -> None:
    """输入：一个禁用对（正反两种写法）。输出：无。预期行为：换顺序也一样被否掉。"""
    assert not features.allow_pair("fact", "fact", "kind", "tool_output")
    assert not features.allow_pair("kind", "tool_output", "fact", "fact")
    assert not features.allow_pair("fact", "chatter", "kind", "system")
    assert not features.allow_pair("kind", "system", "fact", "chatter")
    assert features.allow_pair("fact", "chatter", "kind", "tool_output")
    assert features.allow_pair("fact", "fact", "kind", "user")
    assert features.allow_pair("fact", "chatter", "kind", "user")


def test_pin_overrode_is_false_when_not_pinned() -> None:
    """输入：全空间组合。输出：无。预期行为：没钉住的一律为假。"""
    for tags in all_tags():
        if tags.pins != "pinned":
            assert features.pin_overrode_for(tags) is False


def test_pin_overrode_true_means_the_pin_changed_the_disposition() -> None:
    """输入：全空间组合。输出：无。

    预期行为：真值 ⟺ 钉住之后处置是 `keep`、而把 `pins` 换成 `unpinned` 就不是 `keep`。
    """
    from dataclasses import replace

    for tags in all_tags():
        if not features.pin_overrode_for(tags):
            continue
        assert tags.pins == "pinned"
        assert features.disposition_for(tags) == "keep"
        assert features.disposition_for(replace(tags, pins="unpinned")) != "keep"


def test_pinned_is_always_the_top_priority() -> None:
    """输入：全空间组合。输出：无。预期行为：钉住的一律 3 档。"""
    for tags in all_tags():
        if tags.pins == "pinned":
            assert features.priority_for(tags) == features.PRIORITY_LEVELS - 1
        else:
            assert 0 <= features.priority_for(tags) < features.PRIORITY_LEVELS


def test_priority_table_covers_every_kind_age_pair() -> None:
    """输入：底分表。输出：无。预期行为：五种「这一块是什么」× 三种位置全都定义了。"""
    expected = {(kind, age) for kind in features.KINDS for age in features.AGES}
    assert set(features.BASE_PRIORITY) == expected


def test_priority_raises_on_an_undefined_cell() -> None:
    """输入：一个没定义的「这一块是什么」。输出：无。

    预期行为：缺格抛 KeyError，绝不静默兜底——兜底会把「没定义」伪装成「有定义」。
    """
    tags = TagSet(kind="bogus", fact="chatter", age="current", pins="unpinned", budget="ample")
    with pytest.raises(KeyError):
        features.priority_for(tags)


def test_labels_match_the_pure_functions() -> None:
    """输入：全空间组合。输出：无。预期行为：`label_for` 只是几个纯函数的打包。"""
    for tags in all_tags():
        labels = tags.label()
        assert labels.kind == tags.kind
        assert labels.fact == tags.fact
        assert labels.rule == features.rule_for(tags)
        assert labels.disposition == features.disposition_for(tags)
        assert labels.pin_overrode == features.pin_overrode_for(tags)
        assert labels.priority == features.priority_for(tags)


def test_choice_options_are_a_subset_of_the_dimensions() -> None:
    """输入：choice 选项表。输出：无。预期行为：每题族的选项就是该维取值本身。"""
    assert features.CHOICE_OPTIONS["kind"] == features.KINDS
    assert features.CHOICE_OPTIONS["fact"] == features.FACTS
    assert features.CHOICE_OPTIONS["rule"] == features.RULES
    assert features.CHOICE_OPTIONS["disposition"] == features.DISPOSITIONS


def test_every_compatible_combination_has_fixture_text() -> None:
    """输入：全空间组合。输出：无。

    预期行为：假后端能给每一格写出相称的正文——夹具缺格会让「文本承载两维」这条前提
    在测试里悄悄失效（`body_for` 缺格直接抛 KeyError）。
    """
    for tags in all_tags():
        assert body_for(tags.as_dict()).strip()


def test_tag_set_round_trips_through_dict() -> None:
    """输入：一个组合。输出：无。预期行为：`as_dict` 的键序与 `DIMS` 一致、可重建。"""
    tags = TagSet(kind="note", fact="chatter", age="old", pins="unpinned", budget="tight")
    assert list(tags.as_dict()) == list(features.DIMS)
    assert TagSet(**tags.as_dict()) == tags
    assert tags.label().rule == "tight_old"
    assert tags.label().disposition == "summarize"
