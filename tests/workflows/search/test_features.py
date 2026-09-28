"""检索结果判读的特征层测试：维度表、规则表、可信度表与切分不变量。"""

from __future__ import annotations

from collections import Counter

from workflows.search import features
from workflows.search.features import TagSet

from tests.workflows.search.conftest import TEXTS


def all_tags() -> list[TagSet]:
    """输入：无。输出：全空间里所有相容的特征组合。"""
    return features.build_cases(1024)


def test_every_dimension_value_has_text() -> None:
    """输入：夹具文本表。输出：无。预期行为：每一格（查询意图 × 切题 × 垃圾）都有文本。"""
    for query_kind in features.QUERY_KINDS:
        for topicality in features.TOPICALITIES:
            for spam in features.SPAMS:
                assert (query_kind, topicality, spam) in TEXTS


def test_rules_cover_every_reachable_rule() -> None:
    """输入：全空间组合。输出：无。预期行为：七条规则每一条都至少命中一次。"""
    hit = Counter(features.rule_for(tags) for tags in all_tags())
    for rule in features.RULES:
        assert hit[rule] > 0, f"规则 {rule} 一次都没命中"


def test_trust_reaches_every_level() -> None:
    """输入：全空间组合。输出：无。预期行为：0~3 四档都到得了。"""
    levels = {features.trust_for(tags) for tags in all_tags()}
    assert levels == set(range(features.TRUST_LEVELS))


def test_trust_zero_for_spam_or_off_topic() -> None:
    """输入：全空间组合。输出：无。预期行为：垃圾或跑题一律 0 档。"""
    for tags in all_tags():
        if tags.spam == "spam" or tags.topicality == "off_topic":
            assert features.trust_for(tags) == 0


def test_verdict_matches_raw_table() -> None:
    """输入：全空间组合。输出：无。预期行为：处置恒等于规则表里那一格（没有抬升层）。"""
    for tags in all_tags():
        assert features.verdict_for(tags) == features.RAW_VERDICTS[features.rule_for(tags)]


def test_official_source_is_never_spam() -> None:
    """输入：全空间组合。输出：无。预期行为：禁用对 `官方 × 垃圾` 一组都不出现。"""
    for tags in all_tags():
        assert not (tags.authority == "official" and tags.spam == "spam")


def test_off_topic_is_never_spam() -> None:
    """输入：全空间组合。输出：无。

    预期行为：禁用对 `跑题 × 垃圾` 一组都不出现——垃圾摘要几乎总是「就在讲这个话题」，
    两维同时注入时读的人分不清该答哪一档（真机第一批就卡在这格）。
    """
    for tags in all_tags():
        assert not (tags.topicality == "off_topic" and tags.spam == "spam")


def test_allow_pair_is_order_free() -> None:
    """输入：一个禁用对。输出：无。预期行为：换顺序也一样被否掉。"""
    assert not features.allow_pair("authority", "official", "spam", "spam")
    assert not features.allow_pair("spam", "spam", "authority", "official")
    assert not features.allow_pair("topicality", "off_topic", "spam", "spam")
    assert not features.allow_pair("spam", "spam", "topicality", "off_topic")
    assert features.allow_pair("authority", "community", "spam", "spam")
    assert features.allow_pair("topicality", "on_topic", "spam", "spam")


def test_compatible_matches_forbidden_table() -> None:
    """输入：全空间组合。输出：无。预期行为：产出的组合全都是相容的。"""
    for tags in all_tags():
        assert features.compatible(tags)


def test_choice_options_are_a_subset_of_the_dimensions() -> None:
    """输入：choice 选项表。输出：无。预期行为：每题族的选项都是该维取值本身。"""
    assert features.CHOICE_OPTIONS["query_kind"] == features.QUERY_KINDS
    assert features.CHOICE_OPTIONS["topicality"] == features.TOPICALITIES
    assert features.CHOICE_OPTIONS["rule"] == features.RULES
    assert features.CHOICE_OPTIONS["verdict"] == features.VERDICTS
    assert set(features.RAW_VERDICTS) == set(features.RULES)
    assert set(features.RAW_VERDICTS.values()) <= set(features.VERDICTS)


def test_tag_set_round_trips_through_dict() -> None:
    """输入：一个组合。输出：无。预期行为：`as_dict` 的键序与 `DIMS` 一致、可重建。"""
    tags = TagSet(
        query_kind="informational",
        topicality="on_topic",
        spam="clean",
        authority="community",
        freshness="fresh",
    )
    assert list(tags.as_dict()) == list(features.DIMS)
    assert TagSet(**tags.as_dict()) == tags
    assert tags.label().rule == "community_source"
    assert tags.label().trust == 2
