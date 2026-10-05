"""打分与评分的特征层测试：五维表、评分单、档位表、处置表与「最严重的一条」。"""

from __future__ import annotations

from collections import Counter

import pytest

from workflows.rubric import features, locales, render
from workflows.rubric.features import TagSet

from tests.workflows.rubric.conftest import body_for


def all_tags() -> list[TagSet]:
    """输入：无。输出：全空间里所有相容的特征组合（48 组，走全枚举）。"""
    return features.build_cases(2048)


def tags_of(
    placeholders: str = "no",
    citations: str = "no",
    length: str = "short",
    form: str = "prose",
    data: str = "none",
) -> TagSet:
    """输入：五维取值（都有默认）。输出：一组注入特征。预期行为：只用于构造测试输入。"""
    return TagSet(
        placeholders=placeholders,
        citations=citations,
        length=length,
        form=form,
        data=data,
    )


def test_full_space_is_the_whole_product() -> None:
    """输入：维度表与全空间枚举。输出：无。

    预期行为：2×2×3×2×2 = 48 组，一组不缺——`FORBIDDEN` 是空的（五维说的是材料的五件
    互不相干的事，任意组合都写得出来）。
    """
    total = 1
    for values in features.DIMS.values():
        total *= len(values)
    assert total == 48
    assert len(all_tags()) == 48
    assert features.FORBIDDEN == frozenset()
    for tags in all_tags():
        assert features.compatible(tags)


def test_allow_pair_says_yes_to_every_pair() -> None:
    """输入：任意两维的任意取值对（正反两种写法）。输出：无。预期行为：一个都不禁。"""
    names = list(features.DIMS)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            for left_value in features.DIMS[left]:
                for right_value in features.DIMS[right]:
                    assert features.allow_pair(left, left_value, right, right_value)
                    assert features.allow_pair(right, right_value, left, left_value)


def test_length_bounds_keep_the_floor_and_slack_the_ceiling() -> None:
    """输入：三个篇幅取值。输出：无。

    预期行为：下界照 `LENGTH_SENTENCES` 不放（材料短于下界才是真与清单打架），上界乘
    `LENGTH_SENTENCES_SLACK`——清单那一行是权威，材料比写作指引多写几句不算坏数据。
    """
    assert features.length_bounds("short") == (3, 6 * features.LENGTH_SENTENCES_SLACK)
    assert features.length_bounds("long") == (10, 18 * features.LENGTH_SENTENCES_SLACK)
    for length, (low, high) in features.LENGTH_SENTENCES.items():
        assert features.length_bounds(length)[0] == low
        assert features.length_bounds(length)[1] >= high


def test_count_sentences_counts_whole_paragraphs_and_bare_lines() -> None:
    """输入：几段文本。输出：无。

    预期行为：中文一整个自然段就是几句（按 。！？； 数）；英文句点断句，但 `3.5` 这种
    小数点不算；**结尾没有终止符的最后一段也算一句**——`TODO：补上压测数据` 就是一句，
    不能因为没写句号就当成零句；换行也是一句的边界。
    """
    assert features.count_sentences("一。二。三。") == 3
    assert features.count_sentences("一、二、三，写成一整段。") == 1
    assert features.count_sentences("One. Two. Three.") == 3
    assert features.count_sentences("Version 1.2 ships now. It is fast.") == 2
    assert features.count_sentences("alpha one\nbravo two\ncharlie three") == 3
    assert features.count_sentences("TODO: fill in\nOne. Two.") == 3
    assert features.count_sentences("TODO: fill in the load-test numbers") == 1
    assert features.count_sentences("   ") == 0


def test_every_band_is_reachable() -> None:
    """输入：全空间组合。输出：无。预期行为：低 / 中 / 高三档都到得了。"""
    assert {features.band_for(tags) for tags in all_tags()} == set(features.BANDS)


def test_every_gate_is_reachable() -> None:
    """输入：全空间组合。输出：无。预期行为：通过 / 返修 / 退回三种处置都到得了。"""
    assert {features.gate_for(tags) for tags in all_tags()} == set(features.GATES)


def test_every_criterion_is_both_met_and_unmet() -> None:
    """输入：全空间组合。输出：无。

    预期行为：五条判据每一条都既有达标的组合、也有不达标的组合——一条判据要是全空间里
    只有一个取值，那道题就退化成常量，等于白出。
    """
    for name in features.CRITERIA:
        seen = {features.met_of(name, tags) for tags in all_tags()}
        assert seen == {True, False}, name


def test_band_lines_cover_every_total() -> None:
    """输入：档位表。输出：无。

    预期行为：三档首尾相接盖满 0~11（五条判据权重之和），中间不留缝、不重叠——
    留缝会让 `band_of_total` 抛 KeyError，重叠会让「落在哪一档」没有唯一答案。
    """
    spans = [features.BAND_LINES[band] for band in features.BANDS]
    assert spans[0][0] == 0
    assert spans[-1][1] == sum(features.WEIGHT_OF.values())
    for (low, high), (next_low, _next_high) in zip(spans, spans[1:]):
        assert low <= high
        assert next_low == high + 1
    for total in range(sum(features.WEIGHT_OF.values()) + 1):
        assert features.band_of_total(total) in features.BANDS


def test_gate_table_covers_exactly_every_cell() -> None:
    """输入：处置表。输出：无。

    预期行为：六个格子一个不多一个不少（三档 × 红线两态）；多出来的格子会让「缺格抛
    KeyError」这条保护失效。
    """
    assert set(features.GATE_TABLE) == {
        (band, state) for band in features.BANDS for state in features.REDLINE_STATES
    }
    assert set(features.GATE_TABLE.values()) == set(features.GATES)


def test_redline_is_the_grounded_criterion() -> None:
    """输入：构造出的组合。输出：无。预期行为：红线态只看「有据」那条达标没有。"""
    assert features.REDLINE == "grounded"
    assert features.REDLINE_STATES == ("clean", "redline")
    assert features.redline_state(tags_of(citations="yes")) == "clean"
    assert features.redline_state(tags_of(citations="no")) == "redline"
    assert features.redline_hit(tags_of(citations="no"))
    assert not features.redline_hit(tags_of(citations="yes"))


def test_obs_table_stays_in_range_and_met_means_observed_at_least_need() -> None:
    """输入：观测表与要求表。输出：无。

    预期行为：观测都是 0~4 的整数；达标口径就是「观测 ≥ 要求」这一条，不掺别的。
    """
    for dim, table in features.OBS_TABLE.items():
        assert set(table) == set(features.DIMS[dim]), dim
        assert all(0 <= value <= 4 for value in table.values()), dim
    for tags in all_tags():
        for name in features.CRITERIA:
            assert features.met_of(name, tags) is (
                features.obs_for(name, tags) >= features.NEED_OF[name]
            )


def test_top_is_unique_or_absent() -> None:
    """输入：构造出的组合。输出：无。

    预期行为：没达标的判据里权重最大的那条**唯一**时才给答案；并列或全达标时给 None
    ——那时这道题没有唯一解，硬出一个就是错数据。
    """
    # 完整（权重 3）与有据（权重 3）同时不达标：并列 → 没有答案
    assert features.top_for(tags_of(placeholders="yes", citations="no")) is None
    # 只有「有据」一条不达标 → 就是它
    assert features.top_for(tags_of(citations="no")) == "grounded"
    # 五条全达标 → 也没有「最严重的一条」
    assert (
        features.top_for(
            tags_of(placeholders="no", citations="yes", length="short", form="bulleted", data="numbers")
        )
        is None
    )
    # 全空间里两种情形都出现过：既有唯一解，也有并列
    tops = Counter(features.top_for(tags) for tags in all_tags())
    assert tops[None] > 0
    for name in features.CRITERIA:
        assert tops[name] > 0, name


def test_total_is_the_weight_of_the_met_criteria() -> None:
    """输入：全空间组合。输出：无。预期行为：总分就是达标那几条的权重之和。"""
    for tags in all_tags():
        expected = sum(
            features.WEIGHT_OF[name]
            for name in features.CRITERIA
            if features.met_of(name, tags)
        )
        assert features.total_for(tags) == expected


def test_gate_follows_band_and_redline() -> None:
    """输入：构造出的组合。输出：无。

    预期行为：高档 / 中档且没踩红线才通过；低档没踩红线返修；踩了红线一律退回
    （中档踩红线是退回，不是返修）。
    """
    accept = tags_of(placeholders="no", citations="yes", length="short", form="bulleted", data="numbers")
    assert features.band_for(accept) == "high"
    assert features.gate_for(accept) == "accept"
    # 中档踩红线：完整（3）+ 简洁（2）+ 格式（2）= 7，有据那条没达标
    redline_mid = tags_of(placeholders="no", citations="no", length="short", form="bulleted", data="none")
    assert features.band_for(redline_mid) == "mid"
    assert features.gate_for(redline_mid) == "reject"
    redline_low = tags_of(placeholders="yes", citations="no", length="long", form="prose", data="none")
    assert features.band_for(redline_low) == "low"
    assert features.gate_for(redline_low) == "reject"


def test_body_marks_are_mechanical() -> None:
    """输入：几段文本。输出：无。

    预期行为：留空占位只认清单里那几种写法、出处标记只认方括号编号——口径机械可判，
    渲染期、判别器与审计三处共用这一份。
    """
    assert features.has_placeholder("TODO: 补上压测数据")
    assert features.has_placeholder("FIXME 待补")
    assert not features.has_placeholder("一切都写完了")
    assert features.has_citation("比上版快了一倍[1]")
    assert not features.has_citation("比上版快了一倍(1)")
    assert features.bullet_lines(["- 一条", "  - 缩进的也算", "行文"]) == ["- 一条", "  - 缩进的也算"]


def test_has_number_ignores_the_citation_digit() -> None:
    """输入：只有出处标记的数字的文本。输出：无。

    预期行为：`[1]` 里那个 1 是出处编号，不是量化数据——不抹掉的话「有出处 × 无数据」
    这一格永远写不出来，两个维度会互相打架。
    """
    assert not features.has_number("比上版快了一倍[1]")
    assert features.has_number("延迟从 120ms 降到 80ms[1]")
    assert not features.has_number("没有任何数字")


def test_labels_match_the_pure_functions() -> None:
    """输入：全空间组合。输出：无。预期行为：`label_for` 只是几个纯函数的打包。"""
    for tags in all_tags():
        labels = tags.label()
        assert labels.placeholders == tags.placeholders
        assert labels.citations == tags.citations
        assert labels.met == tuple(features.met_of(name, tags) for name in features.CRITERIA)
        assert labels.total == features.total_for(tags)
        assert labels.band == features.band_for(tags)
        assert labels.gate == features.gate_for(tags)
        assert labels.top == features.top_for(tags)
        for index, name in enumerate(features.CRITERIA, start=1):
            assert labels.met_at(index) is features.met_of(name, tags)


def test_score_from_obs_raises_on_a_missing_criterion() -> None:
    """输入：缺一条判据的观测。输出：无。预期行为：当场抛 KeyError，不猜一个总分。"""
    with pytest.raises(KeyError):
        features.score_from_obs({"complete": 4})


def test_tag_set_round_trips_through_dict() -> None:
    """输入：一个组合。输出：无。预期行为：`as_dict` 的键序与 `DIMS` 一致、可重建。"""
    tags = tags_of(placeholders="yes", citations="yes", length="long", form="bulleted", data="numbers")
    assert list(tags.as_dict()) == list(features.DIMS)
    assert TagSet(**tags.as_dict()) == tags
    assert tags.label().band == "mid"
    assert tags.label().gate == "accept"


def test_choice_options_are_a_subset_of_the_dimensions() -> None:
    """输入：choice 选项表。输出：无。预期行为：每题族的选项就是该族的取值表本身。"""
    assert features.CHOICE_OPTIONS["gate"] == features.GATES
    assert features.CHOICE_OPTIONS["top"] == features.CRITERIA


def test_cover_limit_forces_full_enumeration() -> None:
    """输入：默认 `cover_limit`。输出：无。

    预期行为：全空间只有 48 组，512 之下走的是**全枚举**——成对覆盖在这里只会白扔组合。
    """
    assert features.build_cases(512) == all_tags()
