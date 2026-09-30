"""特征层测试：四条判据的字面口径、权重定级、命中组合与判定计划。"""

from __future__ import annotations

import random

import pytest

from tests.workflows.depcheck.conftest import BASE_LINES, HIT_LINES, packet_of, render_lines
from workflows.depcheck import features
from workflows.depcheck.features import DepCheckCase


def test_each_hit_line_trips_only_its_own_tag() -> None:
    """输入：四条「取到形状」各单独成屏。输出：无。

    预期行为：每条判据只认自己那一行的形状——四条各占一行、互不遮蔽（范式 §9.51）。
    """
    for tag in features.TAG_IDS:
        text = render_lines({**BASE_LINES, features.LINE_KEYS[tag]: HIT_LINES[tag]})
        assert features.hit_set(text) == (tag,), tag


def test_the_plain_record_trips_nothing() -> None:
    """输入：六行骨架。输出：无。预期行为：一条判据都不取到（它是 1 级那一档要的东西）。"""
    assert features.hit_set(render_lines(BASE_LINES)) == ()


def test_lgpl_does_not_trip_the_copyleft_mark() -> None:
    """输入：`license:` 行写成 LGPL。输出：无。

    预期行为：不算命中——`\\b` 让 `GPL` 前面是词字符 `L`，词边界不成立（范式 §9.5x 的字面口径）。
    """
    text = render_lines({**BASE_LINES, "license": "license: LGPL-3.0-only"})
    assert "D1" not in features.hit_set(text)


def test_zero_advisories_does_not_trip() -> None:
    """输入：`count=0`。输出：无。预期行为：没有已知漏洞不算命中。"""
    text = render_lines({**BASE_LINES, "advisories": "advisories: count=0"})
    assert "D3" not in features.hit_set(text)


def test_one_dot_zero_version_does_not_trip() -> None:
    """输入：`version: 1.0.3`。输出：无。

    预期行为：不算「没到 1.0」——`0.` 只是版本号中间的一段，判据看的是冒号后紧跟的版本号。
    """
    text = render_lines({**BASE_LINES, "version": "version: 1.0.3"})
    assert "D4" not in features.hit_set(text)


def test_every_combination_is_reachable_by_its_packet() -> None:
    """输入：全空间 16 种命中组合。输出：无。

    预期行为：`packet_of` 造出来的记录按字面重算，得到的正好是那一组——两条路径（计划 / 文本）
    在这一层就已经对得上，后面 `check_draft` 与审计才有意义。
    """
    for value in features.HIT_SETS:
        tags = features.parse_hits(value)
        assert features.hits_of(packet_of(tags)) == value, value


def test_hits_of_and_parse_hits_round_trip() -> None:
    """输入：全空间 16 种命中组合。输出：无。预期行为：两种表示互相还原。"""
    for value in features.HIT_SETS:
        assert features.hits_of(packet_of(features.parse_hits(value))) == value
    assert features.parse_hits("") == ()


def test_missing_keys_finds_only_absent_ones() -> None:
    """输入：完整记录、少两行的记录。输出：无。预期行为：只报缺掉的那几个键。"""
    full = render_lines(BASE_LINES)
    assert features.missing_keys(full) == ()
    lines = {key: value for key, value in BASE_LINES.items() if key not in {"package", "status"}}
    assert set(features.missing_keys(render_lines(lines))) == {"package", "status"}


def test_missing_keys_needs_the_colon_on_its_own_line() -> None:
    """输入：把键名写在句子中间的记录。输出：无。

    预期行为：算缺——判据是钉在**行首**的，键不单独成行时那四条正则连定位都定不到。
    """
    text = "license: MIT\nstatus: 见 package: 那一行\nadvisories: count=0\nversion: 1.2.3\nnote: 一句"
    assert "package" in features.missing_keys(text)


def test_line_value_reads_the_first_matching_line() -> None:
    """输入：含 `note:` 的记录。输出：无。预期行为：取到冒号后面的内容，取不到给空串。"""
    text = render_lines(BASE_LINES)
    assert features.line_value(text, "note").startswith("内部工具链")
    assert features.line_value(text, "nope") == ""


def test_level_weights_and_level_of() -> None:
    """输入：几个命中组合。输出：无。

    预期行为：等级 = `1 + 权重和`、封顶 5 级；两个轻标记（`D2+D4`）与一个重标记（`D1`）同档。
    """
    assert features.LEVEL_WEIGHTS == {"D1": 2, "D2": 1, "D3": 2, "D4": 1}
    assert features.level_of(()) == 1
    assert features.level_of(("D2",)) == 2
    assert features.level_of(("D1",)) == 3
    assert features.level_of(("D2", "D4")) == 3
    assert features.level_of(("D1", "D2", "D3", "D4")) == features.MAX_LEVEL


def test_level_of_is_monotone_in_every_tag() -> None:
    """输入：全空间 16 组命中组合。输出：无。

    预期行为：多命中一个标记，等级**不会降**——加法形式保证的单调性，改权重时这一条会当场拦下。
    """
    for value in features.HIT_SETS:
        tags = features.parse_hits(value)
        level = features.level_of(tags)
        for tag in features.TAG_IDS:
            if tag in tags:
                continue
            raised = tuple(item for item in features.TAG_IDS if item in (*tags, tag))
            assert features.level_of(raised) >= level, (tags, tag)


def test_level_distribution_is_one_two_three_four_six() -> None:
    """输入：全空间 16 组。输出：无。

    预期行为：五档的分布是 1 / 2 / 3 / 4 / 6——最低那一档只有一个组合（四个标记全不命中）。这条
    钉住的是「小批量里五档都出得来」：某一档变成 0 格时，`grade` 那一族就永远取不到那个答案。
    """
    counts = dict.fromkeys(features.LEVELS, 0)
    for value in features.HIT_SETS:
        counts[features.level_of(features.parse_hits(value))] += 1
    assert counts == {1: 1, 2: 2, 3: 3, 4: 4, 5: 6}


def test_act_for_collapses_the_top_levels() -> None:
    """输入：五个等级。输出：无。预期行为：最后一档吃掉 4 / 5 两级。"""
    assert [features.act_for(level) for level in features.LEVELS] == [
        "approve",
        "watch",
        "review",
        "replace",
        "replace",
    ]


def test_combo_text_key_marks_every_tag() -> None:
    """输入：几个命中组合。输出：无。预期行为：四个标记都出现，按 `TAG_IDS` 顺序，用 hit / miss。"""
    assert features.combo_text_key(()) == "D1=miss D2=miss D3=miss D4=miss"
    assert features.combo_text_key(("D1", "D3")) == "D1=hit D2=miss D3=hit D4=miss"
    assert features.combo_text_key(("D1", "D2", "D3", "D4")) == "D1=hit D2=hit D3=hit D4=hit"


def test_case_helpers_are_stable() -> None:
    """输入：一个 case。输出：无。预期行为：字典键序、指纹与可读描述都固定。"""
    case = DepCheckCase("D1+D3", features.ECO_NPM)
    assert case.as_dict() == {"flags": "D1+D3", "ecosystem": "npm"}
    assert case.fingerprint() == "flags=D1+D3|ecosystem=npm"
    assert case.describe() == "D1+D3-npm"
    assert case.tags() == ("D1", "D3")
    assert DepCheckCase("", features.ECO_PYPI).describe() == "无风险标记-pypi"


def test_plan_is_reproducible_and_distractors_are_near() -> None:
    """输入：全空间 case × 两种问法。输出：无。

    预期行为：同一 (case, 问法) 的计划永远相同；三个干扰组合彼此不同、都不等于真值、都按规范顺序
    排好，而且**都与真值只差一两处**（差得越少越难分辨，题目才有区分度）。
    """
    for case in features.build_cases(512):
        for phrasing in (0, 1):
            first = features.plan_for(case, phrasing)
            second = features.plan_for(case, phrasing)
            assert first == second
            assert first.hits == case.tags()
            combos = first.distractors
            assert len(combos) == 3
            assert len(set(combos)) == 3
            for combo in combos:
                assert set(combo) != set(first.hits)
                assert combo == tuple(tag for tag in features.TAG_IDS if tag in combo)
                assert len(set(combo) ^ set(first.hits)) <= 3


def test_label_for_and_observed_labels_agree_on_faithful_packets() -> None:
    """输入：全空间 16 组「完全兑现」的记录。输出：无。

    预期行为：构造侧（计划）与审计侧（文本）两条路径给出同一份标签——对不上就说明渲染没兑现计划。
    """
    for value in features.HIT_SETS:
        tags = features.parse_hits(value)
        case = DepCheckCase(value, features.ECO_NPM)
        plan = features.plan_for(case, 0)
        assert features.label_for(plan) == features.observed_labels(packet_of(tags))


def test_labels_expose_level_and_act() -> None:
    """输入：几个命中组合。输出：无。预期行为：等级与处置档位都由权重公式推出来。"""
    assert features.DepCheckLabels(()).level == 1
    assert features.DepCheckLabels(()).act == "approve"
    assert features.DepCheckLabels(("D1", "D2", "D3", "D4")).level == 5
    assert features.DepCheckLabels(("D1", "D2", "D3", "D4")).act == "replace"
    assert features.DepCheckLabels(("D2",)).contains("D2")
    assert not features.DepCheckLabels(("D2",)).contains("D1")


def test_build_cases_enumerates_the_whole_space() -> None:
    """输入：全枚举上限、可选的维度覆盖。输出：无。

    预期行为：全空间 16 × 3 = 48 组，顺序稳定；覆盖掉某个取值之后规模随之变小。
    """
    rows = features.build_cases(512)
    assert len(rows) == len(features.HIT_SETS) * len(features.ECOSYSTEMS)
    assert rows == features.build_cases(512)
    restricted = features.build_cases(
        512, dims={"flags": features.HIT_SETS, "ecosystem": ("npm",)}
    )
    assert len(restricted) == len(features.HIT_SETS)
    assert {case.ecosystem for case in restricted} == {"npm"}


def test_patterns_only_look_at_their_own_line() -> None:
    """输入：把**取值**（不带它那个键名）写到别的行上。输出：无。

    预期行为：不算取到——判据认的是「取值 + 落在它说的那一行」，取值挪到别处就失去定位。
    只挪取值不挪键名，是因为正则的口径本来就是「**某一行里**同时有键名与形状」：把整行连键名
    一起搬到别处，那一行仍然是「`license:` 行」，按口径本来就该算命中。
    """
    values = {"D1": "AGPL-3.0-only", "D2": "archived", "D3": "count=3", "D4": "0.9.1"}
    for tag in features.TAG_IDS:
        key = features.LINE_KEYS[tag]
        other = next(name for name in features.KEY_LINES if name != key)
        lines = {**BASE_LINES, other: f"{other}: {values[tag]} -- 写在别的行上了"}
        assert features.hit_set(render_lines(lines)) == (), tag


def test_license_pools_line_up_with_the_d1_pattern() -> None:
    """输入：无。输出：无。

    预期行为：传染性那一半的每个标识符都真的命中 `D1` 的判据、宽松那一半一个都不命中。旋钮的池子
    与判据是两处独立写的，池子改错（把 `LGPL-3.0` 放进宽松侧、或把 `Apache-2.0` 放进传染性侧）
    会让整批样本的标签与文本对不上，所以拿 `PATTERNS["D1"]` 逐条对一遍。
    """
    pattern = features.PATTERNS["D1"]
    for item in features.LICENSE_COPYLEFT:
        assert pattern.search(f"license: {item}"), item
    for item in features.LICENSE_PERMISSIVE:
        assert not pattern.search(f"license: {item}"), item


def test_license_pool_follows_the_d1_hit() -> None:
    """输入：无。输出：无。预期行为：命中 `D1` 给传染性那一半，否则给宽松那一半。"""
    assert features.license_pool(("D1",)) is features.LICENSE_COPYLEFT
    assert features.license_pool(("D1", "D3")) is features.LICENSE_COPYLEFT
    assert features.license_pool(()) is features.LICENSE_PERMISSIVE
    assert features.license_pool(("D2", "D4")) is features.LICENSE_PERMISSIVE


def test_pick_license_rejects_a_one_sided_pool() -> None:
    """输入：只含一侧的池子。输出：无。预期行为：当场报人话，而不是静默产出标签对不上的样本。"""
    with pytest.raises(ValueError, match="相容"):
        features.pick_license(random.Random(0), ("MIT",), ("D1",))
    with pytest.raises(ValueError, match="相容"):
        features.pick_license(random.Random(0), ("GPL-3.0-only",), ("D2",))
    picked = features.pick_license(random.Random(0), features.LICENSE_PERMISSIVE, ("D2",))
    assert picked in features.LICENSE_PERMISSIVE
