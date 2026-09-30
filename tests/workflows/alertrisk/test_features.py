"""特征层测试：三条判据的字面口径、定级矩阵、命中组合与判定计划。"""

from __future__ import annotations

import itertools


from tests.workflows.alertrisk.conftest import BASE_LINES, HIT_LINES, packet_of, render_lines
from workflows.alertrisk import features
from workflows.alertrisk.features import AlertRiskCase


def test_each_hit_line_trips_only_its_own_factor() -> None:
    """输入：三条「取到形状」各单独成屏。输出：无。

    预期行为：每条判据只认自己那一行的形状——三条各占一行、互不遮蔽（范式 §9.51）。
    """
    for tag in features.TAG_IDS:
        text = render_lines({**BASE_LINES, features.LINE_KEYS[tag]: HIT_LINES[tag]})
        assert features.hit_set(text) == (tag,), tag


def test_the_plain_record_trips_nothing() -> None:
    """输入：五行骨架。输出：无。预期行为：一条判据都不取到（它是 1 级那一档要的东西）。"""
    assert features.hit_set(render_lines(BASE_LINES)) == ()


def test_every_combination_is_reachable_by_its_packet() -> None:
    """输入：全空间 8 种命中组合。输出：无。

    预期行为：`packet_of` 造出来的记录按字面重算，得到的正好是那一组——两条路径（计划 / 文本）
    在这一层就已经对得上，后面 `check_draft` 与审计才有意义。
    """
    for value in features.HIT_SETS:
        tags = features.parse_hits(value)
        assert features.hits_of(packet_of(tags)) == value, value


def test_hits_of_and_parse_hits_round_trip() -> None:
    """输入：全空间 8 种命中组合。输出：无。预期行为：两种表示互相还原。"""
    for value in features.HIT_SETS:
        assert features.hits_of(packet_of(features.parse_hits(value))) == value
    assert features.parse_hits("") == ()


def test_missing_keys_finds_only_absent_ones() -> None:
    """输入：完整记录、少两行的记录。输出：无。预期行为：只报缺掉的那几个键。"""
    full = render_lines(BASE_LINES)
    assert features.missing_keys(full) == ()
    lines = {key: value for key, value in BASE_LINES.items() if key not in {"rule", "exploit"}}
    assert set(features.missing_keys(render_lines(lines))) == {"rule", "exploit"}


def test_missing_keys_needs_the_colon_on_its_own_line() -> None:
    """输入：把键名写在句子中间的记录。输出：无。

    预期行为：算缺——判据是钉在**行首**的，键不单独成行时那三条正则连定位都定不到。
    """
    text = "impact: hosts=3\nexposure: 见 rule: 那一行\nexploit: auth=required\ndetail: 一句"
    assert "rule" in features.missing_keys(text)


def test_line_value_reads_the_first_matching_line() -> None:
    """输入：含 `detail:` 的记录。输出：无。预期行为：取到冒号后面的内容，取不到给空串。"""
    text = render_lines(BASE_LINES)
    assert features.line_value(text, "detail").startswith("研发笔记本")
    assert features.line_value(text, "nope") == ""


def test_flags_and_level_follow_the_tag_order() -> None:
    """输入：几个命中组合。输出：无。预期行为：三格顺序恒为 `TAG_IDS` 的顺序。"""
    assert features.flags_of(()) == (False, False, False)
    assert features.flags_of(("R1", "R3")) == (True, False, True)
    assert features.level_of(("R1",)) == features.LEVEL_TABLE[(True, False, False)]


def test_level_table_covers_all_eight_cells() -> None:
    """输入：无。输出：无。预期行为：矩阵恰好八格，键就是三个因子的真假组合。"""
    expected = set(itertools.product((False, True), repeat=len(features.TAG_IDS)))
    assert set(features.LEVEL_TABLE) == expected
    assert len(features.LEVEL_TABLE) == 8


def test_level_table_is_monotone_in_every_factor() -> None:
    """输入：无。输出：无。

    预期行为：把任一因子从低换成高，等级**不会降**——定级表是单调的，改表时这一条会当场拦下。
    """
    for flags, level in features.LEVEL_TABLE.items():
        for index in range(len(flags)):
            raised = list(flags)
            if raised[index]:
                continue
            raised[index] = True
            assert features.LEVEL_TABLE[tuple(raised)] >= level, (flags, index)


def test_level_distribution_is_one_two_two_two_one() -> None:
    """输入：全空间 8 组。输出：无。

    预期行为：五档的分布是 1 / 2 / 2 / 2 / 1——最轻与最重各只占一格，中间三档各两格。这条钉住
    的是「小批量里五档都出得来」：某一档变成 0 格时，`grade` 那一族就永远取不到那个答案。
    """
    counts = dict.fromkeys(features.LEVELS, 0)
    for value in features.HIT_SETS:
        counts[features.level_of(features.parse_hits(value))] += 1
    assert counts == {1: 1, 2: 2, 3: 2, 4: 2, 5: 1}


def test_act_for_collapses_the_top_levels() -> None:
    """输入：五个等级。输出：无。预期行为：最后一档吃掉 4 / 5 两级。"""
    assert [features.act_for(level) for level in features.LEVELS] == [
        "monitor",
        "ticket",
        "patch_now",
        "isolate",
        "isolate",
    ]


def test_combo_text_key_marks_every_factor() -> None:
    """输入：几个命中组合。输出：无。预期行为：三个因子都出现，按 `TAG_IDS` 顺序，用 high / low。"""
    assert features.combo_text_key(()) == "R1=low R2=low R3=low"
    assert features.combo_text_key(("R1", "R3")) == "R1=high R2=low R3=high"
    assert features.combo_text_key(("R1", "R2", "R3")) == "R1=high R2=high R3=high"


def test_case_helpers_are_stable() -> None:
    """输入：一个 case。输出：无。预期行为：字典键序、指纹与可读描述都固定。"""
    case = AlertRiskCase("R1+R3", features.SOURCE_EDR)
    assert case.as_dict() == {"highs": "R1+R3", "source": "edr"}
    assert case.fingerprint() == "highs=R1+R3|source=edr"
    assert case.describe() == "R1+R3-edr"
    assert case.tags() == ("R1", "R3")
    assert AlertRiskCase("", features.SOURCE_WAF).describe() == "无高危因子-waf"


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
            assert first.highs == case.tags()
            combos = first.distractors
            assert len(combos) == 3
            assert len(set(combos)) == 3
            for combo in combos:
                assert set(combo) != set(first.highs)
                assert combo == tuple(tag for tag in features.TAG_IDS if tag in combo)
                assert len(set(combo) ^ set(first.highs)) <= 3


def test_label_for_and_observed_labels_agree_on_faithful_packets() -> None:
    """输入：全空间 8 组「完全兑现」的记录。输出：无。

    预期行为：构造侧（计划）与审计侧（文本）两条路径给出同一份标签——对不上就说明渲染没兑现计划。
    """
    for value in features.HIT_SETS:
        tags = features.parse_hits(value)
        case = AlertRiskCase(value, features.SOURCE_EDR)
        plan = features.plan_for(case, 0)
        assert features.label_for(plan) == features.observed_labels(packet_of(tags))


def test_labels_expose_level_and_act() -> None:
    """输入：几个命中组合。输出：无。预期行为：等级与处置档位都由矩阵推出来。"""
    assert features.AlertRiskLabels(()).level == 1
    assert features.AlertRiskLabels(()).act == "monitor"
    assert features.AlertRiskLabels(("R1", "R2", "R3")).level == 5
    assert features.AlertRiskLabels(("R1", "R2", "R3")).act == "isolate"
    assert features.AlertRiskLabels(("R2",)).contains("R2")
    assert not features.AlertRiskLabels(("R2",)).contains("R1")


def test_build_cases_enumerates_the_whole_space() -> None:
    """输入：全枚举上限、可选的维度覆盖。输出：无。

    预期行为：全空间 8 × 3 = 24 组，顺序稳定；覆盖掉某个取值之后规模随之变小。
    """
    rows = features.build_cases(512)
    assert len(rows) == len(features.HIT_SETS) * len(features.SOURCES)
    assert rows == features.build_cases(512)
    restricted = features.build_cases(512, dims={"highs": features.HIT_SETS, "source": ("edr",)})
    assert len(restricted) == len(features.HIT_SETS)
    assert {case.source for case in restricted} == {"edr"}


def test_patterns_only_look_at_their_own_line() -> None:
    """输入：把**取值**（不带它那个键名）写到别的行上。输出：无。

    预期行为：不算取到——判据认的是「取值 + 落在它说的那一行」，取值挪到别处就失去定位。
    只挪取值不挪键名，是因为正则的口径本来就是「**某一行里**同时有键名与形状」：把整行连键名
    一起搬到别处，那一行仍然是「`exposure:` 行」，按口径本来就该算命中。
    """
    values = {"R1": "reachable=public", "R2": "auth=none", "R3": "hosts=24"}
    for tag in features.TAG_IDS:
        key = features.LINE_KEYS[tag]
        other = next(name for name in features.KEY_LINES if name != key)
        lines = {**BASE_LINES, other: f"{other}: {values[tag]} -- 写在别的行上了"}
        assert features.hit_set(render_lines(lines)) == (), tag
