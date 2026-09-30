"""特征层：四条规则的纯函数、两条独立标签路径、case 空间与留出维度。"""

from __future__ import annotations

from typing import Final

from workflows.quoteread import features as ft

#: 同分布全空间的 case（默认维度表 = 八条读数模式 × 四条单位模式 × 同分布名集）。
FULL: Final[tuple[ft.QuoteCase, ...]] = tuple(ft.build_cases(512))


def test_case_space_covers_the_whole_matrix() -> None:
    """输入：无。输出：无。预期行为：全空间 = 读数模式数 × 单位模式数，一条不多不少。"""
    assert len(FULL) == len(ft.VALUE_PATTERNS) * len(ft.UNIT_PATTERNS)
    assert len(FULL) >= 30, f"全空间只有 {len(FULL)} 个 case，下面的分布尺子量不到东西"


def test_every_pattern_has_exactly_three_rows() -> None:
    """输入：无。输出：无。

    预期行为：读数模式与单位模式都正好 `ROWS` 行——两边长度对不上时 `zip(strict=True)`
    会当场炸，但那已经是运行期了。
    """
    for pattern in ft.VALUE_PATTERNS:
        assert len(pattern) == ft.ROWS
    for units in ft.UNIT_PATTERNS:
        assert len(units) == ft.ROWS
        assert set(units) <= set(ft.UNITS), units


def test_tier_floors_partition_the_absolute_value() -> None:
    """输入：手写的几个基准点数。输出：无。

    预期行为：档位只看**绝对值**、按 `TIER_FLOORS` 从低到高取「最后一个不大于它的下界」；
    符号不影响档位，边界值归入**上**一档。
    """
    assert [ft.tier_of(size) for size in (0, 19, 20, 49, 50, 99, 100, 199, 200, 999)] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        4,
        4,
    ]
    for size in (7, 45, 150):
        assert ft.tier_of(size) == ft.tier_of(-size), size


def test_leader_compares_signed_values_and_breaks_ties_by_row() -> None:
    """输入：手写的几组读数。输出：无。

    预期行为：`leader` 比的是**带符号**的读数（不是绝对值），并列时取序号靠前的——
    `(-150, -45, 25)` 的领涨是第 3 行而不是跌得最狠的第 1 行。
    """
    assert ft.leader_of((-150, -45, 25)) == 2
    assert ft.leader_of((0, 0, 0)) == 0
    assert ft.leader_of((45, 45, 45)) == 0
    assert ft.leader_of((-15, -30, -8)) == 2


def test_alert_and_divergent_are_two_different_questions() -> None:
    """输入：手写的几组读数。输出：无。

    预期行为：`alert` 只看**绝对值够不够大**、`divergent` 只看**有没有一涨一跌**——
    全零时两者都假，`(-150, -45, 25)` 时两者都真，`(45, 45, 45)` 时 `alert` 假而
    `divergent` 也假（同向），`(300, 300, 300)` 时 `alert` 真而 `divergent` 假。
    """
    assert ft.alert_of((0, 0, 0)) is False
    assert ft.divergent_of((0, 0, 0)) is False
    assert ft.alert_of((-150, -45, 25)) is True
    assert ft.divergent_of((-150, -45, 25)) is True
    assert ft.alert_of((45, 45, 45)) is False
    assert ft.divergent_of((45, 45, 45)) is False
    assert ft.alert_of((300, 300, 300)) is True
    assert ft.divergent_of((300, 300, 300)) is False


def test_alert_line_is_not_a_tier_boundary() -> None:
    """输入：无。输出：无。

    预期行为：警戒线不落在任何档位下界上——落在边界上时 `alert` 就成了 `tier` 的同义词
    （`tier >= 3`），那一族题白出。
    """
    assert ft.ALERT_BP not in ft.TIER_FLOORS
    assert ft.TIER_FLOORS[0] <= ft.ALERT_BP <= ft.TIER_FLOORS[-1]


def test_tier_covers_every_level() -> None:
    """输入：全空间 case。输出：无。

    预期行为：档位把 0~4 每一档都取到——某一档取不到时 `tier` 那一族的等级表就有一个等级
    永远不是答案（均衡闸会标它）。
    """
    assert {ft.label_for(case).tier for case in FULL} == set(range(len(ft.TIER_FLOORS)))


def test_leader_covers_every_row() -> None:
    """输入：全空间 case。输出：无。

    预期行为：领涨的行把三行都取到——只取「第一行最大」那种读数模式会让这一族退化成常量。
    """
    assert {ft.label_for(case).leader for case in FULL} == set(range(ft.ROWS))


def test_label_for_agrees_with_observed_labels() -> None:
    """输入：全空间 case。输出：无。

    预期行为：构造侧（`label_for` 吃 case）与审计侧（`observed_labels` 吃反解出来的读数）
    得到同一批标签——规则只有一份实现，两边对不上就说明有人绕过它另算了一套。
    """
    for case in FULL:
        assert ft.label_for(case) == ft.observed_labels(case.bp_values)


def _shown(bp: int, unit: str) -> float:
    """输入：基准点数、单位键。输出：**直接照屏幕上的数字读**得到的值（不做换算）。

    预期行为：`bp` 行的数字就是基准点，`pct` 行照抄小数点后的那个数——这正是「不换算就比」
    会犯的错。
    """
    return float(bp) if unit == "bp" else bp / 100


def _naive_tier(case: ft.QuoteCase) -> int:
    """输入：一个 case。输出：**不换算**直接比数字得出的档位。"""
    size = max(
        abs(_shown(bp, unit)) for bp, unit in zip(case.bp_values, case.unit_seq, strict=True)
    )
    tier = 0
    for index, floor in enumerate(ft.TIER_FLOORS):
        if size >= floor:
            tier = index
    return tier


def _naive_leader(case: ft.QuoteCase) -> int:
    """输入：一个 case。输出：**不换算**直接比数字得出的最大行下标。"""
    shown = [_shown(bp, unit) for bp, unit in zip(case.bp_values, case.unit_seq, strict=True)]
    best = 0
    for index, value in enumerate(shown):
        if value > shown[best]:
            best = index
    return best


def test_skipping_the_conversion_would_get_answers_wrong() -> None:
    """输入：全空间 case。输出：无。

    预期行为：至少有几条 case 的档位与领涨**在不换算时会算错**——这一族的轴是「单位换算」，
    如果直接比数字永远得出同一个答案，那这一维就白设了（那正是 `receiptcat` 的同量纲情形）。
    """
    tier_bad = sum(1 for case in FULL if ft.label_for(case).tier != _naive_tier(case))
    leader_bad = sum(1 for case in FULL if ft.label_for(case).leader != _naive_leader(case))
    assert tier_bad >= 5, f"不换算时档位会错的只有 {tier_bad} 条，换算这一维白设了"
    assert leader_bad >= 2, f"不换算时领涨会错的只有 {leader_bad} 条，换算这一维白设了"


def test_build_cases_honours_a_dim_override() -> None:
    """输入：只留一条读数模式、两条单位模式的维度表。输出：无。

    预期行为：覆盖维度表后只枚举那一小块——`test_ood` 就是靠这个只收留出名集。
    """
    rows = ft.build_cases(512, dims={"values": (4,), "units": (0, 1), "names": (0,)})
    assert [row.describe() for row in rows] == ["V4-U0-N0", "V4-U1-N0"]


def test_case_as_dict_and_describe_are_stable() -> None:
    """输入：一个 case。输出：无。预期行为：键序固定（哈希分桶要用）、描述可读。"""
    case = ft.QuoteCase(values=1, units=2, names=0)
    assert case.as_dict() == {"values": 1, "units": 2, "names": 0}
    assert list(case.as_dict()) == ["values", "units", "names"]
    assert case.describe() == "V1-U2-N0"
    assert case.fingerprint() == case.fingerprint()


def test_the_name_set_is_held_out_as_a_whole_dimension() -> None:
    """输入：无。输出：无。

    预期行为：同分布侧的标的名集**恒为一个取值**（0），留出侧恒为另一个（1），两组名键
    互不相交——维度留出就是靠这条在数据上成立的，只写注释不写断言的话改一次表就能把它
    悄悄变回同分布。
    """
    assert ft.DIMS["names"] == (0,)
    assert ft.OOD_DIMS["names"] == (1,)
    assert ft.NAME_SETS[0] == ft.NAME_IDS
    assert ft.NAME_SETS[1] == ft.OOD_NAME_IDS
    assert not set(ft.NAME_IDS) & set(ft.OOD_NAME_IDS)
    assert len(ft.NAME_IDS) == ft.ROWS and len(ft.OOD_NAME_IDS) == ft.ROWS
    assert ft.DIMS["values"] == tuple(range(len(ft.VALUE_PATTERNS)))
    assert ft.DIMS["units"] == tuple(range(len(ft.UNIT_PATTERNS)))
    assert ft.OOD_DIMS["values"] == ft.DIMS["values"]
    assert ft.OOD_DIMS["units"] == ft.DIMS["units"]
