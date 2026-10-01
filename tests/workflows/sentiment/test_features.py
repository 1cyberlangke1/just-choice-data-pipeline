"""特征层：四条规则的纯函数、两条独立标签路径、case 空间与留出下标。"""

from __future__ import annotations

from collections import Counter
from typing import Final

import pytest

from workflows.sentiment import features as ft

#: 同分布全空间的 case（15 条同分布弧线 × 5 个场景）。
FULL: Final[tuple[ft.SentimentCase, ...]] = tuple(
    ft.build_cases(512, dims={"arc": ft.ID_ARCS, "topic": ft.ID_TOPICS})
)
#: 留出全空间的 case（4 条留出弧线 × 5 个场景）。
OOD: Final[tuple[ft.SentimentCase, ...]] = tuple(
    ft.build_cases(512, dims={"arc": ft.OOD_ARCS, "topic": ft.ID_TOPICS})
)


def test_case_space_covers_the_whole_matrix() -> None:
    """输入：无。输出：无。预期行为：全空间 = 弧线条数 × 场景数，一条不多不少。"""
    assert len(FULL) == len(ft.ID_ARCS) * len(ft.ID_TOPICS) == 75
    assert len(OOD) == len(ft.OOD_ARCS) * len(ft.ID_TOPICS) == 20


def test_default_dims_are_the_id_space() -> None:
    """输入：无。输出：无。预期行为：`DIMS` 就是同分布侧那 75 组，不给维度表时按它枚举。"""
    assert ft.DIMS == {"arc": ft.ID_ARCS, "topic": ft.ID_TOPICS}
    assert ft.build_cases(512) == list(FULL)


def test_every_arc_has_five_turns_of_known_levels() -> None:
    """输入：全部弧线。输出：无。预期行为：每条正好 `TURNS` 个回合，分值都在词表的档位里。"""
    for index in range(len(ft.ARCS)):
        levels = ft.arc_of(index)
        assert len(levels) == ft.TURNS
        assert set(levels) <= set(ft.LEVELS), index


def test_arc_of_reports_a_bad_index() -> None:
    """输入：越界下标。输出：无。预期行为：当场报错，不静默回绕到别的弧线。"""
    with pytest.raises(ValueError, match="弧线下标"):
        ft.arc_of(len(ft.ARCS))


def test_ood_arcs_are_never_in_the_id_space() -> None:
    """输入：无。输出：无。预期行为：留出下标与同分布下标不相交，合起来正好是全部弧线。"""
    assert not set(ft.OOD_ARCS) & set(ft.ID_ARCS)
    assert set(ft.OOD_ARCS) | set(ft.ID_ARCS) == set(range(len(ft.ARCS)))


def test_tone_is_the_sign_of_the_total() -> None:
    """输入：手写弧线。输出：无。预期行为：规则 2——只看和的符号，与中间怎么摆无关。"""
    assert ft.tone_of((1, 1, 1, 1, 1)) == "pos"
    assert ft.tone_of((-1, -1, -1, -1, -1)) == "neg"
    assert ft.tone_of((0, 0, 0, 0, 0)) == "flat"
    assert ft.tone_of((2, 2, 2, -2, -2)) == "pos"
    assert ft.tone_of((2, -1, -1, -1, 1)) == "flat"


def test_pivot_is_the_first_sign_change_and_zero_never_counts() -> None:
    """输入：手写弧线。输出：无。

    预期行为：规则 3——只认「一正一负」的相邻对；0 与任何符号都不构成转折，所以
    `(1, 0, 1, 1, 1)` 一次转折都没有。
    """
    assert ft.pivot_of((1, 1, 1, 1, 1)) is None
    assert ft.pivot_of((1, 0, 1, 1, 1)) is None
    assert ft.pivot_of((1, -1, 1, -1, 1)) == 1
    assert ft.pivot_of((1, 1, -1, 1, 1)) == 2
    assert ft.pivot_of((2, 2, 2, -2, -2)) == 3
    assert ft.pivot_of((0, 0, 0, -1, 1)) == 4
    assert ft.pivot_key(4) == "t5" and ft.pivot_key(None) == "none"


def test_drop_counts_only_the_worsening_direction() -> None:
    """输入：手写弧线。输出：无。预期行为：规则 4——只看「上一回合减下一回合」的最大值。"""
    assert ft.drop_of((1, 1, 1, 1, 1)) == 0
    assert ft.drop_of((-1, -1, -1, -1, -1)) == 0
    assert ft.drop_of((-2, -1, 1, 2, 2)) == 0
    assert ft.drop_of((1, 1, -1, 1, 1)) == 2
    assert ft.drop_of((2, 2, 2, -2, -2)) == 4


def test_rebound_needs_a_negative_turn_followed_by_a_positive_one() -> None:
    """输入：手写弧线。输出：无。

    预期行为：规则 5——正→负不算回升（那是转折），负→正才算。
    """
    assert ft.rebound_of((-1, -1, 1, 1, -1)) is True
    assert ft.rebound_of((1, 1, -1, 1, 1)) is True
    assert ft.rebound_of((2, 2, 2, -2, -2)) is False
    assert ft.rebound_of((0, 0, 0, 0, 0)) is False


def test_word_slot_only_affects_the_wording() -> None:
    """输入：几个弧线与回合。输出：无。预期行为：取词下标永远落在词表范围内，且是确定的。"""
    for arc in range(len(ft.ARCS)):
        for turn in range(ft.TURNS):
            slot = ft.word_slot(arc, turn)
            assert 0 <= slot < ft.WORD_SLOTS
            assert slot == ft.word_slot(arc, turn)


def test_every_family_takes_every_level() -> None:
    """输入：同分布全空间。输出：无。

    预期行为：四族各自把取值铺满——`tone` 三种、`pivot` 五个位置、`drop` 0~4、`rebound` 真假
    都出现。某一档取不到时那一族的选项/等级表就有一个永远不是答案。
    """
    labels = [ft.label_for(case) for case in FULL]
    assert {item.tone for item in labels} == {"pos", "neg", "flat"}
    assert {item.pivot_key for item in labels} == {"t2", "t3", "t4", "t5", "none"}
    assert {item.drop for item in labels} == set(range(5))
    assert {item.rebound for item in labels} == {True, False}


def test_the_id_space_stays_close_to_uniform() -> None:
    """输入：同分布全空间。输出：无。

    预期行为：15 条弧线逐族铺开（倾向 5 正 / 6 负 / 4 中性），四族的最大偏差都压在 0.1 以内
    （口径见 `test_pipeline.py` 的分布断言；退化线是 0.2）。
    """
    labels = [ft.label_for(case) for case in FULL]
    space: dict[str, tuple[object, ...]] = {
        "tone": ("pos", "neg", "flat"),
        "pivot": ("t2", "t3", "t4", "t5", "none"),
        "drop": (0, 1, 2, 3, 4),
        "rebound": (True, False),
    }
    getters = {
        "tone": lambda item: item.tone,
        "pivot": lambda item: item.pivot_key,
        "drop": lambda item: item.drop,
        "rebound": lambda item: item.rebound,
    }
    for family, options in space.items():
        counts = Counter(getters[family](item) for item in labels)
        assert set(counts) == set(options), family
        worst = max(counts[option] / len(labels) for option in options) - 1 / len(options)
        assert worst <= 0.1, f"{family} 偏斜过大：{dict(counts)}"
    tone_counts = Counter(item.tone for item in labels)
    assert tone_counts["pos"] == 25 and tone_counts["neg"] == 30 and tone_counts["flat"] == 20


def test_the_two_label_paths_agree() -> None:
    """输入：全空间 case。输出：无。

    预期行为：`label_for`（注入特征）与 `observed_labels`（反解出来的分值序列）给同一批标签
    ——构造侧与审计侧共用同一个函数，这条防的是将来有人把它拆成两份实现。
    """
    for case in FULL + OOD:
        assert ft.label_for(case) == ft.observed_labels(case.levels)


def test_fingerprint_and_describe_are_stable() -> None:
    """输入：同一个 case 造两次。输出：无。预期行为：指纹与描述都不随进程变化。"""
    case = ft.SentimentCase(arc=7, topic="review")
    assert case.fingerprint() == "arc=7|topic=review"
    assert case.describe() == "A7-review"
    assert case.as_dict() == {"arc": 7, "topic": "review"}
