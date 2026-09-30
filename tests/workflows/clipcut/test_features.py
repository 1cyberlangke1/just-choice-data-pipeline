"""特征层：两条规则的纯函数、两条独立标签路径、case 空间与留出下标。"""

from __future__ import annotations

from typing import Final

from workflows.clipcut import features as ft

#: 同分布全空间的 case（默认维度表 = 四条主题序列 × 八条高亮序列）。
FULL: Final[tuple[ft.ClipCase, ...]] = tuple(ft.build_cases(512))


def test_case_space_covers_the_whole_matrix() -> None:
    """输入：无。输出：无。预期行为：全空间 = 主题序列条数 × 高亮序列条数，一条不多不少。"""
    assert len(FULL) == len(ft.ID_TOPIC_PATTERNS) * len(ft.FLAG_PATTERNS)
    assert len(FULL) >= 30, f"全空间只有 {len(FULL)} 个 case，下面的分布尺子量不到东西"


def test_every_pattern_has_exactly_four_segments() -> None:
    """输入：无。输出：无。

    预期行为：主题序列与高亮序列都正好 `SEGMENTS` 段——顺序折叠的「前缀」是这一族的轴，
    两边长度对不上时 `zip(strict=True)` 会当场炸，但那已经是运行期了。
    """
    for pattern in ft.TOPIC_PATTERNS:
        assert len(pattern) == ft.SEGMENTS
    for flags in ft.FLAG_PATTERNS:
        assert len(flags) == ft.SEGMENTS


def test_first_positions_are_the_first_occurrence() -> None:
    """输入：两条主题序列。输出：无。预期行为：首现位置只看序列、后出现的覆盖不了。"""
    assert ft.first_positions(("T1", "T2", "T1", "T2")) == {"T1": 0, "T2": 1}
    assert ft.first_positions(("T1", "T2", "T3", "T4")) == {"T1": 0, "T2": 1, "T3": 2, "T4": 3}


def test_kept_needs_both_a_highlight_and_a_first_occurrence() -> None:
    """输入：手写的小序列。输出：无。

    预期行为：两条规则各挡一半——全高亮 + 全不重复时四段全留；主题重复时只有首现有资格。
    """
    assert ft.kept_of(("T1", "T2", "T3", "T4"), (True,) * 4) == (0, 1, 2, 3)
    assert ft.kept_of(("T1", "T1", "T2", "T2"), (True,) * 4) == (0, 2)
    assert ft.kept_of(("T1", "T1", "T2", "T2"), (False, True, True, True)) == (2,)


def test_blocked_is_the_highlighted_but_not_first() -> None:
    """输入：手写的小序列。输出：无。

    预期行为：被挡下 = 含高亮点 **且** 主题不是首现；重复段没有高亮点时它压根没资格，
    不算被挡下。
    """
    assert ft.blocked_of(("T1", "T1", "T2", "T2"), (True,) * 4) == (1, 3)
    assert ft.blocked_of(("T1", "T1", "T2", "T2"), (True, False, True, False)) == ()


def test_kept_and_blocked_partition_the_highlighted_segments() -> None:
    """输入：全空间 case。输出：无。

    预期行为：每一段含高亮点的片段要么留下、要么被挡下——两族不重不漏，合起来正好是
    「高亮位置」这个集合。这是两条规则互相不打架的形式化说法。
    """
    for case in FULL:
        kept = set(ft.kept_of(case.topic_seq, case.flag_seq))
        blocked = set(ft.blocked_of(case.topic_seq, case.flag_seq))
        highlighted = {index for index, flag in enumerate(case.flag_seq) if flag}
        assert not kept & blocked, case.describe()
        assert kept | blocked == highlighted, case.describe()


def test_has_dup_looks_only_at_the_topic_sequence() -> None:
    """输入：两条主题序列。输出：无。预期行为：只看主题名，与高亮无关。"""
    assert ft.has_dup(("T1", "T1", "T2", "T2")) is True
    assert ft.has_dup(("T1", "T2", "T3", "T4")) is False
    assert ft.has_dup(("T1", "T2", "T3", "T2")) is True


def test_dup_and_blocked_are_two_different_questions() -> None:
    """输入：有重复但全不高亮的 case。输出：无。

    预期行为：主题重复 ≠ 挡下了东西——重复的那一段如果本来就没有高亮点，它压根没资格，
    所以 `dup_any` 真而 `blocked_any` 假。这两族不是一个自由度。
    """
    case = ft.ClipCase(topics=0, flags=4)  # P0 有重复；F4 全不高亮
    labels = ft.label_for(case)
    assert labels.dup is True
    assert labels.blocked == ()
    assert labels.kept == ()


def test_keep_count_covers_every_level() -> None:
    """输入：全空间 case。输出：无。

    预期行为：保留段数把 0~4 每一档都取到——某一档取不到时 `keep_count` 那一族的等级表
    就有一个等级永远不是答案（均衡闸会标它）。
    """
    assert {ft.label_for(case).count for case in FULL} == set(range(ft.SEGMENTS + 1))


def test_first_kept_covers_every_option() -> None:
    """输入：全空间 case。输出：无。

    预期行为：最早保留的段把「第 1~4 段」与「一段都没保留」五种答案都取到——只取前缀形状的
    高亮序列会让这一族退化成常量。
    """
    seen: set[int | str] = set()
    for case in FULL:
        first = ft.label_for(case).first
        seen.add("none" if first is None else first + 1)
    assert seen == {1, 2, 3, 4, "none"}


def test_label_for_agrees_with_observed_labels() -> None:
    """输入：全空间 case。输出：无。

    预期行为：构造侧（`label_for` 吃 case）与审计侧（`observed_labels` 吃反解出来的序列）
    得到同一批标签——规则只有一份实现，两边对不上就说明有人绕过它另算了一套。
    """
    for case in FULL:
        assert ft.label_for(case) == ft.observed_labels(case.topic_seq, case.flag_seq)


def test_build_cases_is_deterministic() -> None:
    """输入：同一入参两次。输出：无。预期行为：顺序与内容逐条一致。"""
    first = [case.describe() for case in ft.build_cases(512)]
    second = [case.describe() for case in ft.build_cases(512)]
    assert first == second


def test_build_cases_honours_a_dim_override() -> None:
    """输入：只留一条主题序列、两条高亮序列的维度表。输出：无。

    预期行为：覆盖维度表后只枚举那一小块——`test_ood` 就是靠这个只收含留出主题的序列。
    """
    rows = ft.build_cases(512, dims={"topics": (4,), "flags": (0, 1)})
    assert [row.describe() for row in rows] == ["P4-F0", "P4-F1"]


def test_case_as_dict_and_describe_are_stable() -> None:
    """输入：一个 case。输出：无。预期行为：键序固定（哈希分桶要用）、描述可读。"""
    case = ft.ClipCase(topics=1, flags=2)
    assert case.as_dict() == {"topics": 1, "flags": 2}
    assert list(case.as_dict()) == ["topics", "flags"]
    assert case.describe() == "P1-F2"
    assert case.fingerprint() == case.fingerprint()


def test_ood_pattern_is_absent_from_the_id_side() -> None:
    """输入：无。输出：无。

    预期行为：含留出主题的序列**恰好**是 `OOD_TOPIC_PATTERNS` 指的那几条，同分布下标一条都
    不含它——值留出就是靠这条在数据上成立的，只写注释不写断言的话改一次表就能把它悄悄变回
    同分布。顺带钉住两条更要紧的：留出侧至少两条序列，且**重复与不重复各有一条**——只有一条
    自带重复的序列时 `dup_any` 在留出侧会塌成常量（8 条全 True，范式 §9.79 第六条）。
    """
    with_ood = tuple(
        index for index, pattern in enumerate(ft.TOPIC_PATTERNS) if ft.OOD_TOPIC_ID in pattern
    )
    assert ft.OOD_TOPIC_PATTERNS == with_ood
    assert not set(ft.ID_TOPIC_PATTERNS) & set(ft.OOD_TOPIC_PATTERNS)
    assert len(ft.OOD_TOPIC_PATTERNS) >= 2
    assert {ft.has_dup(ft.TOPIC_PATTERNS[index]) for index in ft.OOD_TOPIC_PATTERNS} == {True, False}
    for index in ft.ID_TOPIC_PATTERNS:
        assert ft.OOD_TOPIC_ID not in ft.TOPIC_PATTERNS[index]
    assert ft.DIMS["topics"] == ft.ID_TOPIC_PATTERNS
    assert ft.DIMS["flags"] == tuple(range(len(ft.FLAG_PATTERNS)))
