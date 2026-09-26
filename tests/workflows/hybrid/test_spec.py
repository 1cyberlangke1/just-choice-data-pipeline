"""骨架规格的测试：可解性过滤、不变量、答案与分布。"""

from __future__ import annotations

import json
import random
from collections import Counter
from itertools import pairwise
from typing import Any

import pytest

from workflows.hybrid import spec as sp


def _problems(spec: sp.HybridSpec) -> list[str]:
    """输入：一份规格。输出：违反的骨架不变量（空 = 全过）。

    预期行为：把「答案唯一」这件事在**数据层**查死——开始时刻 / 主持人 / 会议室
    两两不同（`same_room` 那一对除外），场数、时长档、收尾时刻、星期几、重叠对数都要对。
    """
    meetings = spec.meetings
    out: list[str] = []
    if len(meetings) != spec.case.meetings:
        out.append(f"场数 {len(meetings)} != {spec.case.meetings}")
    starts = [meeting.start for meeting in meetings]
    if len(set(starts)) != len(starts):
        out.append(f"开始时刻有重复：{starts}")
    if len({meeting.lead for meeting in meetings}) != len(meetings):
        out.append("主持人有重复")
    rooms = Counter(meeting.room for meeting in meetings)
    duplicated = [room for room, count in rooms.items() if count > 1]
    if spec.case.clash == "same_room":
        if len(duplicated) != 1 or max(rooms.values()) != 2:
            out.append(f"same_room 的共用房间不对：{dict(rooms)}")
        if not spec.has_room_clash:
            out.append("same_room 却没检测到房间冲突")
    elif duplicated:
        out.append(f"{spec.case.clash} 档却出现重复房间：{dict(rooms)}")
    if meetings[-1].end > sp.DAY_END:
        out.append(f"收尾超过 {sp.DAY_END} 点：{meetings[-1].end}")
    lo_dur, hi_dur = sp.DUR_RANGES[spec.case.duration_mix]
    if any(not lo_dur <= meeting.dur <= hi_dur for meeting in meetings):
        out.append(f"时长越档：{[meeting.dur for meeting in meetings]}")
    overlaps = sum(
        1
        for index, left in enumerate(meetings)
        for right in meetings[index + 1 :]
        if sp.intervals_overlap(left, right)
    )
    want = 0 if spec.case.clash == "none" else 1
    if overlaps != want:
        out.append(f"重叠对数 {overlaps} != {want}")
    if spec.weekday != spec.case.weekday:
        out.append(f"星期几 {spec.weekday} != {spec.case.weekday}")
    return out


def test_every_case_holds_every_invariant() -> None:
    """输入：全枚举 case × 每个 8 条样本。输出：无。

    预期行为：一个组合最多能写出多少种日程，骨架不变量都得成立——
    构造器是随机排布的，靠抽样把「某个分支只有特定组合才走到」的坑压出来。
    """
    bad: list[str] = []
    total = 0
    for case in sp.build_cases(sp.HybridConfig()):
        for seq in range(8):
            rng = random.Random(f"{case.describe()}|{seq}")
            spec = sp.sample_spec(rng, case, "train", seq, 0, 0)
            total += 1
            bad += [f"{case.describe()}#{seq}: {item}" for item in _problems(spec)]
    assert total > 0
    assert bad == []


def test_compatible_rejects_combinations_that_cannot_be_written() -> None:
    """输入：几个写不出来的组合。输出：False。

    预期行为：两场会 + `same_room` 写不出「两场抢同一间房」（整份日程只剩一间房，
    `room_of` 会退化成一个选项）；1 场会时所有「第 n 场」题族都只剩一个选项；
    傍晚才开始 + 4 场长会怎么排都开到第二天。
    """
    assert not sp.compatible(sp.HybridCase(1, "early", "none", "bullet", 1, "short"))
    assert not sp.compatible(sp.HybridCase(2, "early", "same_room", "bullet", 1, "short"))
    assert not sp.compatible(sp.HybridCase(4, "evening", "none", "bullet", 1, "long"))
    assert sp.compatible(sp.HybridCase(3, "early", "same_room", "bullet", 1, "short"))
    assert sp.compatible(sp.HybridCase(2, "evening", "none", "bullet", 1, "short"))


def test_single_meeting_config_yields_no_cases() -> None:
    """输入：只要 1 场会的采样参数。输出：空 case 表。

    预期行为：不是「生成了再拒」，而是**根本不生成**——`meeting_counts` 是公开旋钮，
    填了写不出来的取值时应当得到空表，而不是一整批 `E_CHOICE_OPTION_COUNT` 垃圾。
    """
    assert sp.build_cases(sp.HybridConfig(meeting_counts=(1,))) == []
    mixed = sp.build_cases(sp.HybridConfig(meeting_counts=(1, 2)))
    assert mixed  # 混着填时，写得出来的那一档照常生成
    assert {case.meetings for case in mixed} == {2}


def test_build_cases_is_stable_and_filtered() -> None:
    """输入：默认采样参数。输出：case 列表。

    预期行为：顺序稳定（两次调用逐项相同）、全部可解、组合不重复。
    """
    first = sp.build_cases(sp.HybridConfig())
    second = sp.build_cases(sp.HybridConfig())
    assert [case.as_dict() for case in first] == [case.as_dict() for case in second]
    assert all(sp.compatible(case) for case in first)
    assert len({case.describe() for case in first}) == len(first)
    assert len(first) > 100


def test_ood_only_keeps_exactly_the_unseen_values() -> None:
    """输入：`ood_only=True` 的采样参数。输出：case 列表。

    预期行为：每个 OOD case 至少含一个训练侧没见过的取值，且训练侧取值表里确实没有它们。
    """
    ood = sp.build_cases(sp.HybridConfig(split="test_ood", ood_only=True))
    assert ood and all(case.unseen_dims() for case in ood)
    train = sp.build_cases(
        sp.HybridConfig(
            layouts=tuple(sorted(sp.SEEN_LAYOUTS)), clashes=tuple(sorted(sp.SEEN_CLASHES))
        )
    )
    assert all(case.unseen_dims() == () for case in train)
    assert {case.layout for case in train} == set(sp.SEEN_LAYOUTS)
    assert {case.clash for case in train} == set(sp.SEEN_CLASHES)


def test_duration_ranges_do_not_overlap() -> None:
    """输入：时长档表。输出：无。

    预期行为：两档**不重叠**——重叠的话中间那个时长会被两个档各抽一次，
    `duration_of` 的选项出现次数就不均匀了。
    """
    short_lo, short_hi = sp.DUR_RANGES["short"]
    long_lo, long_hi = sp.DUR_RANGES["long"]
    assert short_hi < long_lo
    assert short_lo <= short_hi < long_lo <= long_hi


def test_openings_are_inside_the_day_and_ordered() -> None:
    """输入：起始时段表。输出：无。预期行为：五个时段都在同一天内且不交叉。"""
    spans = [sp.OPENING_HOURS[name] for name in sp.OPENINGS]
    assert all(0 <= lo <= hi <= sp.DAY_END for lo, hi in spans)
    assert spans == sorted(spans)
    for (_, hi), (next_lo, _) in pairwise(spans):
        assert hi < next_lo


def _handmade(
    meetings: tuple[tuple[int, int, int, int], ...], *, clash: str = "none"
) -> sp.HybridSpec:
    """输入：[(开始, 时长, 会议室下标, 主持人下标)]、冲突档。输出：手搓的规格。

    预期行为：绕开构造器直接给骨架，用来把答案算式的边界钉死。
    """
    return sp.HybridSpec(
        case=sp.HybridCase(len(meetings), "early", clash, "bullet", 3, "short"),
        split="train",
        seq=0,
        date=(2026, 9, 23),
        meetings=tuple(sp.Meeting(*item) for item in meetings),
        phrasing=0,
        theme=0,
    )


def test_busy_level_counts_hours_intersecting_the_window() -> None:
    """输入：几种手搓日程。输出：`busy_level`。

    预期行为：算的是**区间求交**——11:00–13:00 的会占用了窗口里的 12 点；
    15:00–17:00 的会只占用 15 点；窗口外的会不贡献；超过 4 个整点封顶到 4。
    """
    assert _handmade(((11, 2, 0, 0), (17, 1, 1, 1))).busy_level == 1
    assert _handmade(((15, 2, 0, 0), (20, 1, 1, 1))).busy_level == 1
    assert _handmade(((13, 4, 0, 0), (20, 1, 1, 1))).busy_level == 3
    assert _handmade(((12, 4, 0, 0), (16, 4, 1, 1))).busy_level == 4
    assert _handmade(((8, 1, 0, 0), (18, 1, 1, 1))).busy_level == 0


def test_busy_level_deduplicates_overlapping_hours() -> None:
    """输入：两场重叠且都落在窗口里的会。输出：`busy_level`。

    预期行为：重叠的整点只算一次——否则「忙碌度」会随重叠而虚高。
    12:00–14:00 与 13:00–15:00 合起来覆盖 12/13/14 三个整点（加起来是 4，封顶到 4）。
    """
    assert _handmade(((12, 2, 0, 0), (13, 2, 1, 1))).busy_level == 3


def test_has_overlap_and_room_clash() -> None:
    """输入：手搓日程。输出：两个布尔答案。

    预期行为：重叠与房间冲突是两件事——共用一间房但时间错开不算冲突。
    """
    apart = _handmade(((9, 1, 0, 0), (10, 1, 0, 1)))
    assert apart.has_overlap is False and apart.has_room_clash is False
    same_time = _handmade(((9, 2, 0, 0), (10, 2, 0, 1)))
    assert same_time.has_overlap is True and same_time.has_room_clash is True


def test_room_slots_are_deduplicated_in_first_seen_order() -> None:
    """输入：手搓日程。输出：会议室下标序列。预期行为：按首次出现去重。"""
    spec = _handmade(((9, 3, 0, 0), (10, 3, 0, 1), (13, 1, 2, 2)), clash="same_room")
    assert spec.room_slots == (0, 2)


def test_slot_names_and_weekday_of() -> None:
    """输入：下标 / 日期。输出：槽名 / 星期几。预期行为：槽名从 1 起，星期几用 ISO 编号。"""
    assert sp.slot_name(0) == "m1" and sp.slot_name(3) == "m4"
    assert sp.weekday_of((2026, 9, 23)) == 3  # 2026-09-23 是星期三
    assert sp.weekday_of((2026, 9, 27)) == 7  # 2026-09-27 是星期日


def test_sample_spec_is_deterministic_and_records_the_plan() -> None:
    """输入：同种子两次采样。输出：两份规格。

    预期行为：逐字段一致（骨架只用传进来的 RNG）；切分名 / 序号 / 问法照抄计划；
    标题与议题留空等渲染阶段补。
    """
    case = sp.HybridCase(3, "noon", "overlap", "table", 5, "long")
    first = sp.sample_spec(random.Random(7), case, "val", 42, 2, 5)
    second = sp.sample_spec(random.Random(7), case, "val", 42, 2, 5)
    assert first == second
    assert first.split == "val" and first.seq == 42 and first.phrasing == 2 and first.theme == 5
    assert first.written == {} and first.slot_names() == ("m1", "m2", "m3")
    assert first.text_task_id == "hybrid-val-0000042-text"


def test_text_of_raises_before_render() -> None:
    """输入：还没渲染过的规格。输出：KeyError。

    预期行为：文本由后端产生，`ask` 之前必须先走 `render`；拿不到就报错而不是给空串。
    """
    spec = sp.sample_spec(
        random.Random(1), sp.HybridCase(2, "early", "none", "bullet", 1, "short"), "train", 0, 0, 0
    )
    with pytest.raises(KeyError, match="还没写"):
        spec.text_of("m1")


def test_spec_to_dict_is_json_serializable(make_spec: Any) -> None:
    """输入：一份渲染过的规格。输出：字典。

    预期行为：能直接 `json.dumps`（manifest 与调试要用），且带上骨架与写好的文本。
    """
    spec = make_spec(9)
    payload = sp.spec_to_dict(spec)
    assert json.loads(json.dumps(payload, ensure_ascii=False))["meetings"]
    assert len(payload["meetings"]) == spec.case.meetings
    assert set(payload["written"]) == set(spec.slot_names())
    assert payload["weekday"] == spec.weekday


def test_case_helpers() -> None:
    """输入：一个 case。输出：描述与维度字典。预期行为：维度顺序与声明一致且稳定。"""
    case = sp.HybridCase(4, "afternoon", "same_room", "table", 6, "long")
    assert list(case.as_dict()) == [
        "meetings",
        "opening",
        "clash",
        "layout",
        "weekday",
        "duration_mix",
    ]
    assert case.describe() == "4场/afternoon/same_room/table/周6/long"
    assert case.unseen_dims() == ("layout", "clash")


def test_sample_spec_refuses_an_unsolvable_case() -> None:
    """输入：没过 `compatible` 的组合。输出：ValueError（人话）。

    预期行为：直接点名这个组合写不出来——不然会在排布中途炸成
    `empty range for randrange()`，看的人一头雾水。
    """
    case = sp.HybridCase(2, "early", "same_room", "bullet", 1, "short")
    with pytest.raises(ValueError, match="写不出日程"):
        sp.sample_spec(random.Random(1), case, "train", 0, 0, 0)


def test_presentation_order_is_not_always_chronological() -> None:
    """输入：全枚举 case × 每个 4 条样本。输出：无。

    预期行为：展示顺序**不能**总是时间顺序——否则「哪一场开始得最早」的答案永远是 m1，
    模型会学成「答案 = 列表第一行」的位置捷径，而不是真的去比时刻。
    （也不能永远不是：偶尔顺序刚好一致是正常的。）
    """
    chronological = 0
    total = 0
    for case in sp.build_cases(sp.HybridConfig()):
        for seq in range(4):
            spec = sp.sample_spec(
                random.Random(f"{case.describe()}|{seq}"), case, "train", seq, 0, 0
            )
            starts = [meeting.start for meeting in spec.meetings]
            total += 1
            chronological += starts == sorted(starts)
    assert 0 < chronological < total


def test_intervals_overlap_is_symmetric() -> None:
    """输入：两场会（顺序互换）。输出：是否重叠。

    预期行为：判据与入参顺序无关——展示顺序是洗过的，非对称判据会把不相邻的两场误判成重叠。
    """
    early = sp.Meeting(9, 2, 0, 0)  # 9:00–11:00
    late = sp.Meeting(11, 1, 1, 1)  # 11:00–12:00
    assert sp.intervals_overlap(early, late) is False
    assert sp.intervals_overlap(late, early) is False
    touching = sp.Meeting(10, 1, 2, 2)  # 10:00–11:00，与上面两场都重叠
    assert sp.intervals_overlap(early, touching) is True
    assert sp.intervals_overlap(touching, early) is True
