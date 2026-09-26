"""会议日程 workflow 的真值规格：**骨架由构造器造，表述由后端补**。

对应 PLAN 3.2 分类表里的**混合式**，与另外三个 workflow 的分工是：

- `maze` / `invoice` 连文本都是算出来的（纯构造式，零 token）；
- `mail` 先定标签、模型只写外壳（特征注入式）；
- 本 workflow 介于两者之间：日期 / 时间 / 时长 / 会议室 / 主持人由构造器算出来，
  **每场会的标题与议题交给后端写**——这一半是模型「无中生有」的，
  也正是真实日程表里最不可枚举的部分。

正确性靠两条：

1. **答案只来自骨架**。10 个题族里 8 个问骨架字段；`title_of` / `agenda_of` 问的是文本字段，
   而文本字段的真值就是后端写出来的原文——`render` 把草稿记回 `HybridSpec.written`，
   题目与判别器都拿它比对，所以模型改不了任何答案；
2. **文本不许夹带骨架事实**。标题 / 议题里出现时刻或会议室名一律拒掉
   （写进去就可能与骨架矛盾，而答案只认骨架）。

输入/输出：数据类与纯函数；不含领域词表（词表在 `locales/`，这里只存下标）。
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field
from typing import Any, Final

from jc.knobs import require_pool

# ---------- 维度取值 ----------

MEETING_COUNTS: Final[tuple[int, ...]] = (2, 3, 4)
#: 「日程从几点开始」——控制第一场会的开始时刻，也就控制了 `busy_level` 那道题的答案分布。
OPENINGS: Final[tuple[str, ...]] = ("early", "late_morning", "noon", "afternoon", "evening")
CLASHES: Final[tuple[str, ...]] = ("none", "overlap", "same_room")
LAYOUTS: Final[tuple[str, ...]] = ("bullet", "table")
WEEKDAYS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6, 7)
#: 时长档。两档**不重叠**（`short` 是 1~2 小时、`long` 是 3~4 小时），
#: 这样 `duration_of` 四个选项的出现次数才均匀——两档重叠的话中间那个值会双倍出现。
DURATION_MIXES: Final[tuple[str, ...]] = ("short", "long")

#: 训练侧见过的取值。`test_ood` 只收「至少含一个不在下面集合里的取值」的 case，
#: 留出的是**取值**而不是另一批同分布样本（PLAN 3.3 的值留出）：
#: 版式 `table`（换一种排版）与冲突档 `same_room`（两场会抢同一间房）训练侧一个都不出现。
SEEN_LAYOUTS: Final[frozenset[str]] = frozenset({"bullet"})
SEEN_CLASHES: Final[frozenset[str]] = frozenset({"none", "overlap"})

OPENING_HOURS: Final[dict[str, tuple[int, int]]] = {
    "early": (8, 9),
    "late_morning": (10, 11),
    "noon": (12, 13),
    "afternoon": (14, 16),
    "evening": (17, 18),
}
DUR_RANGES: Final[dict[str, tuple[int, int]]] = {"short": (1, 2), "long": (3, 4)}

#: 除了「最小推进量」，日程最多再多花这么多小时（用来加长时长或留间隔）——
#: 不留余量的话同一档下每条日程都一样紧，反而没了多样性。
SPAN_EXTRA: Final[int] = 2
#: 最后一场会最晚的结束时刻（24 小时制）。
DAY_END: Final[int] = 23
#: `busy_level` 的统计窗口（24 小时制半开区间 `[BUSY_FROM, BUSY_TO)`）与封顶等级。
#: 窗口取 4 小时宽是**实测调出来的**：窗口放宽到 6 小时（12:00–18:00）时「被占满」
#: 太容易，最高档会吃掉 37% 的样本；4 小时窗口下五档是 24% / 13% / 18% / 24% / 21%，
#: 最高的那档不再压倒性占多数（脚本见 `tmp/hyb_sweep.py`，tmp 不入库）。
BUSY_FROM: Final[int] = 12
BUSY_TO: Final[int] = 16
BUSY_CAP: Final[int] = 4

#: 词表容量：spec 只存下标，所以每种语言的会议室 / 主持人词表都不小于这个数（测试锁住）。
ROOM_POOL: Final[int] = 6
LEAD_POOL: Final[int] = 8

#: 写文本时的「主题方向」池：spec 只存**下标**（与问法编号同一套约定），每种语言各有一份
#: 等长的词表（由 `test_locales.py` 锁住）。
#: **来由**（实测）：写文本的提示词里只有「共 N 场会」一个变量，于是同场数的样本提示词
#: **逐字相同**，模型把标题迅速塌成几句套话——真端点 37 条样本的 114 个标题里只有 75 个
#: 不同（`产品需求评审会` 一条就占了 15 个），而议题是 114/114 全不重复。
#: 给每条样本抽一个方向，提示词就不再一样了。
THEME_IDS: Final[tuple[int, ...]] = tuple(range(12))

#: 日期从这一天起往后找：保证「同一天是星期几」由 case 决定，而月份与日子仍然随机。
BASE_DATE: Final[dt.date] = dt.date(2026, 9, 1)


def slot_name(index: int) -> str:
    """输入：会议下标（从 0 开始，= **展示顺序**里的位置）。输出：会议槽名（`m1` 起）。"""
    return f"m{index + 1}"


def weekday_of(date: tuple[int, int, int]) -> int:
    """输入：日期三元组。输出：星期几（ISO，1 = 周一）。"""
    return dt.date(*date).isoweekday()


def intervals_overlap(left: Any, right: Any) -> bool:
    """输入：两场会（只要各自有 `start` / `end` 就行）。输出：它们在时间上是否重叠。

    预期行为：用**对称判据** `left.start < right.end and right.start < left.end`，
    而不是「后一场的开始时刻早于本场的结束」——后者要求列表按开始时刻排序，
    而展示顺序是洗过的（见 `sample_spec`），用它会把不相邻的两场误判成重叠。
    """
    return left.start < right.end and right.start < left.end


def _min_advance(count: int, clash: str, lo_dur: int) -> int:
    """输入：会议场数、冲突档、最短时长。输出：第一场到最后一场的**最小推进量**（小时）。

    预期行为：不重叠档里每一步至少推进「本场时长」；重叠档里被选中的那一步只推进 1 小时
    （下一场落在本场之内），其余步同前。这个值就是「日程至少要铺多长」。
    """
    if clash == "none":
        return (count - 1) * lo_dur
    return (count - 2) * lo_dur + 1


@dataclass(frozen=True)
class HybridCase:
    """一个 case：6 个维度的取值组合（组合数 ~10³，直接全枚举 + 过滤不可解组合）。"""

    meetings: int
    opening: str
    clash: str
    layout: str
    weekday: int
    duration_mix: str

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：{维度名: 取值}，顺序与声明一致（哈希分桶按它算，必须稳定）。"""
        return {
            "meetings": self.meetings,
            "opening": self.opening,
            "clash": self.clash,
            "layout": self.layout,
            "weekday": self.weekday,
            "duration_mix": self.duration_mix,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return (
            f"{self.meetings}场/{self.opening}/{self.clash}/{self.layout}"
            f"/周{self.weekday}/{self.duration_mix}"
        )

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这个 case 里训练侧没见过的维度名（空 = 全部见过）。"""
        checks = (
            ("layout", self.layout, SEEN_LAYOUTS),
            ("clash", self.clash, SEEN_CLASHES),
        )
        return tuple(name for name, value, seen in checks if value not in seen)


def compatible(case: HybridCase) -> bool:
    """输入：一个 case。输出：这个组合写得出来吗。

    预期行为：三条硬约束，都是**组合本身写不出来**（不是随机抽不到）：

    - 1 场会：`title_of` / `room_of` / `duration_of` / `lead_of` / `agenda_of` /
      `earliest_start` 这些题族的选项就是「各场会的取值」，只剩一场时选项退化成一个
      （契约要求 2~255 个，选项描述还要两两不同），`duration_of` 那一族更会直接抛
      `randrange` 空区间。实测 `meeting_counts=(1,)` 的 12 条样本**全被拒**（10 条
      `E_CHOICE_OPTION_COUNT` + 2 条 `E_ASSEMBLE`），所以这个取值直接不生成；
    - `same_room` 要求「两场会抢同一间房」，而 2 场会时整份日程只剩一间会议室，
      `room_of` 的选项会退化成一个（选项描述必须两两不同，否则那道题有两个说得通的答案）；
    - 当天放不下：起点取本档最晚、推进量取最小、最后一场取最短，仍然超过 `DAY_END`
      （例如傍晚 18 点才开始、4 场会每场至少 3 小时，怎么排都会开到第二天）。

    无解的组合既不生成、也不算没覆盖。
    """
    if case.meetings < 2:
        return False
    if case.clash == "same_room" and case.meetings < 3:
        return False
    lo_dur, _ = DUR_RANGES[case.duration_mix]
    need = _min_advance(case.meetings, case.clash, lo_dur)
    return need + lo_dur <= DAY_END - OPENING_HOURS[case.opening][1]


@dataclass(frozen=True)
class HybridConfig:
    """一个切分的采样参数。

    字段：各维度的取值表（留出时收窄）、问法编号池、主题方向编号池、
    `ood_only` 是否只留「含没见过的取值」的 case。
    """

    split: str = "train"
    meeting_counts: tuple[int, ...] = MEETING_COUNTS
    openings: tuple[str, ...] = OPENINGS
    clashes: tuple[str, ...] = CLASHES
    layouts: tuple[str, ...] = LAYOUTS
    weekdays: tuple[int, ...] = WEEKDAYS
    duration_mixes: tuple[str, ...] = DURATION_MIXES
    phrasings: tuple[int, ...] = (0, 1, 2)
    themes: tuple[int, ...] = THEME_IDS
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：空取值表在构造期就报人话，不拖到采样阶段。

        以前空表会活到 `rng.choice`，抛 `Cannot choose from an empty sequence`
        （校验规则见 `jc.knobs`）。
        """
        for field, values in (
            ("meeting_counts", self.meeting_counts),
            ("openings", self.openings),
            ("clashes", self.clashes),
            ("layouts", self.layouts),
            ("weekdays", self.weekdays),
            ("duration_mixes", self.duration_mixes),
            ("phrasings", self.phrasings),
            ("themes", self.themes),
        ):
            require_pool(field, values)


def build_cases(config: HybridConfig) -> list[HybridCase]:
    """枚举 case 矩阵。

    输入：采样参数。输出：`HybridCase` 列表（顺序稳定、可复算）。
    预期行为：全枚举后剔掉无解组合；`ood_only` 时只留「至少含一个训练侧没见过的取值」的 case。
    """
    out: list[HybridCase] = []
    for meetings in config.meeting_counts:
        for opening in config.openings:
            for clash in config.clashes:
                for layout in config.layouts:
                    for weekday in config.weekdays:
                        for duration_mix in config.duration_mixes:
                            case = HybridCase(
                                meetings=meetings,
                                opening=opening,
                                clash=clash,
                                layout=layout,
                                weekday=weekday,
                                duration_mix=duration_mix,
                            )
                            if not compatible(case):
                                continue
                            if config.ood_only and not case.unseen_dims():
                                continue
                            out.append(case)
    return out


@dataclass(frozen=True)
class Meeting:
    """一场会的骨架：开始整点、时长、会议室下标、主持人下标。

    标题与议题**不在这里**——那两段是后端写的，写在 `HybridSpec.written` 里。
    """

    start: int
    dur: int
    room: int
    lead: int

    @property
    def end(self) -> int:
        """输入：无。输出：结束整点。"""
        return self.start + self.dur


@dataclass(frozen=True)
class MeetingText:
    """后端为一场会写的两段文本（标题 + 一句话议题）。"""

    title: str
    agenda: str


@dataclass
class HybridSpec:
    """一条样本的真值规格。

    **本类刻意不是 frozen 的**：`render` 会把后端写出来的标题 / 议题记进 `written`
    （`ask` 与判别器都要用）。管线是并发跑的，所以这里只能挂**这条样本自己的**可变状态——
    绝不能挂到 workflow 实例上（那会变成跨线程共享）。
    """

    case: HybridCase
    split: str
    seq: int
    date: tuple[int, int, int]
    meetings: tuple[Meeting, ...]
    phrasing: int
    theme: int
    written: dict[str, MeetingText] = field(default_factory=dict)

    def slot_names(self) -> tuple[str, ...]:
        """输入：无。输出：会议槽名（`m1` 起，按开始时刻升序）。"""
        return tuple(slot_name(index) for index in range(len(self.meetings)))

    @property
    def text_task_id(self) -> str:
        """输入：无。输出：写文本任务的后端任务 id。

        预期行为：同一次 run 内唯一且可重算（断点续跑按它查进度）；进度日志按 run 目录分文件，
        所以不同 run_seed 之间不会串。
        """
        return f"hybrid-{self.split}-{self.seq:07d}-text"

    @property
    def weekday(self) -> int:
        """输入：无。输出：这一天的星期几（ISO，1 = 周一）。"""
        return weekday_of(self.date)

    @property
    def has_overlap(self) -> bool:
        """输入：无。输出：有没有两场会时间重叠（`no_overlap` 的答案）。

        预期行为：任意两场都查一遍（展示顺序与时间顺序无关，不能只看相邻两场）。
        """
        return any(
            intervals_overlap(left, right)
            for index, left in enumerate(self.meetings)
            for right in self.meetings[index + 1 :]
        )

    @property
    def has_room_clash(self) -> bool:
        """输入：无。输出：有没有两场会同时占用同一间会议室（`room_free` 的答案）。

        预期行为：同一间房 **且** 时间重叠才算冲突——共用一个房间但时间错开是正常的。
        """
        return any(
            left.room == right.room and intervals_overlap(left, right)
            for index, left in enumerate(self.meetings)
            for right in self.meetings[index + 1 :]
        )

    @property
    def room_slots(self) -> tuple[int, ...]:
        """输入：无。输出：日程里出现过的会议室下标（按首次出现顺序去重）。"""
        seen: list[int] = []
        for meeting in self.meetings:
            if meeting.room not in seen:
                seen.append(meeting.room)
        return tuple(seen)

    @property
    def busy_level(self) -> int:
        """输入：无。输出：`busy_level` 那道题的答案。

        预期行为：把每场会的时间区间与窗口 `[BUSY_FROM, BUSY_TO)` 求交，
        数窗口内被覆盖的整点（**去重**，重叠的整点只算一次），封顶 `BUSY_CAP`。
        注意是「区间求交」而不是「按开始时刻筛选」：12:00 开始、13:00 结束的会
        也算占用了窗口里的 12 点。
        """
        hours: set[int] = set()
        for meeting in self.meetings:
            hours.update(range(max(meeting.start, BUSY_FROM), min(meeting.end, BUSY_TO)))
        return min(BUSY_CAP, len(hours))

    def text_of(self, slot: str) -> MeetingText:
        """输入：会议槽名。输出：后端为它写的文本。

        预期行为：还没渲染过就直接报错——`ask` 之前必须先走 `render`，
        否则题目会拿不到文本字段的真值。
        """
        if slot not in self.written:
            raise KeyError(
                f"{slot} 的标题 / 议题还没写：`ask` 之前必须先走 `render`（文本由后端产生）"
            )
        return self.written[slot]


def _date_with_weekday(rng: random.Random, weekday: int) -> tuple[int, int, int]:
    """输入：RNG、目标星期几（ISO）。输出：(年, 月, 日)。

    预期行为：在 `BASE_DATE` 之后的 4 周里随机挑一天，再顺延到目标星期几——
    月份与日子都随机，但星期几只由 case 决定。
    """
    day = BASE_DATE + dt.timedelta(days=rng.randrange(0, 28))
    day += dt.timedelta(days=(weekday - day.isoweekday()) % 7)
    return (day.year, day.month, day.day)


def _advances(rng: random.Random, case: HybridCase, pair: int | None, extra: int) -> list[int]:
    """输入：RNG、case、重叠对下标、余量小时数。输出：每一步的推进量（长度 = 场数 - 1）。

    预期行为：总推进量 = 最小推进量 + `extra`；重叠那一步固定 1 小时，
    其余步把余量随机分掉。**先定推进量再定时长**，
    所以「最后一场会不会开到第二天」在选时长之前就已经定死了，不需要重抽。

    例外：两场会 + 冲突档时只有一步、且那一步固定 1 小时，没有别处能塞余量，
    余量直接丢掉（`compatible` 已保证这种组合排得下）。
    """
    lo_dur, _ = DUR_RANGES[case.duration_mix]
    steps = [index for index in range(case.meetings - 1) if index != pair]
    shares = [0] * len(steps)
    usable = extra if steps else 0
    for _ in range(usable):  # 余量按「小时」随机加在某一步上，够简单也够散
        shares[rng.randrange(len(steps))] += 1
    advances: list[int] = []
    for index in range(case.meetings - 1):
        if index == pair:
            advances.append(1)
            continue
        share = shares[steps.index(index)]
        advances.append(lo_dur + share)
    return advances


def _place(rng: random.Random, case: HybridCase) -> tuple[list[tuple[int, int]], int | None]:
    """输入：RNG、case。输出：([(开始整点, 时长)], 重叠那一对的左下标或 None)。

    预期行为：按 `opening` 选起点、按 `clash` 决定哪一对会议重叠。
    `compatible` 已经保证排得下，所以这里不需要重抽：

    - 开始时刻两两不同（否则 `earliest_start` 有两个正确答案）；
    - 不重叠档：相邻两场满足 `start + dur <= next_start`；
    - 重叠档：被选中的那一对满足 `next_start < start + dur`（所以本场时长必须 > 1 小时），
      其余相邻对仍然不重叠——重叠对按开始时刻排序后必然相邻。
    """
    lo_dur, hi_dur = DUR_RANGES[case.duration_mix]
    pair: int | None = None if case.clash == "none" else rng.randrange(case.meetings - 1)
    lo_first, hi_first = OPENING_HOURS[case.opening]
    need = _min_advance(case.meetings, case.clash, lo_dur)
    # 起点也要给「最小推进量 + 最后一场最短时长」留位置（本档最晚的那个起点可能留不下）。
    hi_first = min(hi_first, DAY_END - need - lo_dur)
    first = rng.randint(lo_first, hi_first)
    # 余量上限还要再让「最后一场最短时长」塞得下，否则最后一场会开到第二天。
    extra = rng.randint(0, min(SPAN_EXTRA, DAY_END - lo_dur - first - need))
    starts = [first]
    advances = _advances(rng, case, pair, extra)
    for step in advances:
        starts.append(starts[-1] + step)
    # 时长：每一步的时长不能超过该步的推进量（否则那一对就不是「不重叠」了）；
    # 最后一场没有推进量要付，只要当天结束前收尾即可；重叠那一场必须 ≥ 2 小时。
    durs: list[int] = []
    for index in range(case.meetings):
        if index == case.meetings - 1:
            top = min(hi_dur, DAY_END - starts[index])
        elif index == pair:
            top = min(hi_dur, DAY_END - starts[index])
        else:
            top = min(hi_dur, advances[index])
        durs.append(max(2, rng.randint(lo_dur, top)) if index == pair else rng.randint(lo_dur, top))
    return list(zip(starts, durs, strict=True)), pair


def sample_spec(
    rng: random.Random, case: HybridCase, split: str, seq: int, phrasing: int, theme: int
) -> HybridSpec:
    """输入：RNG、case、切分名、样本序号、问法编号、主题方向编号。输出：这一条的真值规格。

    预期行为：日期按 `case.weekday` 抽；骨架按 `case` 排布，且**同类取值两两不同**——
    开始时刻两两不同（`earliest_start`）、主持人两两不同、会议室两两不同；
    只有 `same_room` 那一对共用一间房（这正是 `room_free` 要问的）。
    标题与议题留空，等 `render` 阶段由后端补。

    问法编号与主题方向编号都只是**下标**，由各自的词表翻译（问法查 `ask`，
    方向查 `prompt.themes`）——这样 spec 与语言无关，同一份 spec 可以换语言渲染。

    **展示顺序刻意与时间顺序无关**（`rng.sample` 洗过一遍）：真实日程表常常按会议室或
    预约人排，而且不洗的话「哪一场开始得最早」的答案永远是列表第一行——
    那会训练出「答案 = 第一行」的位置捷径，而不是真的去比时间。

    写不出来的组合（没过 `compatible`）直接报错并点名——否则会在排布中途炸成
    `empty range for randrange()`，看的人一头雾水（实测踩过）。
    """
    if not compatible(case):
        raise ValueError(f"这个组合写不出日程（先过 compatible 过滤）：{case.describe()}")
    spans, pair = _place(rng, case)
    count = len(spans)
    shared = pair is not None and case.clash == "same_room"
    rooms = list(rng.sample(range(ROOM_POOL), count - (1 if shared else 0)))
    if shared and pair is not None:
        rooms.insert(pair + 1, rooms[pair])  # 重叠那一对共用一间房
    leads = rng.sample(range(LEAD_POOL), count)
    # 先按时间顺序造好（房间 / 主持人的分配都按下标来），再整体打乱成展示顺序
    chronological = [
        Meeting(start=start, dur=dur, room=rooms[index], lead=leads[index])
        for index, (start, dur) in enumerate(spans)
    ]
    meetings = tuple(chronological[index] for index in rng.sample(range(count), count))
    return HybridSpec(
        case=case,
        split=split,
        seq=seq,
        date=_date_with_weekday(rng, case.weekday),
        meetings=meetings,
        phrasing=phrasing,
        theme=theme,
    )


def spec_to_dict(spec: HybridSpec) -> dict[str, Any]:
    """输入：规格。输出：可 JSON 序列化的摘要（写进 manifest 与调试，不进样本行）。"""
    return {
        "case": spec.case.describe(),
        "unseen_dims": list(spec.case.unseen_dims()),
        "split": spec.split,
        "seq": spec.seq,
        "date": list(spec.date),
        "weekday": spec.weekday,
        "meetings": [
            {"start": item.start, "dur": item.dur, "room": item.room, "lead": item.lead}
            for item in spec.meetings
        ],
        "written": {
            slot: {"title": text.title, "agenda": text.agenda}
            for slot, text in spec.written.items()
        },
    }
