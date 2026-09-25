"""混合式 workflow 的题族：把骨架 + 后端文本翻成契约形态的题目。

输入：`HybridSpec`、语言字典、问法编号、RNG。输出：`LabeledQuestion` 列表。

十个题族分两半（这正是「混合式」的意思）：

- **八族问骨架**（`start` / `dur` / `room` / `lead` / 日期 / 重叠 / 房间冲突 / 繁忙度）——
  答案由构造器算出来，模型改不了；
- **两族问文本**（`title_of` / `agenda_of`）——答案就是后端写出来的原文，
  所以 `title_of` 的选项是**全部标题**：两个标题一样的话那道题就有两个正确答案，
  构造时直接报错（渲染阶段已经拦过一道，这里是第二道）。

约定：
- choice 的选项名是**稳定的键**（槽名 `m1`、时长 `h2`、房间 `r1`、标题 `t1`、星期 `mon`），
  描述才是给人看的文本；编号由组装器最后一步映射；
- 同一题里选项描述必须两两不同，否则报错（构造 bug 要在生成时就炸掉）。
"""

from __future__ import annotations

import random
from typing import Any, Final

from jc.workflow import LabeledQuestion

from . import locales, render
from .spec import HybridSpec, slot_name

#: 题族顺序：先定位（哪一场 / 几点 / 多久），再核对文本，再算一致性，最后打分。
CHOICE_FAMILIES: Final[tuple[str, ...]] = (
    "earliest_start",
    "meeting_day",
    "duration_of",
    "room_of",
    "lead_of",
    "title_of",
    "agenda_of",
)
NOUL_FAMILIES: Final[tuple[str, ...]] = ("no_overlap", "room_free")
SCORE_FAMILIES: Final[tuple[str, ...]] = ("busy_level",)
FAMILIES: Final[tuple[str, ...]] = CHOICE_FAMILIES + NOUL_FAMILIES + SCORE_FAMILIES

#: 题面里带「第 {n} 场」的题族——反解靠它把题面映射回会议槽。
SLOT_FAMILIES: Final[tuple[str, ...]] = (
    "duration_of",
    "room_of",
    "lead_of",
    "title_of",
    "agenda_of",
)
#: 题面不带参数的题族（`locate` 直接对整句）。
PLAIN_ASK_FAMILIES: Final[tuple[str, ...]] = ("earliest_start", "meeting_day")

#: 星期的选项名，顺序与 `word.weekday.names` 一致（下标 = 星期几 - 1）。
WEEKDAY_KEYS: Final[tuple[str, ...]] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
#: 时长选项名，下标 = 时长 - 1；选项固定 1~4 小时，短时长档下后两个是干扰项。
DURATION_KEYS: Final[tuple[str, ...]] = ("h1", "h2", "h3", "h4")


def duration_key(hours: int) -> str:
    """输入：时长（小时）。输出：该时长的选项名（`h2`）。"""
    return f"h{hours}"


def room_key(room_index: int) -> str:
    """输入：会议室下标。输出：该会议室的选项名（`r1`）。"""
    return f"r{room_index + 1}"


def title_key(index: int) -> str:
    """输入：会议下标。输出：该会议标题的选项名（`t1`）。"""
    return f"t{index + 1}"


def agenda_key(index: int) -> str:
    """输入：会议下标。输出：该会议议题的选项名（`a1`）。"""
    return f"a{index + 1}"


def _choice(
    family: str, instructions: str, options: dict[str, str], answer: str
) -> LabeledQuestion:
    """输入：题族、题面、选项、金答案（选项名）。输出：一道 choice 题。

    预期行为：选项描述重复、答案不在选项里，都直接报错——那是构造 bug，
    必须在生成时就炸掉而不是产出一条有两个正确答案的坏数据。
    """
    if len(set(options.values())) != len(options):
        raise ValueError(f"{family} 的选项描述重复：{options}")
    if answer not in options:
        raise ValueError(f"{family} 的答案 {answer!r} 不在选项里：{sorted(options)}")
    question = {"type": "choice", "instructions": instructions, "criteria": options}
    return LabeledQuestion(family, question, answer)


def _noul(family: str, loc: dict[str, Any], phrasing: int, answer: bool) -> LabeledQuestion:
    """输入：题族、语言字典、问法编号、金答案。输出：一道 noul 题（不打乱）。"""
    question = {
        "type": "noul",
        "instructions": locales.pick(loc["fam"][family]["q"], phrasing),
        "criteria": {
            "true": loc["fam"][family]["criteria"]["true"],
            "false": loc["fam"][family]["criteria"]["false"],
        },
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def _score(family: str, loc: dict[str, Any], phrasing: int, answer: int) -> LabeledQuestion:
    """输入：题族、语言字典、问法编号、金答案（等级下标）。输出：一道 score 题。"""
    question = {
        "type": "score",
        "instructions": locales.pick(loc["fam"][family]["q"], phrasing),
        "criteria": list(loc["fam"][family]["levels"]),
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def _ask_slot(family: str, loc: dict[str, Any], phrasing: int, index: int) -> str:
    """输入：带「第 n 场」的题族、语言字典、问法编号、会议下标。输出：题面。"""
    return locales.pick(loc["ask"][family]["q"], phrasing).format(n=index + 1)


def build(
    spec: HybridSpec, loc: dict[str, Any], phrasing: int, rng: random.Random
) -> list[LabeledQuestion]:
    """输入：规格、语言字典、问法编号、RNG。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：choice 的答案一律是**选项名**（组装器最后一步才映射成编号）；
    noul 的答案是布尔；score 的答案是等级下标。
    「问哪一场会」由 RNG 决定，所以同一题族的答案每份日程都不同，
    逼着模型真的去读日程而不是背位置。
    """
    slots = spec.slot_names()
    count = len(slots)
    out: list[LabeledQuestion] = []
    for family in FAMILIES:
        if family == "earliest_start":
            options = {
                slot: render.time_text(meeting.start, loc)
                for slot, meeting in zip(slots, spec.meetings, strict=True)
            }
            earliest = min(range(count), key=lambda index: spec.meetings[index].start)
            out.append(
                _choice(
                    family,
                    locales.pick(loc["ask"][family]["q"], phrasing),
                    options,
                    slots[earliest],
                )
            )
        elif family == "meeting_day":
            options = {
                key: render.weekday_text(index + 1, loc)
                for index, key in enumerate(WEEKDAY_KEYS)
            }
            out.append(
                _choice(
                    family,
                    locales.pick(loc["ask"][family]["q"], phrasing),
                    options,
                    WEEKDAY_KEYS[spec.weekday - 1],
                )
            )
        elif family == "duration_of":
            index = rng.randrange(count)
            options = {
                key: render.duration_text(hours, loc)
                for hours, key in enumerate(DURATION_KEYS, start=1)
            }
            out.append(
                _choice(
                    family,
                    _ask_slot(family, loc, phrasing, index),
                    options,
                    duration_key(spec.meetings[index].dur),
                )
            )
        elif family == "room_of":
            index = rng.randrange(count)
            options = {
                room_key(room): locales.word_of(loc, "room", room) for room in spec.room_slots
            }
            out.append(
                _choice(
                    family,
                    _ask_slot(family, loc, phrasing, index),
                    options,
                    room_key(spec.meetings[index].room),
                )
            )
        elif family == "lead_of":
            index = rng.randrange(count)
            options = {
                slot: render.lead_text(spec, position, loc)
                for position, slot in enumerate(slots)
            }
            out.append(
                _choice(family, _ask_slot(family, loc, phrasing, index), options, slots[index])
            )
        elif family in ("title_of", "agenda_of"):
            index = rng.randrange(count)
            texts = [render.meeting_text(spec, position, loc) for position in range(count)]
            keyed = title_key if family == "title_of" else agenda_key
            field = "title" if family == "title_of" else "agenda"
            options = {
                keyed(position): getattr(text, field) for position, text in enumerate(texts)
            }
            out.append(
                _choice(
                    family,
                    _ask_slot(family, loc, phrasing, index),
                    options,
                    keyed(index),
                )
            )
        elif family == "no_overlap":
            out.append(_noul(family, loc, phrasing, not spec.has_overlap))
        elif family == "room_free":
            out.append(_noul(family, loc, phrasing, not spec.has_room_clash))
        else:
            out.append(_score(family, loc, phrasing, spec.busy_level))
    return out


def locate(instructions: str, loc: dict[str, Any], meeting_count: int) -> tuple[str, str] | None:
    """输入：题面、语言字典、会议场数。输出：(题族, 目标槽)，或 None（认不出来）。

    预期行为：把题面与语言文件里的问法逐条对。判别器靠它**从题面反查**「问的是哪一场会」，
    从而只依赖渲染出来的文本与题面本身，不依赖 spec（否则就是拿答案验证答案）。
    带「第 {n} 场」的题族要按场数展开模板再比；其余题族的 `target` 是空串。
    """
    for family in PLAIN_ASK_FAMILIES:
        if instructions in loc["ask"][family]["q"]:
            return (family, "")
    for family in SLOT_FAMILIES:
        for index in range(meeting_count):
            if instructions in [text.format(n=index + 1) for text in loc["ask"][family]["q"]]:
                return (family, slot_name(index))
    for family in NOUL_FAMILIES + SCORE_FAMILIES:
        if instructions in loc["fam"][family]["q"]:
            return (family, "")
    return None
