"""混合式 workflow 的渲染：骨架由构造器给，**文本由后端补**。

两件事各占一半：

1. **骨架 → 文本**（`build`）：把 `HybridSpec` 里的日期 / 时刻 / 时长 / 会议室 / 主持人
   与后端写好的标题 / 议题拼成一份日程（`bullet` 与 `table` 两种版式）；
2. **文本 → 骨架**（`parse_state`）：把日程读回结构，供往返判别器逐字比对。
   反解是**语言无关**的——时刻、竖线、缩进都是结构，标签文本靠 `locales.merged_labels()`。

后端交互只有一步：给它「共几场会 + 规则」，它回每场会的标题与议题。
草稿的**结构校验**在这里（`check_draft`）：场数对不对、有没有夹带骨架事实、
标题 / 议题是不是两两不同——不合格直接抛 `DraftInvalidError`，
这样它会被记成「草稿不合格」而不是 provider 故障，不会误伤熔断。

输入/输出：除 `build_task` 产出的任务对象外，本模块不发网络请求、不碰全局状态。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

from jc.backend.types import DraftInvalidError, GenerationTask
from pydantic import BaseModel, Field

from . import locales
from .spec import HybridSpec, MeetingText

# ---------- 文本卫生：标题 / 议题里不许出现骨架事实 ----------

#: 时刻与日期：`14:00`、`9 点`、`下午 3 时`、`9 月 3 日`、`2026 年`。
CLOCK_OR_DATE: Final[re.Pattern[str]] = re.compile(
    r"\d{1,2}\s*[:：]\s*\d{2}|\d{1,2}\s*[点時时分]|\d{4}\s*年|\d{1,2}\s*[月日号]"
)
#: 中文的星期说法（`周三` / `星期三` / `礼拜天`）；英文的星期名在词表里查。
WEEKDAY_CJK: Final[re.Pattern[str]] = re.compile(r"[周週]\s*[一二三四五六日天末]|星期|礼拜")
#: 表格版式用竖线分列、一行一场会，所以文本里不能有竖线，也不能换行。
COLUMN_BREAKER: Final[str] = "|"

#: `bullet` 版式里，一场会的起始行：`1. 08:00–10:00 · 海棠厅 · 王敏`。
BULLET_START: Final[re.Pattern[str]] = re.compile(
    r"^\s*(\d+)\s*[.、]\s*(\d{1,2}):(\d{2})\s*[–\-~—]\s*(\d{1,2}):(\d{2})\s*(.*)$"
)
#: 分隔符：` · `（渲染时用的那个），以及几个近义写法，反解时一并认。
SEPARATOR: Final[re.Pattern[str]] = re.compile(r"\s*[·|;；]\s*")
#: 文本字段行：`   标题：季度预算评审` / `   Title: Quarterly review`。
LABEL_LINE: Final[re.Pattern[str]] = re.compile(r"^\s*(.+?)\s*[:：]\s*(.*)$")
#: 日期：渲染用的是 ISO 写法（两种语言一致），反解只认这一种。
DATE_PATTERN: Final[re.Pattern[str]] = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


class MeetingTextDraft(BaseModel):
    """后端为**一场**会写的两段文本。"""

    title: str = Field(min_length=1, description="会议标题，4~12 个字，像真实的会议名")
    agenda: str = Field(min_length=1, description="会议议题，一句话，10~30 个字")


class ScheduleDraft(BaseModel):
    """后端要写的全部文本：按日程顺序，一场会一条。

    场数必须与提示词里说的一致——少了 / 多了都算草稿不合格（`check_draft` 查）。
    后端**只写文本**：日期、时刻、时长、会议室、主持人都不在它的输出里。
    """

    meetings: list[MeetingTextDraft] = Field(
        description="按日程顺序，每场会议一条（条数必须等于提示词里的场数）"
    )


@dataclass(frozen=True)
class ParsedMeeting:
    """从日程文本里读回来的一场会（反解结果，也是「期望值」的形态）。"""

    start: int
    end: int
    room: str
    lead: str
    title: str
    agenda: str


@dataclass(frozen=True)
class ParsedSchedule:
    """从日程文本里读回来的整份日程。"""

    date: tuple[int, int, int] | None
    meetings: tuple[ParsedMeeting, ...]


# ---------- 槽 → 文本（唯一出口） ----------


def time_text(hour: int, loc: dict[str, Any]) -> str:
    """输入：整点小时数、语言字典。输出：时刻文本（如 `08:00`）。"""
    return loc["time"]["tpl"].format(h=hour)


def date_text(date: tuple[int, int, int], loc: dict[str, Any]) -> str:
    """输入：(年, 月, 日)、语言字典。输出：日期文本（ISO 写法，两种语言一致）。"""
    year, month, day = date
    return loc["date"]["tpl"].format(y=year, m=month, d=day)


def duration_text(hours: int, loc: dict[str, Any]) -> str:
    """输入：时长（小时）、语言字典。输出：时长文本（如 `2 小时` / `2 hours`）。

    预期行为：下标 = 时长 - 1；越界直接报错——时长只能取 `spec.DUR_RANGES` 覆盖的 1~4。
    """
    items = loc["word"]["duration"]["items"]
    if not 1 <= hours <= len(items):
        raise IndexError(f"时长 {hours} 没有对应文案（词表 {len(items)} 条）")
    return items[hours - 1]


def weekday_text(weekday: int, loc: dict[str, Any]) -> str:
    """输入：星期几（ISO，1 = 周一）、语言字典。输出：星期文本（如 `星期三`）。"""
    names = loc["word"]["weekday"]["names"]
    if not 1 <= weekday <= len(names):
        raise IndexError(f"星期几 {weekday} 越界（词表 {len(names)} 条）")
    return names[weekday - 1]


def room_text(spec: HybridSpec, index: int, loc: dict[str, Any]) -> str:
    """输入：规格、会议下标、语言字典。输出：那场会的会议室名。"""
    return locales.word_of(loc, "room", spec.meetings[index].room)


def lead_text(spec: HybridSpec, index: int, loc: dict[str, Any]) -> str:
    """输入：规格、会议下标、语言字典。输出：那场会的主持人姓名。"""
    return locales.word_of(loc, "lead", spec.meetings[index].lead)


def meeting_text(spec: HybridSpec, index: int, loc: dict[str, Any]) -> MeetingText:
    """输入：规格、会议下标、语言字典。输出：后端为那场会写的标题 + 议题。

    预期行为：`spec.text_of` 找不到就直接报错——`ask` 之前必须先走 `render`，
    否则 `title_of` / `agenda_of` 会拿不到真值。
    """
    _ = loc  # 文本本身与语言无关（是后端按该语言写的），这里只为签名一致
    return spec.text_of(spec.slot_names()[index])


def expected_meetings(spec: HybridSpec, loc: dict[str, Any]) -> list[ParsedMeeting]:
    """输入：规格、语言字典。输出：这份日程**应该**被反解成什么（往返判别器拿它比对）。"""
    return [
        ParsedMeeting(
            start=meeting.start,
            end=meeting.end,
            room=room_text(spec, index, loc),
            lead=lead_text(spec, index, loc),
            title=meeting_text(spec, index, loc).title,
            agenda=meeting_text(spec, index, loc).agenda,
        )
        for index, meeting in enumerate(spec.meetings)
    ]


# ---------- 骨架 → 文本 ----------


def _span_text(meeting: Any, loc: dict[str, Any]) -> str:
    """输入：一场会的骨架、语言字典。输出：`08:00–10:00` 这样的时间跨度文本。"""
    return loc["meta"]["span"].format(
        start=time_text(meeting.start, loc), end=time_text(meeting.end, loc)
    )


def _bullet_lines(spec: HybridSpec, loc: dict[str, Any]) -> list[str]:
    """输入：规格、语言字典。输出：`bullet` 版式的若干行（一行一场会 + 两行文本）。"""
    lines: list[str] = []
    for index, meeting in enumerate(spec.meetings):
        lines.append(
            loc["bullet"]["line"].format(
                n=index + 1,
                span=_span_text(meeting, loc),
                sep=loc["meta"]["sep"],
                room=room_text(spec, index, loc),
                lead=lead_text(spec, index, loc),
            )
        )
        text = meeting_text(spec, index, loc)
        lines.append(f"   {loc['bullet']['title_label']}{loc['meta']['colon']}{text.title}")
        lines.append(f"   {loc['bullet']['agenda_label']}{loc['meta']['colon']}{text.agenda}")
    return lines


def _table_lines(spec: HybridSpec, loc: dict[str, Any]) -> list[str]:
    """输入：规格、语言字典。输出：`table` 版式的若干行（表头 + 分隔行 + 每场一行）。"""
    columns = [
        loc["column"][name]
        for name in ("n", "start", "end", "room", "lead", "title", "agenda")
    ]
    rows: list[list[str]] = [columns]
    for index, meeting in enumerate(spec.meetings):
        text = meeting_text(spec, index, loc)
        rows.append(
            [
                str(index + 1),
                time_text(meeting.start, loc),
                time_text(meeting.end, loc),
                room_text(spec, index, loc),
                lead_text(spec, index, loc),
                text.title,
                text.agenda,
            ]
        )
    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join("---" for _ in rows[0]) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return lines


def build(spec: HybridSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：整份日程文本（契约允许的 state）。

    预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
    版式只改排版，不改任何数值；文本字段必须先由后端写好（`render` 阶段），
    否则这里直接报错而不是写出半份日程。
    """
    header = f"{loc['meta']['title']}{loc['meta']['sep']}{date_text(spec.date, loc)}"
    body = _table_lines(spec, loc) if spec.case.layout == "table" else _bullet_lines(spec, loc)
    return "\n".join([header, "", *body])


# ---------- 文本 → 骨架 ----------


def _hour_of(text: str) -> int | None:
    """输入：`08:00` 这样的时刻文本。输出：小时数，或 None（认不出来）。"""
    match = re.fullmatch(r"\s*(\d{1,2})\s*[:：]\s*(\d{2})\s*", text)
    return int(match.group(1)) if match else None


def _parse_table(lines: list[str]) -> list[ParsedMeeting]:
    """输入：日程文本的行。输出：表格版式读出来的会议列表。"""
    out: list[ParsedMeeting] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith(COLUMN_BREAKER):
            continue
        cells = [cell.strip() for cell in stripped.strip(COLUMN_BREAKER).split(COLUMN_BREAKER)]
        if len(cells) != 7:
            continue
        start, end = _hour_of(cells[1]), _hour_of(cells[2])
        if start is None or end is None:
            continue  # 表头与分隔行没有时刻，跳过
        out.append(
            ParsedMeeting(
                start=start, end=end, room=cells[3], lead=cells[4], title=cells[5], agenda=cells[6]
            )
        )
    return out


def _parse_bullet(lines: list[str]) -> list[ParsedMeeting]:
    """输入：日程文本的行。输出：列表版式读出来的会议列表。"""
    out: list[ParsedMeeting] = []
    texts: dict[str, str] = {}
    for line in lines:
        match = BULLET_START.match(line)
        if match:
            tail = [part for part in SEPARATOR.split(match.group(6)) if part]
            out.append(
                ParsedMeeting(
                    start=int(match.group(2)),
                    end=int(match.group(4)),
                    room=tail[0] if tail else "",
                    lead=tail[1] if len(tail) > 1 else "",
                    title="",
                    agenda="",
                )
            )
            continue
        if not out:
            continue
        labelled = LABEL_LINE.match(line)
        if labelled is None:
            continue
        slot = locales.merged_labels().get(labelled.group(1).strip())
        if slot is not None:
            texts[slot] = labelled.group(2).strip()
        if len(texts) == 2:
            last = out[-1]
            out[-1] = ParsedMeeting(
                start=last.start,
                end=last.end,
                room=last.room,
                lead=last.lead,
                title=texts["title"],
                agenda=texts["agenda"],
            )
            texts = {}
    return out


def parse_state(state: Any) -> ParsedSchedule:
    """输入：state。输出：从日程文本里读回来的结构。

    预期行为：**语言无关**——版式靠结构认（竖线 / 行首编号 + 时刻），
    文本字段靠 `locales.merged_labels()` 认标签。非字符串 state 一律当成空日程，
    让判别器判「与真值不符」而不是在这里崩。
    """
    if not isinstance(state, str):
        return ParsedSchedule(None, ())
    lines = state.splitlines()
    date_match = DATE_PATTERN.search(lines[0]) if lines else None
    date = (
        (int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3)))
        if date_match
        else None
    )
    is_table = any(line.strip().startswith(COLUMN_BREAKER) for line in lines)
    return ParsedSchedule(date, tuple(_parse_table(lines) if is_table else _parse_bullet(lines)))


# ---------- 后端交互 ----------


def build_prompt(spec: HybridSpec, loc: dict[str, Any]) -> str:
    """组装写文本的提示词。

    输入：规格、语言字典。输出：提示词字符串。
    预期行为：**只给场数与规则**，不给任何骨架事实（时刻 / 会议室 / 主持人都不出现）——
    给了就是邀请模型把骨架写进文本，而文本里夹带骨架事实一律会被 `check_draft` 拒掉。
    """
    lines = [loc["prompt"]["intro"], ""]
    lines += [
        "- " + rule.format(count=len(spec.meetings), language=loc["language_name"])
        for rule in loc["prompt"]["rules"]
    ]
    return "\n".join(lines)


def build_task(spec: HybridSpec, loc: dict[str, Any], task_id: str) -> GenerationTask:
    """输入：规格、语言字典、任务 id。输出：后端任务。

    预期行为：`spec` 里只放**场数**（不是骨架）——后端看不到答案，也没法把答案写进文本。
    """
    return GenerationTask(
        task_id=task_id,
        kind="text",  # 不是题型：这一步只产文本，题型的编号与 type 由组装器决定
        domain="schedule",
        prompt=build_prompt(spec, loc),
        response_model=ScheduleDraft,
        spec={"count": len(spec.meetings)},
    )


def text_problems(text: str, label: str) -> list[str]:
    """输入：一段文本、它的名字。输出：问题列表（空 = 这段文本干净）。

    预期行为：空 / 带骨架事实（时刻、日期、星期、会议室名）/ 带分列符或换行，都算问题。
    口径压到最窄的那一类——宁可漏检也不误判，误判会直接拒掉好数据。
    """
    problems: list[str] = []
    stripped = text.strip()
    if not stripped:
        problems.append(f"{label} 是空的")
        return problems
    found = CLOCK_OR_DATE.search(stripped)
    if found:
        problems.append(f"{label} 里夹带了时刻或日期：{found.group(0)!r}")
    weekday = WEEKDAY_CJK.search(stripped)
    if weekday:
        problems.append(f"{label} 里夹带了星期：{weekday.group(0)!r}")
    lowered = stripped.lower()
    for word in sorted(locales.room_words()):
        if word.lower() in lowered:
            problems.append(f"{label} 里夹带了会议室名：{word!r}")
    for word in sorted(locales.weekday_words()):
        if word.lower() in lowered:
            problems.append(f"{label} 里夹带了星期：{word!r}")
    if COLUMN_BREAKER in stripped:
        problems.append(f"{label} 里有竖线（会破坏表格版式）")
    if len(stripped.splitlines()) > 1:
        problems.append(f"{label} 里有多行（一场会的文本只能占一行）")
    return problems


def _duplicate_problems(written: dict[str, MeetingText]) -> list[str]:
    """输入：{槽名: 文本}。输出：标题 / 议题重复的问题列表（空 = 两两不同）。"""
    problems: list[str] = []
    for field_name, values in (
        ("标题", [text.title for text in written.values()]),
        ("议题", [text.agenda for text in written.values()]),
    ):
        if len(set(values)) != len(values):
            problems.append(f"{field_name}有重复：{values}")
    return problems


def check_draft(
    spec: HybridSpec, loc: dict[str, Any], data: dict[str, Any]
) -> dict[str, MeetingText]:
    """输入：规格、语言字典、后端草稿。输出：{槽名: 文本}。

    预期行为：任何不合格都抛 `DraftInvalidError`（人话说明），调用方据此拒绝这一条。
    四条判据：场数对得上、两段文本都干净、标题两两不同、议题两两不同——
    后两条是**题目本身的要求**：`title_of` 的选项就是全部标题，
    两个标题一样的话那道题就有两个正确答案。
    """
    raw = data.get("meetings")
    if not isinstance(raw, list):
        raise DraftInvalidError(f"草稿里没有 meetings 数组：{data!r}")
    if len(raw) != len(spec.meetings):
        raise DraftInvalidError(f"草稿给了 {len(raw)} 场会，日程里有 {len(spec.meetings)} 场")
    problems: list[str] = []
    written: dict[str, MeetingText] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            problems.append(f"第 {index + 1} 场不是对象：{item!r}")
            continue
        title = str(item.get("title", ""))
        agenda = str(item.get("agenda", ""))
        problems += text_problems(title, f"第 {index + 1} 场的标题")
        problems += text_problems(agenda, f"第 {index + 1} 场的议题")
        written[spec.slot_names()[index]] = MeetingText(title=title.strip(), agenda=agenda.strip())
    if not problems:
        problems += _duplicate_problems(written)
    if problems:
        raise DraftInvalidError("；".join(problems[:5]))
    _ = loc  # 校验不依赖语言：词表检查用的是各语言合并后的全集
    return written
