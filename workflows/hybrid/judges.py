"""混合式 workflow 的判别器：三个**零 token** 的规则判别器。

- `SkeletonJudge`：把渲染出来的日程反解回结构，与真值骨架**逐字段**比对
  （抓渲染错位、数值写错、场数不对）；
- `AnswerJudge`：只看日程文本与题面，**独立重算**每道题的答案再与数据里的答案比对
  （抓答案算错、选项与日程对不上）。它不看 spec——否则就是「拿答案验证答案」；
- `TextJudge`：文本卫生与必填（标题 / 议题非空、不夹带骨架事实、两两不同）。

输入/输出：只出 `Verdict`，不改样本。对不上就判不通过——宁可少要几条，
也不要一条「题面和日程对不上」的数据。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from dataclasses import dataclass


from jc.judge import Verdict

from . import locales, questions, render
from . import spec as spec_mod
from .spec import HybridSpec


def skeleton_diff(
    spec: HybridSpec, parsed: render.ParsedSchedule, loc: dict[str, JsonValue]
) -> list[str]:
    """输入：真值规格、反解结果、语言字典。输出：差异列表（空 = 一致）。

    预期行为：日期与每一场会都逐字段比（时刻、时长、会议室名、主持人名、标题、议题），
    场数不同也算差异——少写一场会让「第 n 场」这类题面指向错位。
    """
    diff: list[str] = []
    if parsed.date != spec.date:
        diff.append(f"日期：渲染 {parsed.date} vs 期望 {spec.date}")
    want = render.expected_meetings(spec, loc)
    if len(parsed.meetings) != len(want):
        diff.append(f"场数：渲染 {len(parsed.meetings)} vs 期望 {len(want)}")
    for index, expected in enumerate(want):
        if index >= len(parsed.meetings):
            break
        got = parsed.meetings[index]
        for field in ("start", "end", "room", "lead", "title", "agenda"):
            if getattr(got, field) != getattr(expected, field):
                diff.append(
                    f"第 {index + 1} 场的 {field}：渲染 {getattr(got, field)!r} "
                    f"vs 期望 {getattr(expected, field)!r}"
                )
    return diff


@dataclass(frozen=True)
class SkeletonJudge:
    """结构判别器：渲染文本反解后必须与真值骨架完全一致。"""

    name: str = "skeleton"
    locale: str = "zh"

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：没有 spec 就跳过（不误判）。"""
        if not isinstance(spec, HybridSpec):
            return Verdict(True, reason="没有 spec，跳过结构判别", raw={"skipped": True})
        loc = locales.load(self.locale)
        parsed = render.parse_state(sample["state"])
        diff = skeleton_diff(spec, parsed, loc)
        return Verdict(not diff, reason="; ".join(diff[:5]), raw={"diff": diff})


# ---------- 只看文本的答案重算 ----------


def _overlap(parsed: render.ParsedSchedule) -> bool:
    """输入：反解结果。输出：有没有两场会时间重叠。"""
    meetings = parsed.meetings
    return any(
        spec_mod.intervals_overlap(left, right)
        for index, left in enumerate(meetings)
        for right in meetings[index + 1 :]
    )


def _room_clash(parsed: render.ParsedSchedule) -> bool:
    """输入：反解结果。输出：有没有两场会同时占用同一间会议室。"""
    meetings = parsed.meetings
    return any(
        left.room == right.room and spec_mod.intervals_overlap(left, right)
        for index, left in enumerate(meetings)
        for right in meetings[index + 1 :]
    )


def _busy_level(parsed: render.ParsedSchedule) -> int:
    """输入：反解结果。输出：`[BUSY_FROM, BUSY_TO)` 窗口内被覆盖的整点数（封顶）。"""
    hours: set[int] = set()
    for meeting in parsed.meetings:
        hours.update(
            range(max(meeting.start, spec_mod.BUSY_FROM), min(meeting.end, spec_mod.BUSY_TO))
        )
    return min(spec_mod.BUSY_CAP, len(hours))


def _room_index(parsed: render.ParsedSchedule, index: int, loc: dict[str, JsonValue]) -> int | None:
    """输入：反解结果、会议下标、语言字典。输出：那场会会议室名的词表下标，或 None。"""
    names = loc["word"]["room"]["items"]
    try:
        return names.index(parsed.meetings[index].room)
    except ValueError:
        return None


def recompute_choice(
    family: str, target: str, parsed: render.ParsedSchedule, loc: dict[str, JsonValue]
) -> tuple[str | None, list[str]]:
    """输入：题族、目标槽、反解结果、语言字典。输出：(期望的选项名, 问题列表)。

    预期行为：期望值只能从**渲染出来的文本**推出来。推不出来（文本缺字段、会议室名不认识）
    时返回 None 并把原因写进问题列表，让调用方判不通过而不是瞎猜。
    """
    meetings = parsed.meetings
    index = 0
    if target:
        try:
            index = int(target[1:]) - 1
        except ValueError:
            return None, [f"目标槽 {target!r} 认不出来"]
        if not 0 <= index < len(meetings):
            return None, [f"目标槽 {target!r} 超出日程的 {len(meetings)} 场会"]
    if family == "earliest_start":
        if not meetings:
            return None, ["日程里一场会都没有"]
        earliest = min(range(len(meetings)), key=lambda position: meetings[position].start)
        return spec_mod.slot_name(earliest), []
    if family == "meeting_day":
        if parsed.date is None:
            return None, ["日程里读不出日期"]
        return questions.WEEKDAY_KEYS[spec_mod.weekday_of(parsed.date) - 1], []
    if family == "duration_of":
        return questions.duration_key(meetings[index].end - meetings[index].start), []
    if family == "room_of":
        room = _room_index(parsed, index, loc)
        if room is None:
            return None, [f"会议室名 {meetings[index].room!r} 不在词表里"]
        return questions.room_key(room), []
    if family == "lead_of":
        return target, []
    if family == "title_of":
        return questions.title_key(index), []
    return questions.agenda_key(index), []


def _choice_problems(
    qid: str,
    question: dict[str, JsonValue],
    answer: dict[str, JsonValue],
    parsed: render.ParsedSchedule,
    loc: dict[str, JsonValue],
) -> list[str]:
    """输入：题号、题目、答案、反解结果、语言字典。输出：问题列表（空 = 对得上）。

    预期行为：除了「答案指向的选项」，还要查那个选项的**描述**是否就是日程里那一场的值——
    只验编号会漏掉「选项描述写错」，那会让题目本身有歧义。
    """
    found = questions.locate(question["instructions"], loc, len(parsed.meetings))
    if found is None:
        return [f"{qid}: 认不出题面（{question['instructions']!r}）"]
    family, target = found
    keys = list(question["criteria"])
    index = answer["choice"]
    if not 0 <= index < len(keys):
        return [f"{qid}: 答案编号越界（{index}）"]
    chosen = keys[index]
    expected, problems = recompute_choice(family, target, parsed, loc)
    if problems:
        return [f"{qid}: {problem}" for problem in problems]
    if chosen != expected:
        return [f"{qid}: 答案指向 {chosen}，按日程重算是 {expected}"]
    return []


@dataclass(frozen=True)
class AnswerJudge:
    """答案判别器：只读日程文本与题面，独立重算金答案再逐题比对。"""

    name: str = "answers"
    locale: str = "zh"
    max_problems: int = 5

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值 spec（不用）。输出：Verdict。预期行为：对不上就判不通过。"""
        del spec  # 刻意不用：用了就是拿答案验证答案
        loc = locales.load(self.locale)
        parsed = render.parse_state(sample["state"])
        problems = self._check(sample, parsed, loc)
        return Verdict(
            not problems,
            reason="; ".join(problems[: self.max_problems]),
            raw={"problems": problems},
        )

    def _check(
        self, sample: dict[str, JsonValue], parsed: render.ParsedSchedule, loc: dict[str, JsonValue]
    ) -> list[str]:
        """输入：样本、反解结果、语言字典。输出：问题列表（空 = 全部对得上）。"""
        problems: list[str] = []
        for qid, question in sample["questions"].items():
            answer = sample["answers"][qid]
            if question["type"] == "choice":
                problems += _choice_problems(qid, question, answer, parsed, loc)
                continue
            found = questions.locate(question["instructions"], loc, len(parsed.meetings))
            if found is None:
                problems.append(f"{qid}: 认不出题面（{question['instructions']!r}）")
                continue
            family = found[0]
            if family == "no_overlap":
                want = not _overlap(parsed)
            elif family == "room_free":
                want = not _room_clash(parsed)
            else:
                want = _busy_level(parsed)
            got = answer["noul"] if question["type"] == "noul" else answer["score"]
            if got != want:
                problems.append(f"{qid}: 答案是 {got}，按日程重算是 {want}")
        return problems


@dataclass(frozen=True)
class TextJudge:
    """文本判别器：标题 / 议题必须齐、干净、两两不同（零 token）。

    渲染阶段（`render.check_draft`）已经拦过一道，这里再查一次**渲染出来的 state**——
    两道查的是不同东西：前者查后端草稿，后者查最终产物，能抓出渲染阶段的 bug。
    """

    name: str = "text"
    locale: str = "zh"
    max_problems: int = 5

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值 spec（不用）。输出：Verdict。预期行为：不干净就判不通过。"""
        del spec
        parsed = render.parse_state(sample["state"])
        problems = self._check(parsed)
        return Verdict(
            not problems,
            reason="; ".join(problems[: self.max_problems]),
            raw={"problems": problems},
        )

    @staticmethod
    def _check(parsed: render.ParsedSchedule) -> list[str]:
        """输入：反解结果。输出：问题列表（空 = 文本合格）。"""
        if not parsed.meetings:
            return ["日程里一场会都没有（版式或文本可能坏了）"]
        problems: list[str] = []
        for index, meeting in enumerate(parsed.meetings):
            problems += render.text_problems(meeting.title, f"第 {index + 1} 场的标题")
            problems += render.text_problems(meeting.agenda, f"第 {index + 1} 场的议题")
        for field_name, values in (
            ("标题", [meeting.title for meeting in parsed.meetings]),
            ("议题", [meeting.agenda for meeting in parsed.meetings]),
        ):
            if len(set(values)) != len(values):
                problems.append(f"{field_name}有重复：{values}")
        return problems
