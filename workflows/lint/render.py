"""行级规则检查的文本外壳：规则表与处置办法由代码写，片段原文由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **片段抬头、规则名与判据、处置办法表由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**那 8 行原文**（真实代码 / 日志 / 文档里的行是长尾的，模板拼的脱离现实）；
- 写出来的行必须**兑现行计划**：计划说命中的行要真的命中，计划说干净的行要一条规则都不
  命中。`check_draft` 在渲染期查一遍是为了**重试**，`judges.HygieneJudge` 再查一遍是为了
  兜住不是本 workflow 渲染出来的 state。

提示词里**写明哪几行必须命中**——这是「受控合成」的规格，不是让模型自由发挥；五条规则的
判据整段附在后面，模型才知道怎么写才叫命中、干净行要躲开哪几条。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
**本模块不执行任何代码**——片段是当字符串读写的。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.state_parse import recognize_locale
from jc.workflow import DraftError

from . import features, locales
from .features import LintPlan
from .spec import LintSpec

#: 行号前缀。**这是 state 格式的一部分，不是文案**，所以不进语言文件：渲染与解析共用
#: 这一个常量，两边不可能漂移。
LINE_TEMPLATE: Final[str] = "{n}. {line}"
LINE_PREFIX: Final[re.Pattern[str]] = re.compile(r"^\s*\d+\.\s?")

#: Markdown 围栏残留：模型偶尔把整段包进 ``` 里，那是实打实的数据损坏。
FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"```")

#: 处置办法表的三段边界（闭区间），与 `features.gate_of` 的分段一一对应。
GATE_BOUNDS: Final[tuple[tuple[int, int], ...]] = (
    (0, features.GATE_PASS_MAX),
    (features.GATE_PASS_MAX + 1, features.GATE_REVIEW_MAX),
    (features.GATE_REVIEW_MAX + 1, features.SIZE),
)


class FragmentDraft(BaseModel):
    """后端要填的草稿：**只有那几行原文**。

    刻意只留一个字段——抬头、规则名、判据、处置办法表全由组装器渲染，模型碰不到；字段越少，
    它越没有机会把结构化事实写歪（`check_draft` 会兜底）。
    """

    lines: list[str] = Field(
        min_length=features.SIZE,
        max_length=features.SIZE,
        description="片段的每一行原文，按片段顺序，不要行号前缀、不要围栏、不要空行",
    )


@dataclass(frozen=True)
class ParsedLint:
    """从 state 反解出来的、纯代码看得见的部分。"""

    lines: tuple[str, ...]
    rule: str | None
    genre: str | None
    gate: tuple[str, ...]

    @property
    def found(self) -> bool:
        """输入：无。输出：片段抬头认出来了没有。预期行为：就是 `genre` 在不在。"""
        return self.genre is not None


def rule_name(loc: dict[str, JsonValue], rule: str) -> str:
    """输入：语言字典、规则名。输出：规则的短语（提示词与 state 共用）。"""
    return str(loc["rule"][rule]["name"])


def rule_criteria(loc: dict[str, JsonValue], rule: str) -> str:
    """输入：语言字典、规则名。输出：这条规则的判据（一句话）。"""
    return str(loc["rule"][rule]["criteria"])


def gate_rows(loc: dict[str, JsonValue]) -> tuple[str, ...]:
    """输入：语言字典。输出：处置办法表的三行（规范形态）。

    预期行为：档位名由 `features.gate_of` 按**该行的下界**算——表的写法与判据共用同一份
    分段，改一处两处都变。
    """
    template = str(loc["state"]["gate_row"])
    return tuple(
        template.format(lo=low, hi=high, gate=loc["gate"][features.gate_of(low)])
        for low, high in GATE_BOUNDS
    )


def body_header(loc: dict[str, JsonValue], genre: str, count: int) -> str:
    """输入：语言字典、体裁、行数。输出：片段抬头那一行。"""
    lead = str(loc["state"]["lead"][genre]).format(n=count)
    return f"{loc['state']['body_mark']}{lead}"


def build_prompt(plan: LintPlan, loc: dict[str, JsonValue]) -> str:
    """组装提示词。

    输入：行计划、语言字典。输出：完整提示词（不含答案）。
    预期行为：必须命中的行号（1 起）与「其余行要干净」的规格写在最前面，五条规则的判据
    整段附在后面；命中行数为 0 或 8 时换用对应的两句，不留「其余 0 行」这种读不通的话。
    """
    sep = str(loc["meta"]["list_sep"])
    name = rule_name(loc, plan.rule)
    lines = [
        str(loc["prompt"]["intro"]).format(genre=loc["genre"][plan.genre], n=features.SIZE),
        str(loc["prompt"]["lang_rule"]).format(
            genre=loc["genre"][plan.genre], language=loc["language_name"]
        ),
        (
            str(loc["prompt"]["hit_none"]).format(n=features.SIZE, rule=name)
            if plan.hits == 0
            else str(loc["prompt"]["hit_line"]).format(
                rows=sep.join(str(row + 1) for row in plan.hit_rows), rule=name
            )
        ),
        (
            str(loc["prompt"]["clean_none"]).format(n=features.SIZE, rule=name)
            if plan.hits == features.SIZE
            else str(loc["prompt"]["clean_line"]).format(k=features.SIZE - plan.hits)
        ),
        "",
        str(loc["prompt"]["rules_header"]),
    ]
    for rule in features.RULES:
        lines.append(
            str(loc["prompt"]["rule_line"]).format(
                name=rule_name(loc, rule), criteria=rule_criteria(loc, rule)
            )
        )
    lines.append("")
    lines.append(str(loc["prompt"]["output"]).format(n=features.SIZE))
    return "\n".join(lines)


def build_task(spec: LintSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：规格、语言字典。输出：`GenerationTask`。

    预期行为：`response_model` 是 `FragmentDraft`（instructor 要 pydantic 模型）；`spec` 带上
    注入特征与行计划，便于事后按来源切片。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="lint",
        prompt=build_prompt(spec.plan, loc),
        response_model=FragmentDraft,
        seeds=(),
        spec={"case": spec.case.as_dict(), "hit_rows": list(spec.plan.hit_rows)},
    )


def check_draft(lines: list[str], plan: LintPlan) -> None:
    """输入：后端写的行、行计划。输出：无（不合格抛 `DraftError`，管线据此重渲染）。

    预期行为：三件事逐条查——**行数**必须正好 8；**行原文**不许带行号前缀 / 换行 / 围栏；
    **命中与否必须兑现计划**（计划说命中的行要真的命中该规则，计划说干净的行要一条规则都
    不命中）。任一条不满足就抛人话，管线据此重渲染——模型写歪了要重写，不是改标签。
    """
    if len(lines) != features.SIZE:
        raise DraftError(f"片段必须正好 {features.SIZE} 行，收到 {len(lines)} 行")
    for index, raw in enumerate(lines, start=1):
        if not raw.strip():
            raise DraftError(f"第 {index} 行是空的")
        if "\n" in raw:
            raise DraftError(f"第 {index} 行里还有换行符（一行只能是一行）")
        if LINE_PREFIX.match(raw):
            raise DraftError(f"第 {index} 行自带行号前缀：{raw!r}（行号由代码加）")
        if FENCE_PATTERN.search(raw):
            raise DraftError(f"第 {index} 行里有围栏残留：{raw!r}")
    for row in range(features.SIZE):
        text = lines[row]
        if row in plan.hit_rows:
            if not features.matches(plan.rule, text):
                raise DraftError(f"第 {row + 1} 行该命中「{plan.rule}」，但按判据不命中：{text!r}")
            continue
        hit = features.match_any(text)
        if hit is not None:
            raise DraftError(f"第 {row + 1} 行该是干净行，却命中了「{hit}」：{text!r}")


def to_state(lines: list[str], plan: LintPlan, loc: dict[str, JsonValue]) -> str:
    """输入：行原文、行计划、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：抬头 + 规则名 + 判据 + 编号片段 + 处置办法表；**判据印在 state 里**，所以
    「命中这条规则」是自明的（题面只需要指明问的是哪条规则，范式 §9.8）。
    """
    parts = [
        body_header(loc, plan.genre, len(lines)),
        f"{loc['state']['rule_prefix']}{rule_name(loc, plan.rule)}",
        f"{loc['state']['criteria_prefix']}{rule_criteria(loc, plan.rule)}",
    ]
    parts.extend(
        LINE_TEMPLATE.format(n=index + 1, line=text) for index, text in enumerate(lines)
    )
    parts.append(str(loc["state"]["gate_mark"]))
    parts.extend(gate_rows(loc))
    return "\n".join(parts)


def genre_of_lead(loc: dict[str, JsonValue], lead: str) -> str | None:
    """输入：语言字典、抬头里片段标记之后的那段话。输出：体裁名（认不出给 None）。"""
    for genre in features.GENRES:
        if str(loc["state"]["lead"][genre]).format(n=features.SIZE) == lead:
            return genre
    return None


def rule_of_name(loc: dict[str, JsonValue], name: str) -> str | None:
    """输入：语言字典、规则短语。输出：规则名（认不出给 None）。"""
    for rule in features.RULES:
        if str(loc["rule"][rule]["name"]) == name:
            return rule
    return None


def parse_with(state: str, loc: dict[str, JsonValue]) -> ParsedLint:
    """输入：state、语言字典。输出：按这一份语言反解的结果。

    预期行为：认片段标记、处置办法标记、规则前缀三种行；片段行是编号行里前缀之后的部分。
    标记之外的行（例如判据那一行）一律不当片段——否则判据会被数成一行。
    """
    body_mark = str(loc["state"]["body_mark"])
    gate_mark = str(loc["state"]["gate_mark"])
    rule_prefix = str(loc["state"]["rule_prefix"])
    lines: list[str] = []
    gate: list[str] = []
    rule: str | None = None
    genre: str | None = None
    in_gate = False
    for raw in state.splitlines():
        text = raw.strip()
        if text.startswith(gate_mark):
            in_gate = True
            continue
        if text.startswith(body_mark):
            genre = genre_of_lead(loc, text[len(body_mark) :])
            in_gate = False
            continue
        if text.startswith(rule_prefix):
            rule = rule_of_name(loc, text[len(rule_prefix) :].strip())
            continue
        if in_gate:
            if text:
                gate.append(text)
            continue
        found = LINE_PREFIX.match(raw)
        if found:
            lines.append(raw[found.end() :])
    return ParsedLint(lines=tuple(lines), rule=rule, genre=genre, gate=tuple(gate))


def parse_state(state: object) -> ParsedLint:
    """输入：state。输出：纯代码能反解的部分（片段行 / 规则 / 体裁 / 处置办法表）。

    预期行为：与语言无关——把各语言的段落标记都试一遍，**认出片段抬头**的那一份就用它；
    一份都认不出来时返回空结果（`lines` 与 `genre` 都为空）。

    为什么认的是抬头而不是编号行：编号行（`1. …`）与语言无关，任何一种语言都能把它认下来，
    于是「先试的那一种」会抢走所有 state——中文 state 被英文标记试出来的结果就是
    「有 8 行、但没有规则名也没有体裁」，框架判别器会把它判成渲染串味。抬头是渲染时按语言
    写死的，拿它当判据才分得开。
    """
    lang = detect_locale(state)
    if lang is None:
        return ParsedLint(lines=(), rule=None, genre=None, gate=())
    return parse_with(state, locales.load(lang))


def detect_locale(state: object) -> str | None:
    """输入：state。输出：能认出片段抬头的那一种语言（认不出给 None）。

    预期行为：审计拿它按 state **自己**的语言取规范表——一次 run 里混了 zh / en 时，
    用调用方给的那一种去比会平白报一堆「处置办法表对不上」。
    """
    return recognize_locale(state, locales, parse_with)
