"""根因定位的文本外壳：依赖图、判读规则与现场时刻由代码写，每个服务那一行日志由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **依赖图、规则表、时刻与「哪一行是哪个服务」由代码渲染**——它们是结构化事实，让模型去写
  只会写歪；
- 后端只写**六个服务各自的现场日志**（真实日志是长尾的，模板拼出来的脱离现实分布）；
- **每个服务是什么事件由代码按字面算**（`features.PATTERNS` 那三条正则），模型碰不到：
  提示词里逐条写明「这一行必须出现哪种形状、不许出现另外两种」，`check_draft` 在渲染期按
  同一份判据复核一遍（不合格就重渲染），`judges.HygieneJudge` / `FrameJudge` 再各查一遍。

**为什么依赖图写成一行一条边**：通用层 `jc.state_parse` 的判据表段落按「行前缀 + 分隔符」
取每行的第一个字段，所以 `- S1 > S2` 这种写法让**边表**天然落进 `ParsedState.tags`——
审计拿它重推根因时不必再自己写一遍解析器（范式 §9.13：口径只留一份）。

**事件形状也印进 state**（范式 §9.44 的教训）：反解模型要判「这个服务是超时还是报错」，
而事件名是**压缩过的标签**——「超时」推不出「出现『等待超时 / 连接超时 / timeout』这类词」。
规则表里把三种形状的判据原文印出来，读法就是照着比字面，不是猜。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.state_parse import ParsedState as ParsedContent, parse_any, parse_sections, recognize_locale
from jc.workflow import DraftError

from . import features, locales
from .features import SERVICES, CausePlan
from .spec import CauseSpec

#: 规则表与依赖图每行的前缀。**这是 state 格式的一部分，不是文案**，所以不进语言文件：
#: 渲染与解析共用这一个常量，两边不可能漂移。
LINE_PREFIX: Final[str] = "- "

#: 依赖图里「上游 > 下游」的分隔符。同样属于 state 格式，不进语言文件。
EDGE_SEP: Final[str] = " > "

#: 日志正文的长度上下限。太短读不出「像不像一行真日志」，太长就只是灌水。
MIN_CONTENT_CHARS: Final[int] = features.MIN_CONTENT_CHARS
MAX_CONTENT_CHARS: Final[int] = features.MAX_CONTENT_CHARS


class ContentDraft(BaseModel):
    """后端要填的草稿：六个服务各自的现场日志。

    `logs` 是**六个服务 → 一行文本**（日志本身是长尾的、由模型写）；结构化事实（依赖图、
    规则表、时刻、哪一行是哪个服务）全由组装器渲染，模型碰不到。
    """

    logs: dict[str, str] = Field(
        description="服务编号 → 那一行现场日志（每个服务一条，不要标题、不要围栏）"
    )


def event_copy(loc: dict[str, JsonValue], event: str) -> dict[str, JsonValue]:
    """输入：语言字典、事件类型。输出：该事件的文案（name / rule / avoid）。

    预期行为：缺字段就当场报错——规则表与提示词都靠它，缺一个字段模型就会拿到半句话。
    """
    table = loc.get("event")
    entry = table.get(event) if isinstance(table, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"语言文件里缺事件 {event} 的文案")
    for key in ("name", "rule", "avoid"):
        if key not in entry:
            raise ValueError(f"事件 {event} 的文案缺字段 {key}")
    return entry


def event_line(loc: dict[str, JsonValue], event: str) -> str:
    """输入：语言字典、事件类型。输出：规则表里那一行（不含行前缀）。

    预期行为：`编号 + 分隔符 + 事件名 + 规则分隔符 + 判据原文`。编号是**结构键**，
    排在分隔符前面，与语言的书写顺序无关。
    """
    sep = str(loc["state"]["tag_sep"])
    rule_sep = str(loc["state"]["rule_sep"])
    entry = event_copy(loc, event)
    return f"{event}{sep}{entry['name']}{rule_sep}{entry['rule']}"


def role_copy(loc: dict[str, JsonValue], role: str) -> str:
    """输入：语言字典、角色。输出：该角色的说明（根因选项的描述要用）。"""
    table = loc.get("role")
    entry = table.get(role) if isinstance(table, dict) else None
    if not isinstance(entry, str):
        raise ValueError(f"语言文件里缺角色 {role} 的文案")
    return entry


def rules(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：判读规则表（每条一行）。预期行为：空表当场报错。"""
    table = loc.get("rules")
    if not isinstance(table, list) or not table:
        raise ValueError("语言文件里缺 rules（判读规则表）")
    return [str(item) for item in table]


def graph_line(upstream: str, downstream: str) -> str:
    """输入：上游、下游。输出：依赖图里那一行（不含行前缀）。"""
    return f"{upstream}{EDGE_SEP}{downstream}"


def parse_edges(tags: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    """输入：判据表段落里取出来的那些行。输出：边表（认不出来的行直接跳过）。

    预期行为：只认 `上游 > 下游` 且两端都是合法服务编号的行。审计拿它把 state 里印的依赖图
    还原出来，再与规则表一起重推根因。
    """
    edges: list[tuple[str, str]] = []
    for tag in tags:
        upstream, found, downstream = tag.partition(EDGE_SEP)
        head, tail = upstream.strip(), downstream.strip()
        if found and head in SERVICES and tail in SERVICES:
            edges.append((head, tail))
    return tuple(edges)


def parse_rows(text: str | None, loc: dict[str, JsonValue]) -> list[tuple[str, str, str]]:
    """输入：现场日志那一块的正文、语言字典。输出：`(时刻, 服务, 正文)` 列表（认不出的行跳过）。

    预期行为：每行是 `- 时刻 服务：正文`，按 `state.tag_sep` 切第一刀，左半边再按空白切出
    时刻与服务。服务编号是结构键，所以它必须出现在分隔符左边。
    """
    sep = str(loc["state"]["tag_sep"])
    rows: list[tuple[str, str, str]] = []
    for raw in (text or "").splitlines():
        row = raw.strip()
        if not row.startswith(LINE_PREFIX):
            continue
        head, found, body = row[len(LINE_PREFIX) :].partition(sep)
        if not found:
            continue
        parts = head.split()
        if len(parts) != 2 or parts[1] not in SERVICES:
            continue
        rows.append((parts[0], parts[1], body.strip()))
    return rows


def parse_logs(text: str | None, loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：现场日志那一块的正文、语言字典。输出：{服务: 日志正文}（认不出的行跳过）。"""
    return {service: body for _, service, body in parse_rows(text, loc)}


def build_prompt(loc: dict[str, JsonValue], plan: CausePlan) -> str:
    """输入：语言字典、现场计划。输出：渲染提示词。

    预期行为：先给依赖图（一行一条边），再逐服务点名「这一行必须出现哪种形状、不许出现另外
    两种」，最后是语言、长度与输出格式。**不给成品日志样例**——给了就会被整批抄走
    （范式 §9.64），形状只以判据原文的形式出现。
    """
    prompt = loc["prompt"]
    lines: list[str] = [
        str(prompt["intro"]),
        str(prompt["lang_rule"]).format(language=str(loc["language_name"])),
        str(prompt["graph_header"]),
    ]
    lines.extend(f"{LINE_PREFIX}{graph_line(up, down)}" for up, down in features.GRAPHS[plan.graph])
    lines.append(str(prompt["shape_header"]))
    for service in SERVICES:
        entry = event_copy(loc, plan.events[service])
        # 「不许出现」的那半句要列出**另外两种**形状的词——写成本事件自己的 avoid 会把
        # 「必须出现」和「不许出现」说成同一批词，模型当场无所适从。
        others = [
            str(event_copy(loc, other)["avoid"])
            for other in features.EVENTS
            if other != plan.events[service]
        ]
        lines.append(
            str(prompt["shape_line"]).format(
                service=service,
                event=str(entry["name"]),
                rule=str(entry["rule"]),
                avoid=str(prompt["avoid_joiner"]).join(others),
            )
        )
    lines.extend(
        [
            str(prompt["vary_rule"]),
            str(prompt["length_rule"]).format(min=MIN_CONTENT_CHARS, max=MAX_CONTENT_CHARS),
            str(prompt["self_check"]),
            str(prompt["output"]),
        ]
    )
    return "\n".join(lines)


def build_task(spec: CauseSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：真值规格、语言字典。输出：渲染任务。

    预期行为：任务 id 由规格算（断点续跑按它查进度），`spec` 里带上注入特征——后端实现与假后端
    若想「照着计划答」，不必再解析提示词。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="causechain",
        prompt=build_prompt(loc, spec.plan),
        response_model=ContentDraft,
        seeds=(),
        spec={
            "case": spec.case.as_dict(),
            "events": dict(spec.plan.events),
            "order": list(spec.plan.order),
        },
    )


def pasted_rules(text: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：一行日志、语言字典。输出：被**逐字贴进**日志的事件判据原文。

    预期行为：只认完全相同的整段判据原文（`event_copy(...)["rule"]`）——改一个词就不再算。
    提示词里给模型看的**唯一**一段具体文字就是三种形状的判据原文，所以它能抄的只剩它：
    整段贴进去，正则照样判命中，但那一行读起来不是日志而是判据本身。真实日志不会包含
    整段判据原文，所以这条**不会误伤**正常产物。
    """
    return [
        entry
        for entry in (str(event_copy(loc, event)["rule"]) for event in features.EVENTS)
        if entry in text
    ]


def check_draft(draft: ContentDraft, plan: CausePlan, loc: dict[str, JsonValue]) -> None:
    """校验「现场日志」是否兑现了现场计划。

    输入：后端草稿、现场计划、语言字典。输出：无（不合格抛 `DraftError`，管线据此重渲染）。
    预期行为：逐条查——

    1. 六个服务一条不少、不多；
    2. 每条非空、不带围栏；
    3. **不许把事件判据原文原样贴进日志**（`pasted_rules` 逐字比对）——这一条查在长度之前，
       因为判据原文本身就可能比长度上限长：先报「抄了判据」比先报「太长」更接近实情，
       而且照抄闸从此不受长度上下限改动的影响；
    4. 长度落在 `MIN_CONTENT_CHARS`~`MAX_CONTENT_CHARS`；
    5. **按字面算出来的事件类型与计划逐条相同**（多中一条、少中一条都不行，理由里说清差在哪）。
    第 5 条是这一族的核心：事件类型就是金标签的一半，它由正则从文本算得出来，
    所以「模型写歪了」在渲染期就能拦住，不用等判别器。
    """
    logs = draft.logs
    missing = [service for service in SERVICES if service not in logs]
    extra = sorted(key for key in logs if key not in SERVICES)
    if missing or extra:
        raise DraftError(f"日志的服务对不上：缺 {missing}、多 {extra}")
    for service in SERVICES:
        text = logs[service].strip()
        want = plan.events[service]
        if not text:
            raise DraftError(f"{service} 那一行是空的")
        if "```" in text:
            raise DraftError(f"{service} 那一行带围栏：{text[:60]!r}")
        pasted = pasted_rules(text, loc)
        if pasted:
            raise DraftError(f"{service} 那一行把事件判据原文原样贴了进去：{pasted}")
        if len(text) < MIN_CONTENT_CHARS:
            raise DraftError(f"{service} 那一行太短（至少 {MIN_CONTENT_CHARS} 个字符）：{text!r}")
        if len(text) > MAX_CONTENT_CHARS:
            raise DraftError(f"{service} 那一行太长（最多 {MAX_CONTENT_CHARS} 个字符，实际 {len(text)}）")
        found = features.event_of(text)
        if found != want:
            raise DraftError(
                f"{service} 那一行按字面算出来是 {found or '认不出的形状'}，"
                f"计划要的是 {want}（{event_copy(loc, want)['name']}）"
            )


def to_state(draft: ContentDraft, plan: CausePlan, loc: dict[str, JsonValue]) -> str:
    """输入：后端草稿、现场计划、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：依赖图 + 判读规则 + 事件形状表 + 现场日志。**规则与形状表印在 state 里**，
    所以「这次是谁出错、错误从哪儿往上冒」是自明的——考生只读 state 与题面就能答，
    不必知道外部规则（范式 §9.8）。
    """
    marks = loc["state"]
    times = plan.times()
    sep = str(marks["tag_sep"])
    parts = [str(marks["tag_mark"])]
    parts.extend(
        f"{LINE_PREFIX}{graph_line(upstream, downstream)}"
        for upstream, downstream in features.GRAPHS[plan.graph]
    )
    parts.append(str(marks["rule_mark"]))
    parts.extend(f"{LINE_PREFIX}{rule}" for rule in rules(loc))
    parts.extend(f"{LINE_PREFIX}{event_line(loc, event)}" for event in features.EVENTS)
    parts.append(str(marks["content_mark"]))
    parts.extend(
        f"{LINE_PREFIX}{times[service]} {service}{sep}{draft.logs[service].strip()}"
        for service in plan.order
    )
    return "\n".join(parts)


def parse_with(state: str, loc: dict[str, JsonValue]) -> ParsedContent:
    """输入：state、语言字典。输出：按这一份语言反解的结果。"""
    return parse_sections(state, loc, line_prefix=LINE_PREFIX)


def detect_locale(state: object) -> str | None:
    """输入：state。输出：能认出素材抬头的那一种语言（认不出给 None）。"""
    return recognize_locale(state, locales, parse_with)


def parse_state(state: object) -> ParsedContent:
    """输入：state。输出：纯代码能反解的部分（依赖图边表 / 现场日志原文）。"""
    return parse_any(state, locales, parse_with)
