"""工单分诊的文本外壳：属性行与三张表摊平，工单正文由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：
- **工单属性行、派单表、优先级表、附加规则由代码渲染**——它们是结构化事实（这条工单的
  字段、以及三张判据表），让模型去写只会写歪；
- 后端只写**工单正文**（真实工单里那段话是长尾的，模板拼的脱离现实）；
- 那段正文必须体现注入的 `issue` 与 `urgency` 两维，交给 `judges.TraceJudge` 反解一次
  再比对。

提示词**只给模型这两维**，不给客户等级 / 合规标记 / 产品域：给了它就会把那些字段也写进
正文，而它们归「工单属性」那一行管，两边一冲突（正文写着「我们是企业版」、那一行写着
`free`）这道题该读哪一行就有歧义了。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
**本模块不执行任何东西**——它只把文本当字符串读写。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.workflow import DraftError

from . import features, locales
from .features import TagSet

#: 提示词里特征的固定顺序：只列**由正文承载**的两维。顺序固定才能按提示哈希回放。
DIM_ORDER: Final[tuple[str, ...]] = ("issue", "urgency")

#: Markdown 围栏残留：模型偶尔把整段包进 ``` 里，那是实打实的数据损坏。
FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"```")

#: 段落名在 locale 里的键（渲染与解析共用同一份清单）。
SECTION_KEYS: Final[tuple[str, ...]] = (
    "body_label",
    "attrs_label",
    "queue_label",
    "priority_label",
    "flags_label",
)

#: 「工单属性」那一行的格数（产品域 | 客户等级 | 合规标记）。
ATTR_CELLS: Final[int] = 3

#: 派单表的格数（规则名 | 条件 | 队列）。
QUEUE_CELLS: Final[int] = 3

#: 优先级表的格数（紧急度 | 免费版 | 专业版 | 企业版）。
PRIORITY_CELLS: Final[int] = 4

#: 附加规则的格数（名 | 说明）。
FLAG_CELLS: Final[int] = 2

#: 附加规则的名字（顺序就是渲染顺序；`critical_areas` 那条是给 `needs_human` 用的清单）。
FLAG_TOKENS: Final[tuple[str, ...]] = ("critical_areas", "needs_human")

#: 优先级表列的顺序（与 `features.TIERS` 一致；表头文案取自 locale）。
PRIORITY_COLUMNS: Final[tuple[str, ...]] = ("free", "pro", "enterprise")


class RoutingDraft(BaseModel):
    """后端要填的草稿：**只有工单正文**。

    刻意只留一个字段——属性行与三张表全由组装器渲染，模型碰不到；
    字段越少，它越没有机会把结构化事实写歪（`check_draft` 会兜底）。
    """

    body: str = Field(
        min_length=1,
        description="这条工单的正文：一段话，像用户真的提交上来的那段",
    )


def build_prompt(tags: TagSet, loc: dict[str, JsonValue], phrasing: int) -> str:
    """组装提示词。

    输入：一组注入特征、语言字典、问法编号。
    输出：完整提示词（只描述要写的那两维，不含答案）。
    预期行为：特征按 `DIM_ORDER` 固定顺序逐行列出；规则整段附在后面，每条前面加短横线
    并把 `{language}` 填成语言名——留着占位符会把 `{language}` 原样发给模型。
    """
    values = tags.as_dict()
    lines = [str(loc["prompt"]["intro"]), ""]
    lines.extend(locales.feature_line(loc, dim, values[dim]) for dim in DIM_ORDER)
    lines.append("")
    rules = loc["prompt"]["rules"]
    offset = phrasing % len(rules)
    rotated = list(rules[offset:]) + list(rules[:offset])
    lines.extend(f"- {rule.format(language=loc['language_name'])}" for rule in rotated)
    return "\n".join(lines)


def build_task(tags: TagSet, loc: dict[str, JsonValue], task_id: str) -> GenerationTask:
    """输入：注入特征、语言字典、任务 id。输出：`GenerationTask`。

    预期行为：`response_model` 是 `RoutingDraft`（instructor 要 pydantic 模型）；
    `spec` 带上特征本身，便于事后按来源切片。
    """
    return GenerationTask(
        task_id=task_id,
        kind="choice",
        domain="ticket-triage",
        prompt=build_prompt(tags, loc, 0),
        response_model=RoutingDraft,
        seeds=(),
        spec={"tags": tags.as_dict()},
    )


def label_leak(text: str, loc: dict[str, JsonValue]) -> str:
    """输入：一段文本、语言字典。输出：文本开头误写的段落名（没有就空串）。

    预期行为：只认「段落名 + 冒号」开头这一种写法——正文里正常提到这些词不算。
    """
    stripped = text.strip()
    meta = loc["meta"]
    separators = {str(meta["colon"]).strip(), "：", ":"}
    for key in SECTION_KEYS:
        name = str(meta[key])
        if any(stripped.startswith(f"{name}{separator}") for separator in separators):
            return name
    return ""


def banned_echo(text: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：一段文本、语言字典。输出：文本里出现的属性词 / 表名词（没有就空列表）。

    预期行为：只认 locale 里点名的那几条**高精度**话术——宁可漏检也不误判，
    误判会在渲染期白拒一条本来能用的数据。大小写不敏感（英文里 `Enterprise` 与
    `enterprise` 一样泄底）。
    """
    lowered = text.lower()
    return [
        str(phrase)
        for phrase in loc["banned_intent_phrases"]
        if str(phrase).lower() in lowered
    ]


def check_draft(data: dict[str, JsonValue], loc: dict[str, JsonValue]) -> None:
    """输入：草稿字典、语言字典。输出：无；不合规就抛 `DraftError`（管线据此重渲染）。

    预期行为：正文会**原样**拼进 state，拼进去就洗不掉了，所以拼之前逐条查：
    - 非空、单行、不含 Markdown 围栏；
    - 开头不许再写一遍段落名（模型很爱这么干，写进去就是一行重复标记）；
    - 不许出现客户等级 / 合规标记 / 产品域那几档的词，也不许出现三张表的名字——
      它们归属性行与那几张表管，两边一旦冲突，这道题该读哪一行就有歧义了。
    """
    body = str(data.get("body", "")).strip()
    if not body:
        raise DraftError("草稿的「工单正文」是空的")
    if FENCE_PATTERN.search(body):
        raise DraftError("草稿的「工单正文」里混进了 Markdown 围栏")
    if "\n" in body:
        raise DraftError("草稿的「工单正文」是多行——它必须是一行（或一段）")
    leaked = label_leak(body, loc)
    if leaked:
        raise DraftError(f"草稿的「工单正文」把段落名「{leaked}」也写进去了")
    echoed = banned_echo(body, loc)
    if echoed:
        raise DraftError(f"草稿的「工单正文」把属性行 / 表里管的事写进去了：{'、'.join(echoed)}")


def attr_row(tags: TagSet) -> tuple[str, str, str]:
    """输入：注入特征。输出：「工单属性」那一行的三格（产品域 | 客户等级 | 合规标记）。"""
    return (tags.area, tags.tier, tags.compliance)


def attr_lines(tags: TagSet, loc: dict[str, JsonValue]) -> list[str]:
    """输入：注入特征、语言字典。输出：属性行（只有一行，三格都是 ASCII 词元）。"""
    return [locales.row_line(loc, attr_row(tags))]


def queue_rule_rows(loc: dict[str, JsonValue]) -> tuple[tuple[str, str, str], ...]:
    """输入：语言字典。输出：派单表的规范形态 `((规则名, 条件, 队列), ...)`。

    预期行为：顺序严格按 `features.QUEUE_RULES`——派单表是**从上往下第一条命中**的语义，
    顺序变了语义就变了。条件文案取自 locale，规则名与队列词元保持 ASCII。
    """
    return tuple(
        (name, str(loc["rules"][name]["condition"]), features.RAW_QUEUES[name])
        for name in features.QUEUE_RULES
    )


def queue_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：派单表的每一行（规则名 | 条件 | 队列）。"""
    return [locales.row_line(loc, cells) for cells in queue_rule_rows(loc)]


def queue_rule_row_variants() -> frozenset[tuple[tuple[str, str, str], ...]]:
    """输入：无。输出：各语言下派单表规范形态的集合。

    预期行为：一次 run 里语言按样本轮转（`locales[seq % len]`），所以审计拿到的 `locale`
    未必是产物那一种；「条件」那一列是随语言渲染的文案，判定必须写「等于**任一**语言的
    规范表」，拿调用方给的那一种去比会把另一种语言的产物误判成「派单表与规范表对不上」。
    """
    return frozenset(queue_rule_rows(locales.load(lang)) for lang in locales.available())


def priority_rows(loc: dict[str, JsonValue]) -> tuple[tuple[str, ...], ...]:
    """输入：语言字典。输出：优先级表的规范形态 `((紧急度, 免费版, 专业版, 企业版), ...)`。

    预期行为：三行（每行一个紧急度），三列档位按 `PRIORITY_COLUMNS` 的顺序排；
    档位取自 `features.PRIORITY_TABLE`，表里缺格会在这里当场抛 KeyError。
    """
    del loc
    out: list[tuple[str, ...]] = []
    for urgency in features.URGENCIES:
        levels = tuple(str(features.PRIORITY_TABLE[(urgency, tier)]) for tier in PRIORITY_COLUMNS)
        out.append((urgency, *levels))
    return tuple(out)


def priority_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：优先级表的每一行（紧急度 | 免费版 | 专业版 | 企业版）。"""
    return [locales.row_line(loc, cells) for cells in priority_rows(loc)]


def flag_rows(loc: dict[str, JsonValue]) -> tuple[tuple[str, str], ...]:
    """输入：语言字典。输出：附加规则的规范形态 `((名, 说明), ...)`。

    预期行为：名字保持 ASCII，说明取自 locale；关键域清单那条由代码把域名单填进去
    （`{areas}` 槽位），不靠人把清单抄进 locale——抄一遍就会漂移，而它正是 `needs_human`
    那条规则要用的清单。清单按 `features.AREAS` 的声明顺序取（不是按 `frozenset` 的迭代
    顺序），否则同一份输入在不同进程里会渲染出不同的字序，「逐字节可重建」就破了。
    """
    flags = loc["flags"]
    ordered = [name for name in features.AREAS if name in features.CRITICAL_AREAS]
    areas = str(loc["meta"]["list_sep"]).join(str(loc["area"][name]) for name in ordered)
    return (
        ("critical_areas", str(flags["critical_areas"]).format(areas=areas)),
        ("needs_human", str(flags["needs_human"])),
    )


def flag_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：附加规则的每一行（名 | 说明）。"""
    return [locales.row_line(loc, cells) for cells in flag_rows(loc)]


def flag_row_variants() -> frozenset[tuple[tuple[str, str], ...]]:
    """输入：无。输出：各语言下附加规则规范形态的集合。

    预期行为：与 `queue_rule_row_variants` 同一个道理——「说明」那一列随语言变
    （关键域清单那条还把域名单按语言填进 `{areas}` 槽位），判定要走「任一语言」。
    """
    return frozenset(flag_rows(locales.load(lang)) for lang in locales.available())


def to_state(data: dict[str, JsonValue], tags: TagSet, loc: dict[str, JsonValue]) -> str:
    """把草稿与结构化事实拼成 state。

    输入：草稿字典（`body`）、注入特征、语言字典。
    输出：五段文本——工单正文 / 工单属性 / 派单表 / 优先级表 / 附加规则。
    预期行为：只拼接与去首尾空白，不改写模型写的字（反解校验要靠原文）；
    后四段**由代码写**（它们是注入特征与固定表，不是模型的发挥），所以从 `tags` 推出来。
    """
    meta = loc["meta"]
    header = locales.header
    lines = [f"{header(loc, meta['body_label'])}{str(data['body']).strip()}"]
    lines.append(f"{header(loc, meta['attrs_label'])}{meta['paren'].format(text=meta['attrs_columns'])}")
    lines.extend(attr_lines(tags, loc))
    lines.append(f"{header(loc, meta['queue_label'])}{meta['paren'].format(text=meta['queue_columns'])}")
    lines.extend(queue_lines(loc))
    lines.append(
        f"{header(loc, meta['priority_label'])}"
        f"{meta['paren'].format(text=meta['priority_columns'])}"
    )
    lines.extend(priority_lines(loc))
    lines.append(f"{header(loc, meta['flags_label'])}{meta['paren'].format(text=meta['flags_columns'])}")
    lines.extend(flag_lines(loc))
    return "\n".join(lines)


@dataclass(frozen=True)
class AttrsState:
    """从「工单属性」那一行里反解出来的三格。"""

    area: str
    tier: str
    compliance: str


@dataclass(frozen=True)
class ParsedTicket:
    """从 state 里纯代码反解出来的事实。"""

    body: str
    attrs: AttrsState | None
    queue_rules: tuple[tuple[str, str, str], ...]
    priority: tuple[tuple[str, ...], ...]
    flags: tuple[tuple[str, str], ...]

    @property
    def is_empty(self) -> bool:
        """输入：无。输出：正文那一段是不是空的。"""
        return not self.body.strip()


def _sections(state: str, loc: dict[str, JsonValue]) -> dict[str, list[str]]:
    """输入：state、语言字典。输出：`{段落名: 该段的行列表}`（行已去首尾空白）。"""
    meta = loc["meta"]
    titles = tuple(str(meta[key]) for key in SECTION_KEYS)
    out: dict[str, list[str]] = {}
    current: str | None = None
    for raw in state.splitlines():
        line = raw.strip()
        matched = next(
            (title for title in titles if line.startswith(locales.header(loc, title))), None
        )
        if matched is not None:
            current = matched
            out.setdefault(current, [])
            rest = line[len(locales.header(loc, matched)) :].strip()
            if rest:
                out[current].append(rest)
            continue
        if current is not None and line:
            out[current].append(line)
    return out


def _cells(line: str, loc: dict[str, JsonValue]) -> list[str] | None:
    """输入：一行、语言字典。输出：按分隔符切开的格子（不是数据行就 None）。"""
    prefix = str(loc["meta"]["row_prefix"])
    if not line.startswith(prefix):
        return None
    return [cell.strip() for cell in line[len(prefix) :].split(locales.CELL_SEP)]


def body_lines(state: object, loc: dict[str, JsonValue]) -> list[str]:
    """输入：state、语言字典。输出：正文那一段的行列表。

    预期行为：正常只有一行；模型把正文写成多行时这里会有多行——卫生判别器就靠这个
    认出「正文是多行」。非文本 state 返回空列表。
    """
    if not isinstance(state, str):
        return []
    return _sections(state, loc).get(str(loc["meta"]["body_label"]), [])


def body_of(state: object, loc: dict[str, JsonValue]) -> str:
    """输入：state、语言字典。输出：正文那一行（没有就空串）。"""
    lines = body_lines(state, loc)
    return lines[0] if lines else ""


def attrs_of(state: object, loc: dict[str, JsonValue]) -> AttrsState | None:
    """输入：state、语言字典。输出：解析出来的属性行（解不出就 None）。

    预期行为：只认形如 `- payments | enterprise | none` 的三格行；词元不在取值表里就当
    解不出来——判别器与审计会拿它跟注入特征逐格比对，串味会当场暴露。
    """
    if not isinstance(state, str):
        return None
    for line in _sections(state, loc).get(str(loc["meta"]["attrs_label"]), []):
        cells = _cells(line, loc)
        if cells is None:
            continue
        if len(cells) != ATTR_CELLS:
            return None
        area, tier, compliance = cells
        if area in features.AREAS and tier in features.TIERS and compliance in features.COMPLIANCES:
            return AttrsState(area=area, tier=tier, compliance=compliance)
        return None
    return None


def queue_rules_of(state: object, loc: dict[str, JsonValue]) -> tuple[tuple[str, str, str], ...]:
    """输入：state、语言字典。输出：解析出来的派单表 `((规则名, 条件, 队列), ...)`。

    预期行为：只认「规则名 | 条件 | 队列」三格且首格是规则名、末格是队列词元的行；
    一行都对不上就返回空元组，判别器会把空元组当成「表没渲染出来」。
    """
    if not isinstance(state, str):
        return ()
    out: list[tuple[str, str, str]] = []
    for line in _sections(state, loc).get(str(loc["meta"]["queue_label"]), []):
        cells = _cells(line, loc)
        if cells is None:
            continue  # 段落头那一行（列名）不是数据行，跳过
        if len(cells) != QUEUE_CELLS:
            continue
        name, condition, queue = cells
        if name in features.QUEUE_RULES and queue in features.QUEUES:
            out.append((name, condition, queue))
    return tuple(out)


def priority_of(state: object, loc: dict[str, JsonValue]) -> tuple[tuple[str, ...], ...]:
    """输入：state、语言字典。输出：解析出来的优先级表（解不出就空元组）。

    预期行为：只认四格行且首格是紧急度、后三格都是 0~9 的单个数字；一行对不上就跳过。
    """
    if not isinstance(state, str):
        return ()
    out: list[tuple[str, ...]] = []
    for line in _sections(state, loc).get(str(loc["meta"]["priority_label"]), []):
        cells = _cells(line, loc)
        if cells is None:
            continue
        if len(cells) != PRIORITY_CELLS:
            continue
        if cells[0] not in features.URGENCIES:
            continue
        if not all(cell.isdigit() and len(cell) == 1 for cell in cells[1:]):
            continue
        out.append(tuple(cells))
    return tuple(out)


def flags_of(state: object, loc: dict[str, JsonValue]) -> tuple[tuple[str, str], ...]:
    """输入：state、语言字典。输出：解析出来的附加规则（解不出就空元组）。

    预期行为：只认「名 | 说明」两格且名字在 `FLAG_TOKENS` 里的行。
    """
    if not isinstance(state, str):
        return ()
    out: list[tuple[str, str]] = []
    for line in _sections(state, loc).get(str(loc["meta"]["flags_label"]), []):
        cells = _cells(line, loc)
        if cells is None:
            continue
        if len(cells) != FLAG_CELLS:
            continue
        name, text = cells
        if name in FLAG_TOKENS:
            out.append((name, text))
    return tuple(out)


def parse_state(state: object) -> ParsedTicket:
    """输入：state。输出：纯代码能反解的部分（正文 + 属性行 + 三张表）。

    预期行为：与语言无关——把各语言的段落名都试一遍，命中即返回。
    """
    empty = ParsedTicket(body="", attrs=None, queue_rules=(), priority=(), flags=())
    if not isinstance(state, str):
        return empty
    for lang in locales.available():
        loc = locales.load(lang)
        body = body_of(state, loc)
        attrs = attrs_of(state, loc)
        if body or attrs is not None:
            return ParsedTicket(
                body=body,
                attrs=attrs,
                queue_rules=queue_rules_of(state, loc),
                priority=priority_of(state, loc),
                flags=flags_of(state, loc),
            )
    return empty
