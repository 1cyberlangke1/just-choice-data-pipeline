"""会话上下文压缩的文本外壳：处置表与「本块位置」摊平，那一块正文由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：
- **处置表与「本块位置」那一行由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**一块上下文正文**（真实会话里的那一段是长尾的，模板拼的脱离现实）；
- 那段正文必须体现注入的「这一块是什么」与「有没有承载持久事实」两维，交给
  `judges.TraceJudge` 反解一次再比对。

提示词**只给模型这两维**，不给轮次位置 / 是否钉住 / 剩余预算：给了它就会把那三件事也
写进正文，而它们归「本块位置」那一行管，两边一冲突（正文写着「刚才那一轮」、那一行写着
`old`）这道题该读哪一行就有歧义了。

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

from . import locales
from .features import AGES, BUDGETS, DISPOSITIONS, PINS, RULES, TagSet

#: 提示词里特征的固定顺序：只列**由文本承载**的两维。顺序固定才能按提示哈希回放。
DIM_ORDER: Final[tuple[str, ...]] = ("kind", "fact")

#: Markdown 围栏残留：模型偶尔把整段包进 ``` 里，那是实打实的数据损坏。
FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"```")

#: 三格之间的分隔符。选竖线是因为它在中英里写法相同，反解不用跨语言试。
CELL_SEP: Final[str] = "|"

#: 段落名在 locale 里的键（渲染与解析共用同一份清单）。
SECTION_KEYS: Final[tuple[str, ...]] = (
    "content_label",
    "where_label",
    "rules_label",
)

#: 「本块位置」那一行的格数（轮次位置 / 是否钉住 / 剩余预算）。
WHERE_CELLS: Final[int] = 3


class CompactDraft(BaseModel):
    """后端要填的草稿：**只有一块上下文正文**。

    刻意只留一个字段——处置表与「本块位置」全由组装器渲染，模型碰不到；
    字段越少，它越没有机会把结构化事实写歪（`check_draft` 会兜底）。
    """

    content: str = Field(
        min_length=1,
        description="这一块上下文正文：一行（或一段），像真实会话里那一段那样",
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

    预期行为：`response_model` 是 `CompactDraft`（instructor 要 pydantic 模型）；
    `spec` 带上特征本身，便于事后按来源切片。
    """
    return GenerationTask(
        task_id=task_id,
        kind="choice",
        domain="context-compaction",
        prompt=build_prompt(tags, loc, 0),
        response_model=CompactDraft,
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
    """输入：一段文本、语言字典。输出：文本里出现的位置词 / 钉住词 / 预算词（没有就空列表）。

    预期行为：只认 locale 里点名的那几条**高精度**话术——宁可漏检也不误判，
    误判会在渲染期白拒一条本来能用的数据。大小写不敏感（英文里 `Pinned` 与
    `pinned` 一样泄底）。
    """
    lowered = text.lower()
    return [
        str(phrase)
        for phrase in loc["banned_intent_phrases"]
        if str(phrase).lower() in lowered
    ]


def check_draft(data: dict[str, JsonValue], loc: dict[str, JsonValue]) -> None:
    """输入：草稿字典、语言字典。输出：无；不合规就抛 `ValueError`。

    预期行为：正文会**原样**拼进 state，拼进去就洗不掉了，所以拼之前逐条查：
    - 非空、单行、不含 Markdown 围栏；
    - 开头不许再写一遍段落名（模型很爱这么干，写进去就是一行重复标记）；
    - 不许出现位置词 / 钉住词 / 预算词（它们归「本块位置」那一行）——
      两边一旦冲突，这道题该读哪一行就有歧义了。
    """
    content = str(data.get("content", "")).strip()
    if not content:
        raise ValueError("草稿的「本块内容」是空的")
    if FENCE_PATTERN.search(content):
        raise ValueError("草稿的「本块内容」里混进了 Markdown 围栏")
    if "\n" in content:
        raise ValueError("草稿的「本块内容」是多行——它必须是一行（或一段）")
    leaked = label_leak(content, loc)
    if leaked:
        raise ValueError(f"草稿的「本块内容」把段落名「{leaked}」也写进去了")
    echoed = banned_echo(content, loc)
    if echoed:
        raise ValueError(f"草稿的「本块内容」把位置行管的事写进去了：{'、'.join(echoed)}")


def where_line(tags: TagSet, loc: dict[str, JsonValue]) -> str:
    """输入：注入特征、语言字典。输出：「本块位置」那一行（三格都是 ASCII 词元）。"""
    meta = loc["meta"]
    cells = [tags.age, tags.pins, tags.budget]
    return f"{meta['row_prefix']}{f' {CELL_SEP} '.join(cells)}"


def rule_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：处置表的每一行（规则名 | 条件 | 处置）。

    预期行为：顺序严格按 `features.RULES`——处置表是**从上往下第一条命中**的语义，
    顺序变了语义就变了。条件与处置文案取自 locale，规则名与处置词元保持 ASCII。
    """
    meta = loc["meta"]
    out: list[str] = []
    for name in RULES:
        spec = loc["rules"][name]
        cells = [name, str(spec["condition"]), str(spec["verdict"])]
        out.append(f"{meta['row_prefix']}{f' {CELL_SEP} '.join(cells)}")
    return out


def canonical_rules(loc: dict[str, JsonValue]) -> tuple[tuple[str, str], ...]:
    """输入：语言字典。输出：处置表的规范形态 `((规则名, 处置), ...)`。

    预期行为：给 `FrameJudge` 与审计用——state 里那几行必须与它逐格相同，
    否则渲染挂错了表，答案会跟着一起错。
    """
    return tuple((name, str(loc["rules"][name]["verdict"])) for name in RULES)


def to_state(data: dict[str, JsonValue], tags: TagSet, loc: dict[str, JsonValue]) -> str:
    """把草稿与结构化事实拼成 state。

    输入：草稿字典（`content`）、注入特征、语言字典。
    输出：三段文本——本块内容 / 本块位置 / 处置表。
    预期行为：只拼接与去首尾空白，不改写模型写的字（反解校验要靠原文）；
    后两段**由代码写**（它们是注入特征与固定表，不是模型的发挥），所以从 `tags` 推出来。
    """
    meta = loc["meta"]
    header = locales.header
    lines = [f"{header(loc, meta['content_label'])}{str(data['content']).strip()}"]
    lines.append(
        f"{header(loc, meta['where_label'])}"
        f"{meta['paren'].format(text=meta['where_columns'])}"
    )
    lines.append(where_line(tags, loc))
    lines.append(
        f"{header(loc, meta['rules_label'])}"
        f"{meta['paren'].format(text=meta['rules_columns'])}"
    )
    lines.extend(rule_lines(loc))
    return "\n".join(lines)


@dataclass(frozen=True)
class WhereState:
    """从「本块位置」那一行里反解出来的三格。"""

    age: str
    pins: str
    budget: str


@dataclass(frozen=True)
class ParsedFrame:
    """从 state 里纯代码反解出来的事实。"""

    content: str
    where: WhereState | None
    rules: tuple[tuple[str, str], ...]

    @property
    def is_empty(self) -> bool:
        """输入：无。输出：正文那一段是不是空的。"""
        return not self.content.strip()


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
    return [cell.strip() for cell in line[len(prefix) :].split(CELL_SEP)]


def content_lines(state: object, loc: dict[str, JsonValue]) -> list[str]:
    """输入：state、语言字典。输出：正文那一段的行列表。

    预期行为：正常只有一行；模型把正文写成多行时这里会有多行——卫生判别器就靠这个
    认出「正文是多行」。非文本 state 返回空列表。
    """
    if not isinstance(state, str):
        return []
    return _sections(state, loc).get(str(loc["meta"]["content_label"]), [])


def content_of(state: object, loc: dict[str, JsonValue]) -> str:
    """输入：state、语言字典。输出：正文那一行（没有就空串）。"""
    lines = content_lines(state, loc)
    return lines[0] if lines else ""


def where_of(state: object, loc: dict[str, JsonValue]) -> WhereState | None:
    """输入：state、语言字典。输出：解析出来的位置行（解不出就 None）。

    预期行为：只认形如 `- recent | unpinned | tight` 的三格行；词元不在取值表里就当
    解不出来——判别器与审计会拿它跟注入特征逐格比对，串味会当场暴露。
    """
    if not isinstance(state, str):
        return None
    for line in _sections(state, loc).get(str(loc["meta"]["where_label"]), []):
        cells = _cells(line, loc)
        if cells is None or len(cells) != WHERE_CELLS:
            continue
        age, pins, budget = cells
        if age in AGES and pins in PINS and budget in BUDGETS:
            return WhereState(age=age, pins=pins, budget=budget)
        return None
    return None


def rules_of(state: str, loc: dict[str, JsonValue]) -> tuple[tuple[str, str], ...]:
    """输入：state、语言字典。输出：解析出来的处置表 `((规则名, 处置), ...)`。

    预期行为：只认「规则名 | 条件 | 处置」三格且首格是规则名的行；一行都对不上就返回空元组，
    判别器会把空元组当成「表没渲染出来」。
    """
    out: list[tuple[str, str]] = []
    for line in _sections(state, loc).get(str(loc["meta"]["rules_label"]), []):
        cells = _cells(line, loc)
        if cells is None or len(cells) != 3:
            continue
        name, _condition, verdict = cells
        if name in RULES and verdict in DISPOSITIONS:
            out.append((name, verdict))
    return tuple(out)


def parse_state(state: object) -> ParsedFrame:
    """输入：state。输出：纯代码能反解的部分（正文 + 位置行 + 处置表）。

    预期行为：与语言无关——把各语言的段落名都试一遍，命中即返回。
    """
    if not isinstance(state, str):
        return ParsedFrame(content="", where=None, rules=())
    for lang in locales.available():
        loc = locales.load(lang)
        content = content_of(state, loc)
        where = where_of(state, loc)
        if content or where is not None:
            return ParsedFrame(content=content, where=where, rules=rules_of(state, loc))
    return ParsedFrame(content="", where=None, rules=())