"""检索结果判读的文本外壳：规则表与「来源与时效」摊平，查询与结果条目由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：
- **规则表与「来源与时效」那一行由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**一句查询 + 一个结果条目的标题与摘要**（真实结果页的措辞是长尾的，
  模板拼的脱离现实）；
- 那三段文本必须体现注入的「查询意图 / 是否切题 / 是不是垃圾」三维，交给
  `judges.TraceJudge` 反解一次再比对。

提示词**只给模型这三维**，不给来源类型与时效：给了它就会把那两件事也写进标题或摘要，
而它们归「来源与时效」那一行管，两边一冲突（摘要写着「官方文档」、那一行写着
`unknown`）这道题该读哪一行就有歧义了。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
**本模块不发任何请求、也不执行任何东西**——它只把文本当字符串读写。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue

from . import locales
from .features import AUTHORITIES, FRESHNESSES, RULES, VERDICTS, TagSet

#: 提示词里特征的固定顺序：只列**由文本承载**的三维。顺序固定才能按提示哈希回放。
DIM_ORDER: Final[tuple[str, ...]] = ("query_kind", "topicality", "spam")

#: Markdown 围栏残留：模型偶尔把整段包进 ``` 里，那是实打实的数据损坏。
FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"```")

#: 两格之间的分隔符。选竖线是因为它在中英里写法相同，反解不用跨语言试。
CELL_SEP: Final[str] = "|"

#: 段落名在 locale 里的键（渲染与解析共用同一份清单）。
SECTION_KEYS: Final[tuple[str, ...]] = (
    "query_label",
    "source_label",
    "title_label",
    "snippet_label",
)

#: 「来源与时效」那一行的格数（来源类型 / 时效）。
SOURCE_CELLS: Final[int] = 2

#: 草稿里的三个文本字段。前两个走结构检查，后两个还要过禁用语——
#: 查询本身不描述「来源」，所以禁用语只查标题与摘要（见 `check_draft`）。
TEXT_FIELDS: Final[tuple[str, ...]] = ("query", "title", "snippet")

#: 要过禁用语的字段：它们描述的是**结果**，而来源类型与时效归那一行管。
BANNED_FIELDS: Final[tuple[str, ...]] = ("title", "snippet")

#: 草稿字段在 locale 里的中文名（报错信息用）。
FIELD_LABELS: Final[dict[str, str]] = {
    "query": "查询",
    "title": "标题",
    "snippet": "摘要",
}


class SearchDraft(BaseModel):
    """后端要填的草稿：**一句查询 + 一个结果条目的标题与摘要**。

    刻意只留三个字段——规则表与「来源与时效」全由组装器渲染，模型碰不到；
    字段越少，它越没有机会把结构化事实写歪（`check_draft` 会兜底）。
    """

    query: str = Field(
        min_length=1,
        description="用户在搜索框里敲的那句查询：一句话，说清要找什么",
    )
    title: str = Field(
        min_length=1,
        description="这一条搜索结果的标题：一行，像真实网页标题那样",
    )
    snippet: str = Field(
        min_length=1,
        description="这一条搜索结果的摘要：一到两句，搜索引擎抓下来的那段正文",
    )


def build_prompt(tags: TagSet, loc: dict[str, JsonValue], phrasing: int) -> str:
    """组装提示词。

    输入：一组注入特征、语言字典、问法编号。
    输出：完整提示词（只描述要写的那三维，不含答案）。
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

    预期行为：`response_model` 是 `SearchDraft`（instructor 要 pydantic 模型）；
    `spec` 带上特征本身，便于事后按来源切片。
    """
    return GenerationTask(
        task_id=task_id,
        kind="choice",
        domain="search-result-reading",
        prompt=build_prompt(tags, loc, 0),
        response_model=SearchDraft,
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
    """输入：一段文本、语言字典。输出：文本里出现的来源词 / 时效词（没有就空列表）。

    预期行为：只认 locale 里点名的那几条**高精度**话术——宁可漏检也不误判，
    误判会在渲染期白拒一条本来能用的数据。大小写不敏感（英文里 `Official` 与
    `official` 一样泄底）。
    """
    lowered = text.lower()
    return [
        str(phrase)
        for phrase in loc["banned_intent_phrases"]
        if str(phrase).lower() in lowered
    ]


def check_draft(data: dict[str, JsonValue], loc: dict[str, JsonValue]) -> None:
    """输入：草稿字典、语言字典。输出：无；不合规就抛 `ValueError`。

    预期行为：三段文本会**原样**拼进 state，拼进去就洗不掉了，所以拼之前逐条查：
    - 三段都非空、单行、不含 Markdown 围栏；
    - 开头不许再写一遍段落名（模型很爱这么干，写进去就是一行重复标记）；
    - **标题与摘要**不许出现来源词 / 时效词（它们归「来源与时效」那一行）——
      两边一旦冲突，这道题该读哪一行就有歧义了。查询本身不描述来源，所以不查它。
    """
    for field in TEXT_FIELDS:
        text = str(data.get(field, "")).strip()
        name = FIELD_LABELS[field]
        if not text:
            raise ValueError(f"草稿的「{name}」是空的")
        if FENCE_PATTERN.search(text):
            raise ValueError(f"草稿的「{name}」里混进了 Markdown 围栏")
        if "\n" in text:
            raise ValueError(f"草稿的「{name}」是多行——它必须是一行")
        leaked = label_leak(text, loc)
        if leaked:
            raise ValueError(f"草稿的「{name}」把段落名「{leaked}」也写进去了")
        if field in BANNED_FIELDS:
            echoed = banned_echo(text, loc)
            if echoed:
                raise ValueError(f"草稿的「{name}」把来源行管的事写进去了：{'、'.join(echoed)}")


def source_line(tags: TagSet, loc: dict[str, JsonValue]) -> str:
    """输入：注入特征、语言字典。输出：「来源与时效」那一行（两格都是 ASCII 词元）。"""
    meta = loc["meta"]
    cells = [tags.authority, tags.freshness]
    return f"{meta['row_prefix']}{f' {CELL_SEP} '.join(cells)}"


def rule_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：规则表的每一行（规则名 | 条件 | 处置）。

    预期行为：顺序严格按 `features.RULES`——规则表是**从上往下第一条命中**的语义，
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
    """输入：语言字典。输出：规则表的规范形态 `((规则名, 处置), ...)`。

    预期行为：给 `FrameJudge` 与审计用——state 里那几行必须与它逐格相同，
    否则渲染挂错了表，答案会跟着一起错。
    """
    return tuple((name, str(loc["rules"][name]["verdict"])) for name in RULES)


def to_state(data: dict[str, JsonValue], tags: TagSet, loc: dict[str, JsonValue]) -> str:
    """把草稿与结构化事实拼成 state。

    输入：草稿字典（`query` / `title` / `snippet`）、注入特征、语言字典。
    输出：四段文本——查询 / 来源与时效 / 标题 / 摘要 / 规则表。
    预期行为：只拼接与去首尾空白，不改写模型写的字（反解校验要靠原文）；
    来源行与规则表**由代码写**（它们是注入特征与固定表，不是模型的发挥），所以从 `tags` 推。
    """
    meta = loc["meta"]
    header = locales.header
    lines = [f"{header(loc, meta['query_label'])}{str(data['query']).strip()}"]
    lines.append(
        f"{header(loc, meta['source_label'])}"
        f"{meta['paren'].format(text=meta['source_columns'])}"
    )
    lines.append(source_line(tags, loc))
    lines.append(f"{header(loc, meta['title_label'])}{str(data['title']).strip()}")
    lines.append(f"{header(loc, meta['snippet_label'])}{str(data['snippet']).strip()}")
    lines.append(
        f"{header(loc, meta['rules_label'])}"
        f"{meta['paren'].format(text=meta['rules_columns'])}"
    )
    lines.extend(rule_lines(loc))
    return "\n".join(lines)


@dataclass(frozen=True)
class SourceState:
    """从「来源与时效」那一行里反解出来的两格。"""

    authority: str
    freshness: str


@dataclass(frozen=True)
class ParsedFrame:
    """从 state 里纯代码反解出来的事实。"""

    query: str
    title: str
    snippet: str
    source: SourceState | None
    rules: tuple[tuple[str, str], ...]

    @property
    def is_empty(self) -> bool:
        """输入：无。输出：三段文本是不是都空。"""
        return not (self.query.strip() or self.title.strip() or self.snippet.strip())


def _sections(state: str, loc: dict[str, JsonValue]) -> dict[str, list[str]]:
    """输入：state、语言字典。输出：`{段落名: 该段的行列表}`（行已去首尾空白）。"""
    meta = loc["meta"]
    titles = tuple(str(meta[key]) for key in SECTION_KEYS) + (str(meta["rules_label"]),)
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


def _field(state: object, loc: dict[str, JsonValue], key: str) -> str:
    """输入：state、语言字典、段落键。输出：该段的第一行（没有就空串）。"""
    if not isinstance(state, str):
        return ""
    lines = field_lines(state, loc, key)
    return lines[0] if lines else ""


def field_lines(state: object, loc: dict[str, JsonValue], key: str) -> list[str]:
    """输入：state、语言字典、段落键（如 `title_label`）。输出：该段的行列表。

    预期行为：正常只有一行；模型把某段写成多行时这里会有多行——卫生判别器就靠这个
    认出「这一段是多行」。非文本 state 返回空列表。
    """
    if not isinstance(state, str):
        return []
    return _sections(state, loc).get(str(loc["meta"][key]), [])


def query_of(state: object, loc: dict[str, JsonValue]) -> str:
    """输入：state、语言字典。输出：查询那一行（没有就空串）。"""
    return _field(state, loc, "query_label")


def title_of(state: object, loc: dict[str, JsonValue]) -> str:
    """输入：state、语言字典。输出：标题那一行（没有就空串）。"""
    return _field(state, loc, "title_label")


def snippet_of(state: object, loc: dict[str, JsonValue]) -> str:
    """输入：state、语言字典。输出：摘要那一行（没有就空串）。"""
    return _field(state, loc, "snippet_label")


def source_of(state: object, loc: dict[str, JsonValue]) -> SourceState | None:
    """输入：state、语言字典。输出：解析出来的来源行（解不出就 None）。

    预期行为：只认形如 `- community | fresh` 的两格行；词元不在取值表里就当解不出来——
    判别器与审计会拿它跟注入特征逐格比对，串味会当场暴露。
    """
    if not isinstance(state, str):
        return None
    for line in _sections(state, loc).get(str(loc["meta"]["source_label"]), []):
        cells = _cells(line, loc)
        if cells is None or len(cells) != SOURCE_CELLS:
            continue
        authority, freshness = cells
        if authority in AUTHORITIES and freshness in FRESHNESSES:
            return SourceState(authority=authority, freshness=freshness)
        return None
    return None


def rules_of(state: str, loc: dict[str, JsonValue]) -> tuple[tuple[str, str], ...]:
    """输入：state、语言字典。输出：解析出来的规则表 `((规则名, 处置), ...)`。

    预期行为：只认「规则名 | 条件 | 处置」三格且首格是规则名的行；一行都对不上就返回空元组，
    判别器会把空元组当成「表没渲染出来」。
    """
    out: list[tuple[str, str]] = []
    for line in _sections(state, loc).get(str(loc["meta"]["rules_label"]), []):
        cells = _cells(line, loc)
        if cells is None or len(cells) != 3:
            continue
        name, _condition, verdict = cells
        if name in RULES and verdict in VERDICTS:
            out.append((name, verdict))
    return tuple(out)


def parse_state(state: object) -> ParsedFrame:
    """输入：state。输出：纯代码能反解的部分（三段文本 + 来源行 + 规则表）。

    预期行为：与语言无关——把各语言的段落名都试一遍，命中即返回。
    """
    if not isinstance(state, str):
        return ParsedFrame(query="", title="", snippet="", source=None, rules=())
    for lang in locales.available():
        loc = locales.load(lang)
        query = query_of(state, loc)
        source = source_of(state, loc)
        if query or source is not None:
            return ParsedFrame(
                query=query,
                title=title_of(state, loc),
                snippet=snippet_of(state, loc),
                source=source,
                rules=rules_of(state, loc),
            )
    return ParsedFrame(query="", title="", snippet="", source=None, rules=())
