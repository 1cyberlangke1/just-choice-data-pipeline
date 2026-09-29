"""浏览器 / GUI 操作的文本外壳：元素表与两张判据表摊平，那句话由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：
- **界面元素表、门控表、意图表由代码渲染**——它们是结构化事实（这一屏上有什么、每行什么
  状态、哪一档意图该动谁、什么算敏感、风险几档），让模型去写只会写歪；
- 后端只写**用户这一轮说的那一句话**（真实 agent 里用户那句话是长尾的，模板拼的脱离现实）；
- 那句话必须体现注入的 `intent` 与「用户指的是哪个元素」，交给 `judges.TraceJudge`
  反解一次再比对。

提示词**只给模型这两件事**（想干什么 + 指的是哪个元素），不给元素表里的状态 / 敏感 /
干扰行 / 屏型：给了它就会把那几件事也写进那句话，而它们归元素表管，两边一冲突
（那句话写着「那个置灰的按钮」、表里写着 `ready`）这道题该读哪一行就有歧义了。

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

#: 提示词里特征的固定顺序：只列**由那句话承载**的那一维。顺序固定才能按提示哈希回放。
DIM_ORDER: Final[tuple[str, ...]] = ("intent",)

#: Markdown 围栏残留：模型偶尔把整句话包进 ``` 里，那是实打实的数据损坏。
FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"```")

#: 格子之间的分隔符。选竖线是因为它在中英里写法相同，反解不用跨语言试。
CELL_SEP: Final[str] = "|"

#: 段落名在 locale 里的键（渲染与解析共用同一份清单）。
SECTION_KEYS: Final[tuple[str, ...]] = (
    "request_label",
    "elements_label",
    "gates_label",
    "criteria_label",
)

#: 元素表的格数（行号 | 角色 | 元素名 | 状态）。
ELEMENT_CELLS: Final[int] = 4

#: 门控表的格数（判据 | 说明）。
GATE_CELLS: Final[int] = 2

#: 意图表的格数（意图 | 角色 | 动作 | 不敏感时风险 | 敏感时风险）。
CRITERIA_CELLS: Final[int] = 5

#: 门控表的判据词元（顺序就是渲染顺序；`sensitive` 那一行的说明由代码把敏感词表填进去）。
GATE_TOKENS: Final[tuple[str, ...]] = ("sensitive", "done", "blocked", "confirm")

#: 元素表一行的形态：`(行号, 角色, 元素名, 状态)`。行号从 1 数起。
ElementRow = tuple[int, str, str, str]


class BrowserDraft(BaseModel):
    """后端要填的草稿：**只有用户这一轮说的那一句话**。

    刻意只留一个字段——元素表与两张判据表全由组装器渲染，模型碰不到；
    字段越少，它越没有机会把结构化事实写歪（`check_draft` 会兜底）。
    """

    request: str = Field(
        min_length=1,
        description="用户这一轮说的那一句话：一行，像用户真的敲进来的那句",
    )


def build_prompt(tags: TagSet, loc: dict[str, JsonValue], phrasing: int) -> str:
    """组装提示词。

    输入：一组注入特征、语言字典、问法编号。
    输出：完整提示词（只描述要写的那一维 + 要指的那个元素，不含答案）。
    预期行为：特征按 `DIM_ORDER` 固定顺序逐行列出，紧接着一行「用户指的是哪个元素」；
    规则整段附在后面，每条前面加短横线并把 `{language}` 填成语言名——留着占位符
    会把 `{language}` 原样发给模型。
    """
    values = tags.as_dict()
    lines = [str(loc["prompt"]["intro"]), ""]
    lines.extend(locales.feature_line(loc, dim, values[dim]) for dim in DIM_ORDER)
    lines.append(locales.target_line(loc, features.wanted_label(tags)))
    lines.append("")
    rules = loc["prompt"]["rules"]
    offset = phrasing % len(rules)
    rotated = list(rules[offset:]) + list(rules[:offset])
    lines.extend(f"- {rule.format(language=loc['language_name'])}" for rule in rotated)
    return "\n".join(lines)


def build_task(tags: TagSet, loc: dict[str, JsonValue], task_id: str) -> GenerationTask:
    """输入：注入特征、语言字典、任务 id。输出：`GenerationTask`。

    预期行为：`response_model` 是 `BrowserDraft`（instructor 要 pydantic 模型）；
    `spec` 带上特征本身，便于事后按来源切片。
    """
    return GenerationTask(
        task_id=task_id,
        kind="choice",
        domain="browser-gui-action",
        prompt=build_prompt(tags, loc, 0),
        response_model=BrowserDraft,
        seeds=(),
        spec={"tags": tags.as_dict()},
    )


def label_leak(text: str, loc: dict[str, JsonValue]) -> str:
    """输入：一段文本、语言字典。输出：文本开头误写的段落名（没有就空串）。

    预期行为：只认「段落名 + 冒号」开头这一种写法——那句话里正常提到这些词不算。
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
    """输入：一段文本、语言字典。输出：文本里出现的那几段表的话术（没有就空列表）。

    预期行为：只认 locale 里点名的那几条**高精度**短语——宁可漏检也不误判，
    误判会在渲染期白拒一条本来能用的数据。大小写不敏感（英文里 `Disabled` 与
    `disabled` 一样泄底）。
    """
    lowered = text.lower()
    return [
        str(phrase)
        for phrase in loc["banned_intent_phrases"]
        if str(phrase).lower() in lowered
    ]


def check_draft(data: dict[str, JsonValue], tags: TagSet, loc: dict[str, JsonValue]) -> None:
    """输入：草稿字典、注入特征、语言字典。输出：无；不合规就抛 `DraftError`（管线据此重渲染）。

    预期行为：那句话会**原样**拼进 state，拼进去就洗不掉了，所以拼之前逐条查：
    - 非空、单行、不含 Markdown 围栏；
    - 开头不许再写一遍段落名（模型很爱这么干，写进去就是一行重复标记）；
    - 不许出现元素表 / 门控表 / 意图表那几段的话术（它们归那几张表管）；
    - 必须**原样**出现 `features.wanted_label` 那个元素名——反解判别器要拿它比对，
      少了它这道题连「用户指的是谁」都读不出来。
    """
    request = str(data.get("request", "")).strip()
    if not request:
        raise DraftError("草稿的「用户请求」是空的")
    if FENCE_PATTERN.search(request):
        raise DraftError("草稿的「用户请求」里混进了 Markdown 围栏")
    if "\n" in request:
        raise DraftError("草稿的「用户请求」是多行——它必须是一行")
    leaked = label_leak(request, loc)
    if leaked:
        raise DraftError(f"草稿的「用户请求」把段落名「{leaked}」也写进去了")
    echoed = banned_echo(request, loc)
    if echoed:
        raise DraftError(f"草稿的「用户请求」把表里管的事写进去了：{'、'.join(echoed)}")
    label = features.wanted_label(tags)
    if label not in request:
        raise DraftError(f"草稿的「用户请求」里没有原样出现元素名「{label}」")


def _row_line(loc: dict[str, JsonValue], cells: tuple[str, ...]) -> str:
    """输入：语言字典、一行格子。输出：形如 `- fill | field | type` 的一行。"""
    return f"{loc['meta']['row_prefix']}{f' {CELL_SEP} '.join(cells)}"


def canonical_elements(tags: TagSet) -> tuple[ElementRow, ...]:
    """输入：注入特征。输出：这一屏元素表的规范形态（行号 1 起，按显示顺序）。

    预期行为：给 `FrameJudge` 与审计用——state 里那几行必须与它逐格相同，
    否则渲染挂错了 tags，答案会跟着一起错。
    """
    return tuple(
        (index, row.role, row.label, row.state)
        for index, row in enumerate(features.rows_for(tags), start=1)
    )


def element_lines(tags: TagSet, loc: dict[str, JsonValue]) -> list[str]:
    """输入：注入特征、语言字典。输出：元素表的每一行（行号 | 角色 | 元素名 | 状态）。"""
    return [
        _row_line(loc, (str(index), role, label, state))
        for index, role, label, state in canonical_elements(tags)
    ]


def gate_rows(loc: dict[str, JsonValue]) -> tuple[tuple[str, str], ...]:
    """输入：语言字典。输出：门控表的规范形态 `((判据, 说明), ...)`。

    预期行为：敏感词表**由代码填进说明里**（`{words}` 槽位），不靠人把词表抄进
    locale——抄一遍就会漂移，而「什么算敏感」正是 `confirm` 那一族的判据。
    """
    gates = loc["gates"]
    words = "、".join(features.SENSITIVE_WORDS)
    return (
        ("sensitive", str(gates["sensitive"]).format(words=words)),
        ("done", str(gates["done"])),
        ("blocked", str(gates["blocked"])),
        ("confirm", str(gates["confirm"])),
    )


def gate_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：门控表的每一行（判据 | 说明）。"""
    return [_row_line(loc, cells) for cells in gate_rows(loc)]


def gate_row_variants() -> frozenset[tuple[tuple[str, str], ...]]:
    """输入：无。输出：各语言下门控表规范形态的集合。

    预期行为：一次 run 里语言按样本轮转（`locales[seq % len]`），所以审计拿到的 `locale`
    未必是产物那一种；「说明」那一列是随语言渲染的文案，判定必须写「等于**任一**语言的
    规范表」，拿调用方给的那一种去比会把另一种语言的产物误判成「门控表与规范表对不上」。
    """
    return frozenset(gate_rows(locales.load(lang)) for lang in locales.available())


def criteria_rows() -> tuple[tuple[str, ...], ...]:
    """输入：无。输出：意图表的规范形态 `((意图, 角色, 动作, 不敏感风险, 敏感风险), ...)`。

    预期行为：全 ASCII 词元，所以**与语言无关**——它就是 `features` 里那三张表的
    摊平（意图表 / 动作表 / 风险表），没有第二种写法。
    """
    out: list[tuple[str, ...]] = []
    for intent in features.INTENTS:
        role = features.role_for_intent(intent)
        action = features.action_for_intent(intent)
        risk_no = features.RISK_TABLE[(intent, "no")]
        risk_yes = features.RISK_TABLE[(intent, "yes")]
        out.append((intent, role, action, str(risk_no), str(risk_yes)))
    return tuple(out)


def criteria_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：意图表的每一行（意图 | 角色 | 动作 | 不敏感风险 | 敏感风险）。"""
    return [_row_line(loc, cells) for cells in criteria_rows()]


def to_state(data: dict[str, JsonValue], tags: TagSet, loc: dict[str, JsonValue]) -> str:
    """把草稿与结构化事实拼成 state。

    输入：草稿字典（`request`）、注入特征、语言字典。
    输出：四段文本——用户请求 / 界面元素 / 门控表 / 意图表。
    预期行为：只拼接与去首尾空白，不改写模型写的字（反解校验要靠原文）；
    后三段**由代码写**（它们是注入特征与固定表，不是模型的发挥），所以从 `tags` 推出来。
    """
    meta = loc["meta"]
    header = locales.header
    lines = [f"{header(loc, meta['request_label'])}{str(data['request']).strip()}"]
    for key, columns, rows in (
        ("elements_label", "elements_columns", element_lines(tags, loc)),
        ("gates_label", "gates_columns", gate_lines(loc)),
        ("criteria_label", "criteria_columns", criteria_lines(loc)),
    ):
        lines.append(f"{header(loc, meta[key])}{meta['paren'].format(text=meta[columns])}")
        lines.extend(rows)
    return "\n".join(lines)


@dataclass(frozen=True)
class ParsedScreen:
    """从 state 里纯代码反解出来的事实。"""

    request: str
    elements: tuple[ElementRow, ...] | None
    gates: tuple[tuple[str, str], ...] | None
    criteria: tuple[tuple[str, ...], ...] | None

    @property
    def is_empty(self) -> bool:
        """输入：无。输出：那句话那一段是不是空的。"""
        return not self.request.strip()


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


def request_lines(state: object, loc: dict[str, JsonValue]) -> list[str]:
    """输入：state、语言字典。输出：那句话那一段的行列表。

    预期行为：正常只有一行；模型把那句话写成多行时这里会有多行——卫生判别器就靠这个
    认出「那句话是多行」。非文本 state 返回空列表。
    """
    if not isinstance(state, str):
        return []
    return _sections(state, loc).get(str(loc["meta"]["request_label"]), [])


def request_of(state: object, loc: dict[str, JsonValue]) -> str:
    """输入：state、语言字典。输出：那句话（没有就空串）。"""
    lines = request_lines(state, loc)
    return lines[0] if lines else ""


def elements_of(state: object, loc: dict[str, JsonValue]) -> tuple[ElementRow, ...] | None:
    """输入：state、语言字典。输出：解析出来的元素表（解不出就 None）。

    预期行为：只认「行号 | 角色 | 元素名 | 状态」四格行，而且行号必须正好是 1..N——
    词元不在取值表里、行号跳号或重号，都当解不出来：判别器与审计会拿它跟代码算出来的
    规范表逐格比对，串味会当场暴露。
    """
    if not isinstance(state, str):
        return None
    out: list[ElementRow] = []
    for line in _sections(state, loc).get(str(loc["meta"]["elements_label"]), []):
        cells = _cells(line, loc)
        if cells is None:
            continue  # 段落头那一行（列名）不是数据行，跳过
        if len(cells) != ELEMENT_CELLS:
            return None
        index_text, role, label, state_value = cells
        if not index_text.isdigit():
            return None
        if role not in features.ROLES or state_value not in features.STATES:
            return None
        out.append((int(index_text), role, label, state_value))
    if not out:
        return None
    if [row[0] for row in out] != list(range(1, len(out) + 1)):
        return None
    return tuple(out)


def gates_of(state: object, loc: dict[str, JsonValue]) -> tuple[tuple[str, str], ...] | None:
    """输入：state、语言字典。输出：解析出来的门控表（解不出就 None）。

    预期行为：只认「判据 | 说明」两格且判据在 `GATE_TOKENS` 里的行；一行对不上就返回 None，
    判别器会把 None 当成「表没渲染出来」。
    """
    if not isinstance(state, str):
        return None
    out: list[tuple[str, str]] = []
    for line in _sections(state, loc).get(str(loc["meta"]["gates_label"]), []):
        cells = _cells(line, loc)
        if cells is None:
            continue  # 段落头那一行（列名）不是数据行，跳过
        if len(cells) != GATE_CELLS:
            return None
        token, text = cells
        if token not in GATE_TOKENS:
            return None
        out.append((token, text))
    return tuple(out) if out else None


def criteria_of(state: object, loc: dict[str, JsonValue]) -> tuple[tuple[str, ...], ...] | None:
    """输入：state、语言字典。输出：解析出来的意图表（解不出就 None）。

    预期行为：只认五格行且首格在 `features.INTENTS` 里；一行对不上就返回 None。
    """
    if not isinstance(state, str):
        return None
    out: list[tuple[str, ...]] = []
    for line in _sections(state, loc).get(str(loc["meta"]["criteria_label"]), []):
        cells = _cells(line, loc)
        if cells is None:
            continue  # 段落头那一行（列名）不是数据行，跳过
        if len(cells) != CRITERIA_CELLS:
            return None
        if cells[0] not in features.INTENTS:
            return None
        out.append(tuple(cells))
    return tuple(out) if out else None


def parse_state(state: object) -> ParsedScreen:
    """输入：state。输出：纯代码能反解的部分（那句话 + 三张表）。

    预期行为：与语言无关——把各语言的段落名都试一遍，命中即返回。
    """
    empty = ParsedScreen(request="", elements=None, gates=None, criteria=None)
    if not isinstance(state, str):
        return empty
    for lang in locales.available():
        loc = locales.load(lang)
        request = request_of(state, loc)
        elements = elements_of(state, loc)
        if request or elements is not None:
            return ParsedScreen(
                request=request,
                elements=elements,
                gates=gates_of(state, loc),
                criteria=criteria_of(state, loc),
            )
    return empty
