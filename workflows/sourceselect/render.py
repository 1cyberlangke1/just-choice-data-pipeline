"""来源选择的文本外壳：候选表与选源口径摊平，查询由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：
- **候选表与选源口径由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**一句查询**（真实请求的语气是长尾的，模板拼出来的脱离现实）；
- 那句查询必须体现注入的**首要要求**，交给 `judges.TraceJudge` 反解一次再比对。

提示词**只给模型这一维**，不给表长 / 形状 / 诱饵：给了它就会把这些结构也写进查询，
而它们归候选表管，两边一冲突（查询里写着「第 3 行不错」、表却另有其事）这道题该读哪边
就有歧义了。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
**本模块不发任何请求、也不执行任何东西**——它只把文本当字符串读写。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc import locale_files
from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.state_parse import split_sections
from jc.workflow import DraftError

from . import locales
from .features import AXIS_ORDER, AXIS_VALUES, RULES, SourceCase, SourcePlan, SourceRow

#: 提示词里特征的固定顺序：只列**由文本承载**的那一维。顺序固定才能按提示哈希回放。
DIM_ORDER: Final[tuple[str, ...]] = ("need",)

#: Markdown 围栏残留：模型偶尔把整句包进 ``` 里，那是实打实的数据损坏。
FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"```")

#: 「没填的槽位」：模型把提示词里的占位名照抄进查询，括号里还是那句占位话。
#: 全角/半角方括号都算，括号内容限 1~20 字——正常的一句查询不会这么写。
#: 依据：`sourceselect-4x-zh` 的 21 条合格样本里逐条读出来 3 条（`【主题】` / `[代号]` / `[未知主题]`）。
SLOT_PATTERN: Final[re.Pattern[str]] = re.compile(r"【[^】\n]{1,20}】|［[^］\n]{1,20}］|\[[^\]\n]{1,20}\]")

#: 两格之间的分隔符。选竖线是因为它在中英里写法相同，反解不用跨语言试。
CELL_SEP: Final[str] = "|"

#: 段落名在 locale 里的键（渲染与解析共用同一份清单）。
SECTION_KEYS: Final[tuple[str, ...]] = ("query_label", "sources_label", "rules_label")

#: 候选表那一行的格数（序号 / 类型 / 时效 / 覆盖度）。
ROW_CELLS: Final[int] = 4

#: 口径那一行的格数（规则名 / 说明）。
RULE_CELLS: Final[int] = 2

#: 草稿里的文本字段。查询不描述候选表，所以禁用语只查它有没有在谈那张表。
TEXT_FIELDS: Final[tuple[str, ...]] = ("query",)


class SourceDraft(BaseModel):
    """后端要填的草稿：**一句查询**。

    刻意只留一个字段——候选表与选源口径全由组装器渲染，模型碰不到；
    字段越少，它越没有机会把结构化事实写歪（`check_draft` 会兜底）。
    """

    query: str = Field(
        description="这次要找什么：一句话，写清楚在找什么、对来源有什么要求",
    )


def build_prompt(tags: SourceCase, loc: dict[str, JsonValue], phrasing: int) -> str:
    """组装提示词。

    输入：一组注入特征、语言字典、问法编号。
    输出：完整提示词（只描述要写的那一维，不含答案）。
    预期行为：特征按 `DIM_ORDER` 固定顺序逐行列出；规则整段附在后面，每条前面加短横线
    并把 `{language}` 填成语言名——留着占位符会把 `{language}` 原样发给模型。
    """
    values = tags.as_dict()
    lines = [str(loc["prompt"]["intro"]), ""]
    lines.extend(locales.feature_line(loc, dim, str(values[dim])) for dim in DIM_ORDER)
    lines.append("")
    rules = loc["prompt"]["rules"]
    offset = phrasing % len(rules)
    rotated = list(rules[offset:]) + list(rules[:offset])
    lines.extend(f"- {rule.format(language=loc['language_name'])}" for rule in rotated)
    return "\n".join(lines)


def build_task(tags: SourceCase, loc: dict[str, JsonValue], task_id: str) -> GenerationTask:
    """输入：注入特征、语言字典、任务 id。输出：`GenerationTask`。

    预期行为：`response_model` 是 `SourceDraft`（instructor 要 pydantic 模型）；
    `spec` 带上特征本身，便于事后按来源切片。
    """
    return GenerationTask(
        task_id=task_id,
        kind="choice",
        domain="source-selection",
        prompt=build_prompt(tags, loc, 0),
        response_model=SourceDraft,
        seeds=(),
        spec={"tags": tags.as_dict()},
    )


#: 转出去：查段落名的机制只有一份，住在 `jc.locale_files.label_leak`；本家只声明键表。
def label_leak(text: str, loc: dict[str, JsonValue]) -> str:
    """输入：一段文本、语言字典。输出：文本开头误写的段落名（没有就空串）。

    预期行为：只认「段落名 + 冒号」开头这一种写法——正文里正常提到这些词不算。
    """
    return locale_files.label_leak(text, loc, SECTION_KEYS)



#: 转出去：实现只有一份，住在 `jc.locale_files`（7 家逐字同构，只留那一份）。
banned_echo = locale_files.banned_echo


def check_draft(data: dict[str, JsonValue], loc: dict[str, JsonValue]) -> None:
    """输入：草稿字典、语言字典。输出：无；不合规就抛 `DraftError`（管线据此重渲染）。

    预期行为：那句查询会**原样**拼进 state，拼进去就洗不掉了，所以拼之前逐条查：
    - 非空、单行、不含 Markdown 围栏；
    - 不许有没填的槽位（`【主题】` / `[代号]` 这种）；
    - 开头不许再写一遍段落名（模型很爱这么干，写进去就是一行重复标记）；
    - 不许出现禁用话术：谈那张候选表（列名、序号、竖线）——那是代码渲染的地盘，写进去就是
      两边抢同一件事；以及**元话术**（`我需要输出…` / `根据以上要求，我会写出…`）——
      那种正文压根不是查询，反解判别器却照样能反解出一个首要要求，只能在渲染期按字面拦。
    """
    for field in TEXT_FIELDS:
        text = str(data.get(field, "")).strip()
        if not text:
            raise DraftError("草稿的「查询」是空的")
        if FENCE_PATTERN.search(text):
            raise DraftError("草稿的「查询」里混进了 Markdown 围栏")
        slot = SLOT_PATTERN.search(text)
        if slot:
            raise DraftError(f"草稿的「查询」里有没填的槽位：{slot.group(0)}")
        if "\n" in text:
            raise DraftError("草稿的「查询」是多行——它必须是一行")
        leaked = label_leak(text, loc)
        if leaked:
            raise DraftError(f"草稿的「查询」把段落名「{leaked}」也写进去了")
        echoed = banned_echo(text, loc)
        if echoed:
            raise DraftError(f"草稿的「查询」里出现了禁用话术：{'、'.join(echoed)}")


def row_line(index: int, row: SourceRow, loc: dict[str, JsonValue]) -> str:
    """输入：行号（1 起）、一行、语言字典。输出：候选表里的一行（四格都是 ASCII 词元）。"""
    meta = loc["meta"]
    cells = [str(index), *(row.get(axis) for axis in AXIS_ORDER)]
    return f"{meta['row_prefix']}{f' {CELL_SEP} '.join(cells)}"


def table_lines(plan: SourcePlan, loc: dict[str, JsonValue]) -> list[str]:
    """输入：候选表计划、语言字典。输出：候选表的每一行（序号从 1 起）。

    预期行为：顺序严格按计划里的展示顺序——答案「第几行」就是在这个顺序里的行号。
    """
    return [row_line(index, row, loc) for index, row in enumerate(plan.rows, start=1)]


def rule_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：选源口径的每一行（规则名 | 说明）。

    预期行为：顺序严格按 `features.RULES`——口径是**从上往下**读的，顺序变了语义就变了。
    规则名保持 ASCII，说明文案取自 locale。
    """
    meta = loc["meta"]
    return [
        f"{meta['row_prefix']}{f' {CELL_SEP} '.join([name, str(loc['rules'][name]['text'])])}"
        for name in RULES
    ]


def canonical_rules(loc: dict[str, JsonValue]) -> tuple[str, ...]:
    """输入：语言字典。输出：选源口径的规范形态（逐行文本）。

    预期行为：给 `FrameJudge` 与审计用——state 里那几行必须与它逐行相同，
    否则渲染挂错了口径，答案会跟着一起错。
    """
    return tuple(rule_lines(loc))


def to_state(data: dict[str, JsonValue], plan: SourcePlan, loc: dict[str, JsonValue]) -> str:
    """把草稿与结构化事实拼成 state。

    输入：草稿字典（`query`）、候选表计划、语言字典。
    输出：三段——查询 / 候选来源 / 选源口径。
    预期行为：只拼接与去首尾空白，不改写模型写的字（反解校验要靠原文）；
    候选表与选源口径**由代码写**（它们是计划的产物与固定表，不是模型的发挥）。
    """
    meta = loc["meta"]
    header = locales.header
    lines = [f"{header(loc, meta['query_label'])}{str(data['query']).strip()}"]
    lines.append(
        f"{header(loc, meta['sources_label'])}"
        f"{meta['paren'].format(text=meta['sources_columns'])}"
    )
    lines.extend(table_lines(plan, loc))
    lines.append(f"{header(loc, meta['rules_label'])}")
    lines.extend(rule_lines(loc))
    return "\n".join(lines)


@dataclass(frozen=True)
class ParsedFrame:
    """从 state 里纯代码反解出来的事实。

    字段：`query` 查询原文；`rows` 候选表（顺序即表序）；`rules` 口径那几行的规则名。
    """

    query: str
    rows: tuple[SourceRow, ...]
    rules: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        """输入：无。输出：查询是不是空的。"""
        return not self.query.strip()


#: 转出去：切段机制只有一份，住在 `jc.state_parse.split_sections`（11 家逐字同构，只留那一份）。
def _sections(state: str, loc: dict[str, JsonValue]) -> dict[str, list[str]]:
    """输入：state、语言字典。输出：{段落标题: 该段的正文行}。

    预期行为：只认「行首等于该语言的段落头」这一种切法——与 `render.to_state` 对称。
    """
    return split_sections(state, loc, SECTION_KEYS)


def _cells(line: str, loc: dict[str, JsonValue]) -> list[str] | None:
    """输入：一行、语言字典。输出：按分隔符切开的格子（不是数据行就 None）。"""
    prefix = str(loc["meta"]["row_prefix"])
    if not line.startswith(prefix):
        return None
    return [cell.strip() for cell in line[len(prefix) :].split(CELL_SEP)]


def section_lines(state: object, loc: dict[str, JsonValue], key: str) -> list[str]:
    """输入：state、语言字典、段落键（如 `query_label`）。输出：该段的行列表。

    预期行为：正常只有一行；模型把某段写成多行时这里会有多行——卫生判别器就靠这个
    认出「这一段是多行」。非文本 state 返回空列表。
    """
    if not isinstance(state, str):
        return []
    return _sections(state, loc).get(str(loc["meta"][key]), [])


def query_of(state: object, loc: dict[str, JsonValue]) -> str:
    """输入：state、语言字典。输出：查询那一行（没有就空串）。"""
    lines = section_lines(state, loc, "query_label")
    return lines[0] if lines else ""


def rows_of(state: object, loc: dict[str, JsonValue]) -> tuple[SourceRow, ...]:
    """输入：state、语言字典。输出：解析出来的候选表（解不出就是空元组）。

    预期行为：只认「序号 | 类型 | 时效 | 覆盖度」四格、序号从 1 起连续、三个词元都在取值表里
    的行；任何一行不合规就当整张表解不出来——判别器与审计拿它跟计划逐行比对，串味会当场暴露。
    """
    if not isinstance(state, str):
        return ()
    out: list[SourceRow] = []
    for line in _sections(state, loc).get(str(loc["meta"]["sources_label"]), []):
        cells = _cells(line, loc)
        if cells is None or len(cells) != ROW_CELLS:
            continue
        number, *values = cells
        if number != str(len(out) + 1):
            return ()
        row = SourceRow(**dict(zip(AXIS_ORDER, values)))
        if any(row.get(axis) not in AXIS_VALUES[axis] for axis in AXIS_ORDER):
            return ()
        out.append(row)
    return tuple(out)


def rules_of(state: object, loc: dict[str, JsonValue]) -> tuple[str, ...]:
    """输入：state、语言字典。输出：解析出来的口径规则名（解不出就是空元组）。

    预期行为：只认「规则名 | 说明」两格且首格是规则名的行；一行都对不上就返回空元组，
    判别器会把空元组当成「口径没渲染出来」。
    """
    if not isinstance(state, str):
        return ()
    out: list[str] = []
    for line in _sections(state, loc).get(str(loc["meta"]["rules_label"]), []):
        cells = _cells(line, loc)
        if cells is None or len(cells) != RULE_CELLS:
            continue
        name = cells[0]
        if name in RULES:
            out.append(name)
    return tuple(out)


def parse_state(state: object) -> ParsedFrame:
    """输入：state。输出：纯代码能反解的部分（查询 + 候选表 + 口径）。

    预期行为：与语言无关——把各语言的段落名都试一遍，命中即返回。
    """
    if not isinstance(state, str):
        return ParsedFrame(query="", rows=(), rules=())
    for lang in locales.available():
        loc = locales.load(lang)
        query = query_of(state, loc)
        rows = rows_of(state, loc)
        if query or rows:
            return ParsedFrame(query=query, rows=rows, rules=rules_of(state, loc))
    return ParsedFrame(query="", rows=(), rules=())
