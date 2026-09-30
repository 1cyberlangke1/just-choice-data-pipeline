"""改动引入性判定的文本外壳：改动记录与判定口径摊平，那条发现与修法由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：
- **改动记录与判定口径由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**一条审查发现 + 一条建议修法**（真实审查意见的措辞是长尾的，模板拼出来的
  「第 N 行有问题」不像人话）；
- 那条发现必须写清**指向哪一行**、**引用哪个符号**，修法必须写清**点不点名具体符号**，
  这三件事分别由 `judges.TraceJudge` 反解一次再比对。

提示词**只给模型由文本承载的那两维**（指向何处、修法点不点名），不给证据落在哪 / 表长 /
有没有诱饵行：给了它就会把这些结构也写进发现，而它们归记录表管，两边一冲突这道题该读哪边
就有歧义了。

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
from jc.workflow import DraftError

from . import locales
from .features import RULES, TOUCH_KINDS, RecordPlan, RecordRow

#: 提示词里特征的固定顺序：只列**由文本承载**的那两维。顺序固定才能按提示哈希回放。
DIM_ORDER: Final[tuple[str, ...]] = ("cite", "fix")

#: Markdown 围栏残留：模型偶尔把整句包进 ``` 里，那是实打实的数据损坏。
FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"```")

#: 反引号包起来的记号。审查意见里点名符号的正常写法，也是审计认「点了名」的唯一依据。
BACKTICK_PATTERN: Final[re.Pattern[str]] = re.compile(r"`([^`\n]+)`")

#: 两格之间的分隔符。选竖线是因为它在中英里写法相同，反解不用跨语言试。
CELL_SEP: Final[str] = "|"

#: 段落名在 locale 里的键（渲染与解析共用同一份清单）。
SECTION_KEYS: Final[tuple[str, ...]] = (
    "finding_label",
    "fix_label",
    "record_label",
    "rules_label",
)

#: 改动记录那一行的格数（序号 / 行号 / 本次改动 / 符号）。
ROW_CELLS: Final[int] = 4

#: 口径那一行的格数（规则名 / 说明）。
RULE_CELLS: Final[int] = 2

#: 草稿里的文本字段。
TEXT_FIELDS: Final[tuple[str, ...]] = ("finding", "fix")


class FindingDraft(BaseModel):
    """后端要填的草稿：**一条发现 + 一条修法**，各一行。

    刻意只留两个字段——改动记录与判定口径全由组装器渲染，模型碰不到；
    字段越少，它越没有机会把结构化事实写歪（`check_draft` 会兜底）。
    """

    finding: str = Field(
        min_length=1,
        description="一条审查发现：写清指向哪一行、引用的符号（用反引号括起来）、问题是什么",
    )
    fix: str = Field(
        min_length=1,
        description="这条发现建议怎么改：点名具体符号就用反引号括起来，落不了地就只写泛泛的话",
    )


def build_prompt(plan: RecordPlan, loc: dict[str, JsonValue], phrasing: int) -> str:
    """组装提示词。

    输入：改动记录计划、语言字典、问法编号。
    输出：完整提示词（只描述要写的那两维，不含答案）。
    预期行为：特征按 `DIM_ORDER` 固定顺序逐行列出，再补三行**场景事实**（指向第几行、
    引用哪个符号、修法点不点名）——这三件不给，模型写不出与计划对得上的那两行；
    规则整段附在后面，每条前面加短横线并把 `{language}` 填成语言名
    ——留着占位符会把 `{language}` 原样发给模型。
    """
    prompt = loc["prompt"]
    values = plan.case.as_dict()
    lines = [str(prompt["intro"]), ""]
    lines.extend(locales.feature_line(loc, dim, str(values[dim])) for dim in DIM_ORDER)
    lines.append("")
    lines.append(str(prompt["cite_line"]).format(n=plan.cited_line))
    lines.append(str(prompt["symbol_line"]).format(symbol=f"`{plan.evidence_symbol}`"))
    if plan.case.fix == "concrete":
        lines.append(str(prompt["fix_symbol_line"]).format(symbol=f"`{plan.fix_symbol}`"))
    else:
        lines.append(str(prompt["fix_vague_line"]))
    lines.append("")
    rules = prompt["rules"]
    offset = phrasing % len(rules)
    rotated = list(rules[offset:]) + list(rules[:offset])
    lines.extend(f"- {rule.format(language=loc['language_name'])}" for rule in rotated)
    return "\n".join(lines)


def build_task(plan: RecordPlan, loc: dict[str, JsonValue], task_id: str) -> GenerationTask:
    """输入：改动记录计划、语言字典、任务 id。输出：`GenerationTask`。

    预期行为：`response_model` 是 `FindingDraft`（instructor 要 pydantic 模型）；
    `spec` 带上特征本身与**这一条的场景事实**（指向哪一行、引用哪个符号、修法点名哪个符号）
    ——事实要带上是因为计划由 case 指纹与问法共同决定，光有 tags 重算不出来；
    测试里的假后端也靠它现造草稿（零 token）。
    """
    return GenerationTask(
        task_id=task_id,
        kind="choice",
        domain="finding-triage",
        prompt=build_prompt(plan, loc, 0),
        response_model=FindingDraft,
        seeds=(),
        spec={
            "tags": plan.case.as_dict(),
            "locale": str(loc["lang"]),
            "plan": {
                "cite_line": plan.cited_line,
                "evidence_symbol": plan.evidence_symbol,
                "fix_symbol": plan.fix_symbol,
            },
        },
    )


def backticked(text: str) -> list[str]:
    """输入：一段文本。输出：反引号包起来的记号（按出现顺序，可能重复）。

    预期行为：这是「点了名」的唯一判据——提示词里就是这么要求的，审计也照这一条重推。
    """
    return [match.strip() for match in BACKTICK_PATTERN.findall(text)]


def cite_line_of(finding: str, loc: dict[str, JsonValue]) -> int | None:
    """输入：发现原文、语言字典。输出：它指向的行号（没写就是 `None`）。

    预期行为：按 locale 里的 `cite_pattern` 抓**第一处**——提示词要求只写一处；
    抓不到就返回 `None`，由 `check_draft` 报人话。
    """
    pattern = re.compile(str(loc["meta"]["cite_pattern"]))
    match = pattern.search(finding)
    return None if match is None else int(match.group(1))


def cite_line_any(finding: str) -> int | None:
    """输入：发现原文。输出：它指向的行号（与语言无关地试一遍）。

    预期行为：审计拿不到「调用方给的哪一种语言」，所以把各语言的行号写法都试一遍；
    都认不出来就返回 `None`（审计会把它记成「解析不出来」，不硬编一个假的答案）。
    """
    for lang in locales.available():
        line = cite_line_of(finding, locales.load(lang))
        if line is not None:
            return line
    return None


def check_draft(data: dict[str, JsonValue], plan: RecordPlan, loc: dict[str, JsonValue]) -> None:
    """输入：草稿字典、改动记录计划、语言字典。输出：无；不合规就抛 `DraftError`。

    预期行为：两行文本会**原样**拼进 state，拼进去就洗不掉了，所以拼之前逐条查：
    - 非空、单行、不含 Markdown 围栏；
    - 发现必须写清指向哪一行（且与计划一致）、必须用反引号括起证据符号；
    - 修法点名符号（`concrete`）时该符号必须在反引号里；泛泛的话（`vague`）时一个反引号都不许有
      ——这条是**可判**的根基：审计就靠「修法里有没有反引号记号」重推「落不落地」。
    """
    finding = str(data.get("finding", "")).strip()
    fix = str(data.get("fix", "")).strip()
    for label, text in zip(TEXT_FIELDS, (finding, fix)):
        if not text:
            raise DraftError(f"草稿的「{label}」是空的")
        if FENCE_PATTERN.search(text):
            raise DraftError(f"草稿的「{label}」里混进了 Markdown 围栏")
        if "\n" in text:
            raise DraftError(f"草稿的「{label}」是多行——它必须是一行")
    cited = cite_line_of(finding, loc)
    if cited is None:
        raise DraftError(f"草稿的「finding」没写清指向哪一行（要写成 {loc['meta']['cite_hint']}）")
    if cited != plan.cited_line:
        raise DraftError(f"草稿的「finding」指向第 {cited} 行，应为第 {plan.cited_line} 行")
    if plan.evidence_symbol not in backticked(finding):
        raise DraftError(f"草稿的「finding」没用反引号括起证据符号 {plan.evidence_symbol}")
    marks = backticked(fix)
    if plan.case.fix == "concrete":
        if plan.fix_symbol not in marks:
            raise DraftError(f"草稿的「fix」该点名符号 {plan.fix_symbol}（用反引号括起来）")
    elif marks:
        raise DraftError(f"草稿的「fix」是泛泛的话，不该出现反引号记号：{'、'.join(marks)}")


def row_line(index: int, row: RecordRow, loc: dict[str, JsonValue]) -> str:
    """输入：序号（1 起）、一行、语言字典。输出：改动记录里的一行（四格都是 ASCII 词元）。"""
    meta = loc["meta"]
    cells = [str(index), str(row.line), row.touch, row.symbol]
    return f"{meta['row_prefix']}{f' {CELL_SEP} '.join(cells)}"


def table_lines(plan: RecordPlan, loc: dict[str, JsonValue]) -> list[str]:
    """输入：记录计划、语言字典。输出：改动记录的每一行（序号从 1 起）。

    预期行为：顺序严格按计划里的展示顺序——答案「第几行」就是在这个顺序里的序号。
    """
    return [row_line(index, row, loc) for index, row in enumerate(plan.rows, start=1)]


def rule_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：判定口径的每一行（规则名 | 说明）。

    预期行为：顺序严格按 `features.RULES`——口径是**从上往下**读的，顺序变了语义就变了。
    规则名保持 ASCII，说明文案取自 locale。
    """
    meta = loc["meta"]
    return [
        f"{meta['row_prefix']}{f' {CELL_SEP} '.join([name, str(loc['rules'][name]['text'])])}"
        for name in RULES
    ]


def canonical_rules(loc: dict[str, JsonValue]) -> tuple[str, ...]:
    """输入：语言字典。输出：判定口径的规范形态（逐行文本）。

    预期行为：给 `FrameJudge` 与审计用——state 里那几行必须与它逐行相同，
    否则渲染挂错了口径，答案会跟着一起错。
    """
    return tuple(rule_lines(loc))


def to_state(data: dict[str, JsonValue], plan: RecordPlan, loc: dict[str, JsonValue]) -> str:
    """把草稿与结构化事实拼成 state。

    输入：草稿字典（`finding` / `fix`）、记录计划、语言字典。
    输出：四段——审查发现 / 建议修法 / 改动记录 / 判定口径。
    预期行为：只拼接与去首尾空白，不改写模型写的字（反解校验要靠原文）；
    改动记录与判定口径**由代码写**（它们是计划的产物与固定表，不是模型的发挥）。
    """
    meta = loc["meta"]
    header = locales.header
    lines = [f"{header(loc, meta['finding_label'])}{str(data['finding']).strip()}"]
    lines.append(f"{header(loc, meta['fix_label'])}{str(data['fix']).strip()}")
    lines.append(
        f"{header(loc, meta['record_label'])}"
        f"{meta['paren'].format(text=meta['record_columns'])}"
    )
    lines.extend(table_lines(plan, loc))
    lines.append(f"{header(loc, meta['rules_label'])}")
    lines.extend(rule_lines(loc))
    return "\n".join(lines)


@dataclass(frozen=True)
class ParsedFinding:
    """从 state 里纯代码反解出来的事实。

    字段：`finding` 发现原文；`fix` 修法原文；`rows` 改动记录（顺序即表序）；
    `rules` 口径那几行的规则名。
    """

    finding: str
    fix: str
    rows: tuple[RecordRow, ...]
    rules: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        """输入：无。输出：发现是不是空的。"""
        return not self.finding.strip()


def _sections(state: str, loc: dict[str, JsonValue]) -> dict[str, list[str]]:
    """输入：state、语言字典。输出：{段落标题: 该段的正文行}。

    预期行为：只认「行首等于该语言的段落头」这一种切法——与 `render.to_state` 对称。
    """
    titles = [str(loc["meta"][key]) for key in SECTION_KEYS]
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


def section_lines(state: object, loc: dict[str, JsonValue], key: str) -> list[str]:
    """输入：state、语言字典、段落键（如 `finding_label`）。输出：该段的行列表。

    预期行为：正常只有一行；模型把某段写成多行时这里会有多行——卫生判别器就靠这个
    认出「这一段是多行」。非文本 state 返回空列表。
    """
    if not isinstance(state, str):
        return []
    return _sections(state, loc).get(str(loc["meta"][key]), [])


def finding_of(state: object, loc: dict[str, JsonValue]) -> str:
    """输入：state、语言字典。输出：发现那一行（没有就空串）。"""
    lines = section_lines(state, loc, "finding_label")
    return lines[0] if lines else ""


def fix_of(state: object, loc: dict[str, JsonValue]) -> str:
    """输入：state、语言字典。输出：修法那一行（没有就空串）。"""
    lines = section_lines(state, loc, "fix_label")
    return lines[0] if lines else ""


def rows_of(state: object, loc: dict[str, JsonValue]) -> tuple[RecordRow, ...]:
    """输入：state、语言字典。输出：解析出来的改动记录（解不出就是空元组）。

    预期行为：只认「序号 | 行号 | 本次改动 | 符号」四格、序号从 1 起连续、
    行号是正整数、本次改动在 `TOUCH_KINDS` 里的行；任何一行不合规就当整张表解不出来
    ——判别器与审计拿它跟计划逐行比对，串味会当场暴露。
    """
    if not isinstance(state, str):
        return ()
    out: list[RecordRow] = []
    for line in _sections(state, loc).get(str(loc["meta"]["record_label"]), []):
        cells = _cells(line, loc)
        if cells is None or len(cells) != ROW_CELLS:
            continue
        number, raw_line, touch, symbol = cells
        if number != str(len(out) + 1):
            return ()
        if not raw_line.isdigit() or not symbol or touch not in TOUCH_KINDS:
            return ()
        out.append(RecordRow(line=int(raw_line), touch=touch, symbol=symbol))
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
        if cells[0] in RULES:
            out.append(cells[0])
    return tuple(out)


def parse_state(state: object) -> ParsedFinding:
    """输入：state。输出：纯代码能反解的部分（发现 + 修法 + 改动记录 + 口径）。

    预期行为：与语言无关——把各语言的段落名都试一遍，命中即返回。
    """
    if not isinstance(state, str):
        return ParsedFinding(finding="", fix="", rows=(), rules=())
    for lang in locales.available():
        loc = locales.load(lang)
        finding = finding_of(state, loc)
        rows = rows_of(state, loc)
        if finding or rows:
            return ParsedFinding(
                finding=finding, fix=fix_of(state, loc), rows=rows, rules=rules_of(state, loc)
            )
    return ParsedFinding(finding="", fix="", rows=(), rules=())
