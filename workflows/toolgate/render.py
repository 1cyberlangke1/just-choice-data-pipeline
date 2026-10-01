"""工具调用门控的文本外壳：策略表与状态行摊平，那句调用意图由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：
- **「策略与状态」那一行与整张策略表由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写一句**调用意图**（真实 agent 的调用意图是长尾的，模板拼的脱离现实）；
- 那句意图必须体现注入的「工具」与「内容性质」两维，交给 `judges.TraceJudge` 反解一次
  再比对。

提示词**只给模型这两维**，不给作用范围 / 策略档位 / 审批状态：给了它就会把那三件事也
写进意图，而它们归「策略与状态」那一行管，两边一冲突（意图说「读本机的文件」、状态行写着
`external`）这道题该读哪一行就有歧义了。意图只回答「调哪个工具 + 动什么内容」，
作用范围与档位一律由代码那一行负责。

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

from . import locales
from .features import APPROVALS, DECISIONS, POLICIES, REACHES, RULES, TagSet

#: 提示词里特征的固定顺序：只列**由文本承载**的两维。顺序固定才能按提示哈希回放。
DIM_ORDER: Final[tuple[str, ...]] = ("tool", "payload")

#: Markdown 围栏残留：模型偶尔把整句话包进 ``` 里，那是实打实的数据损坏。
FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"```")

#: 三格之间的分隔符。选竖线是因为它在中英里写法相同，反解不用跨语言试。
CELL_SEP: Final[str] = "|"

#: 段落名在 locale 里的键（渲染与解析共用同一份清单）。
SECTION_KEYS: Final[tuple[str, ...]] = ("intent_label", "state_label", "rules_label")

#: 状态行的格数（作用范围 / 策略档位 / 审批状态）。
STATE_CELLS: Final[int] = 3


class ToolGateDraft(BaseModel):
    """后端要填的草稿：**只有一句调用意图**。

    刻意只留一个字段——策略表与状态行全由组装器渲染，模型碰不到；
    字段越少，它越没有机会把结构化事实写歪（`check_draft` 会兜底）。
    """

    intent: str = Field(
        description="agent 的调用意图：一句话，说清楚要调哪个工具、动的是什么内容",
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

    预期行为：`response_model` 是 `ToolGateDraft`（instructor 要 pydantic 模型）；
    `spec` 带上特征本身，便于事后按来源切片。
    """
    return GenerationTask(
        task_id=task_id,
        kind="choice",
        domain="tool-call-gating",
        prompt=build_prompt(tags, loc, 0),
        response_model=ToolGateDraft,
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


def banned_echo(intent: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：一句意图、语言字典。输出：意图里出现的位置词 / 处置词（没有就空列表）。

    预期行为：只认 locale 里点名的那几条**高精度**话术——宁可漏检也不误判，
    因为误判会在渲染期直接拒掉一条本来能用的数据（重渲染要再花一次配额）。
    这些词一旦出现，意图就会与「策略与状态」那一行抢同一件事。
    """
    lowered = intent.lower()
    return [
        str(phrase)
        for phrase in loc["banned_intent_phrases"]
        if str(phrase).lower() in lowered
    ]


def check_draft(data: dict[str, JsonValue], loc: dict[str, JsonValue]) -> None:
    """输入：草稿字典、语言字典。输出：无；不合规就抛 `DraftError`（管线据此重渲染）。

    预期行为：意图会**原样**拼进 state，拼进去就洗不掉了，所以拼之前逐条查：
    - 非空、单行、不含 Markdown 围栏；
    - 开头不许再写一遍段落名（模型很爱这么干，写进去就是一行重复标记）；
    - 不许出现位置词 / 处置词（作用范围与档位归「策略与状态」那一行）——
      两边一旦冲突，这道题该读哪一行就有歧义了。
    """
    intent = str(data.get("intent", "")).strip()
    if not intent:
        raise DraftError("草稿的「调用意图」是空的")
    if FENCE_PATTERN.search(intent):
        raise DraftError("草稿的「调用意图」里混进了 Markdown 围栏")
    if "\n" in intent:
        raise DraftError("草稿的「调用意图」是多行——它必须是一句话")
    leaked = label_leak(intent, loc)
    if leaked:
        raise DraftError(f"草稿的「调用意图」把段落名「{leaked}」也写进去了")
    echoed = banned_echo(intent, loc)
    if echoed:
        raise DraftError(f"草稿的「调用意图」把状态行管的事写进去了：{'、'.join(echoed)}")


def state_line(tags: TagSet, loc: dict[str, JsonValue]) -> str:
    """输入：注入特征、语言字典。输出：状态那一行（三格都是 ASCII 词元，中英一致）。"""
    meta = loc["meta"]
    cells = [tags.reach, tags.policy, tags.approval]
    return f"{meta['row_prefix']}{f' {CELL_SEP} '.join(cells)}"


def rule_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：策略表的每一行（规则名 | 条件 | 处置）。

    预期行为：顺序严格按 `features.RULES`——策略表是**从上往下第一条命中**的语义，
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
    """输入：语言字典。输出：策略表的规范形态 `((规则名, 处置), ...)`。

    预期行为：给 `FrameJudge` 与审计用——state 里那几行必须与它逐格相同，
    否则渲染挂错了表，答案会跟着一起错。
    """
    return tuple((name, str(loc["rules"][name]["verdict"])) for name in RULES)


def to_state(data: dict[str, JsonValue], tags: TagSet, loc: dict[str, JsonValue]) -> str:
    """把草稿与结构化事实拼成 state。

    输入：草稿字典（`intent`）、注入特征、语言字典。
    输出：三段文本——调用意图 / 策略与状态 / 策略表。
    预期行为：只拼接与去首尾空白，不改写模型写的字（反解校验要靠原文）；
    后两段**由代码写**（它们是注入特征与固定表，不是模型的发挥），所以从 `tags` 推出来。
    """
    meta = loc["meta"]
    header = locales.header
    lines = [f"{header(loc, meta['intent_label'])}{str(data['intent']).strip()}"]
    lines.append(
        f"{header(loc, meta['state_label'])}"
        f"{meta['paren'].format(text=meta['state_columns'])}"
    )
    lines.append(state_line(tags, loc))
    lines.append(
        f"{header(loc, meta['rules_label'])}"
        f"{meta['paren'].format(text=meta['rules_columns'])}"
    )
    lines.extend(rule_lines(loc))
    return "\n".join(lines)


@dataclass(frozen=True)
class CallState:
    """从状态行里反解出来的三格。"""

    reach: str
    policy: str
    approval: str


@dataclass(frozen=True)
class ParsedFrame:
    """从 state 里纯代码反解出来的事实。"""

    intent: str
    call: CallState | None
    rules: tuple[tuple[str, str], ...]

    @property
    def is_empty(self) -> bool:
        """输入：无。输出：意图那一行是不是空的。"""
        return not self.intent.strip()


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


def intent_of(state: object, loc: dict[str, JsonValue]) -> str:
    """输入：state、语言字典。输出：意图那一行（没有就空串）。"""
    lines = intent_lines(state, loc)
    return lines[0] if lines else ""


def intent_lines(state: object, loc: dict[str, JsonValue]) -> list[str]:
    """输入：state、语言字典。输出：意图那一段的行列表。

    预期行为：正常只有一行；模型把意图写成多行时这里会有多行——卫生判别器就靠这个
    认出「意图是多行」。
    """
    if not isinstance(state, str):
        return []
    return _sections(state, loc).get(str(loc["meta"]["intent_label"]), [])


def call_of(state: str, loc: dict[str, JsonValue]) -> CallState | None:
    """输入：state、语言字典。输出：解析出来的状态行（解不出就 None）。

    预期行为：只认形如 `- workspace | balanced | absent` 的三格行；词元不在取值表里
    就当解不出来——判别器与审计会拿它跟注入特征逐格比对，串味会当场暴露。
    """
    for line in _sections(state, loc).get(str(loc["meta"]["state_label"]), []):
        cells = _cells(line, loc)
        if cells is None or len(cells) != STATE_CELLS:
            continue
        reach, policy, approval = cells
        if reach in REACHES and policy in POLICIES and approval in APPROVALS:
            return CallState(reach=reach, policy=policy, approval=approval)
        return None
    return None


def rules_of(state: str, loc: dict[str, JsonValue]) -> tuple[tuple[str, str], ...]:
    """输入：state、语言字典。输出：解析出来的策略表 `((规则名, 处置), ...)`。

    预期行为：只认「规则名 | 条件 | 处置」三格且首格是规则名的行；一行都对不上就返回空元组，
    判别器会把空元组当成「表没渲染出来」。
    """
    out: list[tuple[str, str]] = []
    for line in _sections(state, loc).get(str(loc["meta"]["rules_label"]), []):
        cells = _cells(line, loc)
        if cells is None or len(cells) != 3:
            continue
        name, _condition, verdict = cells
        if name in RULES and verdict in DECISIONS:
            out.append((name, verdict))
    return tuple(out)


def parse_state(state: object) -> ParsedFrame:
    """输入：state。输出：纯代码能反解的部分（意图 + 状态行 + 策略表）。

    预期行为：与语言无关——把各语言的段落名都试一遍，命中即返回。
    """
    if not isinstance(state, str):
        return ParsedFrame(intent="", call=None, rules=())
    for lang in locales.available():
        loc = locales.load(lang)
        intent = intent_of(state, loc)
        call = call_of(state, loc)
        if intent or call is not None:
            return ParsedFrame(intent=intent, call=call, rules=rules_of(state, loc))
    return ParsedFrame(intent="", call=None, rules=())
