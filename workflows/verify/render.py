"""完成声明核验的文本外壳：自报事实与验收门槛摊平成两行，那句声明由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：
- **自报事实与验收门槛由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写一句**完成声明**（这活儿只有模型干得出来：真实 agent 的收尾话术是长尾的）；
- 那句声明必须体现注入的「声称做完的事」与「口气」两维，交给 `judges.TraceJudge`
  反解一次再比对。

提示词**只给模型这两维**，不给自报事实：给了它就会把证据也写进声明，而证据那一行是
代码写的，两边一冲突（声明说「测试全过」、事实行写着 `all_fail`）这道题该读哪一行就
有歧义了。声明只回答「做完了什么 + 口气有多满」，证据一律由代码那一行负责。

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
from .features import ARTIFACTS, CHECKS, COVERAGES, Facts, TagSet, facts_for, gates_from

#: 提示词里特征的固定顺序：只列**由文本承载**的两维。顺序固定才能按提示哈希回放。
DIM_ORDER: Final[tuple[str, ...]] = ("claim_kind", "tone")

#: Markdown 围栏残留：模型偶尔把整句话包进 ``` 里，那是实打实的数据损坏。
FENCE_PATTERN: Final[re.Pattern[str]] = re.compile(r"```")

#: 三格之间的分隔符。选竖线是因为它在中英里写法相同，反解不用跨语言试。
CELL_SEP: Final[str] = "|"

#: 段落名在 locale 里的键（渲染与解析共用同一份清单）。
SECTION_KEYS: Final[tuple[str, ...]] = ("claim_label", "facts_label", "gates_label")

#: 三条门槛的个数（与 `features.GATE_COUNT` 一致，写在这里是为了让解析器不必 import）。
GATE_COUNT: Final[int] = 3


class VerifyDraft(BaseModel):
    """后端要填的草稿：**只有一句完成声明**。

    刻意只留一个字段——自报事实与验收门槛全由组装器渲染，模型碰不到；
    字段越少，它越没有机会把结构化事实写歪（`check_draft` 会兜底）。
    """

    claim: str = Field(
        min_length=1,
        description="agent 的收尾话：一句话，说清楚这次做完了什么、口气有多满",
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

    预期行为：`response_model` 是 `VerifyDraft`（instructor 要 pydantic 模型）；
    `spec` 带上特征本身，便于事后按来源切片。
    """
    return GenerationTask(
        task_id=task_id,
        kind="choice",
        domain="claim-verification",
        prompt=build_prompt(tags, loc, 0),
        response_model=VerifyDraft,
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


def evidence_echo(claim: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：一句声明、语言字典。输出：声明里出现的证据话术（没有就空列表）。

    预期行为：只认 locale 里点名的那几条**高精度**话术——宁可漏检也不误判，
    因为误判会在渲染期直接拒掉一条本来能用的数据（重渲染要再花一次配额）。
    """
    lowered = claim.lower()
    return [
        str(phrase)
        for phrase in loc["banned_claim_phrases"]
        if str(phrase).lower() in lowered
    ]


def check_draft(data: dict[str, JsonValue], loc: dict[str, JsonValue]) -> None:
    """输入：草稿字典、语言字典。输出：无；不合规就抛 `DraftError`（管线据此重渲染）。

    预期行为：声明那一句会**原样**拼进 state，拼进去就洗不掉了，所以拼之前逐条查：
    - 非空、单行、不含 Markdown 围栏；
    - 开头不许再写一遍段落名（模型很爱这么干，写进去就是一行重复标记）；
    - 不许把证据写进声明（「测试全过了」这类话是「自报事实」那一行的活）——
      两边一旦冲突，这道题该读哪一行就有歧义了。
    """
    claim = str(data.get("claim", "")).strip()
    if not claim:
        raise DraftError("草稿的「声明」是空的")
    if FENCE_PATTERN.search(claim):
        raise DraftError("草稿的「声明」里混进了 Markdown 围栏")
    if "\n" in claim:
        raise DraftError("草稿的「声明」是多行——它必须是一句话")
    leaked = label_leak(claim, loc)
    if leaked:
        raise DraftError(f"草稿的「声明」把段落名「{leaked}」也写进去了")
    echoed = evidence_echo(claim, loc)
    if echoed:
        raise DraftError(f"草稿的「声明」把证据写进去了：{'、'.join(echoed)}")


def facts_line(facts: Facts, loc: dict[str, JsonValue]) -> str:
    """输入：自报事实、语言字典。输出：事实那一行（三格都是 ASCII 词元，中英一致）。"""
    meta = loc["meta"]
    cells = [facts.checks, facts.coverage, facts.artifact]
    return f"{meta['row_prefix']}{f' {CELL_SEP} '.join(cells)}"


def gates_line(facts: Facts, loc: dict[str, JsonValue]) -> str:
    """输入：自报事实、语言字典。输出：门槛那一行（三条门槛各写 yes / no）。

    预期行为：门槛是固定的三条，写出来的是**过没过**——这一行与事实那一行由同一套
    纯函数算出，是同一份事实的两个视图；审计拿它们互相校验，不一致就是渲染串味了。
    """
    meta = loc["meta"]
    marks = [str(meta["yes"]) if gate else str(meta["no"]) for gate in gates_from(facts)]
    return f"{meta['row_prefix']}{f' {CELL_SEP} '.join(marks)}"


def to_state(data: dict[str, JsonValue], tags: TagSet, loc: dict[str, JsonValue]) -> str:
    """把草稿与结构化事实拼成 state。

    输入：草稿字典（`claim`）、注入特征、语言字典。
    输出：三段文本——声明 / 自报事实 / 验收门槛。
    预期行为：只拼接与去首尾空白，不改写模型写的字（反解校验要靠原文）；
    后两段**由代码写**（它们是注入特征，不是模型的发挥），所以从 `tags` 推出来。
    """
    meta = loc["meta"]
    facts = facts_for(tags)
    header = locales.header
    lines = [f"{header(loc, meta['claim_label'])}{str(data['claim']).strip()}"]
    lines.append(
        f"{header(loc, meta['facts_label'])}"
        f"{meta['paren'].format(text=meta['facts_columns'])}"
    )
    lines.append(facts_line(facts, loc))
    lines.append(
        f"{header(loc, meta['gates_label'])}"
        f"{meta['paren'].format(text=meta['gates_columns'])}"
    )
    lines.append(gates_line(facts, loc))
    return "\n".join(lines)


@dataclass(frozen=True)
class ParsedFrame:
    """从 state 里纯代码反解出来的事实。"""

    claim: str
    facts: Facts | None
    gates: tuple[bool, bool, bool] | None

    @property
    def is_empty(self) -> bool:
        """输入：无。输出：声明那一行是不是空的。"""
        return not self.claim.strip()


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


def claim_of(state: object, loc: dict[str, JsonValue]) -> str:
    """输入：state、语言字典。输出：声明那一行（没有就空串）。"""
    lines = claim_lines(state, loc)
    return lines[0] if lines else ""


def claim_lines(state: object, loc: dict[str, JsonValue]) -> list[str]:
    """输入：state、语言字典。输出：声明那一段的行列表。

    预期行为：正常只有一行；模型把声明写成多行时这里会有多行——卫生判别器就靠这个
    认出「声明是多行」。
    """
    if not isinstance(state, str):
        return []
    return _sections(state, loc).get(str(loc["meta"]["claim_label"]), [])


def facts_of(state: str, loc: dict[str, JsonValue]) -> Facts | None:
    """输入：state、语言字典。输出：解析出来的自报事实（解不出就 None）。

    预期行为：只认形如 `- all_pass | covered | committed` 的三格行；词元不在取值表里
    就当解不出来——判别器与审计会拿它跟注入特征逐格比对，串味会当场暴露。
    """
    for line in _sections(state, loc).get(str(loc["meta"]["facts_label"]), []):
        cells = _cells(line, loc)
        if cells is None or len(cells) != 3:
            continue
        checks, coverage, artifact = cells
        if checks in CHECKS and coverage in COVERAGES and artifact in ARTIFACTS:
            return Facts(checks=checks, coverage=coverage, artifact=artifact)
        return None
    return None


def gates_of(state: str, loc: dict[str, JsonValue]) -> tuple[bool, bool, bool] | None:
    """输入：state、语言字典。输出：解析出来的三条门槛（解不出就 None）。"""
    meta = loc["meta"]
    yes, no = str(meta["yes"]), str(meta["no"])
    for line in _sections(state, loc).get(str(meta["gates_label"]), []):
        cells = _cells(line, loc)
        if cells is None or len(cells) != GATE_COUNT:
            continue
        if not all(cell in {yes, no} for cell in cells):
            return None
        first, second, third = cells
        return (first == yes, second == yes, third == yes)
    return None


def parse_state(state: object) -> ParsedFrame:
    """输入：state。输出：纯代码能反解的部分（声明 + 自报事实 + 验收门槛）。

    预期行为：与语言无关——把各语言的段落名都试一遍，命中即返回。
    """
    if not isinstance(state, str):
        return ParsedFrame(claim="", facts=None, gates=None)
    for lang in locales.available():
        loc = locales.load(lang)
        claim = claim_of(state, loc)
        facts = facts_of(state, loc)
        if claim or facts is not None:
            return ParsedFrame(claim=claim, facts=facts, gates=gates_of(state, loc))
    return ParsedFrame(claim="", facts=None, gates=None)
