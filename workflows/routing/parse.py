"""工单分诊 workflow 的反解：把工单文本读回结构化字段。

输入：`state` 文本 + 标签表。输出：`ParsedTicket`。
预期行为：**不复用渲染时的任何计算**——只按文本本身解析，所以能抓到字段漏写、
标签写错、正文被截断这类硬伤。认不出的行（标题行）直接忽略。

这里只做**结构**反解（各格都是原文），把「原文 → 键」的翻译留给判别器——
判别器手上有语言文件，而 `Workflow.reparse` 没有语言参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from . import locales

#: 语言文件 `[meta]` 里那几个标签（反解要认全，缺一格就报错）。
META_KEYS: Final[tuple[str, ...]] = (
    "ticket_label",
    "area_label",
    "tier_label",
    "impact_label",
    "urgency_label",
    "compliance_label",
    "created_label",
    "current_label",
    "body_label",
)


@dataclass(frozen=True)
class ParsedTicket:
    """反解出来的一条工单：全是**原始文本**。"""

    ticket_id: str
    area: str
    tier: str
    impact: str
    urgency: str
    compliance: str
    created: str
    current: str
    body: str


def label_index(loc: dict[str, Any]) -> dict[str, str]:
    """输入：语言字典。输出：{标签原文: 槽位名}（只收 `META_KEYS` 里那几个）。

    预期行为：两个标签文案相同时报错——否则反解会把两格认成一格。
    """
    out: dict[str, str] = {}
    for key in META_KEYS:
        text = loc["meta"][key]
        if text in out:
            raise ValueError(f"标签文案重复：{text!r}（{out[text]} 与 {key}）")
        out[text] = key
    return out


def merged_labels() -> dict[str, str]:
    """输入：无。输出：全部语言的标签合并表。

    预期行为：供语言无关的 `reparse` 用；各语言的标签互不重叠（由 test_locales 锁住），
    重叠了这里会报错而不是静默取其一。
    """
    out: dict[str, str] = {}
    for lang in locales.available():
        for text, slot in label_index(locales.load(lang)).items():
            if text in out:
                raise ValueError(f"标签 {text!r} 在多种语言里都出现（{out[text]} / {slot}）")
            out[text] = slot
    return out


def _match_label(line: str, labels: dict[str, str]) -> tuple[str, str] | None:
    """输入：一行文本、标签表。输出：(槽位, 值) 或 None。

    预期行为：取**最长**的匹配标签（标签之间有前缀关系时才不会认错），
    并且要求标签后面紧跟分隔符——否则标题行「工单分诊台」会被认成工单号那一格
    （它确实以「工单」开头）。这是与检索 workflow 不同的地方，故意收紧。
    """
    best: tuple[str, str] | None = None
    for text, slot in labels.items():
        if not line.startswith(text):
            continue
        rest = line[len(text) :].lstrip()
        if not rest[:1] in ("：", ":"):
            continue
        if best is None or len(text) > len(best[0]):
            best = (text, slot)
    if best is None:
        return None
    value = line[len(best[0]) :].strip().lstrip("：:").strip()
    return best[1], value


def parse_ticket(state: str, labels: dict[str, str]) -> ParsedTicket:
    """输入：state 文本、标签表。输出：反解结果。

    预期行为：逐行认标签；认不出的行忽略；`META_KEYS` 缺任何一格都报人话——
    缺格说明渲染漏写了字段，是硬伤，不能当成「空值」蒙过去。
    """
    fields: dict[str, str] = {}
    for raw in state.splitlines():
        line = raw.strip()
        if not line:
            continue
        found = _match_label(line, labels)
        if found is None:
            continue
        fields[found[0]] = found[1]
    missing = [key for key in META_KEYS if key not in fields]
    if missing:
        raise ValueError(f"工单缺这几格：{missing}")
    return ParsedTicket(
        ticket_id=fields["ticket_label"],
        area=fields["area_label"],
        tier=fields["tier_label"],
        impact=fields["impact_label"],
        urgency=fields["urgency_label"],
        compliance=fields["compliance_label"],
        created=fields["created_label"],
        current=fields["current_label"],
        body=fields["body_label"],
    )

