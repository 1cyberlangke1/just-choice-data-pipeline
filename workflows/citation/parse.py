"""引文核查 workflow 的反解：把文本读回「文档 + 主张 + 引文 + 声称出处」。

输入：`state` 文本 + 标签表。输出：`ParsedCitation`。
预期行为：**不复用渲染时的任何计算**——只按文本本身解析，所以能抓到字段漏写、小节错位、
句子被折行这类硬伤。认不出的行（抬头）直接忽略。

两种小节编号写法（`[1] 标题` / `1. 标题`）都在这里认；两种语言的标签靠**合并表**认，
所以反解与语言无关（各语言的标签互不重叠，由 `test_locales.py` 锁住）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

from . import locales

#: 标签键（语言文件 `[meta]` 里那几个）。
META_KEYS: Final[tuple[str, ...]] = ("doc_label", "claim_label", "quote_label", "source_label")

_HEAD_BRACKET: Final[re.Pattern[str]] = re.compile(r"^\[(\d+)\]\s*(.*)$")
_HEAD_DOTTED: Final[re.Pattern[str]] = re.compile(r"^(\d+)\.\s+(.*)$")
_NUMBER: Final[re.Pattern[str]] = re.compile(r"\d+")


@dataclass(frozen=True)
class ParsedSection:
    """反解出来的一个小节：编号、标题原文、句子原文（顺序保持）。"""

    index: int
    heading: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class ParsedCitation:
    """反解出来的整页：全是**原始文本**（外加从编号行读出来的小节编号）。"""

    doc: str
    sections: tuple[ParsedSection, ...]
    claim: str
    quote: str
    source: str

    def section(self, index: int) -> ParsedSection:
        """输入：小节编号。输出：那个小节。预期行为：找不到直接报错，不静默返回空。"""
        for section in self.sections:
            if section.index == index:
                return section
        raise ValueError(f"文本里没有编号为 {index} 的小节")


def label_index(loc: dict[str, Any]) -> dict[str, str]:
    """输入：语言字典。输出：{标签文本: 槽名}。"""
    return {loc["meta"][key]: key for key in META_KEYS}


def merged_labels() -> dict[str, str]:
    """输入：无。输出：合并全部语言的 {标签文本: 槽名}。

    预期行为：各语言的标签**互不重叠**——重叠会让语言无关的反解把两种语言的字段串成一行
    （由 `tests/workflows/citation/test_locales.py` 锁住）。
    """
    merged: dict[str, str] = {}
    for lang in locales.available():
        for text, slot in label_index(locales.load(lang)).items():
            if text in merged:
                raise ValueError(f"标签 {text!r} 在 {lang} 与别的语言里重复了")
            merged[text] = slot
    return merged


def _match_label(line: str, labels: dict[str, str]) -> tuple[str, str] | None:
    """输入：一行文本、标签表。输出：(槽名, 值)，或 None（这行不是字段行）。

    预期行为：取「行首能匹配上的**最长**标签」——只按前缀匹配时，一个标签是另一个的
    前缀就会读错槽位。标签后面的冒号（中文全角 / 英文半角）一并去掉。
    """
    best: tuple[str, str] | None = None
    for text, slot in labels.items():
        if not line.startswith(text):
            continue
        if best is None or len(text) > len(best[0]):
            best = (text, slot)
    if best is None:
        return None
    value = line[len(best[0]) :].strip().lstrip("：:").strip()
    return best[1], value


def section_head(line: str) -> tuple[int, str] | None:
    """输入：一行文本。输出：(小节编号, 标题)，或 None。

    预期行为：认 `[1] 标题` 与 `1. 标题` 两种写法。句子行不以数字开头（由
    `test_locales.py` 锁住），所以这两条正则不会把句子吃成小节行。
    """
    for pattern in (_HEAD_BRACKET, _HEAD_DOTTED):
        found = pattern.match(line)
        if found is not None:
            return int(found.group(1)), found.group(2).strip()
    return None


def parse_citation(state: str, labels: dict[str, str]) -> ParsedCitation:
    """输入：state 文本、标签表。输出：反解结果。

    预期行为：逐行认——小节行开一节，字段行进 `fields`，其余行在「已经进了小节」时算句子。
    一句话都没解析出来（或一个小节都没有）说明这不是这份契约的文本，直接报错。
    """
    fields: dict[str, str] = {}
    sections: list[ParsedSection] = []
    in_section = False
    for raw in state.splitlines():
        line = raw.strip()
        if not line:
            continue
        head = section_head(line)
        if head is not None:
            index, heading = head
            sections.append(ParsedSection(index=index, heading=heading, sentences=()))
            in_section = True
            continue
        found = _match_label(line, labels)
        if found is not None:
            fields[found[0]] = found[1]
            in_section = False
            continue
        if in_section:
            last = sections[-1]
            sections[-1] = ParsedSection(
                index=last.index, heading=last.heading, sentences=(*last.sentences, line)
            )
    if not sections:
        raise ValueError(f"文本里一个小节都没有：{state[:80]!r}")
    return ParsedCitation(
        doc=fields.get("doc_label", ""),
        sections=tuple(sections),
        claim=fields.get("claim_label", ""),
        quote=fields.get("quote_label", ""),
        source=fields.get("source_label", ""),
    )


def claimed_index(source: str) -> int:
    """输入：「声称出处」那一行的值。输出：里面写的小节编号。

    预期行为：取里面**唯一的**一个数字——一个都没有或有两个（说明文案里混进了别的数字）
    直接报错，不猜。这条是语言无关的：两种语言都写「小节 / section + 数字」。
    """
    found = _NUMBER.findall(source)
    if len(found) != 1:
        raise ValueError(f"声称出处里的小节编号不是恰好一个：{source!r}")
    return int(found[0])
