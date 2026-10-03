"""片段筛选与去冗余（clipcut）的结构反解：把 `state` 读回片段表。

输入：`state` 文本 + 一张表（抬头 / 区块抬头）。输出：`ParsedClip`。
预期行为：只做结构反解，**不认语言也不做归一化**——「原文 → 主题键 / 高亮布尔」留给判别器，
它手上有语言文件。反解不复用渲染器的任何计算：区块靠抬头认、片段号靠行首编号认，所以渲染
把片段写漏了、写重了、编号跳号了，这里都会当场报人话。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from jc.jsontypes import JsonValue
from jc.state_parse import read_after_label

from . import locales

#: 圆括号对：中文用全角 `（）`，英文用半角 `()`。**两种都认**——反解先统一成半角再切。
PARENS: Final[tuple[tuple[str, str], ...]] = (("（", "）"), ("(", ")"))

#: 片段行：行首编号 + 主题名 + 圆括号里的高亮词。编号后的点号是结构，中英共用。
_LINE: Final[re.Pattern[str]] = re.compile(r"^(\d+)\.\s*(.+?)\s*\((.+)\)$")


@dataclass(frozen=True)
class Tables:
    """反解用的表：抬头标签的几种写法、{区块抬头原文: segments}。"""

    request: tuple[str, ...]
    sections: dict[str, str]


@dataclass(frozen=True)
class ParsedClip:
    """反解出来的片段表：请求号 + 每段的 (主题原文, 高亮词原文) 序列。"""

    request_id: str
    segments: tuple[tuple[str, str], ...]

    @property
    def count(self) -> int:
        """输入：无。输出：这一屏有几段片段。"""
        return len(self.segments)


def tables_of(loc: dict[str, JsonValue]) -> Tables:
    """输入：语言字典。输出：`Tables`。

    预期行为：只收**标签**（抬头 / 区块），主题名与高亮词的**值**原样留给判别器——
    「原文 → 键」的翻译在它那里做，反解这一步不认语言。
    """
    meta = loc["meta"]
    request = str(meta["request"])
    if not request.strip():
        raise ValueError("抬头不能是空串")
    if not str(meta["segments"]).strip():
        raise ValueError("区块抬头不能是空串")
    return Tables(request=(request,), sections={f"[{meta['segments']}]": "segments"})


def merged_tables() -> Tables:
    """输入：无。输出：全部语言合并后的表。

    预期行为：`Workflow.reparse` 没有语言参数，只能靠合并表；各语言的标签互不重叠
    （由 `tests/workflows/clipcut/test_locales.py` 锁住），重叠了这里报错而不是静默取其一。
    """
    sections: dict[str, str] = {}
    requests: list[str] = []
    for lang in locales.available():
        tables = tables_of(locales.load(lang))
        for text, slot in tables.sections.items():
            if text in sections:
                raise ValueError(f"区块抬头 {text!r} 在多种语言里都出现（{sections[text]} / {slot}）")
            sections[text] = slot
        requests.extend(tables.request)
    if len(set(requests)) != len(requests):
        raise ValueError(f"抬头标签在多种语言里重复：{requests}")
    if set(sections) & set(requests):
        raise ValueError(f"区块抬头与抬头标签撞车：{sorted(set(sections) & set(requests))}")
    labels = [*requests, *sections]
    for left in labels:
        for right in labels:
            if left != right and right.startswith(left):
                raise ValueError(f"标签 {left!r} 是 {right!r} 的前缀，反解会认错行")
    return Tables(request=tuple(requests), sections=sections)


def _unparen(line: str) -> str:
    """输入：一行文本。输出：把各语言的圆括号统一成半角之后的那一行。"""
    out = line
    for open_paren, close_paren in PARENS:
        out = out.replace(open_paren, "(").replace(close_paren, ")")
    return out


def parse_state(state: object, tables: Tables) -> ParsedClip:
    """输入：渲染出来的 `state`、两张表。输出：`ParsedClip`。

    预期行为：抬头 → 区块。区块内只认**带编号的片段行**，别的行（空行）一律跳过；
    片段号必须从 1 连续，跳号说明渲染漏写了片段，当场报错。
    """
    if not isinstance(state, str):
        raise ValueError(f"state 不是字符串：{type(state).__name__}")
    lines = [_unparen(line) for line in state.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) < 3:
        raise ValueError(f"state 太短，认不出片段表：{state!r}")
    request_id = read_after_label(lines[0], tables.request, "请求号")
    segments: list[tuple[str, str]] = []
    section = ""
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in tables.sections:
            section = tables.sections[stripped]
            continue
        if not section:
            raise ValueError(f"片段出现在任何区块抬头之前：{line!r}")
        matched = _LINE.match(stripped)
        if matched is None:
            raise ValueError(f"认不出的片段行：{line!r}")
        number = int(matched.group(1))
        expected = len(segments) + 1
        if number != expected:
            raise ValueError(f"片段号跳号：期望 {expected}，实际 {number}")
        segments.append((matched.group(2).strip(), matched.group(3).strip()))
    if not segments:
        raise ValueError("片段表里一段都没有")
    return ParsedClip(request_id=request_id, segments=tuple(segments))
