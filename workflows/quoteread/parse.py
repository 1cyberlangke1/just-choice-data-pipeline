"""行情读数判读（quoteread）的结构反解：把 `state` 读回读数表。

输入：`state` 文本 + 一张表（抬头 / 区块抬头）。输出：`ParsedQuote`。
预期行为：只做结构反解，**不认语言也不做换算**——「原文 → 标的名键 / 读数」留给判别器，
它手上有语言文件。反解不复用渲染器的任何计算：区块靠抬头认、行号靠行首编号认，所以渲染
把某一行写漏了、写重了、编号跳号了，这里都会当场报人话。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from jc.jsontypes import JsonValue

from . import locales

#: 竖线的几种写法：全角与半角。**两种都认**——反解先统一成半角再切。
SEPARATORS: Final[tuple[tuple[str, str], ...]] = (("｜", "|"),)

#: 读数行：行首编号 + 标的名 + 竖线 + 读数。编号后的点号是结构，中英共用。
_LINE: Final[re.Pattern[str]] = re.compile(r"^(\d+)\.\s*(.+?)\s*\|\s*(.+)$")


@dataclass(frozen=True)
class Tables:
    """反解用的表：抬头标签的几种写法、{区块抬头原文: table}。"""

    request: tuple[str, ...]
    sections: dict[str, str]


@dataclass(frozen=True)
class ParsedQuote:
    """反解出来的读数表：请求号 + 每行的 (标的名原文, 读数原文) 序列。"""

    request_id: str
    rows: tuple[tuple[str, str], ...]

    @property
    def count(self) -> int:
        """输入：无。输出：这一屏有几行读数。"""
        return len(self.rows)


def tables_of(loc: dict[str, JsonValue]) -> Tables:
    """输入：语言字典。输出：`Tables`。

    预期行为：只收**标签**（抬头 / 区块），标的名与单位的**值**原样留给判别器——
    「原文 → 键」的翻译在它那里做，反解这一步不认语言。
    """
    meta = loc["meta"]
    request = str(meta["request"])
    if not request.strip():
        raise ValueError("抬头不能是空串")
    if not str(meta["table"]).strip():
        raise ValueError("区块抬头不能是空串")
    return Tables(request=(request,), sections={f"[{meta['table']}]": "table"})


def merged_tables() -> Tables:
    """输入：无。输出：全部语言合并后的表。

    预期行为：`Workflow.reparse` 没有语言参数，只能靠合并表；各语言的标签互不重叠
    （由 `tests/workflows/quoteread/test_locales.py` 锁住），重叠了这里报错而不是静默取其一。
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


def _normalize(line: str) -> str:
    """输入：一行文本。输出：把各语言的竖线写法统一成半角之后的那一行。"""
    out = line
    for full, half in SEPARATORS:
        out = out.replace(full, half)
    return out


def _header(line: str, labels: tuple[str, ...], what: str) -> str:
    """输入：一行文本、抬头标签的几种写法、报错用的名字。输出：抬头后面的值。

    预期行为：**最长前缀优先**（标签之间有前缀关系时才不会认错）；后面必须是空白，
    否则更长的抬头会被认成标签本身。
    """
    best = ""
    for label in labels:
        if line.startswith(label) and len(label) > len(best):
            rest = line[len(label) :].lstrip(" \t")
            if rest:
                best = label
    if not best:
        raise ValueError(f"{what}抬头认不出来：{line!r}")
    return line[len(best) :].lstrip(" \t")


def parse_state(state: object, tables: Tables) -> ParsedQuote:
    """输入：渲染出来的 `state`、两张表。输出：`ParsedQuote`。

    预期行为：抬头 → 区块。区块内只认**带编号的读数行**，别的行（空行）一律跳过；
    行号必须从 1 连续，跳号说明渲染漏写了读数，当场报错。
    """
    if not isinstance(state, str):
        raise ValueError(f"state 不是字符串：{type(state).__name__}")
    lines = [_normalize(line) for line in state.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) < 3:
        raise ValueError(f"state 太短，认不出读数表：{state!r}")
    request_id = _header(lines[0], tables.request, "请求号")
    rows: list[tuple[str, str]] = []
    section = ""
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in tables.sections:
            section = tables.sections[stripped]
            continue
        if not section:
            raise ValueError(f"读数出现在任何区块抬头之前：{line!r}")
        matched = _LINE.match(stripped)
        if matched is None:
            raise ValueError(f"认不出的读数行：{line!r}")
        number = int(matched.group(1))
        expected = len(rows) + 1
        if number != expected:
            raise ValueError(f"行号跳号：期望 {expected}，实际 {number}")
        rows.append((matched.group(2).strip(), matched.group(3).strip()))
    if not rows:
        raise ValueError("读数表里一行都没有")
    return ParsedQuote(request_id=request_id, rows=tuple(rows))
