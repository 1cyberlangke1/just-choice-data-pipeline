"""缓存复用判定（cachereuse）workflow 的结构反解：把 `state` 读回两条轨迹。

输入：`state` 文本 + 一张表（抬头 / 来源 / 两个区块抬头）。输出：`ParsedReuse`。
预期行为：只做结构反解，**不认语言也不做归一化**——「原文 → 动作键 / 元素键」留给判别器，
它手上有语言文件（每个键的几种写法都在那里）。

反解不复用渲染器的任何计算：区块靠抬头认、步号靠行首编号认，所以渲染把步骤写漏了、
写重了、编号跳号了，这里都会当场报人话。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue
from jc.state_parse import read_after_label

import re
from dataclasses import dataclass
from typing import Final

from . import locales

#: 元素名的引号对：中文用 `「」`，英文用 `"`。**两种都认**——反解先把它统一成 `"` 再切。
QUOTES: Final[tuple[tuple[str, str], ...]] = (("「", "」"), ('"', '"'))

#: 标签与值之间的分隔符（来源那一行用），两种语言各一种写法。
COLONS: Final[tuple[str, ...]] = ("：", ":")

#: 步骤行：行首编号 + 动作 + 引号里的元素。编号后的点号是结构，中英共用。
_STEP: Final[re.Pattern[str]] = re.compile(r"^(\d+)\.\s*(.+?)\s*\"(.+)\"$")


@dataclass(frozen=True)
class Tables:
    """反解用的表：抬头标签的几种写法、来源标签的几种写法、{区块抬头原文: cached / current}。"""

    request: tuple[str, ...]
    sources: tuple[str, ...]
    sections: dict[str, str]


@dataclass(frozen=True)
class ParsedReuse:
    """反解出来的两条轨迹：请求号 + 来源原文 + 两侧的 (动作原文, 元素原文) 序列。"""

    request_id: str
    flavor: str
    cached: tuple[tuple[str, str], ...]
    current: tuple[tuple[str, str], ...]

    def side(self, index: int) -> tuple[tuple[str, str], ...]:
        """输入：0 = 缓存轨迹 / 1 = 当前轨迹。输出：那一侧的步骤序列。"""
        return (self.cached, self.current)[index]


def tables_of(loc: dict[str, JsonValue]) -> Tables:
    """输入：语言字典。输出：`Tables`。

    预期行为：只收**标签**（抬头 / 来源 / 两个区块），来源那一行的**值**（「网页版」）原样留给
    判别器——「原文 → 键」的翻译在它那里做，反解这一步不认语言。两个区块抬头撞车直接报错。
    """
    meta = loc["meta"]
    sections = {
        f"[{meta['cached']}]": "cached",
        f"[{meta['current']}]": "current",
    }
    if len(sections) != 2:
        raise ValueError("两个区块抬头的文案相同，反解会认错")
    return Tables(request=(meta["request"],), sources=(meta["source"],), sections=sections)


def merged_tables() -> Tables:
    """输入：无。输出：全部语言合并后的表。

    预期行为：`Workflow.reparse` 没有语言参数，只能靠合并表；各语言的标签互不重叠
    （由 `tests/workflows/cachereuse/test_locales.py` 锁住），重叠了这里报错而不是静默取其一。
    """
    sources: list[str] = []
    sections: dict[str, str] = {}
    requests: list[str] = []
    for lang in locales.available():
        tables = tables_of(locales.load(lang))
        for text in tables.sources:
            if text in sources:
                raise ValueError(f"来源标签在多种语言里重复：{text!r}")
            sources.append(text)
        for text, slot in tables.sections.items():
            if text in sections:
                raise ValueError(f"区块抬头 {text!r} 在多种语言里都出现（{sections[text]} / {slot}）")
            sections[text] = slot
        requests.extend(tables.request)
    if len(set(requests)) != len(requests):
        raise ValueError(f"抬头标签在多种语言里重复：{requests}")
    return Tables(request=tuple(requests), sources=tuple(sources), sections=sections)


def _unquote(line: str) -> str:
    """输入：一行文本。输出：把各语言的引号统一成 `"` 之后的那一行。

    预期行为：先统一引号再匹配，正则就只有一种写法；`「」` 与 `"` 都映射到 `"`。
    """
    out = line
    for open_quote, close_quote in QUOTES:
        out = out.replace(open_quote, '"').replace(close_quote, '"')
    return out


def parse_state(state: object, tables: Tables) -> ParsedReuse:
    """输入：渲染出来的 `state`、三张表。输出：`ParsedReuse`。

    预期行为：抬头 → 来源 → 两个区块。区块内只认**带编号的步骤行**，别的行（空行、注释行）
    一律跳过；步号必须从 1 连续，跳号说明渲染漏写了步骤，当场报错。
    """
    if not isinstance(state, str):
        raise ValueError(f"state 不是字符串：{type(state).__name__}")
    lines = [_unquote(line) for line in state.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) < 4:
        raise ValueError(f"state 太短，认不出两块轨迹：{state!r}")
    request_id = read_after_label(lines[0], tables.request, "请求号")
    flavor = _flavor_of(lines[1], tables)
    sides: dict[str, list[tuple[str, str]]] = {"cached": [], "current": []}
    section = ""
    for line in lines[2:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in tables.sections:
            section = tables.sections[stripped]
            continue
        if not section:
            raise ValueError(f"步骤出现在任何区块抬头之前：{line!r}")
        matched = _STEP.match(stripped)
        if matched is None:
            raise ValueError(f"认不出的步骤行：{line!r}")
        number = int(matched.group(1))
        expected = len(sides[section]) + 1
        if number != expected:
            raise ValueError(f"{section} 的步号跳号：期望 {expected}，实际 {number}")
        sides[section].append((matched.group(2).strip(), matched.group(3).strip()))
    for name in ("cached", "current"):
        if not sides[name]:
            raise ValueError(f"{name} 区块一步都没有")
    if len(sides["cached"]) != len(sides["current"]):
        raise ValueError(
            f"两条轨迹步数不同：{len(sides['cached'])} vs {len(sides['current'])}"
        )
    return ParsedReuse(
        request_id=request_id,
        flavor=flavor,
        cached=tuple(sides["cached"]),
        current=tuple(sides["current"]),
    )


def _flavor_of(line: str, tables: Tables) -> str:
    """输入：来源那一行、表。输出：分隔符后面的**原文**（「网页版」这种）。

    预期行为：按「标签 + 分隔符」切；标签认不出来、或分隔符后面是空的，都当场报错。
    """
    best = ""
    for label in tables.sources:
        if line.startswith(label) and len(label) > len(best):
            rest = line[len(label) :].lstrip()
            if rest[:1] in COLONS:
                best = label
    if not best:
        raise ValueError(f"来源那一行认不出来：{line!r}")
    text = line[len(best) :].lstrip()[1:].strip()
    if not text:
        raise ValueError(f"来源那一行没写值：{line!r}")
    return text
