"""记录匹配（match）workflow 的结构反解：把 `state` 读回两条记录。

输入：`state` 文本 + 三张表（栏标签 / 侧标题 / 抬头标签）。输出：`ParsedMatch`。
预期行为：只做结构反解，**不认语言也不做归一化**——「原文 → 归一化后的值」留给判别器，
它手上有语言文件（单位换算、大小写、同义词表都在那里）。

反解不复用渲染器的任何计算：栏标签来自语言文件、栏序靠标签认（不靠行号），
所以渲染把栏写漏了、写重了、写歪了，这里都会当场报人话。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from dataclasses import dataclass
from typing import Final

from . import locales
from . import spec as spec_mod

#: 标签与值之间的分隔符：两种语言各用一种写法，反解两种都认。
COLONS: Final[tuple[str, ...]] = ("：", ":")

#: 侧标题后面允许跟的分隔符（渲染时写了，反解时可有可无）。
SIDE_TAIL: Final[str] = "：:"


@dataclass(frozen=True)
class Tables:
    """反解用的三张表：{栏标签原文: 槽位}、{侧标题原文: 侧名}、抬头标签（可能有多种语言）。"""

    labels: dict[str, str]
    sides: dict[str, str]
    request: tuple[str, ...]


@dataclass(frozen=True)
class ParsedMatch:
    """反解出来的两条记录：请求号 + 两侧每栏的原文 + 两侧各自的额外栏。

    额外栏按侧分开收：两侧都会写一行（比如都写「产地」），合成一个 dict 会让后写的
    覆盖先写的，于是 A 的产地永远读成 B 的——实测踩过。
    """

    request_id: str
    left: dict[str, str]
    right: dict[str, str]
    extra: dict[str, dict[str, str]]

    def side(self, index: int) -> dict[str, str]:
        """输入：0 = A / 1 = B。输出：那一侧的「栏 → 原文」。"""
        return (self.left, self.right)[index]

    def extra_of(self, index: int) -> dict[str, str]:
        """输入：0 = A / 1 = B。输出：那一侧的额外栏（没写就是空表）。"""
        return self.extra.get(("left", "right")[index], {})


def tables_of(loc: dict[str, JsonValue]) -> Tables:
    """输入：语言字典。输出：`Tables`。

    预期行为：槽位名与语言无关——栏带 `field:` 前缀、额外栏是 `extra:origin`，
    于是「原文 → 槽位」的翻译只在判别器里做。标签撞车直接报错（反解会把两栏认成一栏）。
    """
    labels = {loc["field"][field]: f"field:{field}" for field in spec_mod.FIELDS}
    extra_label = loc["extra"]["origin_label"]
    if extra_label in labels:
        raise ValueError(f"额外栏标签与某一栏重名：{extra_label!r}")
    labels[extra_label] = "extra:origin"
    sides = {loc["meta"]["left"]: "left", loc["meta"]["right"]: "right"}
    if len(sides) != 2:
        raise ValueError("两条记录的侧标题文案相同，反解会认错")
    return Tables(labels=labels, sides=sides, request=(loc["meta"]["request"],))


def merged_tables() -> Tables:
    """输入：无。输出：全部语言合并后的三张表。

    预期行为：`Workflow.reparse` 没有语言参数，只能靠合并表；各语言的标签互不重叠
    （由 `tests/workflows/match/test_locales.py` 锁住），重叠了这里报错而不是静默取其一。
    """
    labels: dict[str, str] = {}
    sides: dict[str, str] = {}
    requests: list[str] = []
    for lang in locales.available():
        tables = tables_of(locales.load(lang))
        for text, slot in tables.labels.items():
            if text in labels:
                raise ValueError(f"栏标签 {text!r} 在多种语言里都出现（{labels[text]} / {slot}）")
            labels[text] = slot
        for text, side in tables.sides.items():
            if text in sides:
                raise ValueError(f"侧标题 {text!r} 在多种语言里都出现（{sides[text]} / {side}）")
            sides[text] = side
        requests.extend(tables.request)
    if len(set(requests)) != len(requests):
        raise ValueError(f"抬头标签在多种语言里重复：{requests}")
    return Tables(labels=labels, sides=sides, request=tuple(requests))


def match_label(line: str, labels: dict[str, str]) -> tuple[str, str] | None:
    """输入：一行文本、栏标签表。输出：(槽位, 值) 或 None。

    预期行为：取**最长**的匹配标签（标签之间有前缀关系时才不会认错），并且要求标签后面
    紧跟分隔符——否则抬头「匹配请求 M-42590」会被认成某一栏。
    """
    best: tuple[str, str] | None = None
    for text, slot in labels.items():
        if not line.startswith(text):
            continue
        rest = line[len(text) :].lstrip()
        if rest[:1] not in COLONS:
            continue
        if best is None or len(text) > len(best[0]):
            best = (text, slot)
    if best is None:
        return None
    return best[1], line[len(best[0]) :].strip().lstrip("：:").strip()


def parse_state(state: str, tables: Tables) -> ParsedMatch:
    """输入：`state` 文本、三张表。输出：反解结果。

    预期行为：逐行认——抬头给请求号、侧标题切记录、`栏: 值` 进当前侧、额外栏单独收。
    少一条记录、少一栏、出现认不出的行，都报人话：渲染漏写了是硬伤，不能当空值蒙过去。
    """
    request_id: str | None = None
    records: dict[str, dict[str, str]] = {}
    extra: dict[str, dict[str, str]] = {}
    current: str | None = None
    for raw in state.splitlines():
        line = raw.strip()
        if not line:
            continue
        if request_id is None:
            for label in tables.request:
                if line.startswith(label):
                    request_id = line[len(label) :].strip()
                    break
            if request_id is not None:
                continue
        head = line.rstrip(SIDE_TAIL).strip()
        if head in tables.sides:
            current = tables.sides[head]
            if current in records:
                raise ValueError(f"侧标题 {head!r} 出现了两次")
            records[current] = {}
            continue
        found = match_label(line, tables.labels)
        if found is None:
            raise ValueError(f"这一行认不出来：{line!r}")
        slot, value = found
        if current is None:
            raise ValueError(f"还没出现侧标题就先写了栏：{line!r}")
        if slot.startswith("field:"):
            records[current][slot[len("field:") :]] = value
        else:
            extra.setdefault(current, {})[slot[len("extra:") :]] = value
    if request_id is None:
        raise ValueError("state 里没有抬头（匹配请求）那一行")
    missing_sides = [name for name in ("left", "right") if name not in records]
    if missing_sides:
        raise ValueError(f"state 里缺这几条记录：{missing_sides}")
    for name in ("left", "right"):
        missing = [field for field in spec_mod.FIELDS if field not in records[name]]
        if missing:
            raise ValueError(f"{name} 这条记录缺这几栏：{missing}")
    return ParsedMatch(request_id, records["left"], records["right"], extra)
