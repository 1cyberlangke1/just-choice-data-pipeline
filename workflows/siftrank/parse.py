"""多批次排名聚合（siftrank）workflow 的结构反解：把 `state` 读回四个批次。

输入：`state` 文本 + 一张表（四个抬头标签）。输出：`ParsedRank`。
预期行为：只做结构反解，**不认语言也不做归一化**——「条目名 → 条目键」留给判别器，
它手上有语言文件（每个键的名字都在那里）。

反解不复用渲染器的任何计算：抬头靠标签认、批次靠编号认、条目靠箭头切，所以渲染把批次
写漏了、写重了、编号跳号了，这里都会当场报人话。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import re
from dataclasses import dataclass
from typing import Final

from . import locales
from . import spec as spec_mod

#: 标签与值之间的分隔符：中文用 `：`，英文用 `:`。**两种都认**。
COLONS: Final[tuple[str, ...]] = ("：", ":")

#: 批次的条目分隔符（结构，中英共用一套写法）。
ARROW: Final[re.Pattern[str]] = re.compile(r"\s*(?:>|→)\s*")

#: 条目清单的分隔符（结构）：中文顿号，英文逗号加空格。
LIST_SEP: Final[re.Pattern[str]] = re.compile(r"\s*、\s*|,\s*")

#: 批次行：编号 + 分隔符 + 条目串。
_BATCH: Final[re.Pattern[str]] = re.compile(r"^(\d+)\s*[：:]\s*(.+)$")


@dataclass(frozen=True)
class Tables:
    """反解用的表：四个抬头标签的几种写法（合并表里会同时装着中英两套）。"""

    request: tuple[str, ...]
    prompt: tuple[str, ...]
    items: tuple[str, ...]
    batch: tuple[str, ...]


@dataclass(frozen=True)
class ParsedRank:
    """反解出来的东西：请求号 + 排序提示原文 + 条目清单原文 + 四个批次（条目名原文）。"""

    request_id: str
    prompt: str
    listing: tuple[str, ...]
    batches: tuple[tuple[str, ...], ...]


def tables_of(loc: dict[str, JsonValue]) -> Tables:
    """输入：语言字典。输出：`Tables`。

    预期行为：只收**标签**（四个抬头），值（请求号 / 提示原文 / 条目名）原样留给判别器。
    四个标签撞车直接报错——撞车会让反解按最长前缀认错行。
    """
    meta = loc["meta"]
    labels = (str(meta["request"]), str(meta["prompt"]), str(meta["items"]), str(meta["batch"]))
    if len(set(labels)) != len(labels):
        raise ValueError(f"四个抬头标签有重复：{labels}")
    return Tables(
        request=(labels[0],),
        prompt=(labels[1],),
        items=(labels[2],),
        batch=(labels[3],),
    )


def merged_tables() -> Tables:
    """输入：无。输出：全部语言合并后的表。

    预期行为：`Workflow.reparse` 没有语言参数，只能靠合并表；各语言的标签互不重叠
    （由 `tests/workflows/siftrank/test_locales.py` 锁住），重叠了这里报错而不是静默取其一。
    """
    buckets: dict[str, list[str]] = {"request": [], "prompt": [], "items": [], "batch": []}
    for lang in locales.available():
        tables = tables_of(locales.load(lang))
        pairs: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("request", tables.request),
            ("prompt", tables.prompt),
            ("items", tables.items),
            ("batch", tables.batch),
        )
        for name, texts in pairs:
            for text in texts:
                if text in buckets[name]:
                    raise ValueError(f"{name} 标签在多种语言里重复：{text!r}")
                buckets[name].append(text)
    return Tables(
        request=tuple(buckets["request"]),
        prompt=tuple(buckets["prompt"]),
        items=tuple(buckets["items"]),
        batch=tuple(buckets["batch"]),
    )


def _header(line: str, labels: tuple[str, ...], what: str) -> str:
    """输入：一行文本、标签的几种写法、报错用的名字。输出：标签后面的原文。

    预期行为：**最长前缀优先**（标签之间有前缀关系时才不会认错）；后面必须跟东西，
    否则「排序请求号」这种更长的标签会被认成更短的标签本身。
    """
    best = ""
    for label in labels:
        if line.startswith(label) and len(label) > len(best) and line[len(label) :].lstrip(" \t"):
            best = label
    if not best:
        raise ValueError(f"{what}那一行的抬头认不出来：{line!r}")
    return line[len(best) :].lstrip(" \t")


def _value(line: str, labels: tuple[str, ...], what: str) -> str:
    """输入：一行文本、标签写法、名字。输出：去掉标签与分隔符之后的值。

    预期行为：分隔符可有可无；去掉之后是空的就报错——「标签写了但没写值」是渲染 bug，
    不能当成空字符串往下走。
    """
    rest = _header(line, labels, what)
    if rest[:1] in COLONS:
        rest = rest[1:].strip()
    if not rest:
        raise ValueError(f"{what}那一行没写值：{line!r}")
    return rest


def _listing(line: str, tables: Tables) -> tuple[str, ...]:
    """输入：条目清单那一行、表。输出：条目名的元组。条目数不对就报错。"""
    text = _value(line, tables.items, "条目清单")
    parts = tuple(part.strip() for part in LIST_SEP.split(text) if part.strip())
    if len(parts) != len(spec_mod.ITEMS):
        raise ValueError(
            f"条目清单的条目数不对：期望 {len(spec_mod.ITEMS)}，实际 {len(parts)}"
        )
    return parts


def _batch(line: str, tables: Tables) -> tuple[int, str]:
    """输入：一个批次那一行、表。输出：(编号, 条目串原文)。编号认不出来就报错。"""
    rest = _header(line, tables.batch, "批次")
    matched = _BATCH.match(rest)
    if matched is None:
        raise ValueError(f"批次那一行的编号认不出来：{line!r}")
    return int(matched.group(1)), matched.group(2).strip()


def _items(text: str) -> tuple[str, ...]:
    """输入：批次里那一串条目原文。输出：条目名元组。一个都没有就报错。"""
    parts = tuple(part.strip() for part in ARROW.split(text) if part.strip())
    if not parts:
        raise ValueError(f"批次里一个条目都没有：{text!r}")
    return parts


def parse_state(state: object, tables: Tables) -> ParsedRank:
    """输入：渲染出来的 `state`、表。输出：`ParsedRank`。

    预期行为：抬头三行 → 若干批次行。批次编号必须从 1 连续、个数等于 `spec.TRIALS`、
    每批条目数等于 `spec.BATCH_SIZE`——跳号说明渲染漏写了批次，个数不对说明批次被吞了。
    """
    if not isinstance(state, str):
        raise ValueError(f"state 不是字符串：{type(state).__name__}")
    lines = [line.strip() for line in state.splitlines() if line.strip()]
    if len(lines) < 3 + spec_mod.TRIALS:
        raise ValueError(f"state 太短，认不出四个批次：{state!r}")
    request_id = _value(lines[0], tables.request, "排序请求号")
    prompt = _value(lines[1], tables.prompt, "排序提示")
    listing = _listing(lines[2], tables)
    batches: list[tuple[str, ...]] = []
    for line in lines[3:]:
        number, text = _batch(line, tables)
        expected = len(batches) + 1
        if number != expected:
            raise ValueError(f"批次的编号跳号：期望 {expected}，实际 {number}")
        batches.append(_items(text))
    if len(batches) != spec_mod.TRIALS:
        raise ValueError(f"批次的个数不对：期望 {spec_mod.TRIALS}，实际 {len(batches)}")
    for index, batch in enumerate(batches, start=1):
        if len(batch) != spec_mod.BATCH_SIZE:
            raise ValueError(
                f"批次 {index} 的条目数不对：期望 {spec_mod.BATCH_SIZE}，实际 {len(batch)}"
            )
    return ParsedRank(
        request_id=request_id, prompt=prompt, listing=listing, batches=tuple(batches)
    )
