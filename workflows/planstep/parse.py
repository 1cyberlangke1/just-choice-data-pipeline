"""有界任务推进（planstep）的结构反解：把 `state` 读回任务表。

输入：`state` 文本 + 一张表（抬头 / 区块抬头 / 格分隔符 / 状态词 / 前置写法）。
输出：`ParsedPlan`。
预期行为：只做结构反解，**不做归一化**——状态词直接映射成布尔、前置项编号直接映射成下标，
因为这两件在题面里就是字面规则的一部分。反解不复用渲染器的任何计算：区块靠抬头认、
任务行靠行首编号认、格靠分隔符切，所以渲染把任务写漏了、写重了、编号跳号了，这里都会
当场报人话。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from jc.jsontypes import JsonValue

from . import locales

#: 任务行：行首编号 + 后面的格子。编号后的点号是结构，中英共用。
_ROW: Final[re.Pattern[str]] = re.compile(r"^(\d+)\.\s*(.+)$")

#: 一行任务必须有几格（任务名 / 状态 / 前置）。
ROW_CELLS: Final[int] = 3


@dataclass(frozen=True)
class Tables:
    """反解用的表：抬头标签的几种写法、{区块抬头原文: slot}、格分隔符、状态词与前置写法。"""

    request: tuple[str, ...]
    sections: dict[str, str]
    cell_sep: str
    deps_sep: str
    done_words: frozenset[str]
    pending_words: frozenset[str]
    deps_none: frozenset[str]
    deps_prefix: tuple[str, ...]
    rules_prefix: str


@dataclass(frozen=True)
class ParsedTask:
    """反解出来的一项任务：任务名原文 + 状态 + 前置项下标（升序）。"""

    name: str
    done: bool
    deps: tuple[int, ...]


@dataclass(frozen=True)
class ParsedPlan:
    """反解出来的任务表：请求号 + 每项任务 + 推进规则原文。"""

    request_id: str
    tasks: tuple[ParsedTask, ...]
    rules: tuple[str, ...]

    @property
    def count(self) -> int:
        """输入：无。输出：这张表有几项任务。"""
        return len(self.tasks)

    @property
    def deps(self) -> tuple[tuple[int, ...], ...]:
        """输入：无。输出：逐项的依赖表（与 `features` 的入参同形）。"""
        return tuple(task.deps for task in self.tasks)

    @property
    def done_set(self) -> frozenset[int]:
        """输入：无。输出：已完成项的下标集合。"""
        return frozenset(index for index, task in enumerate(self.tasks) if task.done)


def tables_of(loc: dict[str, JsonValue]) -> Tables:
    """输入：语言字典。输出：`Tables`（只收这一种语言的写法）。"""
    meta = loc["meta"]
    request = str(meta["request"])
    if not request.strip():
        raise ValueError("抬头不能是空串")
    sections = {
        f"[{meta['tasks']}]": "tasks",
        f"[{meta['rules']}]": "rules",
    }
    cell_sep = str(meta["cell_sep"])
    if not cell_sep.strip():
        raise ValueError("格分隔符不能是空白")
    return Tables(
        request=(request,),
        sections=sections,
        cell_sep=cell_sep,
        deps_sep=str(meta["deps_sep"]),
        done_words=frozenset({str(loc["status"]["done"])}),
        pending_words=frozenset({str(loc["status"]["pending"])}),
        deps_none=frozenset({str(meta["deps_none"])}),
        deps_prefix=(str(meta["deps_prefix"]),),
        rules_prefix=str(meta["rules_prefix"]),
    )


def merged_tables() -> Tables:
    """输入：无。输出：全部语言合并后的表。

    预期行为：`Workflow.reparse` 没有语言参数，只能靠合并表；各语言的标签与结构词元互不
    重叠（由 `tests/workflows/planstep/test_locales.py` 锁住），重叠了这里报错而不是静默取其一。
    """
    sections: dict[str, str] = {}
    requests: list[str] = []
    cell_seps: set[str] = set()
    deps_seps: set[str] = set()
    done: set[str] = set()
    pending: set[str] = set()
    none_words: set[str] = set()
    prefixes: set[str] = set()
    rules_prefixes: set[str] = set()
    for lang in locales.available():
        tables = tables_of(locales.load(lang))
        for text, slot in tables.sections.items():
            if text in sections:
                raise ValueError(f"区块抬头 {text!r} 在多种语言里都出现（{sections[text]} / {slot}）")
            sections[text] = slot
        requests.extend(tables.request)
        cell_seps.add(tables.cell_sep)
        deps_seps.add(tables.deps_sep)
        done |= set(tables.done_words)
        pending |= set(tables.pending_words)
        none_words |= set(tables.deps_none)
        prefixes |= set(tables.deps_prefix)
        rules_prefixes.add(tables.rules_prefix)
    if len(set(requests)) != len(requests):
        raise ValueError(f"抬头标签在多种语言里重复：{requests}")
    if done & pending:
        raise ValueError(f"状态词在多种语言里撞车：{sorted(done & pending)}")
    if len(cell_seps) != 1:
        raise ValueError(f"格分隔符各语言不一致：{sorted(cell_seps)}——反解按它切格，必须唯一")
    if len(deps_seps) != 1:
        raise ValueError(f"前置分隔符各语言不一致：{sorted(deps_seps)}")
    if len(rules_prefixes) != 1:
        raise ValueError(f"规则行前缀各语言不一致：{sorted(rules_prefixes)}")
    labels = [*requests, *sections]
    for left in labels:
        for right in labels:
            if left != right and right.startswith(left):
                raise ValueError(f"标签 {left!r} 是 {right!r} 的前缀，反解会认错行")
    return Tables(
        request=tuple(requests),
        sections=sections,
        cell_sep=next(iter(cell_seps)),
        deps_sep=next(iter(deps_seps)),
        done_words=frozenset(done),
        pending_words=frozenset(pending),
        deps_none=frozenset(none_words),
        deps_prefix=tuple(sorted(prefixes)),
        rules_prefix=next(iter(rules_prefixes)),
    )


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


def _deps_of(cell: str, tables: Tables, count: int, index: int) -> tuple[int, ...]:
    """输入：前置那一格的原文、表、这张表有几项、这一项自己的下标。输出：前置项下标（升序）。

    预期行为：先认「没有前置」的写法；再逐个试前置前缀，把编号（1 起）换成下标（0 起）。
    编号越界 / 重复 / 自己指向自己都当场报人话——这些会让 `features` 算出没有意义的答案。
    """
    if cell in tables.deps_none:
        return ()
    for prefix in tables.deps_prefix:
        if not cell.startswith(prefix):
            continue
        rest = cell[len(prefix) :].strip()
        parts = [part.strip() for part in rest.split(tables.deps_sep)]
        if not parts or any(not part.isdigit() for part in parts):
            raise ValueError(f"前置那一格里的编号不是整数：{cell!r}")
        numbers = sorted(int(part) for part in parts)
        if len(set(numbers)) != len(numbers):
            raise ValueError(f"前置项重复：{cell!r}")
        if any(number < 1 or number > count for number in numbers):
            raise ValueError(f"前置项编号越界：{cell!r}（本表共 {count} 项）")
        if index + 1 in numbers:
            raise ValueError(f"第 {index + 1} 项把自己列成了前置：{cell!r}——依赖图有环，推不动")
        return tuple(number - 1 for number in numbers)
    raise ValueError(f"前置那一格认不出来：{cell!r}")


def parse_state(state: object, tables: Tables) -> ParsedPlan:
    """输入：渲染出来的 `state`、反解用的表。输出：`ParsedPlan`。

    预期行为：抬头 → 区块。任务区块内只认**带编号的行**，别的行（空行）一律跳过；
    编号必须从 1 连续，跳号说明渲染漏写了任务，当场报错。规则区块内的非空行收进 `rules`。
    """
    if not isinstance(state, str):
        raise ValueError(f"state 不是字符串：{type(state).__name__}")
    lines = state.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) < 3:
        raise ValueError(f"state 太短，认不出任务表：{state!r}")
    request_id = _header(lines[0], tables.request, "请求号")
    # 前置那一格要等整张表读完才知道「共几项」（编号越界要报错），所以先存原文、最后一并解析。
    raw: list[tuple[str, bool, str]] = []
    rules: list[str] = []
    section = ""
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in tables.sections:
            section = tables.sections[stripped]
            continue
        if not section:
            raise ValueError(f"内容出现在任何区块抬头之前：{line!r}")
        if section == "rules":
            rules.append(stripped[len(tables.rules_prefix) :].strip())
            continue
        matched = _ROW.match(stripped)
        if matched is None:
            raise ValueError(f"认不出的任务行：{line!r}")
        number = int(matched.group(1))
        expected = len(raw) + 1
        if number != expected:
            raise ValueError(f"任务号跳号：期望 {expected}，实际 {number}")
        cells = [cell.strip() for cell in matched.group(2).split(tables.cell_sep)]
        if len(cells) != ROW_CELLS:
            raise ValueError(f"任务行有 {len(cells)} 格，应为 {ROW_CELLS} 格：{line!r}")
        name, status, deps_cell = cells
        if status in tables.done_words:
            done = True
        elif status in tables.pending_words:
            done = False
        else:
            raise ValueError(f"状态词认不出来：{status!r}（第 {number} 项）")
        raw.append((name, done, deps_cell))
    if not raw:
        raise ValueError("任务表里一项都没有")
    return ParsedPlan(
        request_id=request_id,
        tasks=tuple(
            ParsedTask(name=name, done=done, deps=_deps_of(cell, tables, len(raw), index))
            for index, (name, done, cell) in enumerate(raw)
        ),
        rules=tuple(rules),
    )
