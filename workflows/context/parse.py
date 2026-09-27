"""会话上下文压缩 workflow 的反解：把会话文本读回结构化字段。

输入：`state` 文本 + 标签表。输出：`ParsedSession`。
预期行为：**不复用渲染时的任何计算**——只按文本本身解析，所以能抓到字段漏写、标签写错、
某一块的抬头缺格这类硬伤。认不出的行（标题行）直接忽略。

标签表里每一格存的都是**候选原文元组**：单语言表只有一项，`merged()` 给的合并表把各语言
的写法都收进来（`Workflow.reparse` 没有语言参数，只能靠合并表做语言无关的反解）。
四张属性词表也是「原文 → 键」的并集，判别器与审计拿**键**去查处置表。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from . import locales

#: 抬头行里四格的顺序（与渲染一致，反解按它取格）。
CELL_FIELDS: Final[tuple[str, ...]] = ("kind", "size", "age", "topic")
#: 四个行标签在合并表里的名字（跨语言必须互不相同，否则合并表有歧义）。
LABEL_FIELDS: Final[tuple[str, ...]] = ("session", "budget", "task")


@dataclass(frozen=True)
class Labels:
    """反解要用的全部原文：三格标签的候选、预算单位候选、分隔符、四张属性词表（原文 → 键）。"""

    session_labels: tuple[str, ...]
    budget_labels: tuple[str, ...]
    task_labels: tuple[str, ...]
    token_units: tuple[str, ...]
    block_sep: str
    kind: dict[str, str]
    size: dict[str, str]
    age: dict[str, str]
    topic: dict[str, str]

    def table(self, field: str) -> dict[str, str]:
        """输入：属性名。输出：该属性的「原文 → 键」表。预期行为：认不出的属性名直接报错。"""
        if field not in CELL_FIELDS:
            raise KeyError(f"没有属性 {field!r}（可用：{', '.join(CELL_FIELDS)}）")
        return getattr(self, field)

    def labels_of(self, field: str) -> tuple[str, ...]:
        """输入：标签名（`session` / `budget` / `task`）。输出：候选原文。"""
        if field not in LABEL_FIELDS:
            raise KeyError(f"没有标签 {field!r}（可用：{', '.join(LABEL_FIELDS)}）")
        return getattr(self, f"{field}_labels")


@dataclass(frozen=True)
class ParsedBlock:
    """反解出来的一个块：四格属性（**键**）+ 正文原文。"""

    kind: str
    size: str
    age: str
    topic: str
    content: str

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（`类型/大小/年龄/主题`）。"""
        return f"{self.kind}/{self.size}/{self.age}/{self.topic}"


@dataclass(frozen=True)
class ParsedSession:
    """反解出来的一段会话。"""

    session_id: str
    budget: int
    blocks: tuple[ParsedBlock, ...]


def _reverse(table: dict[str, str]) -> dict[str, str]:
    """输入：{键: 原文}。输出：{原文: 键}。

    预期行为：原文重复时报错——重复会让反解有歧义，属于语言文件的硬伤。
    """
    out: dict[str, str] = {}
    for key, text in table.items():
        if text in out:
            raise ValueError(f"文案重复：{text!r}（{out[text]} 与 {key}）")
        out[text] = key
    return out


def labels_of(loc: dict[str, Any]) -> Labels:
    """输入：语言字典。输出：这一门语言的反解标签表（每格只有一项候选）。"""
    meta = loc["meta"]
    return Labels(
        session_labels=(meta["session_label"],),
        budget_labels=(meta["budget_label"],),
        task_labels=(meta["task_label"],),
        token_units=(meta["token_unit"],),
        block_sep=meta["block_sep"],
        kind=_reverse(loc["kind"]),
        size=_reverse(loc["size"]),
        age=_reverse(loc["age"]),
        topic=_reverse(loc["topic"]),
    )


def merged() -> Labels:
    """输入：无。输出：全部语言合并出来的标签表。

    预期行为：供语言无关的 `reparse` 用。三条不变量由 `tests/workflows/context/test_locales.py`
    锁住——三格标签各语言互不相同（重叠了合并表就有歧义）、四张属性词表各语言互不重叠、
    分隔符各语言**必须一致**（不一致的话抬头行没法只按一个分隔符切格）。
    """
    labels_list = [labels_of(locales.load(lang)) for lang in locales.available()]
    if not labels_list:  # pragma: no cover —— 一个语言文件都没有时才会走到
        raise FileNotFoundError("没有任何语言文件")
    separators = {labels.block_sep for labels in labels_list}
    if len(separators) != 1:
        raise ValueError(f"各语言的分隔符必须一致，现在是 {sorted(separators)}")
    seen: dict[str, str] = {}
    tables: dict[str, dict[str, str]] = {field: {} for field in CELL_FIELDS}
    for labels in labels_list:
        for field in LABEL_FIELDS:
            for text in labels.labels_of(field):
                if text in seen:
                    raise ValueError(f"标签 {text!r} 在多种语言里都出现（{seen[text]} / {field}）")
                seen[text] = field
        for field in CELL_FIELDS:
            for text, key in labels.table(field).items():
                if text in tables[field]:
                    raise ValueError(f"{field} 的文案 {text!r} 在多种语言里都出现")
                tables[field][text] = key
    first = labels_list[0]
    return Labels(
        session_labels=tuple(
            dict.fromkeys(t for labels in labels_list for t in labels.session_labels)
        ),
        budget_labels=tuple(
            dict.fromkeys(t for labels in labels_list for t in labels.budget_labels)
        ),
        task_labels=tuple(
            dict.fromkeys(t for labels in labels_list for t in labels.task_labels)
        ),
        token_units=tuple(
            dict.fromkeys(t for labels in labels_list for t in labels.token_units)
        ),
        block_sep=first.block_sep,
        kind=tables["kind"],
        size=tables["size"],
        age=tables["age"],
        topic=tables["topic"],
    )


def _field_value(line: str, labels: tuple[str, ...]) -> str | None:
    """输入：一行文本、候选标签。输出：冒号后的值；都不是这一格时给 None。

    预期行为：取**最长**的匹配标签（标签之间有前缀关系时才不会认错），并且要求标签后面
    紧跟分隔符——否则标题行会被认成某一格（它确实可能以某个标签开头）。故意收紧。
    """
    best: str | None = None
    for label in labels:
        if not line.startswith(label):
            continue
        if line[len(label) :].lstrip()[:1] not in ("：", ":"):
            continue
        if best is None or len(label) > len(best):
            best = label
    if best is None:
        return None
    return line[len(best) :].strip().lstrip("：:").strip()


def _decode(field: str, text: str, labels: Labels) -> str:
    """输入：属性名、原文、标签表。输出：属性键。预期行为：认不出的原文直接报人话。"""
    table = labels.table(field)
    if text not in table:
        raise ValueError(f"{field} 认不出来：{text!r}")
    return table[text]


def _parse_budget(value: str, units: tuple[str, ...]) -> int:
    """输入：预算那一格的值、单位候选。输出：整数 token 数。

    预期行为：必须是「整数 + 某个单位」；写成别的形状直接报错，不当成 0 蒙过去。
    """
    text = value.strip()
    for unit in units:
        if not text.endswith(unit):
            continue
        digits = text[: -len(unit)].strip()
        if digits.isdigit():
            return int(digits)
    raise ValueError(f"预算不是「整数 + 单位」：{value!r}")


def _parse_header(line: str, labels: Labels) -> tuple[str, ...]:
    """输入：抬头行、标签表。输出：四格属性键（顺序固定）。

    预期行为：`[序号] 格 · 格 · 格 · 格`；格数不对直接报错（少一格说明渲染漏写了属性，
    那是硬伤，不能当成「空值」蒙过去）。
    """
    end = line.find("]")
    if end < 0:
        raise ValueError(f"抬头行缺右方括号：{line!r}")
    cells = [cell.strip() for cell in line[end + 1 :].strip().split(labels.block_sep)]
    if len(cells) != len(CELL_FIELDS):
        raise ValueError(f"抬头行应该是 {len(CELL_FIELDS)} 格，收到 {len(cells)} 格：{line!r}")
    return tuple(
        _decode(field, cell, labels) for field, cell in zip(CELL_FIELDS, cells, strict=True)
    )


def parse_session(state: str, labels: Labels) -> ParsedSession:
    """输入：state 文本、标签表。输出：反解结果。

    预期行为：逐行认——会话号、预算各一格（当前任务行认出来就跳过）；`[` 开头的是抬头行，
    紧跟的一行是它的正文；其余行忽略。三样缺任何一样都报人话。
    """
    if not isinstance(state, str):
        raise TypeError(f"state 必须是字符串，收到 {type(state).__name__}")
    lines = state.splitlines()
    session_id: str | None = None
    budget: int | None = None
    blocks: list[ParsedBlock] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        value = _field_value(line, labels.session_labels)
        if value is not None:
            session_id = value
            continue
        value = _field_value(line, labels.budget_labels)
        if value is not None:
            budget = _parse_budget(value, labels.token_units)
            continue
        if _field_value(line, labels.task_labels) is not None:
            continue
        if not line.startswith("["):
            continue
        cells = _parse_header(line, labels)
        if index >= len(lines) or lines[index].strip().startswith("["):
            raise ValueError(f"这一块缺正文行：{line!r}")
        blocks.append(ParsedBlock(*cells, lines[index].strip()))
        index += 1
    missing = [name for name, got in (("会话号", session_id), ("预算", budget)) if got is None]
    if missing:
        raise ValueError(f"会话缺这几格：{missing}")
    if not blocks:
        raise ValueError("会话里一个块都没有")
    assert session_id is not None and budget is not None  # 上面已经查过
    return ParsedSession(session_id=session_id, budget=budget, blocks=tuple(blocks))
