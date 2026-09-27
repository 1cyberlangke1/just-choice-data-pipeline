"""改动审查 workflow 的反解：把一屏改动读回结构化字段。

输入：`state` 文本 + 标签表。输出：`ParsedChange`。
预期行为：**不复用渲染时的任何计算**——只按文本本身解析，所以能抓到抬头漏写、
格数不对、正文写歪这类硬伤。认不出的行（标题行）直接忽略。

标签表里每一格存的都是**候选原文元组**：单语言表只有一项，`merged()` 给的合并表把各语言
的写法都收进来（`Workflow.reparse` 没有语言参数，只能靠合并表做语言无关的反解）。
三张词表（文件 / 变更 / 规模）也是「原文 → 键」的并集，判别器与审计拿**键**去查规则。

版式：标题行、改动号行、空行，然后每处改动两行——抬头行（序号 + 文件 + 变更 + 规模）
与正文行。**抬头三格与正文是判据的全部输入**，反解只认这两样。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from . import locales

#: 抬头行里三格的顺序（与渲染一致，反解按它取格）。
CELL_FIELDS: Final[tuple[str, ...]] = ("file", "change", "size")
#: 行标签在合并表里的名字（跨语言必须互不相同，否则合并表有歧义）。
LABEL_FIELDS: Final[tuple[str, ...]] = ("change_id",)
#: 词表名（合并表要逐张做「跨语言不重叠」检查）。
VOCAB_FIELDS: Final[tuple[str, ...]] = ("file", "change", "size")


@dataclass(frozen=True)
class Labels:
    """反解要用的全部原文：改动号标签的候选、三格分隔符、三张「原文 → 键」词表。"""

    change_labels: tuple[str, ...]
    cell_sep: str
    file: dict[str, str]
    change: dict[str, str]
    size: dict[str, str]

    def table(self, field: str) -> dict[str, str]:
        """输入：词表名。输出：该词表的「原文 → 键」映射。预期行为：认不出直接报错。"""
        if field not in VOCAB_FIELDS:
            raise KeyError(f"没有词表 {field!r}（可用：{', '.join(VOCAB_FIELDS)}）")
        return getattr(self, field)

    def labels_of(self, field: str) -> tuple[str, ...]:
        """输入：标签名（`change_id`）。输出：候选原文。预期行为：认不出直接报错。"""
        if field not in LABEL_FIELDS:
            raise KeyError(f"没有标签 {field!r}（可用：{', '.join(LABEL_FIELDS)}）")
        return self.change_labels


@dataclass(frozen=True)
class ParsedRow:
    """反解出来的一处改动：文件 / 变更 / 规模（都是键）+ 正文原文。"""

    file: str
    change: str
    size: str
    body: str


@dataclass(frozen=True)
class ParsedChange:
    """反解出来的一屏改动：改动号 + 六处改动。"""

    change_id: str
    rows: tuple[ParsedRow, ...]


def _pairs(loc: dict[str, Any], field: str) -> dict[str, str]:
    """输入：语言字典、词表名。输出：{原文: 键}。预期行为：原文重复直接报错。"""
    table = loc[field]
    out: dict[str, str] = {}
    for key, text in table.items():
        if text in out:
            raise ValueError(f"{field} 里有两个键共用同一个原文 {text!r}")
        out[str(text)] = str(key)
    return out


def labels_of(loc: dict[str, Any]) -> Labels:
    """输入：语言字典。输出：这一门语言的反解表（每一格只有一项候选）。"""
    meta = loc["meta"]
    return Labels(
        change_labels=(str(meta["change_label"]),),
        cell_sep=str(meta["cell_sep"]),
        file=_pairs(loc, "file"),
        change=_pairs(loc, "change"),
        size=_pairs(loc, "size"),
    )


def _label_union(entries: list[str], field: str) -> tuple[str, ...]:
    """输入：各语言的标签候选、标签名。输出：合并后的候选元组。

    预期行为：同一个标签出现在多种语言里直接报人话——合并表有歧义时反解会把某一格
    认成另一格，那比直接报错危险得多。
    """
    out: list[str] = []
    seen: set[str] = set()
    for text in entries:
        if text in seen:
            raise ValueError(f"{field} 的标签 {text!r} 在多种语言里都出现（合并表有歧义）")
        seen.add(text)
        out.append(text)
    return tuple(out)


def _union(entries: list[tuple[str, str]], field: str) -> dict[str, str]:
    """输入：[(原文, 键)]、词表名。输出：{原文: 键}。

    预期行为：同一个原文出现在多种语言里、或者同一门语言里两个键撞车，都直接报人话——
    合并表有歧义时反解会认错键，那比直接报错危险得多。
    """
    out: dict[str, str] = {}
    for text, key in entries:
        if text in out:
            raise ValueError(f"{field} 的原文 {text!r} 在多种语言里都出现（合并表有歧义）")
        out[text] = key
    return out


def merged() -> Labels:
    """输入：无。输出：所有语言的合并反解表（`reparse` 用它做语言无关的结构反解）。

    预期行为：三格分隔符各语言必须一致（合并表只按一个分隔符切行），标签与三张词表
    必须跨语言不重叠。
    """
    langs = locales.available()
    if not langs:
        raise ValueError("一个语言文件都没有")
    per_lang = [(lang, labels_of(locales.load(lang))) for lang in langs]
    seps = {labels.cell_sep for _, labels in per_lang}
    if len(seps) != 1:
        raise ValueError(f"各语言的三格分隔符必须一致，现在是 {sorted(seps)}")
    first = per_lang[0][1]
    return Labels(
        change_labels=_label_union(
            [text for _, labels in per_lang for text in labels.change_labels], "change_id"
        ),
        cell_sep=first.cell_sep,
        file=_union(
            [(text, key) for _, labels in per_lang for text, key in labels.file.items()],
            "file",
        ),
        change=_union(
            [(text, key) for _, labels in per_lang for text, key in labels.change.items()],
            "change",
        ),
        size=_union(
            [(text, key) for _, labels in per_lang for text, key in labels.size.items()],
            "size",
        ),
    )


def _field_value(line: str, labels: tuple[str, ...]) -> str | None:
    """输入：一行文本、候选标签。输出：冒号后的值；都不是这一格时给 None。

    预期行为：取**最长**的匹配标签（标签之间有前缀关系时才不会认错），并且要求标签后面
    紧跟分隔符——否则标题行会被认成改动号（标题确实以「改动」开头）。故意收紧。
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
    """输入：词表名、原文、标签表。输出：键。预期行为：认不出的原文直接报人话。"""
    table = labels.table(field)
    if text not in table:
        raise ValueError(f"{field} 认不出来：{text!r}")
    return table[text]


def _parse_cells(line: str, labels: Labels) -> tuple[str, ...]:
    """输入：抬头行、标签表。输出：三格词键（顺序固定）。

    预期行为：`[序号] 格 · 格 · 格`；格数不对直接报错（少一格说明渲染漏写了属性，
    那是硬伤，不能当成「空值」蒙过去）。
    """
    end = line.find("]")
    if end < 0:
        raise ValueError(f"抬头行缺右方括号：{line!r}")
    cells = [cell.strip() for cell in line[end + 1 :].strip().split(labels.cell_sep)]
    if len(cells) != len(CELL_FIELDS):
        raise ValueError(f"抬头行应该是 {len(CELL_FIELDS)} 格，收到 {len(cells)} 格：{line!r}")
    return tuple(
        _decode(field, cell, labels) for field, cell in zip(CELL_FIELDS, cells, strict=True)
    )


def parse_change(state: str, labels: Labels) -> ParsedChange:
    """输入：state 文本、标签表。输出：反解结果。

    预期行为：逐行认——`[` 开头的是抬头行，抬头行**后面紧跟的第一条非空行**是它的正文，
    其余行（标题 / 改动号）按标签取，取不到就忽略。改动号、抬头行、正文缺任何一样
    都报人话：正文缺席说明渲染漏写了一行，那是硬伤。
    """
    if not isinstance(state, str):
        raise TypeError(f"state 必须是字符串，收到 {type(state).__name__}")
    change_id: str | None = None
    rows: list[ParsedRow] = []
    pending: tuple[str, ...] | None = None
    for raw in state.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("["):
            if pending is not None:
                raise ValueError(f"上一处改动还没写正文，又来了抬头行：{line!r}")
            pending = _parse_cells(line, labels)
            continue
        if pending is not None:
            rows.append(ParsedRow(pending[0], pending[1], pending[2], line))
            pending = None
            continue
        value = _field_value(line, labels.change_labels)
        if value is not None:
            change_id = value
    if pending is not None:
        raise ValueError("最后一处改动没有正文")
    if change_id is None:
        raise ValueError("改动单缺改动号")
    if not rows:
        raise ValueError("改动单里一处改动都没有")
    return ParsedChange(change_id=change_id, rows=tuple(rows))
