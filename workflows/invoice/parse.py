"""单据 workflow 的反解：把渲染出来的文本读回结构化字段。

输入：`state` 文本 + 语言字典。输出：`ParsedInvoice`。
预期行为：**不复用渲染时的任何计算**——只按文本本身解析，所以能抓到渲染错位、
字段漏写、数值写错这类硬伤。认不出的标签（备注之类的噪声字段）直接忽略；
「该有的槽没解出来」由 `judges.RoundtripJudge` 判定为不合格。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import re
from dataclasses import dataclass


#: 段落版式里字段之间的分隔（渲染用 `；`，句子用 `。` 收尾）。
_SEPARATORS: re.Pattern[str] = re.compile(r"[。；;\n]+")
#: 标签后面允许紧跟的分隔符（对齐表格没有分隔符，靠两个以上空白分列）。
_LABEL_SEPS: tuple[str, ...] = ("：", ":", "|")
#: 对齐表格里标签与值之间的空白。
_ALIGNED_GAP: re.Pattern[str] = re.compile(r"\s{2,}")
#: 对齐表格的列分隔：两个以上空白。
_CELL_SPLIT: re.Pattern[str] = re.compile(r"\s{2,}")
#: 日期里的三段数字（年在前，允许年月日这种带字的写法）。
_DATE_PARTS: re.Pattern[str] = re.compile(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})")


@dataclass(frozen=True)
class ParsedItem:
    """反解出来的一条明细。"""

    name: str
    qty: int
    unit: int
    amount: int


@dataclass(frozen=True)
class ParsedInvoice:
    """反解出来的单据：字段是**原始文本**，按槽分类解析交给下面的取值函数。"""

    fields: dict[str, str]
    items: tuple[ParsedItem, ...]

    def money(self, slot: str) -> int | None:
        """输入：金额槽。输出：整数金额，或 None（没解出来 / 该槽不存在）。"""
        return as_int(self.fields.get(slot, ""))

    def date(self, slot: str) -> tuple[int, int, int] | None:
        """输入：日期槽。输出：(年, 月, 日)，或 None。"""
        return as_date(self.fields.get(slot, ""))

    def rate(self) -> int | None:
        """输入：无。输出：税率（百分数），或 None。"""
        return as_int(self.fields.get("tax_rate", ""))

    def text(self, slot: str) -> str:
        """输入：任意槽。输出：原始文本（不存在时是空串）。"""
        return self.fields.get(slot, "")


def label_index(loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：语言字典。输出：{标签文本: 槽名}（含正名与全部别名）。

    预期行为：只收**字段**标签；明细表的列名（品名 / 数量 / …）不算字段标签，
    否则表头会被当成一行字段读进来。
    """
    index: dict[str, str] = {}
    for slot, entry in loc["field"].items():
        for label in entry["labels"]:
            index[label] = slot
    return index


def _segments(text: str) -> list[str]:
    """输入：整段文本。输出：候选行 / 段。预期行为：换行、分号、句号都当分隔符。"""
    return [part for part in _SEPARATORS.split(text) if part.strip()]


def _cells(line: str) -> list[str]:
    """输入：一行文本。输出：单元格列表。

    预期行为：竖线表格按 `|` 切，对齐表格按两个以上空白切，两种归一化成同一形态。
    """
    stripped = line.strip().strip("|").strip()
    if "|" in stripped:
        return [cell.strip() for cell in stripped.split("|") if cell.strip()]
    return [cell.strip() for cell in _CELL_SPLIT.split(stripped) if cell.strip()]


def _match_label(line: str, labels: dict[str, str]) -> tuple[str, str] | None:
    """输入：一行文本、标签表。输出：(标签, 槽名)，或 None（这行不是字段行）。

    预期行为：取「行首能匹配上的**最长**标签」，且标签后面必须跟分隔符或两个以上空白。
    两条都必要——英文里 `Tax` 是 `Tax Rate` 的前缀，只看「行首词」会把
    `Tax Rate: 6%` 读成 `Tax` 加一段值；只看前缀又会把 `Total lines: 3` 这类噪声读成字段。
    """
    text = line.strip().lstrip("|").strip()
    best: tuple[str, str] | None = None
    for label, slot in labels.items():
        if not text.startswith(label):
            continue
        rest = text[len(label) :]
        stripped = rest.strip()
        ok = bool(_ALIGNED_GAP.match(rest)) or stripped[:1] in ("", *_LABEL_SEPS)
        if not ok:
            continue
        if best is None or len(label) > len(best[0]):
            best = (label, slot)
    return best


def _value_after_label(line: str, label: str) -> str:
    """输入：一行文本、该行开头的标签。输出：标签后面的值（去掉分隔符与空白）。"""
    text = line.strip().lstrip("|").strip()
    rest = text[len(label) :].strip().lstrip("|").strip().lstrip("：:").strip()
    return rest.strip("|").strip()


def as_int(text: str) -> int | None:
    """输入：任意文本。输出：去掉非数字后的整数，或 None（一个数字都没有）。

    预期行为：货币符号、千分位、百分号都丢掉——金额与税率都是整数，不带小数。
    """
    digits = re.sub(r"[^\d]", "", text)
    return int(digits) if digits else None


def as_date(text: str) -> tuple[int, int, int] | None:
    """输入：任意文本。输出：(年, 月, 日)，或 None。预期行为：年必须在最前面。"""
    match = _DATE_PARTS.search(text)
    if match is None:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _looks_like_item(cells: list[str]) -> bool:
    """输入：一行的单元格。输出：它是不是一条明细行。

    预期行为：判据是「恰好四格，后三格都能解出数字」——不靠表头文字，
    所以表头行（数量 / 单价 / 金额）不会被误判成明细。
    """
    if len(cells) != 4:
        return False
    return all(as_int(cell) is not None for cell in cells[1:])


def parse_document(text: str, loc: dict[str, JsonValue]) -> ParsedInvoice:
    """输入：单据文本、语言字典。输出：`ParsedInvoice`。

    预期行为：逐段扫描——先认「四格且后三格是数字」的明细行，再认「行首是已知字段标签」
    的字段行；其余一律忽略（标题、明细表头、备注、段落引言都在此列）。
    """
    labels = label_index(loc)
    fields: dict[str, str] = {}
    items: list[ParsedItem] = []
    for segment in _segments(text):
        cells = _cells(segment)
        if _looks_like_item(cells):
            qty = as_int(cells[1])
            unit = as_int(cells[2])
            amount = as_int(cells[3])
            assert qty is not None and unit is not None and amount is not None  # 判据已保证
            items.append(ParsedItem(name=cells[0], qty=qty, unit=unit, amount=amount))
            continue
        matched = _match_label(segment, labels)
        if matched is not None and matched[1] not in fields:
            fields[matched[1]] = _value_after_label(segment, matched[0])
    return ParsedInvoice(fields=fields, items=tuple(items))


def merged_labels() -> dict[str, JsonValue]:
    """输入：无。输出：所有语言文件的「标签 → 槽」合并表（`{"field": …}` 形态）。

    预期行为：`Workflow.reparse` 没有语言参数，靠合并表做到语言无关；
    不同语言之间标签不得重叠（由 `tests/workflows/invoice/test_locales.py` 锁住），
    否则同一个标签会指向两个槽，反解就不可靠了。
    """
    from . import locales

    merged: dict[str, dict[str, JsonValue]] = {}
    for lang in locales.available():
        for slot, entry in locales.load(lang)["field"].items():
            bucket = merged.setdefault(slot, {"labels": []})
            for label in entry["labels"]:
                if label not in bucket["labels"]:
                    bucket["labels"].append(label)
    return {"field": merged}
