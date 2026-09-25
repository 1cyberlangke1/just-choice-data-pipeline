"""单据 workflow 的渲染：把真值规格写成一份可读的单据文本。

输入：`InvoiceSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
四种版式只改字段区的排版，不改任何数值；「哪个值写在哪」由 `field_text` 唯一决定，
渲染、选项描述、判别器全都走它，避免三处各写一遍。
"""

from __future__ import annotations

import unicodedata
from typing import Any, Final

from . import locales
from .spec import InvoiceSpec

#: 字段顺序：先是抬头（编号 / 日期 / 当事方），再是明细，最后是金额小计。
#: 顺序固定是为了「两次生成逐字节一致」，也是报告里好读。
HEAD_FIELDS: Final[tuple[str, ...]] = (
    "doc",
    "po",
    "contract",
    "issue",
    "due",
    "received",
    "issuer",
    "payer",
    "ship_to",
    "handler",
)
TAIL_FIELDS: Final[tuple[str, ...]] = ("subtotal", "shipping", "tax_rate", "tax", "total")
#: 全部字段槽（含只在有税额行时出现的 `tax`）。
ALL_FIELDS: Final[tuple[str, ...]] = HEAD_FIELDS + TAIL_FIELDS

#: 金额槽 / 日期槽 / 文本槽的划分（反解要按类解析，见 `parse.py`）。
DATE_SLOTS: Final[tuple[str, ...]] = ("issue", "due", "received")
MONEY_SLOTS: Final[tuple[str, ...]] = ("subtotal", "shipping", "tax", "total")


def format_date(date: tuple[int, int, int], fmt: str, loc: dict[str, Any]) -> str:
    """输入：日期、日期格式名、语言字典。输出：渲染后的日期文本。

    预期行为：模板由语言文件给（中文是「年月日」，英文是短横线），但**必须年在前**——
    反解按「四位年 + 月 + 日」的顺序读，这条由测试对每种格式做往返校验锁住。
    """
    year, month, day = date
    spec = loc["date"].get(fmt)
    if spec is None:
        raise KeyError(f"语言文件里没有日期格式 {fmt!r}；可用：{sorted(loc['date'])}")
    return spec["tpl"].format(y=year, m=month, d=day)


def format_money(amount: int, currency: str, loc: dict[str, Any]) -> str:
    """输入：金额（整数）、货币名、语言字典。输出：带符号与千分位的金额文本。"""
    entry = loc["currency"].get(currency)
    if entry is None:
        raise KeyError(f"语言文件里没有货币 {currency!r}；可用：{sorted(loc['currency'])}")
    return loc["money"]["tpl"].format(symbol=entry["symbol"], amount=f"{amount:,}")


def format_rate(rate: int, loc: dict[str, Any]) -> str:
    """输入：税率（百分数）、语言字典。输出：税率文本（如 `6%`）。"""
    return loc["rate"]["tpl"].format(rate=rate)


def display_width(text: str) -> int:
    """输入：任意文本。输出：终端显示宽度（东亚宽/全角字符按 2 格算）。

    预期行为：`aligned_table` 的补位必须按**显示宽度**算——中文标签是双宽字符，
    按字符数补齐会让整张表参差不齐（`len("票号") == 2` 但占 4 格）。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    """输入：文本、目标显示宽度。输出：右侧补空格到该宽度（已够宽就原样返回）。"""
    return text + " " * max(0, width - display_width(text))


def field_text(spec: InvoiceSpec, slot: str, loc: dict[str, Any]) -> str:
    """输入：规格、字段槽、语言字典。输出：该槽渲染出来的文本。

    预期行为：这是「槽 → 文本」的唯一出口——渲染、选项描述、往返判别器都调它，
    三处一旦各写一遍，就会出现「数据里是这个值、判别器按另一个值比」的假阳性。
    """
    case = spec.case
    if slot in ("doc", "po", "contract"):
        return spec.id_of(slot)
    if slot in DATE_SLOTS:
        return format_date(spec.date_of(slot), case.date_format, loc)
    if slot in ("issuer", "payer", "ship_to"):
        return locales.word_of(loc, "vendor", spec.party_of(slot))
    if slot == "handler":
        return locales.word_of(loc, "handler", spec.handler_index)
    if slot in MONEY_SLOTS:
        return format_money(spec.money_of(slot), case.currency, loc)
    if slot == "tax_rate":
        return format_rate(spec.tax_rate, loc)
    raise KeyError(f"未知字段槽：{slot!r}")


def item_name(spec: InvoiceSpec, index: int, loc: dict[str, Any]) -> str:
    """输入：规格、明细行下标、语言字典。输出：该行的品名。"""
    return locales.word_of(loc, "service", spec.items[index].name_index)


def is_visible(spec: InvoiceSpec, slot: str) -> bool:
    """输入：规格、字段槽。输出：这份单据会不会写这个槽。

    预期行为：只有 `tax` 是有条件的——没有税额行就整行不写。
    渲染、选项描述、判别器都问这一个函数，避免三处各写一遍「有没有税额行」。
    """
    return slot != "tax" or spec.has_tax


def visible_fields(spec: InvoiceSpec) -> tuple[str, ...]:
    """输入：规格。输出：这份单据真的会写出来的字段槽（没有税额行就没有 `tax`）。"""
    return tuple(slot for slot in ALL_FIELDS if is_visible(spec, slot))


def expected_fields(spec: InvoiceSpec, loc: dict[str, Any]) -> dict[str, str]:
    """输入：规格、语言字典。输出：{槽: 该写出来的文本}（往返判别器拿它比对）。"""
    return {slot: field_text(spec, slot, loc) for slot in visible_fields(spec)}


def expected_items(spec: InvoiceSpec, loc: dict[str, Any]) -> list[tuple[str, int, int, int]]:
    """输入：规格、语言字典。输出：[(品名, 数量, 单价, 金额)]（往返判别器拿它比对）。"""
    return [
        (item_name(spec, index, loc), item.qty, item.unit, item.amount)
        for index, item in enumerate(spec.items)
    ]


def _label(slot: str, loc: dict[str, Any], style: str) -> str:
    """输入：槽名、语言字典、标签风格。输出：本次渲染用的标签文本。"""
    return locales.label_of(loc, slot, style)


def _field_block(
    slots: tuple[str, ...], spec: InvoiceSpec, loc: dict[str, Any], *, with_intro: bool = False
) -> list[str]:
    """输入：要渲染的槽、规格、语言字典。输出：字段区的若干行。

    预期行为：版式只影响排版——`paragraph` 把整个字段区压成一行（分号分隔，
    引言单独占一行，这样反解不依赖句号——英文的 `.` 与日期、标签里的点会撞车），
    `aligned_table` 把标签补齐到等宽，`pipe_table` 用竖线，`labeled_lines` 一行一个。
    冒号与分号取自语言文件：英文单据用 `: ` / `; `，不混中文标点。
    引言只在 `with_intro` 时输出：抬头与金额区各调一次本函数，引言只能出现一次。
    """
    layout = spec.case.layout
    # 先按可见性过滤：没有税额行的单据不该写出 `tax` 行（写出来会让往返判别器判它「多出字段」）
    pairs = [(slot, field_text(spec, slot, loc)) for slot in slots if is_visible(spec, slot)]
    colon = loc["meta"]["colon"]
    if layout == "paragraph":
        joined = loc["meta"]["sep"].join(
            f"{_label(slot, loc, spec.case.labels)}{colon}{value}" for slot, value in pairs
        )
        return [loc["meta"]["intro"], joined] if with_intro else [joined]
    if layout == "aligned_table":
        # 列宽按**全部**会写出来的字段算，而不是只按本段（抬头 / 金额区）——
        # 否则两段的列宽各算各的，同一份单据里字段值会从两个不同的列开始。
        width = (
            max(display_width(_label(slot, loc, spec.case.labels)) for slot in visible_fields(spec))
            + 3
        )
        return [
            f"{_pad(_label(slot, loc, spec.case.labels), width)}{value}"
            for slot, value in pairs
        ]
    if layout == "pipe_table":
        return [
            f"| {_label(slot, loc, spec.case.labels)} | {value} |" for slot, value in pairs
        ]
    return [f"{_label(slot, loc, spec.case.labels)}{colon}{value}" for slot, value in pairs]


def _item_block(spec: InvoiceSpec, loc: dict[str, Any]) -> list[str]:
    """输入：规格、语言字典。输出：明细表的若干行（表头 + 每行 + 无表尾）。

    预期行为：`aligned_table` 用等宽列，其余版式用竖线表格——两种形态反解都认，
    认列的方式是「后三格是数字」，不依赖表头文字。
    """
    columns = [loc["column"][name]["labels"][0] for name in ("name", "qty", "unit", "amount")]
    rows: list[list[str]] = [columns]
    for index, item in enumerate(spec.items):
        rows.append(
            [
                item_name(spec, index, loc),
                str(item.qty),
                format_money(item.unit, spec.case.currency, loc),
                format_money(item.amount, spec.case.currency, loc),
            ]
        )
    if spec.case.layout == "aligned_table":
        widths = [max(display_width(row[col]) for row in rows) for col in range(4)]
        return [
            "".join(_pad(cell, widths[col] + 3) for col, cell in enumerate(row)).rstrip()
            for row in rows
        ]
    return ["| " + " | ".join(row) + " |" for row in rows]


def build(spec: InvoiceSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：整份单据文本。

    预期行为：抬头字段 → 明细表 → 金额字段 → 备注；字段与数值一一对应，
    不多写也不少写（少写会被往返判别器抓出来）。
    """
    lines: list[str] = [loc["meta"]["title"]]
    lines.extend(_field_block(HEAD_FIELDS, spec, loc, with_intro=True))
    lines.append(loc["meta"]["items_title"])
    lines.extend(_item_block(spec, loc))
    lines.extend(_field_block(TAIL_FIELDS, spec, loc))
    lines.append(f"{loc['meta']['note_label']}{loc['meta']['colon']}{loc['meta']['note_value']}")
    return "\n".join(lines)
