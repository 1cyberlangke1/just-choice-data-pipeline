"""记录匹配（match）workflow 的渲染：把真值规格写成一屏「两条目录记录」。

输入：`MatchSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
版式固定成「抬头（含请求号）→ 记录 A 一块 → 记录 B 一块」，每块是「侧标题 + 若干 `栏: 值`」。

关键约束：**两边逐字相同的栏必须真的逐字相同**（否则 `same` 会被读成 `variant`），
所以只有 `variant` 那一栏才让两边用不同的写法（见 `spelling_of`）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from . import spec as spec_mod
from .spec import MatchCase, MatchSpec


def cell(field: str, value: Any, spelling: int, loc: dict[str, Any]) -> str:
    """输入：栏名、归一化后的值、写法下标、语言字典。输出：这一格写出来的原文。

    预期行为：名称与品牌用商品键查表；规格 / 度数 / 包装的键是数值或包装键，
    语言文件里统一按 `str(值)` 存（`750` / `5.0` / `glass`）。
    """
    if field in ("name", "brand"):
        return loc[field][value][spelling]
    return loc[field][str(value)][spelling]


def cells_of(case: MatchCase, pool: Sequence[str]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """输入：case、`diff` 取值池。输出：(记录 A 的每栏值, 记录 B 的每栏值)，顺序同 `spec.FIELDS`。

    预期行为：`diff` 那一栏给 B 换成**池子里另一个商品**在这一栏上的值；换过去的值得确实不同，
    否则「不一致」就是假的（`spec.other_product` 逐个跳过同值的商品）。池子由
    `spec.diff_pool(case 所在的切分)` 给——同分布侧不给留出商品，留出才在文本上也成立。
    """
    left = [spec_mod.field_value(field, case.product) for field in spec_mod.FIELDS]
    right = list(left)
    for index, field in enumerate(spec_mod.FIELDS):
        if case.pattern[index] == spec_mod.DIFF:
            other = spec_mod.other_product(field, case.product, pool)
            right[index] = spec_mod.field_value(field, other)
    return tuple(left), tuple(right)


def spelling_of(case: MatchCase, side: int, kind: str) -> int:
    """输入：case、哪条记录（0 = A / 1 = B）、这一栏的关系。输出：这一格用第几种写法。

    预期行为：只有 `variant` 那一栏才让两边用不同的写法；`same` 两边必须逐字相同，
    `diff` 的值本来就不同、写法统一用第 0 种（不然「不一致」的理由会被写法搅浑）。
    """
    return case.spelling[side] if kind == spec_mod.VARIANT else 0


def record_lines(
    case: MatchCase, side: int, loc: dict[str, Any], pool: Sequence[str]
) -> tuple[str, ...]:
    """输入：case、哪条记录、语言字典、`diff` 取值池。输出：这一条记录的行（不含侧标题）。

    预期行为：五栏按 `LAYOUT_ORDERS[case.layout]` 排；额外栏（产地）固定排在**最后**
    ——它不在比较范围内，是「看着像可比的一栏、其实不算」的干扰项。
    """
    left, right = cells_of(case, pool)
    values = (left, right)[side]
    sep = loc["meta"]["sep"]
    lines: list[str] = []
    for index in spec_mod.LAYOUT_ORDERS[case.layout]:
        field = spec_mod.FIELDS[index]
        spelling = spelling_of(case, side, case.pattern[index])
        lines.append(f"{loc['field'][field]}{sep}{cell(field, values[index], spelling, loc)}")
    if case.extra == "origin":
        side_key = ("origin_left", "origin_right")[side]
        lines.append(f"{loc['extra']['origin_label']}{sep}{loc['extra'][side_key]}")
    return tuple(lines)


def build(spec: MatchSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：`state` 全文。

    预期行为：抬头一行（标签 + 空格 + 请求号），随后两条记录各一块；每条记录的行缩进两格
    （反解按 `strip()` 后的内容认标签，缩进只是给人看的）。
    """
    sep = loc["meta"]["sep"]
    pool = spec_mod.diff_pool(spec.split)
    lines = [f"{loc['meta']['request']} {spec.query_id}"]
    for side, key in enumerate(("left", "right")):
        # 侧标题后面不接值，`sep` 里的空格要收掉（英文的 `": "` 会留一个行尾空格）
        lines.append(f"{loc['meta'][key]}{sep}".rstrip())
        lines.extend(f"  {line}" for line in record_lines(spec.case, side, loc, pool))
    return "\n".join(lines) + "\n"
