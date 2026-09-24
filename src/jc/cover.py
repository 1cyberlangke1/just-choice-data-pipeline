"""维度组合枚举器（通用层）：给「维度 → 取值」生成覆盖用的组合。

用途：格式 case 的正例（本仓库现在用）与 M4 的 tag 组合留出是**同一个实现**
（一个枚举 schema 维度、一个枚举 tag 维度，见 PLAN 3.3）。

输入/输出：`cover_dims` 收 `{维度名: 取值序列}`，返回 `[{维度名: 取值}]`。
预期行为：
- 组合数不超过 `max_full` 时**全枚举**，顺序与 `itertools.product` 一致；
- 超过时用**成对覆盖（pairwise / covering array）**：任意两维的任意取值组合
  至少出现一次，行数远小于全枚举；
- 同一输入必须给出同一输出（case 库要能逐字节重建）。

`missing_pairs` 是给测试与自检用的反向检查：算出还没被覆盖的取值对。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import product
from typing import Any, Final

# 全枚举的规模上限：超过就换成对覆盖（PLAN 3.3 定的口径）。
FULL_ENUM_LIMIT: Final[int] = 10_000

Pair = tuple[str, Any, str, Any]


def cover_dims(
    dims: Mapping[str, Sequence[Any]],
    *,
    max_full: int = FULL_ENUM_LIMIT,
) -> list[dict[str, Any]]:
    """枚举维度组合。

    输入：维度表（每个维度至少一个取值）、全枚举上限。输出：组合列表。
    预期行为：小空间全枚举；大空间成对覆盖；两者都保持维度顺序与确定性。
    """
    names = list(dims)
    values = [list(dims[name]) for name in names]
    total = 1
    for item in values:
        total *= len(item)
    if total <= max_full:
        return [dict(zip(names, combo, strict=True)) for combo in product(*values)]
    return _pairwise(names, values)


def missing_pairs(
    dims: Mapping[str, Sequence[Any]],
    rows: Iterable[Mapping[str, Any]],
) -> set[Pair]:
    """算出还没被覆盖到的取值对。

    输入：维度表、组合列表。输出：`{(维度A, 取值A, 维度B, 取值B)}`，空集表示成对覆盖达成。
    预期行为：只认维度表里声明的取值——行里缺维度或取值不在表里都算没覆盖。
    """
    names = list(dims)
    need: set[Pair] = set()
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            for left_value in dims[left]:
                for right_value in dims[right]:
                    need.add((left, left_value, right, right_value))
    for row in rows:
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                need.discard((left, row.get(left), right, row.get(right)))
    return need


def _new_pairs(
    index: int,
    value: Any,
    row: list[Any | None],
    pending: set[tuple[int, Any, int, Any]],
) -> int:
    """数一数「把 index 维定成 value」能新覆盖多少未覆盖对（只跟已定的维比）。

    输入：维度下标、候选取值、当前行（None = 还没定）、未覆盖对集合。输出：新覆盖的对数。
    """
    total = 0
    for other, assigned in enumerate(row):
        if assigned is None or other == index:
            continue
        if other < index:
            key = (other, assigned, index, value)
        else:
            key = (index, value, other, assigned)
        if key in pending:
            total += 1
    return total


def _pairwise(names: list[str], values: list[list[Any]]) -> list[dict[str, Any]]:
    """贪心成对覆盖（AETG 式，确定性）。

    输入：维度名列表、各维取值列表（都非空）。输出：组合列表。
    预期行为：每轮先固定一对还没覆盖的取值当种子（保证本轮一定覆盖到东西、
    一定终止），其余维度按维度顺序挑「新增覆盖最多」的取值，并列取下标小的。
    """
    pending: set[tuple[int, Any, int, Any]] = set()
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            for left_value in range(len(values[left])):
                for right_value in range(len(values[right])):
                    pending.add((left, left_value, right, right_value))

    rows: list[dict[str, Any]] = []
    while pending:
        seed = min(pending)
        row: list[Any | None] = [None] * len(names)
        row[seed[0]] = seed[1]
        row[seed[2]] = seed[3]
        for index in range(len(names)):
            if row[index] is not None:
                continue
            row[index] = max(
                range(len(values[index])),
                key=lambda value, idx=index: (_new_pairs(idx, value, row, pending), -value),
            )
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                pending.discard((left, row[left], right, row[right]))
        rows.append({names[index]: values[index][row[index]] for index in range(len(names))})
    return rows
