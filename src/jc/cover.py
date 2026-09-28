"""维度组合枚举器（通用层）：给「维度 → 取值」生成覆盖用的组合。

用途：格式 case 的正例（本仓库现在用）与 M4 的 tag 组合留出是**同一个实现**
（一个枚举 schema 维度、一个枚举 tag 维度，见 PLAN 3.3）。

输入/输出：`cover_dims` 收 `{维度名: 取值序列}`，返回 `[{维度名: 取值}]`。
预期行为：
- 组合数不超过 `max_full` 时**全枚举**，顺序与 `itertools.product` 一致；
- 超过时用**成对覆盖（pairwise / covering array）**：任意两维的任意取值组合
  至少出现一次，行数远小于全枚举；
- 同一输入必须给出同一输出（case 库要能逐字节重建）；
- 可选的 `allow_pair` 用来表达「有些取值天生不能同时出现」（例如清单同时说
  「要求点击链接」和「邮件里没有任何链接」）。被否掉的取值对**不进结果，
  也不算没覆盖**——否则枚举器会一直追一个不可能达成的目标。

`missing_pairs` 是给测试与自检用的反向检查：算出还没被覆盖的取值对。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import product
from typing import Final

# 全枚举的规模上限：超过就换成对覆盖（PLAN 3.3 定的口径）。
FULL_ENUM_LIMIT: Final[int] = 10_000

Pair = tuple[str, JsonValue, str, JsonValue]

#: 取值对约束：收 `(维度A, 取值A, 维度B, 取值B)`（A 在 `dims` 里更靠前），返回能否共存。
PairRule = Callable[[str, object, str, object], bool]


def _always(dim_a: str, value_a: object, dim_b: str, value_b: object) -> bool:
    """默认约束：任何取值对都允许。输入：两对维度取值。输出：True。"""
    del dim_a, value_a, dim_b, value_b
    return True


def _row_allowed(names: list[str], row: Mapping[str, JsonValue], rule: PairRule) -> bool:
    """输入：维度名顺序、一行组合、取值对约束。输出：这一行的所有取值对是否都允许。"""
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if not rule(left, row[left], right, row[right]):
                return False
    return True


def cover_dims(
    dims: Mapping[str, Sequence[JsonValue]],
    *,
    max_full: int = FULL_ENUM_LIMIT,
    allow_pair: PairRule | None = None,
) -> list[dict[str, JsonValue]]:
    """枚举维度组合。

    输入：维度表（每个维度至少一个取值）、全枚举上限、可选的取值对约束。
    输出：组合列表。
    预期行为：小空间全枚举；大空间成对覆盖；两者都保持维度顺序与确定性，
    且都只产出「被约束允许」的组合。约束太紧（某维没有任何可选取值）时直接报错，
    不静默少给行——那会让成对覆盖悄悄失效。
    """
    names = list(dims)
    values = [list(dims[name]) for name in names]
    rule = allow_pair or _always
    total = 1
    for item in values:
        total *= len(item)
    if total <= max_full:
        rows: list[dict[str, JsonValue]] = []
        for combo in product(*values):
            row = dict(zip(names, combo, strict=True))
            if _row_allowed(names, row, rule):
                rows.append(row)
        return rows
    return _pairwise(names, values, rule)


def missing_pairs(
    dims: Mapping[str, Sequence[JsonValue]],
    rows: Iterable[Mapping[str, JsonValue]],
    *,
    allow_pair: PairRule | None = None,
) -> set[Pair]:
    """算出还没被覆盖到的取值对。

    输入：维度表、组合列表、可选的取值对约束。
    输出：`{(维度A, 取值A, 维度B, 取值B)}`，空集表示成对覆盖达成。
    预期行为：只认维度表里声明的取值——行里缺维度或取值不在表里都算没覆盖；
    被约束否掉的取值对不算「该覆盖而没覆盖」。
    """
    names = list(dims)
    rule = allow_pair or _always
    need: set[Pair] = set()
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            for left_value in dims[left]:
                for right_value in dims[right]:
                    if rule(left, left_value, right, right_value):
                        need.add((left, left_value, right, right_value))
    for row in rows:
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                need.discard((left, row.get(left), right, row.get(right)))
    return need


def _new_pairs(
    index: int,
    value: object,
    row: list[object | None],
    pending: set[tuple[int, object, int, object]],
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


def _pairwise(
    names: list[str], values: list[list[JsonValue]], rule: PairRule
) -> list[dict[str, JsonValue]]:
    """贪心成对覆盖（AETG 式，确定性）。

    输入：维度名列表、各维取值列表（都非空）、取值对约束。输出：组合列表。
    预期行为：每轮先固定一对还没覆盖的取值当种子（保证本轮一定覆盖到东西、
    一定终止），其余维度按维度顺序挑「新增覆盖最多」的取值，并列取下标小的；
    候选取值里先剔掉与已定维度冲突的那些，所以产出行天然满足约束。
    """
    pending: set[tuple[int, object, int, object]] = set()
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            for left_value in range(len(values[left])):
                for right_value in range(len(values[right])):
                    if rule(
                        names[left],
                        values[left][left_value],
                        names[right],
                        values[right][right_value],
                    ):
                        pending.add((left, left_value, right, right_value))

    rows: list[dict[str, JsonValue]] = []
    while pending:
        seed = min(pending)
        row: list[object | None] = [None] * len(names)
        row[seed[0]] = seed[1]
        row[seed[2]] = seed[3]
        for index in range(len(names)):
            if row[index] is not None:
                continue
            choices = [
                value
                for value in range(len(values[index]))
                if all(
                    rule(
                        names[other],
                        values[other][assigned],
                        names[index],
                        values[index][value],
                    )
                    for other, assigned in enumerate(row)
                    if assigned is not None and other != index
                )
            ]
            if not choices:
                raise ValueError(
                    f"维度 {names[index]!r} 在已定的组合下没有任何可选取值："
                    "取值对约束太紧，成对覆盖不可能达成"
                )
            row[index] = max(
                choices,
                key=lambda value, idx=index: (_new_pairs(idx, value, row, pending), -value),
            )
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                pending.discard((left, row[left], right, row[right]))
        rows.append({names[index]: values[index][row[index]] for index in range(len(names))})
    return rows
