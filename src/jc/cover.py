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

`Fingerprinted` 是给 case 类的 mixin：按 `as_dict()` 的键序拼出指纹。27 家的 case 类各抄一份
纯属重复，抄歪了还会让「同一个 case 在不同进程里指纹不同」——分桶与种子派生都靠这一串。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random
import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import combinations, product
from typing import Final, Protocol, TypeVar

# 全枚举的规模上限：超过就换成对覆盖（PLAN 3.3 定的口径）。
FULL_ENUM_LIMIT: Final[int] = 10_000

Pair = tuple[str, JsonValue, str, JsonValue]


class HasDict(Protocol):
    """能报出自己那一组取值的东西（本仓库所有 case 类型都满足）。"""

    def as_dict(self) -> Mapping[str, JsonValue]:
        """输入：无。输出：`{维度名: 取值}`。"""


class Fingerprinted:
    """给 case 类提供 `fingerprint`：按 `as_dict()` 的键序拼成一串（27 家同构，只留这一份）。

    输入：无（读子类自己的 `as_dict()`）。
    输出：形如 `hits=A1+A2|surface=feed` 的一串。
    预期行为：键序由子类的 `as_dict()` 定（各家都保证固定），所以同一个 case 在任何进程、
    任何切分里都得到同一串——分桶与种子派生都靠这个。
    """

    def as_dict(self) -> Mapping[str, JsonValue]:
        """输入：无。输出：`{维度名: 取值}`。预期行为：子类实现，键序固定。"""
        raise NotImplementedError

    def fingerprint(self) -> str:
        """输入：无。输出：case 指纹。预期行为：同一 case 任何进程都得到同一串。"""
        return "|".join(f"{key}={value}" for key, value in self.as_dict().items())


#: case 的占位类型：`slice_by_bucket` 只要求「有桶号、能报取值」，不关心是哪种 case。
CaseT = TypeVar("CaseT", bound=HasDict)


def restricted_dims(
    dims: Mapping[str, tuple[JsonValue, ...]],
    overrides: Mapping[str, tuple[JsonValue, ...]],
) -> dict[str, tuple[JsonValue, ...]]:
    """输入：基础维度表 + 要覆盖的维度。输出：覆盖后的维度表。

    预期行为：不改动入参——`features.DIMS` 是模块级常量，就地改会污染同一进程里的其他调用方。
    38 家 workflow 的 `_restricted` 原来是逐字同构的（只差文档串措辞），收在这里只留这一份。
    """
    merged: dict[str, tuple[JsonValue, ...]] = dict(dims)
    merged.update(overrides)
    return merged


#: 维度取值的占位类型：`restrictor` 不关心取值是字符串还是别的 JSON 值。
DimValue = TypeVar("DimValue")


class DimsRestrictor(Protocol[DimValue]):
    """`restrictor` 造出来的包装函数：形状与原来那 38 份 `_restricted` 一模一样。

    输入：要覆盖的维度（关键字实参）。输出：覆盖后的维度表。
    预期行为：只声明形状、实现在 `restrictor` 里——本仓库的「禁 Any」尺子拦 `Callable[..., X]`
    （那里的 `...` 就是「参数表是 Any」），所以包装函数用 `Protocol` 声明。
    """

    def __call__(self, **overrides: tuple[DimValue, ...]) -> dict[str, tuple[DimValue, ...]]:
        ...


def restrictor(dims: Mapping[str, tuple[DimValue, ...]]) -> DimsRestrictor[DimValue]:
    """输入：基础维度表。输出：`_restricted(**overrides)` 那样的包装函数。

    预期行为：把各家 workflow 里那三行逐字同构的 `_restricted` 收成一行赋值
    （`_restricted = restrictor(features.DIMS)`）；每次调用现算，不改动 `dims`。
    """

    def _restricted(**overrides: tuple[DimValue, ...]) -> dict[str, tuple[DimValue, ...]]:
        return restricted_dims(dims, overrides)

    return _restricted


#: 取值对约束：收 `(维度A, 取值A, 维度B, 取值B)`（A 在 `dims` 里更靠前），返回能否共存。
PairRule = Callable[[str, object, str, object], bool]


def _always(dim_a: str, value_a: object, dim_b: str, value_b: object) -> bool:
    """默认约束：任何取值对都允许。输入：两对维度取值。输出：True。"""
    del dim_a, value_a, dim_b, value_b
    return True


def row_allowed(names: list[str], row: Mapping[str, JsonValue], rule: PairRule) -> bool:
    """输入：维度名顺序、一行组合、取值对约束。输出：这一行的所有取值对是否都允许。

    预期行为：`names` 决定比对的先后（A 在 B 前），只比**两两**取值——单维取值的合法性由
    维度表自己保证。workflow 拿它当「这组特征写得出来吗」的判据（`compatible`）。
    """
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
            if row_allowed(names, row, rule):
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


def all_hit_sets(tag_ids: Sequence[str]) -> tuple[str, ...]:
    """输入：编号表。输出：它的全部命中组合（13 家同构，只留这一份）。

    预期行为：空串在前，然后单条、两条……顺序稳定（`itertools.combinations` 按入参顺序出），
    所以 case 矩阵能逐字节重建。每条编号在 2^n 组里各出现 2^(n-1) 次——这就是「单条判据」
    那一族的答案占比按构造就是 50% 的原因。
    """
    out: list[str] = []
    for size in range(len(tag_ids) + 1):
        for combo in combinations(tag_ids, size):
            out.append("+".join(combo))
    return tuple(out)


def near_miss_combos(
    tag_ids: Sequence[str],
    truth: tuple[str, ...],
    rng: random.Random,
    count: int = 3,
) -> tuple[tuple[str, ...], ...]:
    """输入：判据编号表、真值组合、随机源、要几个。输出：若干个「最像真值」的干扰组合。

    预期行为：先按「与真值只差几处」排序，再从最近的一批里抽 `count` 个——差得越少越难分辨，
    题目才有区分度；抽完按 (条数, 编号) 排好，保证同一入参必得同一结果。
    真值本身永远不在候选里（干扰项不能等于答案）。
    """
    truth_set = set(truth)
    near = sorted(
        (
            combo
            for size in range(len(tag_ids) + 1)
            for combo in combinations(tag_ids, size)
            if set(combo) != truth_set
        ),
        key=lambda combo: (len(set(combo) ^ truth_set), combo),
    )
    picked = rng.sample(near[:8], count)
    return tuple(sorted(picked, key=lambda combo: (len(combo), combo)))


def slice_by_bucket(
    cases: Sequence[CaseT],
    split: str,
    bucket_of: Callable[[CaseT], int],
    buckets: Mapping[str, tuple[int, int]],
) -> list[CaseT]:
    """输入：同分布 case 列表、切分名、桶号函数、切分表。输出：该切分的 case（顺序稳定）。

    预期行为：先按 (桶号, 组合) 排序再切片，比例才是**精确**的——不排序就切，同一批 case
    换个生成顺序就会落进不同切分，切分不可复现。`buckets[split]` 是 `(下界, 上界)` 百分位。
    """
    ordered = sorted(cases, key=lambda case: (bucket_of(case), tuple(case.as_dict().items())))
    low, high = buckets[split]
    total = len(ordered)
    return ordered[low * total // 100 : high * total // 100]


def hash_bucket(case: CaseT, seed: int) -> int:
    """输入：一个 case、种子。输出：0~99 的哈希桶号。

    预期行为：只按 `as_dict()` 报出来的那组取值算，与切分名无关——同一组取值永远落在同一个
    桶里，所以「同一组取值不跨切分」是**算出来的**，不是靠人排的。36 个包各抄一份，只留这一份。
    """
    payload = "|".join(f"{key}={value}" for key, value in case.as_dict().items())
    return int(hashlib.sha256(f"{seed}|{payload}".encode("utf-8")).hexdigest()[:8], 16) % 100
