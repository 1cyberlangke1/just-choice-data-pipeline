"""采样旋钮的通用校验：空取值表 / 少元素的区间。

为什么要有：workflow 的 config 里那些「取值表」和「区间」是最容易被改坏的旋钮。
以前空表会一路活到采样阶段才炸，抛的是 `Cannot choose from an empty sequence`、
`tuple index out of range`、`Random.randint() missing 2 required positional arguments`
这种裸异常——既看不出是哪个字段坏了，也看不出该填什么（4 个 workflow × 每个切分的
边界扫描里，68 个组合全是这类裸异常，见 PLAN 10.32）。

输入/输出：纯函数，不碰任何数据；命中问题抛 `ValueError`（人话、点名字段），正常返回 None。
预期行为：调用方在 config 的 `__post_init__` 里逐字段调用，**错误在开工前暴露**。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from collections.abc import Sequence



def require_pool(field: str, values: Sequence[JsonValue]) -> None:
    """输入：字段名、取值表。输出：无。预期行为：空表报人话（点名是哪个字段）。"""
    if len(values) == 0:
        raise ValueError(f"{field} 不能是空表：至少要有一个取值")


def require_bounds(field: str, bounds: Sequence[int]) -> None:
    """输入：字段名、(下界, 上界)。输出：无。

    预期行为：必须恰好两个整数、且下界 <= 上界；否则报人话——不然会一路拖到
    `rng.randint(*bounds)` 才抛 `missing 2 required positional arguments` 之类的裸异常。
    """
    if len(bounds) != 2:
        raise ValueError(f"{field} 必须是 (下界, 上界) 两个数，现在有 {len(bounds)} 个")
    lo, hi = bounds
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (lo, hi)):
        raise ValueError(f"{field} 的两个元素都必须是整数，现在是 {bounds!r}")
    if lo > hi:
        raise ValueError(f"{field} 的下界不能大于上界：{lo} > {hi}")
