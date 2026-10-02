"""规格字典（通用层）：把各家 spec 摊平成可 JSON 序列化的字典。

用途：写进 manifest 与调试输出（**不进样本行**）。22 家的 `spec_to_dict` 只有类型名与
字段集不同，正文逐字相同——按「有没有标签组 / 有没有判定计划」留三份实现，各家只留一行转调。

输入/输出：只吃关键字参数，返回 `dict[str, JsonValue]`。
预期行为：键序固定（注入特征在前，然后 `phrasing` / `split` / `seq`，判定计划字段垫底），
同一 spec 每次得到同一份字典——manifest 要能逐字节复现。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from jc.cover import HasDict
from jc.jsontypes import JsonValue


class HasPlan(Protocol):
    """能报出判定计划的命中组合与干扰组合的东西（各家 Plan 都是不可变数据类）。"""

    #: 命中的判据编号，如 `("A1", "A3")`。
    hits: Sequence[str]
    #: 干扰组合，每组一个编号元组；按构造保证不等于答案。
    distractors: Sequence[Sequence[str]]


def tagged_spec_dict(*, tags: HasDict, phrasing: int, split: str, seq: int) -> dict[str, JsonValue]:
    """输入：标签组、问法编号、切分名、样本序号。输出：可 JSON 序列化的规格字典。"""
    return {
        "tags": tags.as_dict(),
        "phrasing": phrasing,
        "split": split,
        "seq": seq,
    }


def case_spec_dict(*, case: HasDict, phrasing: int, split: str, seq: int) -> dict[str, JsonValue]:
    """输入：注入特征、问法编号、切分名、样本序号。输出：可 JSON 序列化的规格字典。"""
    return {
        "case": case.as_dict(),
        "phrasing": phrasing,
        "split": split,
        "seq": seq,
    }


def planned_spec_dict(
    *, case: HasDict, plan: HasPlan, phrasing: int, split: str, seq: int
) -> dict[str, JsonValue]:
    """输入：注入特征、判定计划、问法编号、切分名、样本序号。输出：可 JSON 序列化的规格字典。"""
    return {
        "case": case.as_dict(),
        "phrasing": phrasing,
        "split": split,
        "seq": seq,
        "hits": list(plan.hits),
        "distractors": [list(combo) for combo in plan.distractors],
    }
