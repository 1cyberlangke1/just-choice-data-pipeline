"""假后端共用的构造件：造一个填满记账字段的 `Draft`。

输入：草稿数据、原始文本。输出：`Draft`。预期行为：记账字段填假值（`fake`），不进产物——
三十来个包的 conftest 各抄一份纯属重复，抄歪了还会让「假后端长什么样」各包不一致。
"""

from __future__ import annotations

from jc.backend.types import Draft
from jc.jsontypes import JsonValue


def draft(data: dict[str, JsonValue], raw_text: str = "<fake>") -> Draft:
    """输入：草稿数据、原始文本。输出：`Draft`。预期行为：记账字段填假值，不进产物。"""
    return Draft(
        data=data,
        raw_text=raw_text,
        backend="fake",
        provider="fake",
        model="fake",
        mode="fake",
    )
