"""假后端共用的构造件：造一个填满记账字段的 `Draft`，以及各家假后端的共用底座。

输入：草稿数据、原始文本。输出：`Draft`。预期行为：记账字段填假值（`fake`），不进产物——
三十来个包的 conftest 各抄一份纯属重复，抄歪了还会让「假后端长什么样」各包不一致。
"""

from __future__ import annotations

from jc.backend.types import Draft
from jc.jsontypes import JsonValue


class ScriptedBackend:
    """各家脚本化假后端的共用底座：只放「逐字同构」的那一个方法。

    输入：任务列表。输出：与入参顺序一致的草稿列表。预期行为：逐个走子类的 `generate`——
    38 个包的假后端原来各抄一份，而那三行本来就只有一句。
    """

    def generate_many(self, tasks: list[object]) -> list[Draft]:
        """输入：任务列表。输出：与入参顺序一致的草稿列表。"""
        return [self.generate(task) for task in tasks]


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
