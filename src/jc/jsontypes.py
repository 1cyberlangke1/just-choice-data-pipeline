"""JSON 值的类型别名（`jc` 通用层）。

输入/输出：无（纯类型）。
预期行为：把「这是一个 JSON 值」与「这是任意 Python 对象」在类型上分开——
凡是装样本 / 统计 / 素材 payload 的容器一律写 `dict[str, JsonValue]`，
这样 `object` 才能被整体禁掉（见 `tests/core/test_no_any.py`）。
"""

from __future__ import annotations

#: 一个 JSON 值：标量、数组或对象。
JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]