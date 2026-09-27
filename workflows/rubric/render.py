"""打分与评分 workflow 的渲染：把真值规格写成一屏评分单。

输入：`RubricSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
版式固定成「标题 → 单号 → 空行 → 六行判据」，每行四格：判据 · 权重 · 要求 · 观测。
**抬头一行与判据四格是判据的全部输入**。
"""

from __future__ import annotations

from typing import Any

from .spec import RubricRow, RubricSpec


def field_line(label: str, value: str, loc: dict[str, Any]) -> str:
    """输入：标签、值、语言字典。输出：`标签：值` 一行。

    预期行为：分隔符统一从 `meta.colon` 取——反解要求标签后面紧跟分隔符，
    换语言时不会因为冒号写法不同而认不出来。
    """
    return f"{label}{loc['meta']['colon']}{value}"


def row_line(index: int, row: RubricRow, loc: dict[str, Any]) -> str:
    """输入：序号（从 1 起）、一行判据、语言字典。输出：评分单那一行。

    预期行为：四格按固定顺序拼——判据 / 权重 / 要求 / 观测，用语言文件里的分隔符连起来。
    这四格就是 `met` / `band` / `gate` / `top` 四族的全部输入，所以一行里必须都写出来。
    """
    cells = (
        loc["criterion"][row.criterion],
        loc["weight"][row.weight],
        loc["need"][row.need],
        loc["count"][row.count],
    )
    return f"[{index}] " + str(loc["meta"]["cell_sep"]).join(str(cell) for cell in cells)


def build(spec: RubricSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：整张评分单文本。

    预期行为：每一格都只从**规格里已经算好的东西**取（单号、六行判据），渲染器不做任何
    判断——判断全在 `spec` 的规则函数里。
    """
    meta = loc["meta"]
    lines = [
        str(meta["title"]),
        field_line(str(meta["request_label"]), spec.request_id, loc),
        "",
    ]
    for index, row in enumerate(spec.rows, start=1):
        lines.append(row_line(index, row, loc))
    return "\n".join(lines)

