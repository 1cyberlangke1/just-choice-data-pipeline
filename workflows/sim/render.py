"""游戏与模拟控制 workflow 的渲染：把真值规格写成一屏局面。

输入：`SimSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
版式固定成「标题 → 单号 / 回合 / 资源 / 战利品 → 空行 → 六行动作」，每行四格：
动作名 · 耗 · 收益 · 冷却。**抬头三格与六行四格就是判据的全部输入**。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue



from .spec import SimRow, SimSpec


def field_line(label: str, value: str, loc: dict[str, JsonValue]) -> str:
    """输入：标签、值、语言字典。输出：`标签：值` 一行。

    预期行为：分隔符统一从 `meta.colon` 取——反解要求标签后面紧跟分隔符，
    换语言时不会因为冒号写法不同而认不出来。
    """
    return f"{label}{loc['meta']['colon']}{value}"


def row_line(index: int, row: SimRow, loc: dict[str, JsonValue]) -> str:
    """输入：序号（从 1 起）、一行动作、语言字典。输出：局面那一行。

    预期行为：四格按固定顺序拼——动作名 / 耗 / 收益 / 冷却，用语言文件里的分隔符连起来。
    这四格就是 `can_move` / `gain` / `best` / `cost` 四族的全部输入，所以一行里必须都写出来。
    """
    cells = (
        loc["action"][row.action],
        loc["cost"][row.cost],
        loc["gain"][row.gain],
        loc["cooldown"][row.cooldown],
    )
    return f"[{index}] " + str(loc["meta"]["cell_sep"]).join(str(cell) for cell in cells)


def build(spec: SimSpec, loc: dict[str, JsonValue]) -> str:
    """输入：规格、语言字典。输出：整屏局面文本。

    预期行为：每一格都只从**规格里已经算好的东西**取（单号、抬头三格、六行动作），
    渲染器不做任何判断——判断全在 `spec` 的规则函数里。
    """
    meta = loc["meta"]
    labels = loc["label"]
    lines = [
        str(meta["title"]),
        field_line(str(labels["request"]), spec.request_id, loc),
        field_line(str(labels["turn"]), str(spec.turn), loc),
        field_line(str(labels["resources"]), str(spec.resources), loc),
        field_line(str(labels["haul"]), str(spec.haul), loc),
        "",
    ]
    for index, row in enumerate(spec.rows, start=1):
        lines.append(row_line(index, row, loc))
    return "\n".join(lines)
