"""技能选择 workflow 的渲染：把真值规格写成一屏选择单。

输入：`SkillSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
版式固定成「标题 → 请求号 → 标签 / 授权 / 已占用 → 空行 → 六行技能表」，
每行三格：适用场景 · 权限范围 · 资源。**抬头四行与技能表三格是判据的全部输入**。
"""

from __future__ import annotations

from typing import Any

from .spec import CatalogRow, SkillSpec


def field_line(label: str, value: str, loc: dict[str, Any]) -> str:
    """输入：标签、值、语言字典。输出：`标签：值` 一行。

    预期行为：分隔符统一从 `meta.colon` 取——反解要求标签后面紧跟分隔符，
    换语言时不会因为冒号写法不同而认不出来。
    """
    return f"{label}{loc['meta']['colon']}{value}"


def row_line(index: int, row: CatalogRow, loc: dict[str, Any]) -> str:
    """输入：序号（从 1 起）、一行技能、语言字典。输出：技能表那一行。

    预期行为：三格按固定顺序拼——适用场景 / 权限范围 / 资源，用语言文件里的分隔符连起来。
    这三格就是 `pick` / `scope` / `conflict` 三族的全部输入，所以一行里必须都写出来。
    """
    cells = (loc["tag"][row.tag], loc["scope"][row.scope], loc["resource"][row.resource])
    return f"[{index}] " + str(loc["meta"]["cell_sep"]).join(str(cell) for cell in cells)


def build(spec: SkillSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：整屏选择单文本。

    预期行为：每一格都只从**规格里已经算好的东西**取（请求号、标签 / 授权 / 已占用的键、
    技能表），渲染器不做任何判断——判断全在 `spec` 的规则函数里。
    """
    meta = loc["meta"]
    lines = [
        str(meta["title"]),
        field_line(str(meta["request_label"]), spec.request_id, loc),
        field_line(str(meta["tag_label"]), str(loc["tag"][spec.case.tag]), loc),
        field_line(str(meta["grant_label"]), str(loc["grant"][spec.case.grant]), loc),
        field_line(str(meta["inuse_label"]), str(loc["tag"][spec.inuse_tag]), loc),
        "",
    ]
    for index, row in enumerate(spec.rows, start=1):
        lines.append(row_line(index, row, loc))
    return "\n".join(lines)
