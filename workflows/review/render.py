"""改动审查 workflow 的渲染：把真值规格写成一屏改动。

输入：`ChangeSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
版式固定成「标题 → 改动号 → 空行 → 六处改动」，每处两行：
抬头行（序号 + 文件 + 变更 + 规模）与正文行（这一处改了什么）。
**抬头三格与正文是判据的全部输入**。
"""

from __future__ import annotations

from typing import Any

from .spec import ChangeSpec, Finding


def finding_header(index: int, finding: Finding, loc: dict[str, Any]) -> str:
    """输入：序号（从 1 起）、一处改动、语言字典。输出：抬头行。

    预期行为：三格按固定顺序拼——文件 / 变更 / 规模，用语言文件里的分隔符连起来。
    这三格加上正文就是判据的全部输入，所以一行里必须都写出来。
    """
    cells = (
        loc["file"][finding.file],
        loc["change"][finding.change],
        loc["size"][finding.size],
    )
    return f"[{index}] " + str(loc["meta"]["cell_sep"]).join(cells)


def finding_body(finding: Finding, loc: dict[str, Any]) -> str:
    """输入：一处改动、语言字典。输出：正文行（带缩进）。

    预期行为：按正文键取模板——键是 `{规则}_{槽}`（干净槽是 `clean_{槽}`），
    模板里的字面量正是规则判据要抓的东西。
    """
    if finding.body not in loc["body"]:
        raise KeyError(f"语言文件里没有正文模板 {finding.body!r}")
    return str(loc["meta"]["body_indent"]) + str(loc["body"][finding.body])


def build(spec: ChangeSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：整屏改动文本。

    预期行为：每一格都只从**规格里已经算好的东西**取（四格、改动号），渲染器不做任何
    判断——判断全在 `spec` 的规则函数里。
    """
    meta = loc["meta"]
    lines = [
        str(meta["title"]),
        f"{meta['change_label']}{meta['colon']}{spec.change_id}",
        "",
    ]
    for index, finding in enumerate(spec.rows, start=1):
        lines.append(finding_header(index, finding, loc))
        lines.append(finding_body(finding, loc))
    return "\n".join(lines)
