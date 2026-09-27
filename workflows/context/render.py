"""会话上下文压缩 workflow 的渲染：把真值规格写成一屏会话。

输入：`SessionSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
版式固定成「标题 → 会话号 → 预算 → 当前任务 → 空行 → 五个块」，
每块两行：抬头行（序号 + 类型 + 大小 + 年龄 + 主题）与正文行（这一块的内容）。
**抬头行是判据的全部输入**，正文行只提供语境。
"""

from __future__ import annotations

from typing import Any

from .spec import BlockCase, SessionSpec


def block_header(index: int, block: BlockCase, loc: dict[str, Any]) -> str:
    """输入：块序号（从 1 起）、块、语言字典。输出：抬头行。

    预期行为：四格按固定顺序拼——类型 / 大小 / 年龄 / 主题，用语言文件里的分隔符连起来。
    这四格就是判据的全部输入，所以一行里必须都写出来。
    """
    meta = loc["meta"]
    cells = (
        loc["kind"][block.kind],
        loc["size"][block.size],
        loc["age"][block.age],
        loc["topic"][block.topic],
    )
    return f"[{index}] " + meta["block_sep"].join(cells)


def block_content(block: BlockCase, loc: dict[str, Any]) -> str:
    """输入：块、语言字典。输出：正文行（带缩进）。

    预期行为：按「类型_大小_年龄_主题」四元组取模板——四元组在一段会话里两两不同，
    所以同一屏里不会出现两行一模一样的正文（撞了会让人以为渲染漏了块）。
    """
    key = f"{block.kind}_{block.size}_{block.age}_{block.topic}"
    if key not in loc["content"]:
        raise KeyError(f"语言文件里没有正文模板 {key!r}")
    return loc["meta"]["content_indent"] + loc["content"][key]


def build(spec: SessionSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：整屏会话文本。

    预期行为：抬头与正文都只从**规格里已经算好的东西**取（块属性、会话号、预算），
    渲染器不做任何判断——判断全在 `spec` 的规则函数里。
    """
    meta = loc["meta"]
    colon = meta["colon"]
    lines = [
        meta["title"],
        f"{meta['session_label']}{colon}{spec.session_id}",
        f"{meta['budget_label']}{colon}{spec.budget} {meta['token_unit']}",
        f"{meta['task_label']}{colon}{meta['task_text']}",
        "",
    ]
    for index, block in enumerate(spec.blocks, start=1):
        lines.append(block_header(index, block, loc))
        lines.append(block_content(block, loc))
    return "\n".join(lines)
