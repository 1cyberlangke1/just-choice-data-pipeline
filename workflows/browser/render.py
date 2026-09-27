"""界面操作 workflow 的渲染：把真值规格写成一屏界面。

输入：`ScreenSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
版式固定成「标题 → 窗口号 → 任务 → 空行 → 五行元素」，每行三格：
角色 · 对象 · 状态。**任务那一行与三格是判据的全部输入**，别的字都只是语境。
"""

from __future__ import annotations

from typing import Any

from .spec import Element, ScreenSpec


def element_line(index: int, element: Element, loc: dict[str, Any]) -> str:
    """输入：元素序号（从 1 起）、元素、语言字典。输出：这一行。

    预期行为：三格按固定顺序拼——角色 / 对象 / 状态，用语言文件里的分隔符连起来。
    这三格就是判据的全部输入，所以一行里必须都写出来。
    """
    cells = (
        loc["role"][element.role],
        loc["object"][element.object],
        loc["state"][element.state],
    )
    return f"[{index}] " + str(loc["meta"]["cell_sep"]).join(cells)


def task_line(spec: ScreenSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：任务那一行。

    预期行为：意图词 + 对象拼成一句话——意图词由题面里的任务意图表解释，
    对象就是元素表里要操作的那个标签，两边对得上才解得开。
    """
    meta = loc["meta"]
    phrase = str(loc["intent"][spec.task]) + str(loc["object"][spec.object])
    return f"{meta['task_label']}{meta['colon']}{phrase}"


def build(spec: ScreenSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：整屏界面文本。

    预期行为：每一格都只从**规格里已经算好的东西**取（元素三格、窗口号、任务），
    渲染器不做任何判断——判断全在 `spec` 的规则函数里。
    """
    meta = loc["meta"]
    lines = [
        str(meta["title"]),
        f"{meta['screen_label']}{meta['colon']}{spec.screen_id}",
        task_line(spec, loc),
        "",
    ]
    for index, element in enumerate(spec.rows, start=1):
        lines.append(element_line(index, element, loc))
    return "\n".join(lines)
