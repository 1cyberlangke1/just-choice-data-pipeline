"""有界任务推进（planstep）的 state 渲染：任务表与四条规则。

**四条规则写进 state，不写进判别器**：规则是判据原文，题面必须自带（范式 §9.8），而 state
只描述「这张计划长什么样」——读者拿 state + 题面就能自己把四个答案算出来。

输入/输出：纯函数，只有字符串拼装；不碰 RNG、不碰后端、不碰文件。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue

from .spec import PlanSpec

#: 任务名的键（渲染与反解共用同一份清单；`[task]` 表里必须不多不少）。
TASK_KEYS: Final[tuple[str, ...]] = ("T1", "T2", "T3", "T4", "T5")


def task_name(loc: dict[str, JsonValue], key: str) -> str:
    """输入：语言字典、任务键（`T1`~`T5`）。输出：这一项的可读名。

    预期行为：键不在表里直接报错——悄悄退回键名会把「任务名没翻译」这件事藏进产物里。
    """
    table = loc["task"]
    if not isinstance(table, dict) or key not in table:
        raise ValueError(f"任务 {key!r} 不在语言表的 [task] 里")
    return str(table[key])


def status_text(loc: dict[str, JsonValue], done: bool) -> str:
    """输入：语言字典、这一项做没做完。输出：状态词的原文。"""
    table = loc["status"]
    if not isinstance(table, dict):
        raise ValueError("语言表的 [status] 必须是对象（done / pending 两个词）")
    return str(table["done" if done else "pending"])


def deps_text(loc: dict[str, JsonValue], deps: tuple[int, ...]) -> str:
    """输入：语言字典、前置项下标（升序）。输出：前置那一格的原文。

    预期行为：没有前置时**只写那个词**（`无前置` / `none`），有前置时写成
    `前置=2,3` 这样的「前缀 + 编号」，编号按渲染成 1 起——反解照这两种形状各认一遍。
    """
    meta = loc["meta"]
    if not deps:
        return str(meta["deps_none"])
    numbers = str(meta["deps_sep"]).join(str(index + 1) for index in deps)
    return f"{meta['deps_prefix']}{numbers}"


def task_line(
    loc: dict[str, JsonValue], number: int, done: bool, deps: tuple[int, ...]
) -> str:
    """输入：语言字典、任务号（1 起）、状态、前置项下标。输出：任务表里的一行。

    预期行为：三格按「任务名 | 状态 | 前置」固定顺序拼；编号后的点号与格分隔符都是**结构**
    （中英共用），缩进两格只是给人看的——反解按 `strip()` 后的内容认编号。
    """
    cells = [task_name(loc, TASK_KEYS[number - 1]), status_text(loc, done), deps_text(loc, deps)]
    sep = str(loc["meta"]["cell_sep"])
    return f"  {number}. {(' ' + sep + ' ').join(cells)}"


def rules_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：推进规则那一段的每一行（含行首前缀）。

    预期行为：整段从语言文件的 `[rules].lines` 原样取——判据必须让只读 state 与题面的读者
    自己算得出来，所以它不能藏在生成器的代码里。
    """
    prefix = str(loc["meta"]["rules_prefix"])
    return [f"{prefix}{line}" for line in loc["rules"]["lines"]]


def build(spec: PlanSpec, loc: dict[str, JsonValue]) -> str:
    """输入：规格、语言字典。输出：`state` 全文。

    预期行为：区块抬头用方括号包住（与其它 workflow 一致）；任务表按编号 1 起连续、
    每一项一行；四条规则附在最后。
    """
    meta = loc["meta"]
    case = spec.case
    deps = case.deps
    done = case.done_set
    lines = [f"{meta['request']} {spec.query_id}", f"[{meta['tasks']}]"]
    lines.extend(
        task_line(loc, number, number - 1 in done, deps[number - 1])
        for number in range(1, len(deps) + 1)
    )
    lines.append(f"[{meta['rules']}]")
    lines.extend(rules_lines(loc))
    return "\n".join(lines) + "\n"
