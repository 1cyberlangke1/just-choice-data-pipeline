"""命令审批 workflow 的渲染：把真值规格写成一条「待审批的工具调用请求」。

输入：`CommandSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
命令本体单独占一行、缩进两格：反解要能一刀切出「命令从哪开始」，缩进就是那个界碑。
"""

from __future__ import annotations

from typing import Any

from .spec import CommandSpec


def goal_text(spec: CommandSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：用户目标那句话（点名了哪个对象由 case 决定）。

    预期行为：句式按「操作 × 写法」取，`{target}` 填的是**用户目标点名的那个对象**
    在该方言下的写法——它与命令操作的对象相同 = 一致，不同 = 不一致
    （题族 `matched_goal` 的答案）。
    """
    entry = loc["goal"].get(spec.case.op)
    if entry is None:
        raise KeyError(f"语言文件里没有操作 {spec.case.op!r} 的目标句式")
    tpl = entry.get(spec.case.goal_style)
    if tpl is None:
        raise KeyError(
            f"操作 {spec.case.op!r} 缺目标句式 {spec.case.goal_style!r}（可用：{sorted(entry)}）"
        )
    return tpl.format(target=spec.goal_target.path_in(spec.case.dialect))


def env_text(spec: CommandSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：执行环境那一行的值（如 `bash（Linux）`）。

    预期行为：值以方言名开头（`bash` / `powershell`），反解按首个词元认方言，
    所以不需要给每种语言再写一张反查表。
    """
    entry = loc["env"].get(spec.case.dialect)
    if entry is None:
        raise KeyError(f"语言文件里没有方言 {spec.case.dialect!r} 的执行环境文案")
    return entry["text"]


def build(spec: CommandSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：整段请求文本。

    预期行为：抬头 → 工作目录 → 执行环境 → 用户目标 → 已读文件 → 待执行命令（单独一行）。
    标点取自语言文件（英文请求不该出现「：」）。
    """
    meta = loc["meta"]
    colon = meta["colon"]
    lines = [
        meta["title"],
        f"{meta['cwd_label']}{colon}{spec.cwd}",
        f"{meta['env_label']}{colon}{env_text(spec, loc)}",
        f"{meta['goal_label']}{colon}{goal_text(spec, loc)}",
        f"{meta['reads_label']}{colon}{meta['list_sep'].join(spec.reads)}",
        f"{meta['cmd_label']}{colon}",
        f"  {spec.command}",
    ]
    return "\n".join(lines)
