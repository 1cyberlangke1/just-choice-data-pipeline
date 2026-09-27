"""函数调用 workflow 的题族：把绑定翻成契约形态的题目。

输入：`CallPlan`（计划里的绑定，或读法从文本读回来的等价绑定）、语言字典、问法编号、RNG。
输出：`LabeledQuestion` 列表。
预期行为：题面**自带判据**——每个选项的描述写清「这个取值是什么意思」，参数题还点名是哪个
参数（`[arg_desc]`），只读 state 与题面的考生能自己答（范式 §9.8）。`rng` 是协议的一部分，
这里不掷骰子（选项打乱由组装器做）。

四个题族（题号就是族名；同族一题，所以不用 `_2` 后缀）：
- `tool`（choice N）：这句请求要调用哪个函数；
- `arg_<参数>`（choice）：被提到的 `choice` 参数取哪个值；
- `stated_<参数>`（noul）：请求里提到这个可选参数了吗（没提就用函数默认值）；
- `member_<取值>`（noul）：`set` 参数的每个成员，请求里点名要它吗。

为什么必填参数不出 `stated_*`：它恒为 True，是常量答案——教「永远答 True」的题是坏数据
（范式 §5.7）。
"""

from __future__ import annotations

import random
from typing import Any, Final

from jc.workflow import LabeledQuestion

from . import locales
from .spec import (
    KIND_FLAG,
    KIND_SET,
    TOOLS,
    Binding,
    CallPlan,
    phrase_key_of,
)


def families_of(tool: str) -> tuple[str, ...]:
    """输入：工具名。输出：它**可能**出的全部题族（排序后）。

    预期行为：审计按「产物里真的出现过的工具」点名覆盖，所以同分布侧没有 `top_movers` 时，
    它的 `arg_direction` 不会被当成缺题。
    """
    out: set[str] = {"tool"}
    for arg in TOOLS[tool]:
        if arg.kind == KIND_SET:
            out.update(f"member_{value}" for value in arg.values)
        elif arg.kind == KIND_FLAG:
            out.add(f"stated_{arg.name}")
        else:
            out.add(f"arg_{arg.name}")
            if not arg.required:
                out.add(f"stated_{arg.name}")
    return tuple(sorted(out))


def families_for(tool: str, stated: frozenset[str]) -> tuple[str, ...]:
    """输入：工具名、请求里**真的提到过**的参数名集合。输出：这种请求应该出的题族。

    预期行为：与 `build` 的出题规则一一对应（必填参数不出 `stated_*`；`set` 参数逐成员问；
    `choice` 参数只有被提到才出 `arg_*`）。审计拿它算「应有题族」——按工具点名会把
    「这一档语气没提到的参数」误报成缺题，按**实际提到的参数**点名才准。
    """
    out: set[str] = {"tool"}
    for arg in TOOLS[tool]:
        if arg.kind == KIND_SET:
            out.update(f"member_{value}" for value in arg.values)
        elif arg.kind == KIND_FLAG:
            out.add(f"stated_{arg.name}")
        else:
            if not arg.required:
                out.add(f"stated_{arg.name}")
            if arg.name in stated:
                out.add(f"arg_{arg.name}")
    return tuple(sorted(out))


#: 全部合法题族（各工具并集，排序后）。判别器用它认「未知题族」。
FAMILIES: Final[tuple[str, ...]] = tuple(
    sorted({family for tool in TOOLS for family in families_of(tool)})
)
CHOICE_FAMILIES: Final[tuple[str, ...]] = ("tool",)
NOUL_FAMILIES: Final[tuple[str, ...]] = ("stated_", "member_")


def _choice(
    family: str, instructions: str, options: dict[str, str], answer: str
) -> LabeledQuestion:
    """输入：题族、题面、选项、金答案（选项名）。输出：一道 choice 题。

    预期行为：选项描述重复直接报错——两个选项看起来一样，考生没法分辨，是构造 bug。
    """
    if len(set(options.values())) != len(options):
        raise ValueError(f"{family} 的选项描述重复：{options}")
    if answer not in options:
        raise ValueError(f"{family} 的答案 {answer!r} 不在选项里：{sorted(options)}")
    return LabeledQuestion(
        family, {"type": "choice", "instructions": instructions, "criteria": options}, answer
    )


def _noul(
    family: str, instructions: str, criteria: dict[str, str], answer: bool
) -> LabeledQuestion:
    """输入：题族、题面、判据（true / false 两行文案）、金答案。输出：一道 noul 题。"""
    return LabeledQuestion(
        family, {"type": "noul", "instructions": instructions, "criteria": dict(criteria)},
        answer, shuffle=False,
    )


def _fill(text: str, **slots: str) -> str:
    """输入：模板、占位符取值。输出：填好的文本。预期行为：未知占位符保持原样。"""
    out = text
    for key, value in slots.items():
        out = out.replace("{" + key + "}", value)
    return out


def _tool_question(tool: str, loc: dict[str, Any], phrasing: int) -> LabeledQuestion:
    """输入：金工具、语言字典、问法编号。输出：`tool` 题。

    预期行为：选项描述写清每个函数干什么（`[tool_desc]`），题面不点名是哪个函数。
    """
    options = {name: str(loc["tool_desc"][name]) for name in TOOLS}
    return _choice(
        "tool", locales.pick(loc["fam"]["tool"]["q"], phrasing), options, tool
    )


def _arg_question(binding: Binding, loc: dict[str, Any], phrasing: int) -> LabeledQuestion:
    """输入：一个 `choice` 参数的绑定、语言字典、问法编号。输出：`arg_<参数>` 题。

    预期行为：选项就是该参数的取值表（描述来自 `[opt.<短语表键>]`），答案是请求里写的那个值。
    """
    key = phrase_key_of(binding.name)
    options = {value: str(loc["opt"][key][value]) for value in _values_of(binding)}
    instructions = _fill(
        locales.pick(loc["fam"]["arg"]["q"], phrasing),
        arg=str(loc["arg_desc"][binding.name]),
    )
    return _choice(f"arg_{binding.name}", instructions, options, binding.value)


def _values_of(binding: Binding) -> tuple[str, ...]:
    """输入：一个绑定。输出：它所属参数的取值表（按工具表定义）。"""
    for args in TOOLS.values():
        for arg in args:
            if arg.name == binding.name:
                return arg.values
    raise KeyError(f"工具表里没有参数 {binding.name!r}")


def _stated_question(binding: Binding, loc: dict[str, Any], phrasing: int) -> LabeledQuestion:
    """输入：一个可选参数的绑定、语言字典、问法编号。输出：`stated_<参数>` 题。

    预期行为：判据写清「没提 = 用函数默认值」——这条是人为约定，必须写在题面里（范式 §9.8）。
    """
    entry = loc["fam"]["stated"]
    criteria = {
        "true": _fill(str(entry["criteria"]["true"]), arg=str(loc["arg_desc"][binding.name])),
        "false": _fill(str(entry["criteria"]["false"]), arg=str(loc["arg_desc"][binding.name])),
    }
    return _noul(
        f"stated_{binding.name}",
        _fill(locales.pick(entry["q"], phrasing), arg=str(loc["arg_desc"][binding.name])),
        criteria,
        binding.stated,
    )


def _member_question(
    value: str, stated: bool, loc: dict[str, Any], phrasing: int
) -> LabeledQuestion:
    """输入：一个成员、它有没有被点名、语言字典、问法编号。输出：`member_<成员>` 题。"""
    entry = loc["fam"]["member"]
    shown = str(loc["opt"]["symbol"][value])
    return _noul(
        f"member_{value}",
        _fill(locales.pick(entry["q"], phrasing), value=shown),
        {"true": str(entry["criteria"]["true"]), "false": str(entry["criteria"]["false"])},
        stated,
    )


def build(
    plan: CallPlan, loc: dict[str, Any], phrasing: int, rng: random.Random
) -> list[LabeledQuestion]:
    """输入：绑定计划、语言字典、问法编号、RNG。输出：题目 + 金答案（顺序稳定）。"""
    del rng  # 题目完全由绑定与语言文件决定
    out: list[LabeledQuestion] = [_tool_question(plan.tool, loc, phrasing)]
    for binding in plan.args:
        if binding.kind == KIND_SET:
            out.extend(
                _member_question(value, value in binding.members, loc, phrasing)
                for value in _values_of(binding)
            )
        elif binding.kind == KIND_FLAG:
            out.append(_stated_question(binding, loc, phrasing))
        else:
            if not _is_required(binding.name):
                out.append(_stated_question(binding, loc, phrasing))
            if binding.stated:
                out.append(_arg_question(binding, loc, phrasing))
    return out


def _is_required(name: str) -> bool:
    """输入：参数名。输出：它是不是必填参数（必填的不出 `stated_*` 题）。"""
    for args in TOOLS.values():
        for arg in args:
            if arg.name == name:
                return arg.required
    raise KeyError(f"工具表里没有参数 {name!r}")
