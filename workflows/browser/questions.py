"""界面操作 workflow 的题族：把一屏元素表与任务翻成契约形态的题目。

输入：五行元素、任务意图、对象键、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题号就是题族名（`done` / `target` / `action` / `risk`）；**判据（任务意图表 +
可用判据 + 完成判据 + 敏感清单）整段写进每道题的题面里**（范式 §9.8）：只读 state 与
题面的读者必须能自己算出来。

四个题族：
- `done`（noul，总是出）：任务是不是已经完成；
- `target`（choice 5，**只在任务未完成时出**）：下一步该操作第几个元素；
- `action`（choice 4，**只在任务未完成时出**）：这一步做什么动作；
- `risk`（noul，**只在任务未完成时出**）：这一步要不要先人工确认。

「出不出」由「任务完没完成」决定，这条写进 `families_for`，审计拿它算「应有题族」，
不能只写在 `build` 里。
"""

from __future__ import annotations

from typing import Any, Final

from jc.workflow import LabeledQuestion

from . import locales
from .spec import (
    ACTIONS,
    SLOTS,
    RowLike,
    action_of,
    is_done,
    is_sensitive,
    target_position,
)


CHOICE_FAMILIES: Final[tuple[str, ...]] = ("target", "action")
NOUL_FAMILIES: Final[tuple[str, ...]] = ("done", "risk")
#: 全部合法题族（判别器用它认「未知题族」）。
FAMILIES: Final[tuple[str, ...]] = tuple(sorted({*CHOICE_FAMILIES, *NOUL_FAMILIES}))


def families_for(rows: tuple[RowLike, ...], task: str, object_: str) -> tuple[str, ...]:
    """输入：五行元素、任务意图、对象键。输出：这一屏应该出的题族（排序后）。

    预期行为：与 `build` 的出题规则一一对应——任务已经完成时只出完成度题，定位 / 动作 /
    风险三族都不出。这条必须写在这里，不能只写在 `build` 里。
    """
    out = ["done"]
    if not is_done(rows, task, object_):
        out += ["target", "action", "risk"]
    return tuple(sorted(out))


def _fill(text: str, **slots: str) -> str:
    """输入：模板、槽位取值。输出：填好的文本。

    预期行为：填完还剩 `{xxx}` 直接报错——题面少填一个槽会变成一道读不懂的题，
    而答案照样发出去，这种样本比直接报错危险得多。
    """
    out = text
    for name, value in slots.items():
        out = out.replace("{" + name + "}", value)
    if "{" in out and "}" in out:
        raise ValueError(f"题面渲染后还剩占位符：{out!r}")
    return out


def _instructions(loc: dict[str, Any], family: str, phrasing: int) -> str:
    """输入：语言字典、题族、问法编号。输出：题面（问句 + 判据原文）。

    预期行为：判据（`loc["policy"]["text"]`）整段拼在问句后面——题面自带判据，
    考生不需要任何外部知识。
    """
    question = locales.pick(loc["fam"][family]["q"], phrasing)
    return f"{question} {loc['policy']['text']}"


def _choice(
    family: str, instructions: str, options: dict[str, str], answer: str
) -> LabeledQuestion:
    """输入：题族、题面、选项、金答案（选项名）。输出：一道 choice 题。

    预期行为：选项描述重复直接报错——两个选项看起来一样，考生没法分辨，是构造 bug，
    必须在生成时炸掉而不是产出一条坏数据。
    """
    if len(set(options.values())) != len(options):
        raise ValueError(f"{family} 的选项描述重复：{options}")
    if answer not in options:
        raise ValueError(f"{family} 的答案 {answer!r} 不在选项里：{sorted(options)}")
    question = {"type": "choice", "instructions": instructions, "criteria": options}
    return LabeledQuestion(family, question, answer)


def _noul(family: str, loc: dict[str, Any], phrasing: int, answer: bool) -> LabeledQuestion:
    """输入：题族、语言字典、问法编号、金答案。输出：一道 noul 题。

    预期行为：判据（true / false 两行文案）一并写进 `criteria`——官方支持 noul 带描述，
    带上它模型才知道「真」与「假」分别意味着什么。
    """
    entry = loc["fam"][family]
    question = {
        "type": "noul",
        "instructions": _instructions(loc, family, phrasing),
        "criteria": {"true": entry["criteria"]["true"], "false": entry["criteria"]["false"]},
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def _position_options(rows: tuple[RowLike, ...], loc: dict[str, Any]) -> dict[str, str]:
    """输入：五行元素、语言字典。输出：位置选项（`p1`..`p5`）。

    预期行为：描述里带上那一行的三格（角色 · 对象 · 状态）——选项之间因此天然互不相同，
    考生也不必回头数序号。
    """
    template = str(loc["option"]["position"]["element"])
    out: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        out[f"p{index}"] = _fill(
            template,
            n=str(index),
            role=str(loc["role"][row.role]),
            label=str(loc["object"][row.object]),
            state=str(loc["state"][row.state]),
        )
    return out


def build(
    rows: tuple[RowLike, ...], task: str, object_: str, loc: dict[str, Any], phrasing: int
) -> list[LabeledQuestion]:
    """输入：五行元素、任务意图、对象键、语言字典、问法编号。输出：题目 + 金答案（顺序稳定）。

    预期行为：这里不掷骰子（选项打乱由组装器做），所以同一屏界面 + 同一份语言文件必得
    同一批题目。顺序：完成度题、定位题、动作题、风险题。
    """
    if len(rows) != len(SLOTS):
        raise ValueError(f"元素表应该是 {len(SLOTS)} 行，收到 {len(rows)} 行")
    done = is_done(rows, task, object_)
    position = target_position(rows, task, object_)
    if done != (position is None):
        raise ValueError("完成判据与可用判据打架：判成完成时不该还有可操作的目标")
    out: list[LabeledQuestion] = [_noul("done", loc, phrasing, done)]
    if position is None:
        return out
    out.append(
        _choice(
            "target",
            _instructions(loc, "target", phrasing),
            _position_options(rows, loc),
            f"p{position}",
        )
    )
    out.append(
        _choice(
            "action",
            _instructions(loc, "action", phrasing),
            {name: str(loc["option"]["action"][name]) for name in ACTIONS},
            action_of(task),
        )
    )
    out.append(_noul("risk", loc, phrasing, is_sensitive(object_)))
    return out
