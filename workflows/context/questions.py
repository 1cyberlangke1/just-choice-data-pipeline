"""会话上下文压缩 workflow 的题族：把五个块与预算翻成契约形态的题目。

输入：五个块、预算、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题号就是题族名（`action` / `compact` / `spot` / `top`），同族多题由组装器加
`_2`、`_3` 后缀；**判据（处置表 + 尺寸表 + 开销表 + 预算比较）整段写进每道题的题面里**
（范式 §9.8）：只读 state 与题面的读者必须能自己算出来。

四个题族：
- `action`（choice 4，逐块）：第 k 块该怎么处置；
- `compact`（noul）：按处置表算完，这次会话还要不要再压缩；
- `spot`（choice 5，**只在恰好一块被丢时出**）：被丢弃的那一块是第几块；
- `top`（choice 5，**只在占用最多的一块唯一时出**）：处置后占用最多的是第几块。

`spot` / `top` 是**两阶段依赖**：出不出由「有没有唯一解」决定。「出不出」这件事写进
`families_for`，审计拿它算「应有题族」，不能只写在 `build` 里。
"""

from __future__ import annotations

from typing import Any, Final

from jc.workflow import LabeledQuestion

from . import locales
from .spec import ACTIONS, BlockCase, compact_of, drop_position, top_position, total_of

CHOICE_FAMILIES: Final[tuple[str, ...]] = ("action", "spot", "top")
NOUL_FAMILIES: Final[tuple[str, ...]] = ("compact",)
#: 全部合法题族（判别器用它认「未知题族」）。
FAMILIES: Final[tuple[str, ...]] = tuple(sorted({*CHOICE_FAMILIES, *NOUL_FAMILIES}))


def families_for(blocks: tuple[BlockCase, ...]) -> tuple[str, ...]:
    """输入：会话里的五个块。输出：这段会话应该出的题族（排序后）。

    预期行为：与 `build` 的出题规则一一对应——被丢的块不止一块时不出 `spot`，占用最多的
    块并列时不出 `top`。这两条必须写在这里，不能只写在 `build` 里。
    """
    out = ["action", "compact"]
    if drop_position(blocks) is not None:
        out.append("spot")
    if top_position(blocks) is not None:
        out.append("top")
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


def _instructions(loc: dict[str, Any], family: str, phrasing: int, **slots: str) -> str:
    """输入：语言字典、题族、问法编号、槽位。输出：题面（问句 + 判据原文）。

    预期行为：判据（`loc["policy"]["text"]`）整段拼在问句后面——题面自带判据，
    考生不需要任何外部知识。
    """
    question = _fill(locales.pick(loc["fam"][family]["q"], phrasing), **slots)
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


def _noul(
    family: str, loc: dict[str, Any], phrasing: int, answer: bool
) -> LabeledQuestion:
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


def _position_options(blocks: tuple[BlockCase, ...], loc: dict[str, Any]) -> dict[str, str]:
    """输入：五个块、语言字典。输出：位置选项（`p1`..`p5`）。

    预期行为：描述里带上那一块的抬头（类型 · 大小 · 年龄 · 主题）——选项之间因此天然
    互不相同，考生也不必回头数序号。
    """
    template = str(loc["option"]["position"]["block"])
    out: dict[str, str] = {}
    for index, block in enumerate(blocks, start=1):
        out[f"p{index}"] = _fill(
            template,
            n=str(index),
            kind=str(loc["kind"][block.kind]),
            size=str(loc["size"][block.size]),
            age=str(loc["age"][block.age]),
            topic=str(loc["topic"][block.topic]),
        )
    return out


def build(
    blocks: tuple[BlockCase, ...],
    budget: int,
    loc: dict[str, Any],
    phrasing: int,
) -> list[LabeledQuestion]:
    """输入：五个块、预算、语言字典、问法编号。输出：题目 + 金答案（顺序稳定）。

    预期行为：这里不掷骰子（选项打乱由组装器做），所以同一批块 + 同一份语言文件必得
    同一批题目。顺序：逐块处置题（按位置）、门控题、定位题、极值题。
    """
    if not blocks:
        raise ValueError("会话里一个块都没有")
    out: list[LabeledQuestion] = [
        _choice(
            "action",
            _instructions(loc, "action", phrasing, n=str(index)),
            {name: str(loc["option"]["action"][name]) for name in ACTIONS},
            block.action(),
        )
        for index, block in enumerate(blocks, start=1)
    ]
    out.append(_noul("compact", loc, phrasing, compact_of(total_of(blocks), budget)))
    spot = drop_position(blocks)
    if spot is not None:
        out.append(
            _choice(
                "spot",
                _instructions(loc, "spot", phrasing),
                _position_options(blocks, loc),
                f"p{spot}",
            )
        )
    top = top_position(blocks)
    if top is not None:
        out.append(
            _choice(
                "top",
                _instructions(loc, "top", phrasing),
                _position_options(blocks, loc),
                f"p{top}",
            )
        )
    return out
