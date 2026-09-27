"""结构恢复 workflow 的题族：把渲染结果翻成契约形态的题目。

输入：`DocSpec`、语言字典、问法编号、RNG。输出：`LabeledQuestion` 列表。
预期行为：题面**自带上下文**（「内容块」的定义写在每一道题里）——只读 state 与题面的
读者必须能自己把答案算出来（范式 §9.8）。`rng` 是协议的一部分，这里不掷骰子。

四个题族对应官方 structure-recovery cookbook 的两遍：
- `join`（noul）：相邻两行，「下一行是不是接着上一行那句话往下写的」——第一遍「并段」；
- `block_type`（choice 6）：第 k 个内容块属于哪一类——第二遍「逐块分类」；
- `callout_kind`（choice 3）：提示框是备注 / 小技巧 / 警告（只在有提示框时出）；
- `list_ordered`（noul）：那个列表是不是步骤列表（只在有列表时出）。
"""

from __future__ import annotations

import random
from typing import Any, Final

from jc.workflow import LabeledQuestion

from . import locales, render
from .spec import BLOCK_KINDS, KIND_CALLOUT, KIND_LIST_ITEM, DocSpec

CHOICE_FAMILIES: Final[tuple[str, ...]] = ("block_type", "callout_kind")
NOUL_FAMILIES: Final[tuple[str, ...]] = ("join", "list_ordered")
#: 题族顺序：先做第一遍「并段」，再做第二遍「逐块分类」，最后是伴随题。
FAMILIES: Final[tuple[str, ...]] = ("join", "block_type", "callout_kind", "list_ordered")


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
    question = {"type": "noul", "instructions": instructions, "criteria": dict(criteria)}
    return LabeledQuestion(family, question, answer, shuffle=False)


def _fill(text: str, **kwargs: str) -> str:
    """输入：模板、占位符取值。输出：填好的文本。预期行为：未知占位符保持原样。"""
    out = text
    for key, value in kwargs.items():
        out = out.replace("{" + key + "}", value)
    return out


def build(
    spec: DocSpec, loc: dict[str, Any], phrasing: int, rng: random.Random
) -> list[LabeledQuestion]:
    """输入：规格、语言字典、问法编号、RNG。输出：题目 + 金答案（顺序按 `FAMILIES`）。"""
    del rng  # 题目完全由规格与语言文件决定
    rendered = render.layout(spec, loc)
    context = str(loc["doc"]["context"])
    out: list[LabeledQuestion] = []

    join_entry = loc["fam"]["join"]
    for left, right in rendered.asked_pairs(loc):
        instructions = f"{context} " + _fill(
            locales.pick(join_entry["q"], phrasing), a=left, b=right
        )
        criteria = {
            "true": _fill(join_entry["criteria"]["true"], a=left, b=right),
            "false": _fill(join_entry["criteria"]["false"], a=left, b=right),
        }
        out.append(
            _noul("join", instructions, criteria, rendered.same_block(left, right))
        )

    options = {
        kind: f"{loc['type'][kind]}：{loc['type_desc'][kind]}" for kind in BLOCK_KINDS
    }
    block_entry = loc["fam"]["block_type"]
    for index, block in enumerate(rendered.blocks, start=1):
        instructions = f"{context} " + _fill(
            locales.pick(block_entry["q"], phrasing), k=str(index)
        )
        out.append(_choice("block_type", instructions, dict(options), block.kind))

    callouts = [block for block in rendered.blocks if block.kind == KIND_CALLOUT]
    if callouts:
        kinds = loc["callout_kind"]
        kind_options = {kind: str(kinds[kind]) for kind in sorted(kinds)}
        out.append(
            _choice(
                "callout_kind",
                f"{context} " + locales.pick(loc["fam"]["callout_kind"]["q"], phrasing),
                kind_options,
                callouts[0].callout_kind,
            )
        )

    items = [block for block in rendered.blocks if block.kind == KIND_LIST_ITEM]
    if items:
        entry = loc["fam"]["list_ordered"]
        out.append(
            _noul(
                "list_ordered",
                f"{context} " + locales.pick(entry["q"], phrasing),
                {"true": entry["criteria"]["true"], "false": entry["criteria"]["false"]},
                items[0].list_ordered,
            )
        )
    return out
