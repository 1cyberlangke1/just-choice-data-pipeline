"""结构恢复的题族：把渲染结果翻成契约形态的题目。

输入：`DocSpec`（块计划 + 回填的文字）、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题面**自带上下文**（「内容块」的定义写在每一道题里）——只读 state 与题面的读者必须能
自己把答案算出来（范式 §9.8）。

四个题族对应官方 structure-recovery 那类用例的两遍：

- `join`（noul）：相邻两行，「下一行是不是接着上一行那句话往下写的」——第一遍「并段」；
- `block_type`（choice 6）：第 k 个内容块属于哪一类——第二遍「逐块分类」；
- `callout_kind`（choice 3）：提示框是备注 / 小技巧 / 警告（只在有提示框时出）；
- `list_ordered`（noul）：那个列表是不是步骤列表（只在有列表时出）。

**答案来自渲染出来的块**（构造侧那条路径），判别器则从 state 文本**重读**一遍（`parse` 那条路径）
——两条独立实现互相印证。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import locales, parse, render
from .features import BLOCK_KINDS, KIND_CALLOUT, KIND_LIST_ITEM
from .spec import DocSpec

#: 题族的固定顺序：先做第一遍「并段」，再做第二遍「逐块分类」，最后是伴随题。
FAMILIES: Final[tuple[str, ...]] = ("join", "block_type", "callout_kind", "list_ordered")


from jc.assemble import choice as _choice


from jc.assemble import noul as _noul


def _fill(text: str, **kwargs: str) -> str:
    """输入：模板、占位符取值。输出：填好的文本。预期行为：未知占位符保持原样。"""
    out = text
    for key, value in kwargs.items():
        out = out.replace("{" + key + "}", value)
    return out


def block_options(loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：语言字典。输出：块类型那一族的选项表（键是块类型，值是「名字：一句话解释」）。"""
    template = str(loc["fam"]["block_type"]["opt"])
    return {
        kind: _fill(
            template,
            type=str(loc["type"][kind]),
            desc=str(loc["type_desc"][kind]),
        )
        for kind in BLOCK_KINDS
    }


def callout_options(loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：语言字典。输出：提示框种类那一族的选项表（键是种类）。"""
    table = loc["callout_kind"]
    return {kind: str(table[kind]) for kind in sorted(table)}


def build(spec: DocSpec, loc: dict[str, JsonValue], phrasing: int) -> list[LabeledQuestion]:
    """输入：规格、语言字典、问法编号。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：先把回填的文字排成纯文本（`render.layout`），再按它出题；`texts` 是空的说明调用方
    漏了 `render`，直接报错而不是猜。
    """
    if not spec.texts:
        raise ValueError("出题前必须先渲染（spec.texts 是空的）")
    rendered = render.layout(spec.plan, spec.texts, loc)
    context = str(loc["doc"]["context"])
    out: list[LabeledQuestion] = []

    join_entry = loc["fam"]["join"]
    criteria = join_entry["criteria"]
    for left, right in rendered.asked_pairs(loc):
        instructions = f"{context} " + _fill(
            locales.pick(join_entry["q"], phrasing), a=left, b=right
        )
        out.append(
            _noul(
                "join",
                instructions,
                {
                    "true": _fill(str(criteria["true"]), a=left, b=right),
                    "false": _fill(str(criteria["false"]), a=left, b=right),
                },
                rendered.same_block(left, right),
            )
        )

    options = block_options(loc)
    block_entry = loc["fam"]["block_type"]
    for index, block in enumerate(rendered.blocks, start=1):
        instructions = f"{context} " + _fill(
            locales.pick(block_entry["q"], phrasing), k=str(index)
        )
        out.append(_choice("block_type", instructions, dict(options), block.kind))

    callouts = [block for block in rendered.blocks if block.kind == KIND_CALLOUT]
    if callouts:
        out.append(
            _choice(
                "callout_kind",
                f"{context} " + locales.pick(loc["fam"]["callout_kind"]["q"], phrasing),
                callout_options(loc),
                callouts[0].callout_kind,
            )
        )

    items = [block for block in rendered.blocks if block.kind == KIND_LIST_ITEM]
    if items:
        entry = loc["fam"]["list_ordered"]
        # 判据必须点名**顺序词**，不能只问「有没有先后顺序」：有序列表的标签就是按「每一项都带顺序词」
        # 算的，而「Set up your workstation / Meet the team」这种并列项读起来也像有先后顺序
        # （真机上读者答 True、标签是 False，两边吵起来）。问法写死判据，读者才能机械地算。
        steps = str(loc["doc"]["list_sep"]).join(parse.seq_words(loc))
        out.append(
            _noul(
                "list_ordered",
                f"{context} " + _fill(locales.pick(entry["q"], phrasing), steps=steps),
                {
                    "true": _fill(str(entry["criteria"]["true"]), steps=steps),
                    "false": _fill(str(entry["criteria"]["false"]), steps=steps),
                },
                items[0].list_ordered,
            )
        )
    return out
