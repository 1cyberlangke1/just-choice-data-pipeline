"""单据 workflow 的题族：把真值规格翻成契约形态的题目。

输入：`InvoiceSpec`、语言字典、问法编号、RNG。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（`date_field` 等），每族最多一题；
- choice 的**选项名是槽名**（`due` / `total` / `line_2`），选项描述是那个槽的**渲染文本**；
  干扰项因此天然是同一份单据里的同类值（别的日期、别的金额），不是编出来的；
- 目标是**抽**出来的（问「到期日」还是「开票日期」由 RNG 决定），所以同一题族的答案语义
  每份单据都不同，也逼着模型真的去读字段而不是背位置；
- 同一题里选项描述必须两两不同——否则那道题有两个正确答案（构造时已保证，这里再查一次）。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random
from typing import Final

from jc.workflow import LabeledQuestion

from . import locales, render
from .spec import ID_SLOTS, PARTY_SLOTS, InvoiceSpec
from . import spec as spec_mod

CHOICE_FAMILIES: Final[tuple[str, ...]] = (
    "date_field",
    "money_field",
    "party_field",
    "line_field",
    "id_field",
)
NOUL_FAMILIES: Final[tuple[str, ...]] = ("subtotal_consistent", "tax_consistent")
SCORE_FAMILIES: Final[tuple[str, ...]] = ("bad_lines",)
#: 题族顺序：先抽字段（读文本），再核对（算一致性），最后数错（打分）。
FAMILIES: Final[tuple[str, ...]] = CHOICE_FAMILIES + NOUL_FAMILIES + SCORE_FAMILIES

#: `bad_lines` 的档位表是 0/1/2/3+，所以只在**至少这么多行**的单据上问：2 行的单据
#: 取不到第 4 档（2 行不可能错 3 处），问了就是把那一档压到 0.18（实测 0.0774 的偏差，
#: 容差 0.03）——属于「档位取不到」那一类失衡，与 §9.79 第六条同源。
MIN_LINES_FOR_BAD_LINES: Final[int] = 3


def families_for(spec: InvoiceSpec) -> tuple[str, ...]:
    """输入：规格。输出：这份单据会出的题族。

    预期行为：两个条件题族——
    - `tax_consistent`：单据里没有税额行就不问（判据要的字段根本不存在）；
    - `bad_lines`：明细少于 `MIN_LINES_FOR_BAD_LINES` 行就不问（档位表的第 4 档取不到）。
    """
    dropped: set[str] = set()
    if not spec.has_tax:
        dropped.add("tax_consistent")
    if len(spec.items) < MIN_LINES_FOR_BAD_LINES:
        dropped.add("bad_lines")
    return tuple(family for family in FAMILIES if family not in dropped)


def line_option(index: int) -> str:
    """输入：明细行下标（从 0 开始）。输出：该行的选项名（`line_1` 起）。"""
    return f"line_{index + 1}"


def _slot_options(spec: InvoiceSpec, loc: dict[str, JsonValue], slots: tuple[str, ...]) -> dict[str, str]:
    """输入：规格、语言字典、槽名序列。输出：{槽名: 该槽的渲染文本}。"""
    return {slot: render.field_text(spec, slot, loc) for slot in slots}


def _line_options(spec: InvoiceSpec, loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：规格、语言字典。输出：{line_k: 第 k 行单价文本}（第 k 行从 1 数起）。"""
    return {
        line_option(index): render.format_money(item.unit, spec.case.currency, loc)
        for index, item in enumerate(spec.items)
    }


from jc.assemble import choice as _choice


def build(
    spec: InvoiceSpec, loc: dict[str, JsonValue], phrasing: int, rng: random.Random
) -> list[LabeledQuestion]:
    """输入：规格、语言字典、问法编号、RNG。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：choice 的答案一律是**槽名**（组装器最后一步才映射成编号）；
    noul 的答案是布尔；score 的答案是等级下标（「有几行对不上」封顶 3）。
    """
    out: list[LabeledQuestion] = []
    for family in families_for(spec):
        if family == "date_field":
            slots = spec.date_slots
            target = rng.choice(slots)
            out.append(
                _choice(
                    family,
                    locales.pick(loc["ask"][target]["q"], phrasing),
                    _slot_options(spec, loc, slots),
                    target,
                )
            )
        elif family == "money_field":
            slots = spec.money_slots
            target = rng.choice(slots)
            out.append(
                _choice(
                    family,
                    locales.pick(loc["ask"][target]["q"], phrasing),
                    _slot_options(spec, loc, slots),
                    target,
                )
            )
        elif family == "party_field":
            target = rng.choice(PARTY_SLOTS)
            out.append(
                _choice(
                    family,
                    locales.pick(loc["ask"][target]["q"], phrasing),
                    _slot_options(spec, loc, PARTY_SLOTS),
                    target,
                )
            )
        elif family == "id_field":
            target = rng.choice(ID_SLOTS)
            out.append(
                _choice(
                    family,
                    locales.pick(loc["ask"][target]["q"], phrasing),
                    _slot_options(spec, loc, ID_SLOTS),
                    target,
                )
            )
        elif family == "line_field":
            index = rng.randrange(len(spec.items))
            instructions = locales.pick(loc["ask"]["line_unit"]["q"], phrasing).format(n=index + 1)
            out.append(
                _choice(
                    family, instructions, _line_options(spec, loc), line_option(index)
                )
            )
        elif family in NOUL_FAMILIES:
            answer = (
                spec.subtotal_consistent
                if family == "subtotal_consistent"
                else spec.tax_consistent
            )
            question = {
                "type": "noul",
                "instructions": locales.pick(loc["fam"][family]["q"], phrasing),
                "criteria": {
                    "true": loc["fam"][family]["criteria"]["true"],
                    "false": loc["fam"][family]["criteria"]["false"],
                },
            }
            out.append(LabeledQuestion(family, question, answer, shuffle=False))
        else:
            question = {
                "type": "score",
                "instructions": locales.pick(loc["fam"][family]["q"], phrasing),
                "criteria": list(loc["fam"][family]["levels"]),
            }
            out.append(LabeledQuestion(family, question, spec.bad_lines, shuffle=False))
    return out


def locate(
    instructions: str, loc: dict[str, JsonValue], line_count: int
) -> tuple[str, str] | None:
    """输入：题面、语言字典、明细行数。输出：(题族, 目标槽)，或 None（认不出来）。

    预期行为：把题面与语言文件里的问法逐条对。判别器靠它**从题面反查**「问的是哪个字段」，
    从而只依赖渲染出来的文本与题面本身，不依赖 spec（否则就是拿答案验证答案）。
    `line_field` 的题面带行号，所以要按行数展开模板再比。
    """
    pools: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("date_field", spec_mod.DATE_SLOTS),
        ("money_field", spec_mod.MONEY_SLOTS),
        ("party_field", PARTY_SLOTS),
        ("id_field", ID_SLOTS),
    )
    for family, slots in pools:
        for slot in slots:
            if instructions in loc["ask"][slot]["q"]:
                return (family, slot)
    for index in range(line_count):
        if instructions in [text.format(n=index + 1) for text in loc["ask"]["line_unit"]["q"]]:
            return ("line_field", line_option(index))
    for family in NOUL_FAMILIES + SCORE_FAMILIES:
        if instructions in loc["fam"][family]["q"]:
            return (family, "")
    return None
