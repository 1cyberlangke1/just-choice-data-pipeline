"""界面元素定位 workflow 的题族：把一屏元素清单翻成契约形态的题目。

输入：六行元素清单 / 动作标签 / 语言字典 / 问法编号。输出：`LabeledQuestion` 列表。
预期行为：题号就是题族名（`found` / `shadow` / `target` / `role` / `decoy`）；**判据
（完全一致的匹配、可见且可点、近似文本的定义）整段写进每道题的题面里**（范式 §9.8）：
只读 state 与题面的读者必须能自己算出来。

五个题族：
- `found`（noul，**总是出**）：屏幕上有没有能落点的元素；
- `shadow`（noul，**总是出**）：屏幕上有没有名字只是近似的元素；
- `target`（choice 6，**只在有时**）：落点是第几行；
- `role`（choice 3，**只在有时**）：落点的角色属于哪一类；
- `decoy`（score 3 档，**只在有时**）：有几个同名元素被挡住了。

「出不出」写进 `families_for`，审计拿它算「应有题族」，不能只写在 `build` 里。
答案一律**只吃清单里的那四格**（不是从 case 反查），所以判别器把文本读回来重算时，
「渲染把某一格写歪」会直接变成「答案对不上」。
"""

from __future__ import annotations
from jc.assemble import choice
from jc.jsontypes import JsonValue

from collections.abc import Sequence
from typing import Final

from jc.workflow import LabeledQuestion

from . import locales, render
from .spec import (
    ROLE_CLASSES,
    SIZE,
    RowLike,
    blocked_count,
    role_class_of,
    shadow_present,
    target_slot,
)

CHOICE_FAMILIES: Final[tuple[str, ...]] = ("role", "target")
SCORE_FAMILIES: Final[tuple[str, ...]] = ("decoy",)
NOUL_FAMILIES: Final[tuple[str, ...]] = ("found", "shadow")
#: 全部合法题族（判别器用它认「未知题族」）。
FAMILIES: Final[tuple[str, ...]] = tuple(
    sorted({*CHOICE_FAMILIES, *SCORE_FAMILIES, *NOUL_FAMILIES})
)


def families_for(rows: Sequence[RowLike], label: str) -> tuple[str, ...]:
    """输入：元素清单、动作标签。输出：这一屏应该出的题族（排序后）。

    预期行为：与 `build` 的出题规则一一对应——`found` 与 `shadow` 总是出；另外三族只在
    真有落点时才出（没有落点，「第几个 / 哪一类 / 挡住几条」都无从谈起）。
    """
    out = ["found", "shadow"]
    if target_slot(rows, label) is not None:
        out += ["decoy", "role", "target"]
    return tuple(sorted(out))


def _fill(template: str, **slots: str) -> str:
    """输入：模板、槽位取值。输出：填好的文本。

    预期行为：填完还剩 `{xxx}` 直接报错——题面少填一个槽会变成一道读不懂的题，
    而答案照样发出去，这种样本比直接报错危险得多。
    """
    out = template
    for name, value in slots.items():
        out = out.replace("{" + name + "}", value)
    if "{" in out and "}" in out:
        raise ValueError(f"题面渲染后还剩占位符：{out!r}")
    return out


def _instructions(loc: dict[str, JsonValue], family: str, phrasing: int, **slots: str) -> str:
    """输入：语言字典、题族、问法编号、槽位取值。输出：题面（问句 + 判据原文）。

    预期行为：判据（`loc["policy"]["text"]`）整段拼在问句后面——题面自带判据，
    考生不需要任何外部知识。
    """
    question = _fill(locales.pick(loc["fam"][family]["q"], phrasing), **slots)
    return f"{question} {loc['policy']['text']}"


def _choice(
    family: str, instructions: str, options: dict[str, str], answer: str
) -> LabeledQuestion:
    """输入：题族、题面、选项、金答案。输出：一道 choice 题（选项有自然顺序，不打乱）。

    预期行为：校验与构造在通用层（`jc.assemble.choice`），这里只钉住本族的 `shuffle=False`。
    """
    return choice(family, instructions, options, answer, shuffle=False)


def _noul(family: str, loc: dict[str, JsonValue], phrasing: int, answer: bool) -> LabeledQuestion:
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


def _score(
    family: str, loc: dict[str, JsonValue], instructions: str, answer: int
) -> LabeledQuestion:
    """输入：题族、语言字典、题面、金答案（档位下标）。输出：一道 score 题。

    预期行为：`criteria` 是档位描述数组，下标从 0 起、与 `spec` 算出来的条数一一对应。
    """
    question = {
        "type": "score",
        "instructions": instructions,
        "criteria": [str(level) for level in loc["fam"][family]["levels"]],
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def _slot_options(rows: Sequence[RowLike], loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：元素清单、语言字典。输出：位置选项（`s1`..`s6`）。

    预期行为：描述里带上那一行的四格，并且带上位置号——位置号让六个选项天然互不相同。
    """
    template = str(loc["option"]["slot"]["element"])
    out: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        out[f"s{index}"] = _fill(
            template,
            n=str(index),
            role=str(loc["role"][row.role]),
            visibility=str(loc["visibility"][row.visibility]),
            clickable=str(loc["clickable"][row.clickable]),
            text=render.text_of(row.text, loc),
        )
    return out


def build(
    rows: Sequence[RowLike], label: str, loc: dict[str, JsonValue], phrasing: int
) -> list[LabeledQuestion]:
    """输入：元素清单 / 动作标签 / 语言字典 / 问法编号。输出：题目 + 金答案。

    预期行为：这里不掷骰子，所以同一屏清单 + 同一份语言文件必得同一批题目。
    顺序：有没有落点、有没有近似元素、落在第几个、哪一类、挡住几条。
    """
    if len(rows) != SIZE:
        raise ValueError(f"元素表应该是 {SIZE} 行，收到 {len(rows)} 行")
    slot = target_slot(rows, label)
    out: list[LabeledQuestion] = [
        _noul("found", loc, phrasing, slot is not None),
        _noul("shadow", loc, phrasing, shadow_present(rows, label)),
    ]
    if slot is None:
        return out
    out.append(
        _choice(
            "target",
            _instructions(loc, "target", phrasing),
            _slot_options(rows, loc),
            f"s{slot}",
        )
    )
    out.append(
        _choice(
            "role",
            _instructions(loc, "role", phrasing, n=str(slot)),
            {name: str(loc["option"]["role_class"][name]) for name in ROLE_CLASSES},
            role_class_of(rows[slot - 1].role),
        )
    )
    out.append(
        _score("decoy", loc, _instructions(loc, "decoy", phrasing), blocked_count(rows, label))
    )
    return out
