"""游戏与模拟控制 workflow 的题族：把一屏局面翻成契约形态的题目。

输入：六行动作、抬头三格、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题号就是题族名（`can_move` / `finished` / `gain` / `best` / `cost`）；
**判据（合法怎么比、结束怎么判、该用哪个怎么定、收益与耗各几档）整段写进每道题的题面里**
（范式 §9.8）：只读 state 与题面的读者必须能自己算出来。

五个题族：
- `can_move`（noul，逐位置 6 道，**总是出**）：第 n 个动作现在能不能用；
- `finished`（noul，**总是出**）：这一局结束了吗；
- `gain`（score 6 档，逐位置 6 道，**总是出**）：第 n 个动作的收益是第几档；
- `best`（choice 6，**只在还没结束时出**）：该用第几个动作；
- `cost`（score 4 档，**只在还没结束时出**）：该用的那个动作要耗多少。

「出不出」写进 `families_for`，审计拿它算「应有题族」，不能只写在 `build` 里。
答案一律**只吃行里的三格与抬头三格**（不是从动作名反查表），所以判别器把文本读回来重算时，
「渲染把某一格写歪」会直接变成「答案对不上」。
"""

from __future__ import annotations
from jc.assemble import choice
from jc.jsontypes import JsonValue

from collections.abc import Sequence
from typing import Final

from jc.workflow import LabeledQuestion

from . import locales
from .spec import (
    ACTIONS,
    HeadLike,
    RowLike,
    best_slot_of_rows,
    cost_level_of_rows,
    done_of_head,
    gain_level_of_row,
    legal_of_row,
)

CHOICE_FAMILIES: Final[tuple[str, ...]] = ("best",)
NOUL_FAMILIES: Final[tuple[str, ...]] = ("can_move", "finished")
SCORE_FAMILIES: Final[tuple[str, ...]] = ("cost", "gain")
#: 全部合法题族（判别器用它认「未知题族」）。
FAMILIES: Final[tuple[str, ...]] = tuple(
    sorted({*CHOICE_FAMILIES, *NOUL_FAMILIES, *SCORE_FAMILIES})
)


def families_for(rows: Sequence[RowLike], head: HeadLike) -> tuple[str, ...]:
    """输入：六行（键）、抬头。输出：这屏局面应该出的题族（排序后）。

    预期行为：与 `build` 的出题规则一一对应——`can_move` / `finished` / `gain` 总是出；
    `best` 与 `cost` 只在**还没结束、并且至少有一个能用的动作**时出（结束了就没有「这一手
    该用哪个」；一个能用的都没有时同样没有答案）。这条必须写在这里，不能只写在 `build` 里。
    """
    out = ["can_move", "finished", "gain"]
    if not done_of_head(head) and best_slot_of_rows(rows, head.resources) is not None:
        out += ["best", "cost"]
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


def _noul(
    family: str, loc: dict[str, JsonValue], phrasing: int, answer: bool, **slots: str
) -> LabeledQuestion:
    """输入：题族、语言字典、问法编号、金答案、槽位取值。输出：一道 noul 题。

    预期行为：判据（true / false 两行文案）一并写进 `criteria`——官方支持 noul 带描述，
    带上它模型才知道「真」与「假」分别意味着什么。
    """
    entry = loc["fam"][family]
    question = {
        "type": "noul",
        "instructions": _instructions(loc, family, phrasing, **slots),
        "criteria": {"true": entry["criteria"]["true"], "false": entry["criteria"]["false"]},
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def _score(
    family: str, loc: dict[str, JsonValue], phrasing: int, answer: int, **slots: str
) -> LabeledQuestion:
    """输入：题族、语言字典、问法编号、金答案（档位下标）、槽位取值。输出：一道 score 题。

    预期行为：`criteria` 是档位描述数组，下标从 0 起、与 `spec` 算出来的档位一一对应，
    所以等级数组**不能打乱**。
    """
    entry = loc["fam"][family]
    question = {
        "type": "score",
        "instructions": _instructions(loc, family, phrasing, **slots),
        "criteria": list(entry["levels"]),
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def _slot_options(rows: Sequence[RowLike], loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：六行动作、语言字典。输出：位置选项（`p1`..`p6`）。

    预期行为：描述里带上那一行的四格（动作名 · 耗 · 收益 · 冷却）并且带上位置号——
    位置号让六个选项天然互不相同。
    """
    template = str(loc["option"]["slot"]["element"])
    out: dict[str, str] = {}
    for slot, row in enumerate(rows, start=1):
        out[f"p{slot}"] = _fill(
            template,
            n=str(slot),
            action=str(loc["action"][row.action]),
            cost=str(loc["cost"][row.cost]),
            gain=str(loc["gain"][row.gain]),
            cooldown=str(loc["cooldown"][row.cooldown]),
        )
    return out


def build(
    rows: Sequence[RowLike], head: HeadLike, loc: dict[str, JsonValue], phrasing: int
) -> list[LabeledQuestion]:
    """输入：六行动作、抬头、语言字典、问法编号。输出：题目 + 金答案（顺序稳定）。

    预期行为：这里不掷骰子（选项打乱由组装器做），所以同一屏局面 + 同一份语言文件必得
    同一批题目。顺序：逐位置能不能用（6 道）、这一局结没结束、逐位置收益第几档（6 道）、
    该用第几个、该用的那个要耗多少。
    """
    if len(rows) != len(ACTIONS):
        raise ValueError(f"局面应该是 {len(ACTIONS)} 行，收到 {len(rows)} 行")
    out: list[LabeledQuestion] = [
        _noul("can_move", loc, phrasing, legal_of_row(row, head.resources), n=str(slot))
        for slot, row in enumerate(rows, start=1)
    ]
    out.append(_noul("finished", loc, phrasing, done_of_head(head)))
    out.extend(
        _score("gain", loc, phrasing, gain_level_of_row(row), n=str(slot))
        for slot, row in enumerate(rows, start=1)
    )
    best = best_slot_of_rows(rows, head.resources)
    if not done_of_head(head) and best is not None:
        out.append(
            _choice(
                "best",
                _instructions(loc, "best", phrasing),
                _slot_options(rows, loc),
                f"p{best}",
            )
        )
        out.append(_score("cost", loc, phrasing, cost_level_of_rows(rows, head.resources)))
    return out
