"""评测与基准 workflow 的题族：把一屏评测记录翻成契约形态的题目。

输入：五行（键）、抬头（门槛）、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题号就是题族名（`hit` / `count` / `pass` / `best` / `gap`）；
**判据（答对怎么比、答对几道几档、达标怎么判、谁最好、落后怎么算）整段写进每道题的题面里**
（范式 §9.8）：只读 state 与题面的读者必须能自己算出来。

五个题族：
- `hit`（noul，逐题 5 道，**总是出**）：第 n 题，第 1 个候选答对了吗；
- `count`（score 6 档，**总是出**）：第 1 个候选答对了几道；
- `pass`（noul，**总是出**）：第 1 个候选达标了吗；
- `best`（choice 3，**总是出**）：第几个候选答对最多；
- `gap`（score 2 档，**只在第 1 个候选不是最好时出**）：最好的那个比第 1 个候选多答对几道。

「出不出」写进 `families_for`，审计拿它算「应有题族」，不能只写在 `build` 里。
答案一律**只吃行里的四格与抬头那一格**（不是从成绩表名反查），所以判别器把文本读回来重算时，
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
    TOTAL,
    HeadLike,
    RowLike,
    best_column_of_rows,
    count_of_column,
    correct_of_cell,
    gap_level_of_rows,
    pass_of_column,
)

CHOICE_FAMILIES: Final[tuple[str, ...]] = ("best",)
NOUL_FAMILIES: Final[tuple[str, ...]] = ("hit", "pass")
SCORE_FAMILIES: Final[tuple[str, ...]] = ("count", "gap")
#: 全部合法题族（判别器用它认「未知题族」）。
FAMILIES: Final[tuple[str, ...]] = tuple(sorted({*CHOICE_FAMILIES, *NOUL_FAMILIES, *SCORE_FAMILIES}))
#: 「第 1 个候选」是第几列——`hit` / `count` / `pass` / `gap` 四族都问它。
FIRST: Final[int] = 1


def families_for(rows: Sequence[RowLike], head: HeadLike) -> tuple[str, ...]:
    """输入：五行（键）、抬头。输出：这屏记录**该出**哪些题族（顺序稳定）。

    预期行为：与 `build` 的出题规则一一对应——`hit` / `count` / `pass` / `best` 总是出；
    `gap` 只在**第 1 个候选不是最好**时出（第 1 列已经领先时没有「落后」可言）。这条必须
    写在这里，不能只写在 `build` 里。
    """
    del head
    out = ["hit", "count", "pass", "best"]
    if gap_level_of_rows(rows) is not None:
        out.append("gap")
    return tuple(out)


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


def _cand_options(rows: Sequence[RowLike], loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：五行（键）、语言字典。输出：候选选项（`c1`..`c3`）。

    预期行为：描述里带上列号与候选名——列号让三个选项天然互不相同，名字让考生看得出
    选项说的是哪一列（列序随排布变，名字才是稳的那一半）。列序取自第一行；解析时已经验过
    各行列序一致。
    """
    template = str(loc["option"]["cand"]["element"])
    return {
        f"c{column}": _fill(
            template, n=str(column), name=str(loc["cand"][cell.cand])
        )
        for column, cell in enumerate(rows[0].cells, start=1)
    }


def build(
    rows: Sequence[RowLike], head: HeadLike, loc: dict[str, JsonValue], phrasing: int
) -> list[LabeledQuestion]:
    """输入：五行（键）、抬头、语言字典、问法编号。输出：题目 + 金答案（顺序稳定）。

    预期行为：这里不掷骰子（选项打乱由组装器做），所以同一屏记录 + 同一份语言文件必得
    同一批题目。顺序：逐题答没答对（5 道）、答对几道、达标了吗、第几个候选最好、落后几道。
    """
    if len(rows) != TOTAL:
        raise ValueError(f"记录应该是 {TOTAL} 行，收到 {len(rows)} 行")
    out: list[LabeledQuestion] = [
        _noul("hit", loc, phrasing, correct_of_cell(row, FIRST), n=str(index))
        for index, row in enumerate(rows, start=1)
    ]
    out.append(_score("count", loc, phrasing, count_of_column(rows, FIRST)))
    out.append(_noul("pass", loc, phrasing, pass_of_column(rows, FIRST, head.bar)))
    out.append(
        _choice(
            "best",
            _instructions(loc, "best", phrasing),
            _cand_options(rows, loc),
            f"c{best_column_of_rows(rows)}",
        )
    )
    gap = gap_level_of_rows(rows)
    if gap is not None:
        out.append(_score("gap", loc, phrasing, gap))
    return out
