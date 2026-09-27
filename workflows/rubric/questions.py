"""打分与评分 workflow 的题族：把一张评分单翻成契约形态的题目。

输入：六行判据、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题号就是题族名（`met` / `band` / `gate` / `top`）；**判据（达标怎么比、权重表、
门槛表、档位门槛、红线、处置表、最严重怎么定）整段写进每道题的题面里**（范式 §9.8）：
只读 state 与题面的读者必须能自己算出来。

四个题族：
- `met`（noul，逐条 6 道，**总是出**）：这一条判据达标了吗；
- `band`（score 3 档，**总是出**）：总分落在低 / 中 / 高哪一档；
- `gate`（choice 3，**总是出**）：这份评分单该怎么处置；
- `top`（choice 6，**只在最严重的那一条唯一时出**）：最严重的是第几条判据。

「出不出」写进 `families_for`，审计拿它算「应有题族」，不能只写在 `build` 里。
答案一律**只吃评分单里那四格**（不是从判据键反查表），所以判别器把文本读回来重算时，
「渲染把某一格写歪」会直接变成「答案对不上」。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

from jc.workflow import LabeledQuestion

from . import locales
from .spec import (
    BANDS,
    CRITERIA,
    GATES,
    REDLINE,
    RowLike,
    band_of,
    gate_of,
    is_redline,
    met_criteria,
    met_of,
    top_slot,
    total_of,
)

CHOICE_FAMILIES: Final[tuple[str, ...]] = ("gate", "top")
NOUL_FAMILIES: Final[tuple[str, ...]] = ("met",)
SCORE_FAMILIES: Final[tuple[str, ...]] = ("band",)
#: 全部合法题族（判别器用它认「未知题族」）。
FAMILIES: Final[tuple[str, ...]] = tuple(
    sorted({*CHOICE_FAMILIES, *NOUL_FAMILIES, *SCORE_FAMILIES})
)


def families_for(rows: Sequence[RowLike]) -> tuple[str, ...]:
    """输入：六行判据。输出：这张评分单应该出的题族（排序后）。

    预期行为：与 `build` 的出题规则一一对应——`met` / `band` / `gate` 总是出；`top` 只在
    最严重的那一条**唯一**时出（并列时「最严重的是哪一条」没有唯一解，出这种题就是坏数据，
    所以它是两阶段依赖）。这条必须写在这里，不能只写在 `build` 里。
    """
    out = ["band", "gate", "met"]
    if top_slot(rows) is not None:
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
    """输入：语言字典、题族、问法编号、槽位取值。输出：题面（问句 + 判据原文）。

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

    `shuffle=False`：两个 choice 题族的选项描述都自带**自然顺序**（位置族按「第 n 条」
    编号、处置族是「通过 / 返修 / 退回」这条处置阶梯），按契约「有自然顺序的集合不许打乱」。
    附带的好处是答案编号就等于选项名的分布（构造出来就是均匀的），不再带一层打乱的随机抖动。
    """
    if len(set(options.values())) != len(options):
        raise ValueError(f"{family} 的选项描述重复：{options}")
    if answer not in options:
        raise ValueError(f"{family} 的答案 {answer!r} 不在选项里：{sorted(options)}")
    question = {"type": "choice", "instructions": instructions, "criteria": options}
    return LabeledQuestion(family, question, answer, shuffle=False)


def _noul(
    family: str, loc: dict[str, Any], phrasing: int, answer: bool, position: int
) -> LabeledQuestion:
    """输入：题族、语言字典、问法编号、金答案、条号。输出：一道 noul 题。

    预期行为：判据（true / false 两行文案）一并写进 `criteria`——官方支持 noul 带描述，
    带上它模型才知道「真」与「假」分别意味着什么。
    """
    entry = loc["fam"][family]
    question = {
        "type": "noul",
        "instructions": _instructions(loc, family, phrasing, n=str(position)),
        "criteria": {"true": entry["criteria"]["true"], "false": entry["criteria"]["false"]},
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def _score(family: str, loc: dict[str, Any], phrasing: int, answer: int) -> LabeledQuestion:
    """输入：题族、语言字典、问法编号、金答案（档位下标）。输出：一道 score 题。

    预期行为：`criteria` 是档位描述数组，下标从 0 起、与 `spec` 算出来的档位一一对应
    （低 = 0、中 = 1、高 = 2），所以等级数组**不能打乱**。
    """
    entry = loc["fam"][family]
    question = {
        "type": "score",
        "instructions": _instructions(loc, family, phrasing),
        "criteria": list(entry["levels"]),
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def _slot_options(rows: Sequence[RowLike], loc: dict[str, Any]) -> dict[str, str]:
    """输入：六行判据、语言字典。输出：条号选项（`p1`..`p6`）。

    预期行为：描述里带上那一行的四格（判据 · 权重 · 要求 · 观测），并且带上条号——
    条号让六个选项天然互不相同。
    """
    template = str(loc["option"]["slot"]["element"])
    out: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        out[f"p{index}"] = _fill(
            template,
            n=str(index),
            criterion=str(loc["criterion"][row.criterion]),
            weight=str(loc["weight"][row.weight]),
            need=str(loc["need"][row.need]),
            count=str(loc["count"][row.count]),
        )
    return out


def build(rows: Sequence[RowLike], loc: dict[str, Any], phrasing: int) -> list[LabeledQuestion]:
    """输入：六行判据、语言字典、问法编号。输出：题目 + 金答案（顺序稳定）。

    预期行为：这里不掷骰子（选项打乱由组装器做），所以同一张评分单 + 同一份语言文件必得
    同一批题目。顺序：逐条达没达标（6 道）、总分哪一档、该怎么处置、最严重的是哪一条。
    """
    if len(rows) != len(CRITERIA):
        raise ValueError(f"评分单应该是 {len(CRITERIA)} 行，收到 {len(rows)} 行")
    out: list[LabeledQuestion] = [
        _noul("met", loc, phrasing, met_of(row.need, row.count), index)
        for index, row in enumerate(rows, start=1)
    ]
    mets = met_criteria(rows)
    band = band_of(total_of(mets))
    out.append(_score("band", loc, phrasing, BANDS.index(band)))
    out.append(
        _choice(
            "gate",
            _instructions(loc, "gate", phrasing),
            {name: str(loc["option"]["gate"][name]) for name in GATES},
            gate_of(band, is_redline(REDLINE, REDLINE in mets)),
        )
    )
    top = top_slot(rows)
    if top is not None:
        out.append(
            _choice(
                "top",
                _instructions(loc, "top", phrasing),
                _slot_options(rows, loc),
                f"p{top}",
            )
        )
    return out

