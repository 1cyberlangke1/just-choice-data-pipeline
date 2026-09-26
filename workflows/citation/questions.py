"""引文核查 workflow 的题族：把真值规格翻成契约形态的题目。

输入：`CitationSpec`、语言字典、问法编号、RNG。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（`verdict` 等），每族一题；
- choice 的**选项名是稳定标识**（`verified` / `cited_section` / `supports` / `auto_accept`），
  选项描述是本地化文案——判别器靠选项名反查，不靠描述文字比对；
- 判据（整句匹配的口径、四档结论的映射表、处置规则、证据强度量表）**全部写在题面里**
  （范式 §9.8）：只读 state 与题面的读者必须能自己算出来，不能要求他去读生成器代码；
- 答案一律由 `spec.answers_of` 那一套规则算出来——构造侧不另起一套判断。
"""

from __future__ import annotations

import random
from typing import Any, Final

from jc.workflow import LabeledQuestion

from . import locales
from . import spec as spec_mod
from .spec import CitationSpec

CHOICE_FAMILIES: Final[tuple[str, ...]] = ("verdict", "quote_source", "relation", "action")
NOUL_FAMILIES: Final[tuple[str, ...]] = ("found", "located")
SCORE_FAMILIES: Final[tuple[str, ...]] = ("evidence_strength",)
#: 题族顺序：先给结论（要串起整条链），再定位，再问单项事实，最后给等级。
FAMILIES: Final[tuple[str, ...]] = CHOICE_FAMILIES + NOUL_FAMILIES + SCORE_FAMILIES


def _choice(
    family: str, instructions: str, options: dict[str, str], answer: str
) -> LabeledQuestion:
    """输入：题族、题面、选项、金答案（选项名）。输出：一道 choice 题。

    预期行为：选项描述重复直接报错——那意味着两个选项说得通，是构造 bug，
    必须在生成时炸掉而不是产出一条坏数据。
    """
    if len(set(options.values())) != len(options):
        raise ValueError(f"{family} 的选项描述重复：{options}")
    if answer not in options:
        raise ValueError(f"{family} 的答案 {answer!r} 不在选项里：{sorted(options)}")
    question = {"type": "choice", "instructions": instructions, "criteria": options}
    return LabeledQuestion(family, question, answer)


def _noul(family: str, loc: dict[str, Any], phrasing: int, answer: bool) -> LabeledQuestion:
    """输入：题族、语言字典、问法编号、金答案。输出：一道 noul 题。"""
    entry = loc["ask"][family]
    question = {
        "type": "noul",
        "instructions": locales.pick(entry["q"], phrasing),
        "criteria": {"true": entry["criteria"]["true"], "false": entry["criteria"]["false"]},
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def _score(family: str, loc: dict[str, Any], phrasing: int, answer: int) -> LabeledQuestion:
    """输入：题族、语言字典、问法编号、金答案（等级下标）。输出：一道 score 题。"""
    entry = loc["ask"][family]
    question = {
        "type": "score",
        "instructions": locales.pick(entry["q"], phrasing),
        "criteria": list(entry["levels"]),
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def build(
    spec: CitationSpec, loc: dict[str, Any], phrasing: int, rng: random.Random
) -> list[LabeledQuestion]:
    """输入：规格、语言字典、问法编号、RNG。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：`del rng`——七个答案全部由 `spec.truth()` 与 `spec.answers_of` 算出来，
    这里不掷骰子；选项顺序由组装器打乱（choice 的选项名是稳定标识，打乱不影响答案）。
    """
    del rng
    ask = loc["ask"]
    found, located, relation = spec.truth()
    answers = spec_mod.answers_of(found, located, relation)
    out: list[LabeledQuestion] = [
        _choice(
            "verdict",
            locales.pick(ask["verdict"]["q"], phrasing),
            dict(loc["option"]["verdict"]),
            answers["verdict"],
        ),
        _choice(
            "quote_source",
            locales.pick(ask["quote_source"]["q"], phrasing),
            dict(loc["option"]["quote_source"]),
            answers["quote_source"],
        ),
        _choice(
            "relation",
            locales.pick(ask["relation"]["q"], phrasing),
            dict(loc["option"]["relation"]),
            answers["relation"],
        ),
        _choice(
            "action",
            locales.pick(ask["action"]["q"], phrasing),
            dict(loc["option"]["action"]),
            answers["action"],
        ),
    ]
    for family in NOUL_FAMILIES:
        out.append(_noul(family, loc, phrasing, answers[family]))
    out.append(_score("evidence_strength", loc, phrasing, answers["evidence_strength"]))
    return out
