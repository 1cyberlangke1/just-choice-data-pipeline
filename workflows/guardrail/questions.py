"""内容审核（护栏）workflow 的题族：把真值规格翻成契约形态的题目。

输入：`GuardSpec`、语言字典、问法编号、RNG。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（`violates_harass` / `severity` 等），每族一题；
- 判据（严重度九格矩阵、处置六条有序规则、并列规则、两条分界线）**全部写在题面里**
  （范式 §9.8）：只读 state 与题面的读者必须能自己算出来；
- 选项**全部打乱**：处置与条款名都没有「必须按这个顺序读」的理由，打乱能压掉位置捷径；
  选项描述里都写着档名，打乱不影响可读性。
"""

from __future__ import annotations

import random
from typing import Any, Final

from jc.workflow import LabeledQuestion

from . import locales
from . import spec as spec_mod
from .spec import GuardSpec

CHOICE_FAMILIES: Final[tuple[str, ...]] = ("action", "top_clause")
#: 六条条款各一族（题号 `violates_<条款>`）——这正是「一批互相独立的 noul」的形态。
CLAUSE_FAMILIES: Final[tuple[str, ...]] = tuple(
    f"violates_{clause}" for clause in spec_mod.CLAUSES
)
NOUL_FAMILIES: Final[tuple[str, ...]] = CLAUSE_FAMILIES + (
    "minor_exposure",
    "repeat_offender",
)
SCORE_FAMILIES: Final[tuple[str, ...]] = ("severity",)
#: 题族顺序：先给结论（处置、最重条款），再问六条条款，最后两条分界线与档位。
FAMILIES: Final[tuple[str, ...]] = CHOICE_FAMILIES + NOUL_FAMILIES + SCORE_FAMILIES


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
    family: str,
    entry: dict[str, Any],
    phrasing: int,
    answer: bool,
    clause: str | None = None,
) -> LabeledQuestion:
    """输入：题族、词条、问法编号、金答案、条款名。输出：一道 noul 题。

    预期行为：判据（true / false 两行文案）一并写进 `criteria`——官方支持 noul 带描述，
    带上它模型才知道「真」与「假」分别意味着什么；六条条款共用一份词条，
    靠 `{clause}` 把条款名填进去。
    """
    fill = {"clause": clause} if clause is not None else {}
    question = {
        "type": "noul",
        "instructions": locales.pick(entry["q"], phrasing).format(**fill),
        "criteria": {key: text.format(**fill) for key, text in entry["criteria"].items()},
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def _score(family: str, loc: dict[str, Any], phrasing: int, answer: int) -> LabeledQuestion:
    """输入：题族、语言字典、问法编号、金答案（档位下标）。输出：一道 score 题。

    预期行为：`criteria` 是档位描述数组，下标从 0 起、与 `spec` 算出来的档位一一对应。
    """
    entry = loc["fam"][family]
    question = {
        "type": "score",
        "instructions": locales.pick(entry["q"], phrasing),
        "criteria": list(entry["levels"]),
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def build(
    spec: GuardSpec, loc: dict[str, Any], phrasing: int, rng: random.Random
) -> list[LabeledQuestion]:
    """输入：规格、语言字典、问法编号、RNG。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：这里不掷骰子（`rng` 是协议的一部分，选项打乱由组装器做），
    所以同一份规格 + 同一份语言文件必得同一批题目。
    """
    del rng  # 题目完全由规格与语言文件决定
    out: list[LabeledQuestion] = [
        _choice(
            "action",
            locales.pick(loc["fam"]["action"]["q"], phrasing),
            dict(loc["option"]["action"]),
            spec.action,
        ),
        _choice(
            "top_clause",
            locales.pick(loc["fam"]["top_clause"]["q"], phrasing),
            dict(loc["option"]["top_clause"]),
            spec.top_clause,
        ),
    ]
    for clause, family in zip(spec_mod.CLAUSES, CLAUSE_FAMILIES):
        out.append(
            _noul(
                family,
                loc["fam"]["violates"],
                phrasing,
                spec.violated(clause),
                loc["clause"][clause],
            )
        )
    out.append(
        _noul("minor_exposure", loc["fam"]["minor_exposure"], phrasing, spec.minor_exposure)
    )
    out.append(
        _noul("repeat_offender", loc["fam"]["repeat_offender"], phrasing, spec.repeat_offender)
    )
    out.append(_score("severity", loc, phrasing, spec.severity))
    return out
