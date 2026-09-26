"""工单分诊 workflow 的题族：把真值规格翻成契约形态的题目。

输入：`TicketSpec`、语言字典、问法编号、RNG。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（`severity` / `queue` 等），每族一题；
- 判据（严重度矩阵、派单表、升级规则、时限表、优先级表）**全部写在题面里**（范式 §9.8）：
  只读 state 与题面的读者必须能自己算出来；
- 选项**全部打乱**：这三族 choice 的选项名（`s1` / `support_l1` …）没有「必须按这个顺序读」
  的理由，打乱能压掉位置捷径；选项描述里都写着档名，打乱不影响可读性。
"""

from __future__ import annotations

import random
from typing import Any, Final

from jc.workflow import LabeledQuestion

from . import locales
from .spec import TicketSpec

CHOICE_FAMILIES: Final[tuple[str, ...]] = ("severity", "queue", "escalation")
NOUL_FAMILIES: Final[tuple[str, ...]] = (
    "critical_area",
    "needs_human",
    "sla_breach",
    "data_risk",
)
SCORE_FAMILIES: Final[tuple[str, ...]] = ("priority",)
#: 题族顺序：先定级与派单（要通读工单），再问单项事实，最后给档。
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


def _noul(family: str, loc: dict[str, Any], phrasing: int, answer: bool) -> LabeledQuestion:
    """输入：题族、语言字典、问法编号、金答案。输出：一道 noul 题。

    预期行为：判据（true / false 两行文案）一并写进 `criteria`——官方支持 noul 带描述，
    带上它模型才知道「真」与「假」分别意味着什么。
    """
    entry = loc["fam"][family]
    question = {
        "type": "noul",
        "instructions": locales.pick(entry["q"], phrasing),
        "criteria": {"true": entry["criteria"]["true"], "false": entry["criteria"]["false"]},
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
    spec: TicketSpec, loc: dict[str, Any], phrasing: int, rng: random.Random
) -> list[LabeledQuestion]:
    """输入：规格、语言字典、问法编号、RNG。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：这里不掷骰子（`rng` 是协议的一部分，选项打乱由组装器做），
    所以同一份规格 + 同一份语言文件必得同一批题目。
    """
    del rng  # 题目完全由规格与语言文件决定
    out: list[LabeledQuestion] = [
        _choice(
            "severity",
            locales.pick(loc["fam"]["severity"]["q"], phrasing),
            dict(loc["option"]["severity"]),
            spec.severity,
        ),
        _choice(
            "queue",
            locales.pick(loc["fam"]["queue"]["q"], phrasing),
            dict(loc["option"]["queue"]),
            spec.queue,
        ),
        _choice(
            "escalation",
            locales.pick(loc["fam"]["escalation"]["q"], phrasing),
            dict(loc["option"]["escalation"]),
            spec.escalation,
        ),
    ]
    out.extend(
        _noul(family, loc, phrasing, getattr(spec, family)) for family in NOUL_FAMILIES
    )
    out.append(_score("priority", loc, phrasing, spec.priority))
    return out

