"""命令审批 workflow 的题族：把真值规格翻成契约形态的题目。

输入：`CommandSpec`、语言字典、问法编号、RNG。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（`verdict` 等），每族一题；
- choice 的**选项名是稳定标识**（`allow` / `ask` / `deny`、`task` / `home` / …），
  选项描述是本地化文案——判别器靠选项名反查，不靠描述文字比对；
- 判据（审批策略、影响范围量表、相关定义）**全部写在题面里**（范式 §9.8）：
  只读 state 与题面的读者必须能自己算出来，不能要求他去读生成器代码。
"""

from __future__ import annotations

import random
from typing import Any, Final

from jc.workflow import LabeledQuestion

from . import locales
from .spec import CommandSpec

CHOICE_FAMILIES: Final[tuple[str, ...]] = ("verdict", "target_owner")
NOUL_FAMILIES: Final[tuple[str, ...]] = (
    "matched_goal",
    "elevated",
    "network",
    "destructive",
    "touches_secrets",
)
SCORE_FAMILIES: Final[tuple[str, ...]] = ("blast_radius",)
#: 题族顺序：先出结论（要通读策略），再问单项事实，最后给等级。
FAMILIES: Final[tuple[str, ...]] = CHOICE_FAMILIES + NOUL_FAMILIES + SCORE_FAMILIES


def _noul_answer(family: str, spec: CommandSpec) -> bool:
    """输入：noul 题族名、规格。输出：金答案。

    预期行为：每一项都从 `spec.flags`（= 从命令文本反推的事实）读，不另起一套判断。
    """
    flags = spec.flags
    table = {
        "matched_goal": spec.aligned,
        "elevated": flags.elevated,
        "network": flags.network,
        "destructive": flags.destructive,
        "touches_secrets": flags.secret,
    }
    if family not in table:
        raise KeyError(f"未知 noul 题族：{family!r}（可用：{sorted(table)}）")
    return table[family]


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


def build(
    spec: CommandSpec, loc: dict[str, Any], phrasing: int, rng: random.Random
) -> list[LabeledQuestion]:
    """输入：规格、语言字典、问法编号、RNG。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：`verdict` 是**有序阶梯**（放行 → 需确认 → 拒绝），但它照样打乱选项——
    三档的概率天生不均（真实分布就是多数命令要确认或拒绝），不打乱等于把
    「答案位置」白送给考生（范式 §5.8：有序阶梯的均衡闸同时就是捷径闸；
    这里的档位由选项**描述**承载，不靠位置承载，所以打乱才是对的做法）。
    """
    del rng  # 选项顺序由组装器打乱，这里不掷骰子
    fam = loc["fam"]
    out: list[LabeledQuestion] = [
        _choice(
            "verdict",
            locales.pick(fam["verdict"]["q"], phrasing),
            dict(loc["option"]["verdict"]),
            spec.verdict,
        ),
        _choice(
            "target_owner",
            locales.pick(fam["target_owner"]["q"], phrasing),
            dict(loc["option"]["target_owner"]),
            spec.flags.owner,
        ),
    ]
    for family in NOUL_FAMILIES:
        entry = fam[family]
        question = {
            "type": "noul",
            "instructions": locales.pick(entry["q"], phrasing),
            "criteria": {"true": entry["criteria"]["true"], "false": entry["criteria"]["false"]},
        }
        out.append(LabeledQuestion(family, question, _noul_answer(family, spec), shuffle=False))
    entry = fam["blast_radius"]
    question = {
        "type": "score",
        "instructions": locales.pick(entry["q"], phrasing),
        "criteria": list(entry["levels"]),
    }
    out.append(LabeledQuestion("blast_radius", question, spec.blast, shuffle=False))
    return out
