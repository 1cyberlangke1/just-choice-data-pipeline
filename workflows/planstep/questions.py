"""有界任务推进（planstep）的题族：把真值规格翻成契约形态的题目。

输入：`PlanSpec`、语言字典、问法编号、RNG。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（`remaining` / `next` / `rounds` / `unlocks`），每族一题；
- 判据（四条推进规则）**全部写在题面里**（范式 §9.8）：只读 state 与题面的读者必须能自己
  把四个答案算出来；
- `next` 的选项**允许打乱**（组装器做）：五项任务的展示顺序没有语义，打乱能压掉位置捷径；
  选项描述就是任务名，自带含义，打乱不影响可读性。
"""

from __future__ import annotations

import random
from typing import Final

from jc import contract
from jc.assemble import choice as _choice
from jc.assemble import noul as _noul
from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import locales, render
from .spec import PlanSpec

#: 两个 score 题族（还剩几项 / 还能推进几轮）。
SCORE_FAMILIES: Final[tuple[str, ...]] = ("remaining", "rounds")
#: 唯一的 choice 题族（下一步推进哪一项）。
CHOICE_FAMILIES: Final[tuple[str, ...]] = ("next",)
#: 唯一的 noul 题族（做完下一步会不会立刻解锁别的任务）。
NOUL_FAMILIES: Final[tuple[str, ...]] = ("unlocks",)
#: 题族顺序：先给还剩几项，再给下一步，再给还要几轮，最后给解锁。
FAMILIES: Final[tuple[str, ...]] = ("remaining", "next", "rounds", "unlocks")


def _score(family: str, instructions: str, levels: list[str], answer: int) -> LabeledQuestion:
    """输入：题族、题面、档位描述、金答案（档位下标）。输出：一道 score 题。

    预期行为：`criteria` 是档位描述数组，下标从 0 起、与 `features` 算出来的数一一对应；
    档位有自然顺序，所以不打乱。
    """
    question: dict[str, JsonValue] = {
        "type": contract.TYPE_SCORE,
        "instructions": instructions,
        "criteria": list(levels),
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def rules_block(loc: dict[str, JsonValue]) -> str:
    """输入：语言字典。输出：题面末尾那段判据原文。

    预期行为：整段从语言文件的 `[rules].lines` 原样取（与 state 里那一段**同源**），
    行首前缀去掉——题面里不需要那根短横线。
    """
    return "\n".join(str(line) for line in loc["rules"]["lines"])


def next_options(loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：语言字典。输出：`next` 那一族的选项（`t1`~`t5` + `none`）。

    预期行为：五项任务的描述**取自 `[task]` 表**（与 state 里那几行同一个来源），
    外加一个「没有可开工的项」——两项以上写法相同会当场被组装器拒掉。
    """
    options = {f"t{index + 1}": render.task_name(loc, key) for index, key in enumerate(render.TASK_KEYS)}
    options["none"] = str(loc["fam"]["next"]["opt"]["none"])
    return options


def build(
    spec: PlanSpec, loc: dict[str, JsonValue], phrasing: int, rng: random.Random
) -> list[LabeledQuestion]:
    """输入：规格、语言字典、问法编号、RNG（**不用**）。输出：四道题。

    预期行为：**纯函数**——同一份规格 + 同一份语言文件必得同一批题目。
    """
    del rng  # 题目完全由规格与语言文件决定
    rules = rules_block(loc)

    def instructions(family: str) -> str:
        """输入：题族。输出：题面 + 判据原文。"""
        head = locales.pick([str(text) for text in loc["fam"][family]["q"]], phrasing)
        return f"{head}\n\n{rules}"

    return [
        _score(
            "remaining",
            instructions("remaining"),
            [str(level) for level in loc["fam"]["remaining"]["levels"]],
            spec.remaining,
        ),
        _choice("next", instructions("next"), next_options(loc), spec.next_key),
        _score(
            "rounds",
            instructions("rounds"),
            [str(level) for level in loc["fam"]["rounds"]["levels"]],
            spec.rounds,
        ),
        _noul(
            "unlocks",
            instructions("unlocks"),
            {str(key): str(value) for key, value in loc["fam"]["unlocks"]["criteria"].items()},
            spec.unlocks,
        ),
    ]
