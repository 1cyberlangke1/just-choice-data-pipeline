"""情感与情绪（sentiment）的题族：把真值规格翻成契约形态的题目。

输入：`SentimentSpec`、语言字典、问法编号、RNG。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（`tone` / `pivot` / `drop` / `rebound`），每族一题；
- 判据（四条判读规则）**全部写在题面里**（范式 §9.8）：只读 state 与题面的读者必须能自己
  把四个答案算出来；
- 两个 choice 族的选项**允许打乱**（组装器做）：选项顺序没有语义，打乱能压掉位置捷径；
  两个 score 族的档位有自然顺序，不打乱。
"""

from __future__ import annotations

import random
from typing import Final

from jc import contract
from jc.assemble import choice as _choice
from jc.assemble import noul as _noul
from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import locales
from .spec import SentimentSpec

#: 唯一的 score 题族（相邻两回合的最大落差）。
SCORE_FAMILIES: Final[tuple[str, ...]] = ("drop",)
#: 两个 choice 题族（整段倾向 / 首次转折在哪一回合）。
CHOICE_FAMILIES: Final[tuple[str, ...]] = ("tone", "pivot")
#: 唯一的 noul 题族（有没有出现回升）。
NOUL_FAMILIES: Final[tuple[str, ...]] = ("rebound",)
#: 题族顺序：先给整段倾向，再给转折位置，再给落差，最后给回升。
FAMILIES: Final[tuple[str, ...]] = ("tone", "pivot", "drop", "rebound")

#: 首次转折那一族的选项键（`t2`~`t5` 是回合号，`none` 是一次转折都没有）。
PIVOT_KEYS: Final[tuple[str, ...]] = ("t2", "t3", "t4", "t5", "none")


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


def tone_options(loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：语言字典。输出：`tone` 那一族的选项（`pos` / `neg` / `flat`）。

    预期行为：三项描述互不相同——组装器会拒掉重名选项，那是运行期才炸，这里提前钉住。
    """
    return {str(key): str(value) for key, value in loc["fam"]["tone"]["criteria"].items()}


def pivot_options(loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：语言字典。输出：`pivot` 那一族的选项（`t2`~`t5` + `none`）。

    预期行为：键与 `PIVOT_KEYS` 完全一致——少一个键就是某一回合永远不是答案。
    """
    return {str(key): str(value) for key, value in loc["fam"]["pivot"]["criteria"].items()}


def rules_block(loc: dict[str, JsonValue]) -> str:
    """输入：语言字典。输出：题面末尾那段判据原文。

    预期行为：整段从语言文件的 `[rules].lines` 原样取（与 state 里那一段**同源**），
    行首前缀去掉——题面里不需要那根短横线。
    """
    return "\n".join(str(line) for line in loc["rules"]["lines"])


def build(
    spec: SentimentSpec, loc: dict[str, JsonValue], phrasing: int, rng: random.Random
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
        _choice("tone", instructions("tone"), tone_options(loc), spec.tone),
        _choice("pivot", instructions("pivot"), pivot_options(loc), spec.pivot_key),
        _score(
            "drop",
            instructions("drop"),
            [str(level) for level in loc["fam"]["drop"]["levels"]],
            spec.drop,
        ),
        _noul(
            "rebound",
            instructions("rebound"),
            {str(key): str(value) for key, value in loc["fam"]["rebound"]["criteria"].items()},
            spec.rebound,
        ),
    ]
