"""行情读数判读（quoteread）的题族：把真值规格翻成契约形态的题目。

输入：`QuoteSpec`、语言字典、问法编号、RNG。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（`tier` / `leader` / `alert` / `divergent`），每族一题；
- 判据（怎么换算、档位怎么切、警戒线在哪、并列怎么取）**全部写在题面里**（范式 §9.8）：
  只读 state 与题面的读者必须能自己把四个答案算出来；
- choice 的选项**允许打乱**（组装器做）：三个标的在表里的展示顺序没有语义，打乱能压掉
  位置捷径；选项描述自带含义，打乱不影响可读性。
"""

from __future__ import annotations

import random
from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import locales
from .spec import QuoteSpec

#: 唯一的 score 题族（波动最大的那条落在哪一档）。
SCORE_FAMILIES: Final[tuple[str, ...]] = ("tier",)
#: 唯一的 choice 题族（涨幅最大的是哪个标的）。
CHOICE_FAMILIES: Final[tuple[str, ...]] = ("leader",)
#: 两个 noul 题族（有没有越过警戒线 / 有没有方向背离）。
NOUL_FAMILIES: Final[tuple[str, ...]] = ("alert", "divergent")
#: 题族顺序：先给档位，再给标的，最后给两个布尔。
FAMILIES: Final[tuple[str, ...]] = SCORE_FAMILIES + CHOICE_FAMILIES + NOUL_FAMILIES


def rules_block(loc: dict[str, JsonValue]) -> str:
    """输入：语言字典。输出：题面末尾那段判据原文。

    预期行为：整段从语言文件的 `[rules].lines` 原样取——判据必须让只读 state 与题面的
    读者自己算得出来，所以它不能藏在判别器或生成器的代码里。
    """
    return "\n".join(str(line) for line in loc["rules"]["lines"])


def options_of(family: str, spec: QuoteSpec, loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：题族、规格、语言字典。输出：{选项名: 选项描述}。

    预期行为：`leader` 的选项就是这一条 case 的三个标的名（键 = 名键，描述 = 可读名）；
    选项名与 `spec` 算出来的答案是同一套键，答案不在选项里会当场报错（构造 bug，
    不能让它进产物）。
    """
    if family == "leader":
        table = loc["name"]
        if not isinstance(table, dict):
            raise ValueError("语言表的 [name] 必须是对象（名键 → 可读名）")
        return {name_id: str(table[name_id]) for name_id in spec.case.name_ids}
    raise ValueError(f"未知题族：{family!r}（选项只有 leader 一族）")


from jc.assemble import choice as _choice


def _noul(
    family: str, entry: dict[str, JsonValue], instructions: str, answer: bool
) -> LabeledQuestion:
    """输入：题族、词条、题面、金答案。输出：一道 noul 题。

    预期行为：判据（true / false 两行文案）一并写进 `criteria`——官方支持 noul 带描述，
    带上它模型才知道「真」与「假」分别意味着什么。
    """
    question = {
        "type": "noul",
        "instructions": instructions,
        "criteria": {str(key): str(value) for key, value in entry["criteria"].items()},
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def _score(
    family: str, loc: dict[str, JsonValue], instructions: str, answer: int
) -> LabeledQuestion:
    """输入：题族、语言字典、题面、金答案（档位下标）。输出：一道 score 题。

    预期行为：`criteria` 是档位描述数组，下标从 0 起、与 `spec` 算出来的档位一一对应。
    """
    question = {
        "type": "score",
        "instructions": instructions,
        "criteria": [str(level) for level in loc["fam"][family]["levels"]],
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def build(
    spec: QuoteSpec, loc: dict[str, JsonValue], phrasing: int, rng: random.Random
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
        _score("tier", loc, instructions("tier"), spec.tier),
        _choice(
            "leader",
            instructions("leader"),
            options_of("leader", spec, loc),
            spec.leader,
        ),
        _noul("alert", loc["fam"]["alert"], instructions("alert"), spec.alert),
        _noul(
            "divergent",
            loc["fam"]["divergent"],
            instructions("divergent"),
            spec.divergent,
        ),
    ]
