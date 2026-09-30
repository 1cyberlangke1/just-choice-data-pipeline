"""多批次排名聚合（siftrank）的题族：把真值规格翻成契约形态的题目。

输入：`RankSpec`、语言字典、问法编号、RNG。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（`agg_top` / `pair_order` / `cut_size` / `stable`），每族一题；
- 判据（怎么计分、并列怎么算、肘部在哪、什么算「一致」）**全部写在题面里**（范式 §9.8）：
  只读 state 与题面的读者必须能自己算出来；
- choice 的选项**全部打乱**（组装器做）：四个候选条目在题面里的展示顺序没有语义，
  打乱能压掉位置捷径。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random
from typing import Final

from jc.workflow import LabeledQuestion

from . import locales
from . import spec as spec_mod
from .spec import RankSpec

#: 两个 choice 题族。
CHOICE_FAMILIES: Final[tuple[str, ...]] = ("agg_top", "pair_order")
#: 唯一的 score 题族。
SCORE_FAMILIES: Final[tuple[str, ...]] = ("cut_size",)
#: 唯一的 noul 题族。
NOUL_FAMILIES: Final[tuple[str, ...]] = ("stable",)
#: 题族顺序：先给聚合后的头名，再给一对条目的比较，再给相关条目数，最后给那个布尔。
FAMILIES: Final[tuple[str, ...]] = CHOICE_FAMILIES + SCORE_FAMILIES + NOUL_FAMILIES

#: `agg_top` 给几个选项。选项取自总分最高的那几个条目，所以四个都「看起来像」答案。
TOP_OPTIONS: Final[int] = 4


def rules_block(loc: dict[str, JsonValue]) -> str:
    """输入：语言字典。输出：题面末尾那段判据原文。

    预期行为：整段从语言文件的 `[rules].lines` 原样取——判据必须让只读 state 与题面的
    读者自己算得出来，所以它不能藏在判别器或生成器的代码里。
    """
    return "\n".join(loc["rules"]["lines"])


def top_options(spec: RankSpec, loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：规格、语言字典。输出：{条目键: 条目名}（总分最高的 `TOP_OPTIONS` 个条目）。

    预期行为：选项按**展示顺序**取——它们是「最有可能排第一」的那几个，四个都像答案；
    正确答案一定在其中（展示顺序的第一个就是它）。
    """
    ordered = spec_mod.order_of(spec.totals)
    return {key: str(loc["item"][key]) for key in ordered[:TOP_OPTIONS]}


def pair_options(spec: RankSpec, loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：规格、语言字典。输出：{选项键: 选项描述}（`a_first` / `b_first` / `tie`）。

    预期行为：每条描述里**只点名自己那一个条目**（另一条写「对方」）——判别器靠「每个选项
    描述里恰好一个条目名」把被比较的两个条目认出来，两边都写全就会认不出。
    """
    left, right = spec.pair
    names = {"a": str(loc["item"][left]), "b": str(loc["item"][right])}
    table = loc["fam"]["pair_order"]["criteria"]
    return {str(key): str(text).format(**names) for key, text in table.items()}


from jc.assemble import choice as _choice


def _noul(
    family: str, entry: dict[str, JsonValue], instructions: str, answer: bool
) -> LabeledQuestion:
    """输入：题族、词条、题面、金答案。输出：一道 noul 题。

    预期行为：判据（true / false 两行文案）一并写进 `criteria`——官方支持 noul 带描述，
    带上它模型才知道「是」与「否」分别意味着什么。
    """
    question = {
        "type": "noul",
        "instructions": instructions,
        "criteria": dict(entry["criteria"]),
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def _score(
    family: str, loc: dict[str, JsonValue], instructions: str, answer: int
) -> LabeledQuestion:
    """输入：题族、语言字典、题面、金答案（等级下标）。输出：一道 score 题。

    预期行为：`criteria` 是等级描述数组，下标从 0 起、与 `spec.CUTS` 一一对应。
    """
    question = {
        "type": "score",
        "instructions": instructions,
        "criteria": list(loc["fam"][family]["levels"]),
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def build(
    spec: RankSpec, loc: dict[str, JsonValue], phrasing: int, rng: random.Random
) -> list[LabeledQuestion]:
    """输入：规格、语言字典、问法编号、RNG（**不用**）。输出：四道题。

    预期行为：**纯函数**——同一份规格 + 同一份语言文件必得同一批题目。
    """
    del rng  # 题目完全由规格与语言文件决定
    rules = rules_block(loc)
    left, right = spec.pair

    def instructions(family: str, **fill: str) -> str:
        """输入：题族、题面占位符。输出：题面 + 判据原文。"""
        head = locales.pick(loc["fam"][family]["q"], phrasing).format(**fill)
        return f"{head}\n\n{rules}"

    return [
        _choice("agg_top", instructions("agg_top"), top_options(spec, loc), spec.top),
        _choice(
            "pair_order",
            instructions("pair_order", a=str(loc["item"][left]), b=str(loc["item"][right])),
            pair_options(spec, loc),
            spec.relation,
        ),
        _score("cut_size", loc, instructions("cut_size"), spec.cut - 1),
        _noul("stable", loc["fam"]["stable"], instructions("stable"), spec.stable),
    ]
