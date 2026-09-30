"""片段筛选与去冗余（clipcut）的题族：把真值规格翻成契约形态的题目。

输入：`ClipSpec`、语言字典、问法编号、RNG。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（`keep_count` / `first_kept` / `blocked_any` / `dup_any`），每族一题；
- 判据（保留怎么算、去冗余怎么挡、什么叫主题重复）**全部写在题面里**（范式 §9.8）：
  只读 state 与题面的读者必须能自己把四个答案算出来；
- choice 的选项**允许打乱**（组装器做）：五档在题面里的展示顺序没有语义，打乱能压掉
  位置捷径；选项描述自带含义，打乱不影响可读性。
"""

from __future__ import annotations

import random
from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import locales
from .spec import ClipSpec

#: 唯一的 score 题族（保留段数）。
SCORE_FAMILIES: Final[tuple[str, ...]] = ("keep_count",)
#: 唯一的 choice 题族（最早保留的是第几段）。
CHOICE_FAMILIES: Final[tuple[str, ...]] = ("first_kept",)
#: 两个 noul 题族（有没有被挡下的 / 有没有主题重复）。
NOUL_FAMILIES: Final[tuple[str, ...]] = ("blocked_any", "dup_any")
#: 题族顺序：先给段数，再给位置，最后给两个布尔。
FAMILIES: Final[tuple[str, ...]] = SCORE_FAMILIES + CHOICE_FAMILIES + NOUL_FAMILIES


def rules_block(loc: dict[str, JsonValue]) -> str:
    """输入：语言字典。输出：题面末尾那段判据原文。

    预期行为：整段从语言文件的 `[rules].lines` 原样取——判据必须让只读 state 与题面的
    读者自己算得出来，所以它不能藏在判别器或生成器的代码里。
    """
    return "\n".join(str(line) for line in loc["rules"]["lines"])


def options_of(family: str, spec: ClipSpec, loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：题族、规格、语言字典。输出：{选项名: 选项描述}。

    预期行为：`first_kept` 五个选项固定（第 1~4 段 / 一段都没保留）；选项名与 `spec` 算出来的
    答案是同一套键，答案不在选项里会当场报错（构造 bug，不能让它进产物）。
    """
    del spec  # 选项表与这一条的具体答案无关
    if family == "first_kept":
        return {str(key): str(value) for key, value in loc["fam"]["first_kept"]["criteria"].items()}
    raise ValueError(f"未知题族：{family!r}（选项只有 first_kept 一族）")


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

    预期行为：`criteria` 是档位描述数组，下标从 0 起、与 `spec` 算出来的段数一一对应。
    """
    question = {
        "type": "score",
        "instructions": instructions,
        "criteria": [str(level) for level in loc["fam"][family]["levels"]],
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def build(
    spec: ClipSpec, loc: dict[str, JsonValue], phrasing: int, rng: random.Random
) -> list[LabeledQuestion]:
    """输入：规格、语言字典、问法编号、RNG（**不用**）。输出：四道题。

    预期行为：**纯函数**——同一份规格 + 同一份语言文件必得同一批题目。
    """
    del rng  # 题目完全由规格与语言文件决定
    rules = rules_block(loc)

    def instructions(family: str, **fill: str) -> str:
        """输入：题族、题面占位符。输出：题面 + 判据原文。"""
        head = locales.pick([str(text) for text in loc["fam"][family]["q"]], phrasing).format(**fill)
        return f"{head}\n\n{rules}"

    return [
        _score("keep_count", loc, instructions("keep_count"), spec.keep_count),
        _choice(
            "first_kept",
            instructions("first_kept"),
            options_of("first_kept", spec, loc),
            spec.first_kept,
        ),
        _noul(
            "blocked_any", loc["fam"]["blocked_any"], instructions("blocked_any"), spec.blocked_any
        ),
        _noul("dup_any", loc["fam"]["dup_any"], instructions("dup_any"), spec.dup_any),
    ]
