"""缓存复用判定的题族：把真值规格翻成契约形态的题目。

输入：`ReuseSpec`、语言字典、问法编号、RNG。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（`reuse_band` / `diverge_at` / `diff_kind` / `reuse_half`），每族一题；
- 判据（等价怎么算、可复用为什么是前缀、四档怎么划、四个位置档怎么分）**全部写在题面里**
  （范式 §9.8）：只读 state 与题面的读者必须能自己算出来；
- choice 的选项**全部打乱**（组装器做）：四个位置档在题面里的展示顺序没有语义，
  打乱能压掉位置捷径；选项描述自带含义，打乱不影响可读性。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random
from typing import Final

from jc.workflow import LabeledQuestion

from . import locales
from . import spec as spec_mod
from .spec import ReuseSpec

#: 两个 choice 题族。
CHOICE_FAMILIES: Final[tuple[str, ...]] = ("diverge_at", "diff_kind")
#: 唯一的 noul 题族。
NOUL_FAMILIES: Final[tuple[str, ...]] = ("reuse_half",)
#: 唯一的 score 题族。
SCORE_FAMILIES: Final[tuple[str, ...]] = ("reuse_band",)
#: 题族顺序：先给档位与位置，再给分叉原因，最后给那个布尔。
FAMILIES: Final[tuple[str, ...]] = (
    SCORE_FAMILIES + CHOICE_FAMILIES + NOUL_FAMILIES
)


def rules_block(loc: dict[str, JsonValue]) -> str:
    """输入：语言字典。输出：题面末尾那段判据原文。

    预期行为：整段从语言文件的 `[rules].lines` 原样取——判据必须让只读 state 与题面的
    读者自己算得出来，所以它不能藏在判别器或生成器的代码里。
    """
    return "\n".join(loc["rules"]["lines"])


def options_of(family: str, spec: ReuseSpec, loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：题族、规格、语言字典。输出：{选项名: 选项描述}。

    预期行为：`diverge_at` 四个选项固定（第 1 步 / 第 2 步 / 中间某一步 / 最后一步），
    `diff_kind` 两个选项固定（动作 / 元素）；选项名与 `spec` 算出来的答案是同一套键，
    答案不在选项里会当场报错（构造 bug，不能让它进产物）。
    """
    if family == "diverge_at":
        return dict(loc["fam"]["diverge_at"]["criteria"])
    if family == "diff_kind":
        return dict(loc["fam"]["diff_kind"]["criteria"])
    raise ValueError(f"未知题族：{family!r}（选项只有 diverge_at 与 diff_kind 两族）")


from jc.assemble import choice as _choice


def _noul(family: str, entry: dict[str, JsonValue], instructions: str, answer: bool) -> LabeledQuestion:
    """输入：题族、词条、题面、金答案。输出：一道 noul 题。

    预期行为：判据（true / false 两行文案）一并写进 `criteria`——官方支持 noul 带描述，
    带上它模型才知道「真」与「假」分别意味着什么。
    """
    question = {
        "type": "noul",
        "instructions": instructions,
        "criteria": dict(entry["criteria"]),
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def _score(family: str, loc: dict[str, JsonValue], instructions: str, answer: int) -> LabeledQuestion:
    """输入：题族、语言字典、题面、金答案（档位下标）。输出：一道 score 题。

    预期行为：`criteria` 是档位描述数组，下标从 0 起、与 `spec` 算出来的档位一一对应。
    """
    question = {
        "type": "score",
        "instructions": instructions,
        "criteria": list(loc["fam"][family]["levels"]),
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def build(
    spec: ReuseSpec, loc: dict[str, JsonValue], phrasing: int, rng: random.Random
) -> list[LabeledQuestion]:
    """输入：规格、语言字典、问法编号、RNG（**不用**）。输出：四道题。

    预期行为：**纯函数**——同一份规格 + 同一份语言文件必得同一批题目。
    """
    del rng  # 题目完全由规格与语言文件决定
    rules = rules_block(loc)

    def instructions(family: str, **fill: str) -> str:
        """输入：题族、题面占位符。输出：题面 + 判据原文。"""
        head = locales.pick(loc["fam"][family]["q"], phrasing).format(**fill)
        return f"{head}\n\n{rules}"

    out: list[LabeledQuestion] = [
        _score(
            "reuse_band",
            loc,
            instructions("reuse_band"),
            spec_mod.BANDS.index(spec.case.band),
        ),
        _choice(
            "diverge_at",
            instructions("diverge_at"),
            options_of("diverge_at", spec, loc),
            spec.diverge,
        ),
        _choice(
            "diff_kind",
            instructions("diff_kind", step=str(spec.first_diff)),
            options_of("diff_kind", spec, loc),
            spec.diff_kind,
        ),
        _noul("reuse_half", loc["fam"]["reuse_half"], instructions("reuse_half"), spec.reuse_half),
    ]
    return out
