"""行级规则检查的题族：把行计划翻成契约形态的题目。

输入：行计划、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题面**自带判据**——规则名与判据印在 state 的抬头里，题面只需要指明问的是哪条
规则，只读 state 与题面的考生能自己答（范式 §9.8）。四个题族：

- `line`（noul，逐行，共 8 道）：第 k 行命中这条规则吗；
- `count`（choice，9 选项）：这段片段里有几行命中；
- `gate`（choice，3 选项）：按处置办法走哪一档（放行 / 人工复核 / 拦下）；
- `spot`（choice，8 选项，**只在恰好一行命中时出**）：是哪一行。

`spot` 只在「恰好一行命中」时出，是为了让答案唯一——多行命中时「哪一行命中」没有唯一解，
出这种题就是坏数据。它同时是一个**两阶段依赖**：出不出这一族由命中行数决定。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import features, locales
from .features import LintPlan, label_for

#: 题族的固定顺序（先逐行判，再问命中数，再问处置，最后问定位）。
FAMILIES: Final[tuple[str, ...]] = ("line", "count", "gate", "spot")

#: 选项有自然顺序、打乱会破坏语义的题族。本 workflow 一个都没有：`count` 的 9 档虽然天然
#: 有序，但打乱它能防「答案总在第 n 个」这种位置捷径，而选项名（`c0`…`c8`）本身不受影响。
ORDERED: Final[frozenset[str]] = frozenset()

NOUL_FAMILIES: Final[tuple[str, ...]] = ("line",)
CHOICE_FAMILIES: Final[tuple[str, ...]] = ("count", "gate", "spot")

#: `spot` 一族缺席时，其余题族照样出（用来算「这条样本该有哪些题」）。
ALWAYS_FAMILIES: Final[tuple[str, ...]] = tuple(name for name in FAMILIES if name != "spot")


def families_for(plan: LintPlan) -> tuple[str, ...]:
    """输入：行计划。输出：这条样本实际会出的题族（顺序按 `FAMILIES`）。

    预期行为：`spot` 只在「恰好一行命中」时才出——零行或多行命中时这道题没有唯一解，
    硬出就是一条错数据（`features.label_for` 给 None 就是这种情况）。
    """
    if label_for(plan).spot is None:
        return ALWAYS_FAMILIES
    return FAMILIES


from jc.assemble import choice as _choice


def _line_questions(
    spec: dict[str, JsonValue], labels: features.Labels, phrasing: int
) -> list[LabeledQuestion]:
    """输入：`line` 一族的文案、金标签、问法编号。输出：八道 noul 题。

    预期行为：题号由组装器唯一化（`line`、`line_2`…`line_8`），行号与片段行序一一对应。
    """
    template = locales.pick(spec["q"], phrasing)
    out: list[LabeledQuestion] = []
    for row in range(features.SIZE):
        question = {
            "type": "noul",
            "instructions": template.format(n=row + 1),
            "criteria": {
                "true": str(spec["criteria"]["true"]).format(n=row + 1),
                "false": str(spec["criteria"]["false"]).format(n=row + 1),
            },
        }
        out.append(LabeledQuestion("line", question, labels.matched[row], shuffle=False))
    return out


def build(plan: LintPlan, loc: dict[str, JsonValue], phrasing: int) -> list[LabeledQuestion]:
    """输入：行计划、语言字典、问法编号。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：choice 的 `answer` 是**选项名**（编号由组装器最后一步映射）；noul 的 `answer`
    是布尔。`spot` 的八个选项**全都出**（只出答案那一个会让选项集泄露答案）。
    """
    labels = label_for(plan)
    out: list[LabeledQuestion] = []
    for family in families_for(plan):
        spec = loc["fam"][family]
        instructions = locales.pick(spec["q"], phrasing)
        if family in NOUL_FAMILIES:
            out.extend(_line_questions(spec, labels, phrasing))
            continue
        if family == "count":
            options = {
                f"c{value}": str(spec["opt"][f"c{value}"])
                for value in range(features.SIZE + 1)
            }
            out.append(_choice(family, instructions, options, f"c{labels.hits}"))
            continue
        if family == "gate":
            options = {name: str(spec["opt"][name]) for name in features.GATES}
            out.append(_choice(family, instructions, options, labels.gate))
            continue
        options = {
            f"row{row + 1}": str(spec["opt"]["row"]).format(n=row + 1)
            for row in range(features.SIZE)
        }
        out.append(_choice(family, instructions, options, f"row{labels.spot}"))
    return out
