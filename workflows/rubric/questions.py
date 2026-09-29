"""打分与评分的题族：把金标签翻成契约形态的题目。

输入：`TagSet`、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（`met` 逐条问，所以带序号后缀 `met_1`…`met_5`，`family_of` 会剥掉数字）；
- `met` 五道 noul 问的是**评分单那五行**逐条达没达标（观测 ≥ 要求），顺序与评分单一致；
- `band` 是有序阶梯（低 → 中 → 高），禁止打乱；`gate` / `top` 两个 choice 题族允许打乱，
  打乱能防「答案总在第 n 个」这种位置捷径；
- `top` 只在**唯一解**时才出（没达标的判据里权重最大的那条并列、或者全达标，都不出）。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import features, locales
from .features import TagSet

#: 题族的固定顺序（先逐条判达标，再问总分档位，最后问处置与最严重的一条）。
FAMILIES: Final[tuple[str, ...]] = ("met", "band", "gate", "top")

#: 选项有自然顺序、打乱会破坏语义的题族（`band` 是低→中→高的阶梯）。
ORDERED: Final[frozenset[str]] = frozenset({"band"})

#: 题型分组。
NOUL_FAMILIES: Final[tuple[str, ...]] = ("met",)
SCORE_FAMILIES: Final[tuple[str, ...]] = ("band",)
CHOICE_FAMILIES: Final[tuple[str, ...]] = ("gate", "top")

#: `top` 一族缺席时，其余题族照样出（用来算「这条样本该有哪些题」）。
ALWAYS_FAMILIES: Final[tuple[str, ...]] = tuple(
    name for name in FAMILIES if name not in {"top"}
)


def families_for(tags: TagSet) -> tuple[str, ...]:
    """输入：一组注入特征。输出：这条样本实际会出的题族（顺序按 `FAMILIES`）。

    预期行为：`top` 只在「没达标的判据里权重最大的那条唯一」时才出——并列或全达标时
    这道题没有唯一解，硬出就是一条错数据（`features.top_for` 给 None 就是这种情况）。
    """
    if features.top_for(tags) is None:
        return ALWAYS_FAMILIES
    return FAMILIES


def build(tags: TagSet, loc: dict[str, JsonValue], phrasing: int) -> list[LabeledQuestion]:
    """输入：注入特征、语言字典、问法编号。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：choice 的 `answer` 是**选项名**（编号由组装器最后一步映射），
    score 的 `answer` 是等级下标，noul 的 `answer` 是布尔。
    """
    labels = tags.label()
    out: list[LabeledQuestion] = []
    for family in families_for(tags):
        spec = loc["fam"][family]
        if family in NOUL_FAMILIES:
            out.extend(_met_questions(spec, loc, labels, phrasing))
            continue
        instructions = locales.pick(spec["q"], phrasing)
        if family in SCORE_FAMILIES:
            question = {
                "type": "score",
                "instructions": instructions,
                "criteria": list(spec["levels"]),
            }
            out.append(
                LabeledQuestion(
                    family, question, features.BANDS.index(labels.band), shuffle=family not in ORDERED
                )
            )
            continue
        options = features.CHOICE_OPTIONS[family]
        question = {
            "type": "choice",
            "instructions": instructions,
            # 只收本题允许的选项：locale 里可能有别的取值，那是给反解提示词用的
            "criteria": {name: spec["criteria"][name] for name in options},
        }
        answer = labels.gate if family == "gate" else labels.top
        out.append(
            LabeledQuestion(family, question, answer, shuffle=family not in ORDERED)
        )
    return out


def _met_questions(
    spec: dict[str, JsonValue],
    loc: dict[str, JsonValue],
    labels: features.Labels,
    phrasing: int,
) -> list[LabeledQuestion]:
    """输入：`met` 一族的文案、语言字典、金标签、问法编号。输出：五道 noul 题。

    预期行为：题号 `met_1`…`met_5` 与评分单的行序一一对应（`family_of` 会剥掉数字，
    所以审计按题族名统计时五道题都归到 `met`）。
    """
    out: list[LabeledQuestion] = []
    for index, name in enumerate(features.CRITERIA, start=1):
        template = locales.pick(spec["q"], phrasing)
        question = {
            "type": "noul",
            "instructions": template.format(n=index, criterion=loc["criterion"][name]),
            "criteria": {
                "true": spec["criteria"]["true"],
                "false": spec["criteria"]["false"],
            },
        }
        out.append(LabeledQuestion(f"met_{index}", question, labels.met_at(index), shuffle=False))
    return out
