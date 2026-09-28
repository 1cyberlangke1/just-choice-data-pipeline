"""工单分诊的题族：把金标签翻成契约形态的题目。

输入：`TagSet`、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（如 `rule`），同族不重复所以不需要后缀；
- choice 的选项名 = locale 里 `criteria` 的 key，与标签取值一一对应；
- `priority` 是有序阶梯（0 → 3 档），禁止打乱；四个 choice 题族都允许打乱，
  打乱能防「答案总在第 n 个」这种位置捷径。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import features, locales
from .features import TagSet

#: 题族的固定顺序（先读正文承载的两维，再判派单表命中哪条与去哪个队列，最后问表里查出来的两族）。
FAMILIES: Final[tuple[str, ...]] = (
    "issue",
    "urgency",
    "rule",
    "queue",
    "priority",
    "needs_human",
)

#: 选项有自然顺序、打乱会破坏语义的题族（本 workflow 一个都没有）。
ORDERED: Final[frozenset[str]] = frozenset()

#: 题型分组。
CHOICE_FAMILIES: Final[tuple[str, ...]] = ("issue", "urgency", "rule", "queue")
NOUL_FAMILIES: Final[tuple[str, ...]] = ("needs_human",)
SCORE_FAMILIES: Final[tuple[str, ...]] = ("priority",)


def families_for(tags: TagSet) -> tuple[str, ...]:
    """输入：一组注入特征。输出：这条样本实际会出的题族（顺序按 `FAMILIES`）。

    预期行为：本 workflow 的六个题族对任何特征组合都问得出来，所以恒等于 `FAMILIES`。
    留着这个函数是为了与其它 workflow 的形状一致（将来要加条件题族时只改这里）。
    """
    del tags
    return FAMILIES


def build(tags: TagSet, loc: dict[str, JsonValue], phrasing: int) -> list[LabeledQuestion]:
    """输入：注入特征、语言字典、问法编号。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：题号用题族名；choice 的 `answer` 是**选项名**（编号由组装器最后一步映射），
    score 的 `answer` 是等级下标，noul 的 `answer` 是布尔。
    """
    labels = tags.label()
    out: list[LabeledQuestion] = []
    for family in families_for(tags):
        spec = loc["fam"][family]
        instructions = locales.pick(spec["q"], phrasing)
        if family in SCORE_FAMILIES:
            question = {
                "type": "score",
                "instructions": instructions,
                "criteria": list(spec["levels"]),
            }
            out.append(LabeledQuestion(family, question, labels.priority, shuffle=False))
        elif family in NOUL_FAMILIES:
            question = {
                "type": "noul",
                "instructions": instructions,
                "criteria": {"true": spec["criteria"]["true"], "false": spec["criteria"]["false"]},
            }
            out.append(LabeledQuestion(family, question, labels.needs_human, shuffle=False))
        else:
            options = features.CHOICE_OPTIONS[family]
            question = {
                "type": "choice",
                "instructions": instructions,
                # 只收本题允许的选项：locale 里可能有别的取值，那是给反解提示词用的
                "criteria": {name: spec["criteria"][name] for name in options},
            }
            out.append(
                LabeledQuestion(
                    family, question, getattr(labels, family), shuffle=family not in ORDERED
                )
            )
    return out
