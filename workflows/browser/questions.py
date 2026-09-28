"""浏览器 / GUI 操作的题族：把金标签翻成契约形态的题目。

输入：`TagSet`、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（如 `target`），同族不重复所以不需要后缀；
- choice 的选项名 = locale 里 `criteria` 的 key，与标签取值一一对应；
- `target` 那一族的选项**随这一屏的元素表变**（元素名 + 「不操作」），所以它的选项表
  由 `target_options` 现算，不走 `features.CHOICE_OPTIONS`；
- `risk` 是有序阶梯（0 → 3 档），禁止打乱；三个 choice 题族都允许打乱，
  打乱能防「答案总在第 n 个」这种位置捷径。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import features, locales
from .features import TagSet

#: 题族的固定顺序（先读那句话承载的意图，再问该动谁 / 动什么，最后问三档门控与风险）。
FAMILIES: Final[tuple[str, ...]] = (
    "intent",
    "target",
    "action",
    "done",
    "blocked",
    "confirm",
    "risk",
)

#: 选项有自然顺序、打乱会破坏语义的题族（本 workflow 一个都没有）。
ORDERED: Final[frozenset[str]] = frozenset()

#: 题型分组。
CHOICE_FAMILIES: Final[tuple[str, ...]] = ("intent", "target", "action")
NOUL_FAMILIES: Final[tuple[str, ...]] = ("done", "blocked", "confirm")
SCORE_FAMILIES: Final[tuple[str, ...]] = ("risk",)


def families_for(tags: TagSet) -> tuple[str, ...]:
    """输入：一组注入特征。输出：这条样本实际会出的题族（顺序按 `FAMILIES`）。

    预期行为：本 workflow 的七个题族对任何特征组合都问得出来，所以恒等于 `FAMILIES`。
    留着这个函数是为了与其它 workflow 的形状一致（将来要加条件题族时只改这里）。
    """
    del tags
    return FAMILIES


def target_options(tags: TagSet, loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：注入特征、语言字典。输出：`target` 那一族的选项（这一屏的元素名 + 「不操作」）。

    预期行为：选项名就是元素名（反解判别器比的 `wanted_label` 与它们是同一份文本，
    所以两份必须同源）；描述只给**行号**——考生得去读元素表才知道那一行是什么角色、
    什么状态，光看选项猜不出答案。
    """
    template = str(loc["fam"]["target"]["row_option"])
    options = {
        row.label: template.format(index=index)
        for index, row in enumerate(features.rows_for(tags), start=1)
    }
    options[features.NONE_TARGET] = str(loc["fam"]["target"]["none"])
    return options


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
            out.append(LabeledQuestion(family, question, labels.risk, shuffle=False))
        elif family in NOUL_FAMILIES:
            question = {
                "type": "noul",
                "instructions": instructions,
                "criteria": {"true": spec["criteria"]["true"], "false": spec["criteria"]["false"]},
            }
            out.append(LabeledQuestion(family, question, getattr(labels, family), shuffle=False))
        else:
            # 只收本题允许的选项：locale 里可能有别的取值，那是给反解提示词用的
            options = (
                target_options(tags, loc)
                if family == "target"
                else {name: spec["criteria"][name] for name in features.CHOICE_OPTIONS[family]}
            )
            question = {"type": "choice", "instructions": instructions, "criteria": options}
            out.append(
                LabeledQuestion(
                    family, question, getattr(labels, family), shuffle=family not in ORDERED
                )
            )
    return out