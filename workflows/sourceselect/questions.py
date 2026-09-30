"""来源选择的题族：把候选表与金标签翻成契约形态的题目。

输入：`SourcePlan`、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（`need` / `pick` / `enough` / `cover` / `near_miss`），同族不重复所以
  不需要后缀；
- choice 的选项名 = locale 里 `criteria` 的 key；`pick` 的选项随表长变化（`s1`…`sN` +
  `none`），由 `pick_options` 现场拼——选项名不受打乱影响，所以答案永远映射得回去；
- `cover` 是有序阶梯（0 → 2 档），禁止打乱；其余 choice 题族都允许打乱，
  打乱能防「答案总在第 n 个」这种位置捷径。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import features, locales
from .features import SourceCase, SourcePlan

#: 题族的固定顺序（先读文本承载的那一维，再问选哪一行，再问三个直读的事实）。
FAMILIES: Final[tuple[str, ...]] = ("need", "pick", "enough", "cover", "near_miss")

#: 选项有自然顺序、打乱会破坏语义的题族（本 workflow 一个都没有）。
ORDERED: Final[frozenset[str]] = frozenset()

#: 题型分组。
CHOICE_FAMILIES: Final[tuple[str, ...]] = ("need", "pick")
NOUL_FAMILIES: Final[tuple[str, ...]] = ("enough", "near_miss")
SCORE_FAMILIES: Final[tuple[str, ...]] = ("cover",)

#: `cover` 的档数（0 / 1 / 2）。
COVER_LEVELS: Final[int] = 3


def pick_options(size: int, loc: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """输入：表长、语言字典。输出：`pick` 那一族的选项（每个行号一个 + 一个弃权）。

    预期行为：选项名是 `s<行号>`（1 起）与 `none`，描述取自 locale 的行号模板。
    """
    template = str(loc["meta"]["row_ref"])
    options: dict[str, JsonValue] = {
        f"s{index}": template.format(n=index) for index in range(1, size + 1)
    }
    options[features.NO_PICK] = str(loc["fam"]["pick"]["opt"]["none"])
    return options


def families_for(tags: SourceCase) -> tuple[str, ...]:
    """输入：一组注入特征。输出：这条样本实际会出的题族（顺序按 `FAMILIES`）。

    预期行为：本 workflow 的五个题族对任何特征组合都问得出来，所以恒等于 `FAMILIES`。
    留着这个函数是为了与其它 workflow 的形状一致（将来要加条件题族时只改这里）。
    """
    del tags
    return FAMILIES


def build(plan: SourcePlan, loc: dict[str, JsonValue], phrasing: int) -> list[LabeledQuestion]:
    """输入：候选表计划、语言字典、问法编号。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：题号用题族名；choice 的 `answer` 是**选项名**（编号由组装器最后一步映射），
    score 的 `answer` 是等级下标，noul 的 `answer` 是布尔。
    """
    labels = features.label_for(plan)
    out: list[LabeledQuestion] = []
    for family in FAMILIES:
        spec = loc["fam"][family]
        instructions = locales.pick(spec["q"], phrasing)
        if family in SCORE_FAMILIES:
            question: dict[str, JsonValue] = {
                "type": "score",
                "instructions": instructions,
                "criteria": list(spec["levels"]),
            }
            out.append(LabeledQuestion(family, question, labels.cover, shuffle=False))
        elif family in NOUL_FAMILIES:
            question = {
                "type": "noul",
                "instructions": instructions,
                "criteria": {
                    "true": spec["criteria"]["true"],
                    "false": spec["criteria"]["false"],
                },
            }
            answer = labels.enough if family == "enough" else labels.near_miss
            out.append(LabeledQuestion(family, question, answer, shuffle=False))
        else:
            if family == "need":
                options: dict[str, JsonValue] = {
                    name: spec["criteria"][name] for name in features.NEEDS
                }
                answer = labels.need
            else:
                options = pick_options(plan.size, loc)
                answer = labels.pick_key
            question = {
                "type": "choice",
                "instructions": instructions,
                "criteria": options,
            }
            out.append(LabeledQuestion(family, question, answer, shuffle=family not in ORDERED))
    return out
