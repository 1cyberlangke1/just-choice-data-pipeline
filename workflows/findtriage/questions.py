"""改动引入性判定的题族：把改动记录与金标签翻成契约形态的题目。

输入：`RecordPlan`、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（`cite` / `introduced` / `supported` / `actionable` / `triage` /
  `decoy`），同族不重复所以不需要后缀；
- choice 的选项名 = locale 里 `criteria` 的 key；`cite` 的选项随表长变化（`s1`…`sN` +
  `none`），由 `cite_options` 现场拼——选项名不受打乱影响，所以答案永远映射得回去；
- 本 workflow 没有有序阶梯题族，所以所有 choice 题族都允许打乱，
  打乱能防「答案总在第 n 个」这种位置捷径。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import features, locales
from .features import RecordCase, RecordPlan

#: 题族的固定顺序（先问指向哪一行，再问三轴，最后问处置与诱饵）。
FAMILIES: Final[tuple[str, ...]] = (
    "cite",
    "introduced",
    "supported",
    "actionable",
    "triage",
    "decoy",
)

#: 选项有自然顺序、打乱会破坏语义的题族（本 workflow 一个都没有）。
ORDERED: Final[frozenset[str]] = frozenset()

#: 题型分组。
CHOICE_FAMILIES: Final[tuple[str, ...]] = ("cite", "introduced", "supported", "triage")
NOUL_FAMILIES: Final[tuple[str, ...]] = ("actionable", "decoy")


def cite_options(size: int, loc: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """输入：表长、语言字典。输出：`cite` 那一族的选项（每个序号一个 + 一个「记录里没有」）。

    预期行为：选项名是 `s<序号>`（1 起）与 `none`，描述取自 locale 的序号模板。
    """
    template = str(loc["meta"]["row_ref"])
    options: dict[str, JsonValue] = {
        f"s{index}": template.format(n=index) for index in range(1, size + 1)
    }
    options[features.NO_CITE] = str(loc["fam"]["cite"]["opt"]["none"])
    return options


def families_for(tags: RecordCase) -> tuple[str, ...]:
    """输入：一组注入特征。输出：这条样本实际会出的题族（顺序按 `FAMILIES`）。

    预期行为：本 workflow 的六个题族对任何特征组合都问得出来，所以恒等于 `FAMILIES`。
    留着这个函数是为了与其它 workflow 的形状一致（将来要加条件题族时只改这里）。
    """
    del tags
    return FAMILIES


def build(plan: RecordPlan, loc: dict[str, JsonValue], phrasing: int) -> list[LabeledQuestion]:
    """输入：改动记录计划、语言字典、问法编号。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：题号用题族名；choice 的 `answer` 是**选项名**（编号由组装器最后一步映射），
    noul 的 `answer` 是布尔。
    """
    labels = features.label_for(plan)
    out: list[LabeledQuestion] = []
    for family in FAMILIES:
        spec = loc["fam"][family]
        instructions = locales.pick(spec["q"], phrasing)
        if family in NOUL_FAMILIES:
            question: dict[str, JsonValue] = {
                "type": "noul",
                "instructions": instructions,
                "criteria": {
                    "true": spec["criteria"]["true"],
                    "false": spec["criteria"]["false"],
                },
            }
            answer = labels.actionable if family == "actionable" else labels.decoy
            out.append(LabeledQuestion(family, question, answer, shuffle=False))
            continue
        if family == "cite":
            options = cite_options(plan.size, loc)
            answer = labels.cite_key
        else:
            names = features.CHOICE_OPTIONS[family]
            options = {name: spec["criteria"][name] for name in names}
            answer = getattr(labels, family)
        question = {"type": "choice", "instructions": instructions, "criteria": options}
        out.append(LabeledQuestion(family, question, answer, shuffle=family not in ORDERED))
    return out
