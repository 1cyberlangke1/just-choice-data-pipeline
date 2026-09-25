"""邮件 workflow 的题族：把金标签翻成契约形态的题目。

输入：`TagSet`、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（如 `requested`），同族不重复所以不需要后缀；
- choice 的选项名 = locale 里 `criteria` 的 key，与标签取值一一对应；
- `action` 是**有序阶梯**（deliver → review → quarantine → block），禁止打乱；
  其余 choice 允许打乱，打乱能防「答案总在第 n 个」这种位置捷径；
- **没有链接就不出 `link_text` 这道题**：否则「邮件里没有链接」会占掉 1/3 的选项份额，
  把一道二元判断题拖成 11/44/44 的偏斜分布，而那道题本来也没什么好问的。
"""

from __future__ import annotations

from typing import Any, Final

from jc.workflow import LabeledQuestion

from . import features, locales
from .features import TagSet

#: 题族的固定顺序（先读文本，再下判断，最后给处置）。
FAMILIES: Final[tuple[str, ...]] = (
    "requested",
    "pressure",
    "link_text",
    "sender_identity",
    "is_phishing",
    "is_spam",
    "severity",
    "action",
)

#: 选项有自然顺序、打乱会破坏语义的题族。
ORDERED: Final[frozenset[str]] = frozenset({"action"})

#: 题型分组。
CHOICE_FAMILIES: Final[tuple[str, ...]] = (
    "requested",
    "pressure",
    "link_text",
    "sender_identity",
    "action",
)
NOUL_FAMILIES: Final[tuple[str, ...]] = ("is_phishing", "is_spam")
SCORE_FAMILIES: Final[tuple[str, ...]] = ("severity",)


def families_for(tags: TagSet) -> tuple[str, ...]:
    """输入：一组注入特征。输出：这封邮件实际会出的题族（顺序按 `FAMILIES`）。

    预期行为：只有 `link_text` 是条件题族——没有链接就不问链接文字。
    """
    if tags.link == features.LINK_NONE:
        return tuple(family for family in FAMILIES if family != "link_text")
    return FAMILIES


def build(tags: TagSet, loc: dict[str, Any], phrasing: int) -> list[LabeledQuestion]:
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
            out.append(LabeledQuestion(family, question, labels.severity, shuffle=False))
        elif family in NOUL_FAMILIES:
            # noul 的 criteria 只有 true / false 两个 key，值是给模型看的选项描述
            question = {
                "type": "noul",
                "instructions": instructions,
                "criteria": {"true": spec["criteria"]["true"], "false": spec["criteria"]["false"]},
            }
            out.append(LabeledQuestion(family, question, getattr(labels, family), shuffle=False))
        else:
            options = features.CHOICE_OPTIONS[family]
            question = {
                "type": "choice",
                "instructions": instructions,
                # 只收本题允许的选项：locale 里可能有别的取值（如 link_text 的 no_link），
                # 那是给反解提示词用的，不能进选项表
                "criteria": {name: spec["criteria"][name] for name in options},
            }
            out.append(
                LabeledQuestion(
                    family, question, getattr(labels, family), shuffle=family not in ORDERED
                )
            )
    return out
