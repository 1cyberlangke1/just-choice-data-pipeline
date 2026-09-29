"""引文核查的题族：把引文计划翻成契约形态的题目。

输入：引文计划、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题面**自带材料**——文档与引文都印在 state 里，题面只需要指明问的是哪一条引文，
只读 state 与题面的考生能自己答（范式 §9.8）。四个题族：

- `verdict`（choice，3 选项）：这条引文的核查结论（能查到且出处对 / 能查到但出处错 / 查不到）；
- `in_doc`（noul）：引文在文档里原样出现吗；
- `section`（choice，3 选项）：引文真正出自第几节；**只在引文真的在文档里时才出**——查不到
  的引文没有「真正出自第几节」这回事，硬出就是一条坏数据；
- `in_claimed`（noul）：引文能在它声称的那一节里查到吗。这是「引文核查」真正要问的那一问，
  它是前两问的合取，所以比它们都难。

`section` 是一个**两阶段依赖**：出不出这一族由「引文在不在文档里」决定（与 lint 的 `spot`
同构）。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import features, locales, render
from .features import CitationPlan, label_for

#: 题族的固定顺序（先问结论，再问在不在文档里，再问出自哪一节，最后问声称的那一节里有没有）。
FAMILIES: Final[tuple[str, ...]] = ("verdict", "in_doc", "section", "in_claimed")

#: 选项有自然顺序、打乱会破坏语义的题族。本 workflow 一个都没有：`section` 的选项虽然是
#: 「第 1 节」这种天然有序的集合，但打乱它能防「答案总在第 n 个」这种位置捷径，而选项名
#: （`s0`…`s2`）本身不受影响。
ORDERED: Final[frozenset[str]] = frozenset()


def families_for(plan: CitationPlan) -> tuple[str, ...]:
    """输入：引文计划。输出：这条样本实际会出的题族（顺序按 `FAMILIES`）。

    预期行为：`section` 只在引文真的在文档里时才出——`altered` 造出来的引文整篇都查不到，
    「它出自第几节」没有唯一解，硬出就是一条错数据。
    """
    labels = label_for(plan)
    return tuple(name for name in FAMILIES if name != "section" or labels.in_doc)


def _choice(
    family: str, instructions: str, options: dict[str, str], answer: str
) -> LabeledQuestion:
    """输入：题族、题面、选项、金答案（选项名）。输出：一道 choice 题。

    预期行为：选项描述重复直接报错——两个选项看起来一样，考生没法分辨，是构造 bug。
    """
    if len(set(options.values())) != len(options):
        raise ValueError(f"{family} 的选项描述重复：{options}")
    if answer not in options:
        raise ValueError(f"{family} 的答案 {answer!r} 不在选项里：{sorted(options)}")
    return LabeledQuestion(
        family, {"type": "choice", "instructions": instructions, "criteria": options}, answer
    )


def _noul(
    family: str, instructions: str, criteria: dict[str, str], answer: bool
) -> LabeledQuestion:
    """输入：题族、题面、判据（true / false 两句文案）、金答案。输出：一道 noul 题。

    预期行为：判据也印进题目里——考生得知道「查到」在本题里指什么（原样字符，不是同义）。
    """
    return LabeledQuestion(
        family,
        {"type": "noul", "instructions": instructions, "criteria": dict(criteria)},
        answer,
        shuffle=False,
    )


def _verdict(loc: dict[str, JsonValue], phrasing: int, verdict: str) -> LabeledQuestion:
    """输入：语言字典、问法编号、结论。输出：一道 choice 题（三种结论全出）。"""
    spec = loc["fam"]["verdict"]
    options = {
        f"v{index}": str(spec["opt"][name]) for index, name in enumerate(features.VERDICTS)
    }
    return _choice(
        "verdict",
        locales.pick(spec["q"], phrasing),
        options,
        f"v{features.VERDICTS.index(verdict)}",
    )


def _section(loc: dict[str, JsonValue], phrasing: int, section: int) -> LabeledQuestion:
    """输入：语言字典、问法编号、真正出自的节下标。输出：一道 choice 题（三节全出）。"""
    spec = loc["fam"]["section"]
    options = {f"s{index}": render.sec_ref(loc, index) for index in range(features.SECTIONS)}
    return _choice("section", locales.pick(spec["q"], phrasing), options, f"s{section}")


def _yes_no(
    family: str, loc: dict[str, JsonValue], phrasing: int, answer: bool
) -> LabeledQuestion:
    """输入：题族、语言字典、问法编号、金答案。输出：一道 noul 题（判据取该族的 true/false）。"""
    spec = loc["fam"][family]
    criteria = {
        "true": str(spec["criteria_true"]),
        "false": str(spec["criteria_false"]),
    }
    return _noul(family, locales.pick(spec["q"], phrasing), criteria, answer)


def build(plan: CitationPlan, loc: dict[str, JsonValue], phrasing: int) -> list[LabeledQuestion]:
    """输入：引文计划、语言字典、问法编号。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：choice 的 `answer` 是**选项名**（编号由组装器最后一步映射）；noul 的 `answer`
    是布尔。
    """
    labels = label_for(plan)
    out: list[LabeledQuestion] = []
    for family in families_for(plan):
        if family == "verdict":
            out.append(_verdict(loc, phrasing, labels.verdict))
        elif family == "in_doc":
            out.append(_yes_no("in_doc", loc, phrasing, labels.in_doc))
        elif family == "section":
            assert labels.section is not None  # families_for 已保证
            out.append(_section(loc, phrasing, labels.section))
        elif family == "in_claimed":
            out.append(_yes_no("in_claimed", loc, phrasing, labels.at_claimed))
        else:
            raise KeyError(f"未知题族：{family!r}")
    return out
