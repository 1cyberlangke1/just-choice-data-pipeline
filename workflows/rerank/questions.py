"""检索重排的题族：把名次计划翻成契约形态的题目。

输入：名次计划、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题面**自带口径**——排序口径印在 state 里，题面只需要指明问的是哪一批候选，只读
state 与题面的考生能自己答（范式 §9.8）。四个题族：

- `top`（choice，5 选项）：哪条候选最相关；**只在至少有一条相关候选时出**（一条都没有时
  「谁最相关」没有唯一解，出这种题就是坏数据）；
- `count`（choice，5 选项）：有几条候选沾上了至少一个关键词；
- `pair`（noul，20 道）：候选 A 是不是该排在候选 B 前面——**每个无序对两个方向都出**，
  所以这一族的答案按构造就是一半 True 一半 False，不靠采样去凑平衡；
- `order`（choice，4 选项）：前三名的正确顺序。

`top` 是一个**两阶段依赖**：出不出这一族由相关候选条数决定（与 lint 的 `spot` 同构）。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import features, locales, render
from .features import RerankPlan, label_for

#: 题族的固定顺序（先问最相关的是谁，再问相关条数，再两两比较，最后问前三名的顺序）。
FAMILIES: Final[tuple[str, ...]] = ("top", "count", "pair", "order")

#: 选项有自然顺序、打乱会破坏语义的题族。本 workflow 一个都没有：`count` 的档位虽然天然
#: 有序，但打乱它能防「答案总在第 n 个」这种位置捷径，而选项名（`n0`…`n4`）本身不受影响。
ORDERED: Final[frozenset[str]] = frozenset()

#: 无论相关候选有几条都会出的题族。
ALWAYS_FAMILIES: Final[tuple[str, ...]] = tuple(
    name for name in FAMILIES if name != "top"
)


def families_for(plan: RerankPlan) -> tuple[str, ...]:
    """输入：名次计划。输出：这条样本实际会出的题族（顺序按 `FAMILIES`）。

    预期行为：`top` 只在「至少有一条相关候选」时才出——一条都没有时名次全按原有先后排，
    「谁最相关」没有唯一解，硬出就是一条错数据。
    """
    return ALWAYS_FAMILIES if plan.relevant == 0 else FAMILIES


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


def _top(loc: dict[str, JsonValue], phrasing: int, top: int) -> LabeledQuestion:
    """输入：语言字典、问法编号、第一名的下标。输出：一道 choice 题（5 个候选全出）。"""
    spec = loc["fam"]["top"]
    options = {f"s{index + 1}": render.cand_ref(loc, index) for index in range(features.CANDIDATES)}
    return _choice("top", locales.pick(spec["q"], phrasing), options, f"s{top + 1}")


def _count(loc: dict[str, JsonValue], phrasing: int, relevant: int) -> LabeledQuestion:
    """输入：语言字典、问法编号、相关候选条数。输出：一道 choice 题（0~4 五个档位全出）。"""
    spec = loc["fam"]["count"]
    options = {f"n{v}": str(spec["opt"][f"n{v}"]) for v in range(features.CANDIDATES)}
    return _choice("count", locales.pick(spec["q"], phrasing), options, f"n{relevant}")


def _order(loc: dict[str, JsonValue], phrasing: int, order: tuple[int, ...]) -> LabeledQuestion:
    """输入：语言字典、问法编号、完整名次。输出：一道 choice 题（正确顺序 + 三个扰动）。

    预期行为：四个选项两两不同——`_choice` 兜底校验；正确的那一个固定叫 `o0`（打乱由组装器
    负责，选项名不受影响）。
    """
    spec = loc["fam"]["order"]
    head = order[:3]
    variants = (
        head,
        (head[1], head[0], head[2]),
        (head[0], head[2], head[1]),
        (head[2], head[1], head[0]),
    )
    template = str(spec["opt"]["row"])
    options = {
        f"o{index}": template.format(
            a=render.cand_ref(loc, item[0]),
            b=render.cand_ref(loc, item[1]),
            c=render.cand_ref(loc, item[2]),
        )
        for index, item in enumerate(variants)
    }
    return _choice("order", locales.pick(spec["q"], phrasing), options, "o0")


def _pairs(loc: dict[str, JsonValue], phrasing: int, labels: features.RerankLabels) -> list[LabeledQuestion]:
    """输入：语言字典、问法编号、金标签。输出：20 道 noul 题（每个无序对两个方向）。"""
    spec = loc["fam"]["pair"]
    template = locales.pick(spec["q"], phrasing)
    out: list[LabeledQuestion] = []
    for left, right in features.PAIRS:
        # 判据模板里也带 {a} / {b}：不填的话两条 noul 的判据会一模一样，考生分不清「谁排前面」
        # 说的是哪两条。
        criteria = {
            "true": str(spec["criteria"]["true"]).format(
                a=render.cand_ref(loc, left), b=render.cand_ref(loc, right)
            ),
            "false": str(spec["criteria"]["false"]).format(
                a=render.cand_ref(loc, left), b=render.cand_ref(loc, right)
            ),
        }
        question = {
            "type": "noul",
            "instructions": template.format(
                a=render.cand_ref(loc, left), b=render.cand_ref(loc, right)
            ),
            "criteria": dict(criteria),
        }
        out.append(LabeledQuestion("pair", question, labels.above(left, right), shuffle=False))
    return out


def build(plan: RerankPlan, loc: dict[str, JsonValue], phrasing: int) -> list[LabeledQuestion]:
    """输入：名次计划、语言字典、问法编号。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：choice 的 `answer` 是**选项名**（编号由组装器最后一步映射）；noul 的 `answer`
    是布尔。
    """
    labels = label_for(plan)
    out: list[LabeledQuestion] = []
    for family in families_for(plan):
        if family == "top":
            assert labels.top is not None  # families_for 已保证
            out.append(_top(loc, phrasing, labels.top))
        elif family == "count":
            out.append(_count(loc, phrasing, labels.relevant))
        elif family == "pair":
            out.extend(_pairs(loc, phrasing, labels))
        elif family == "order":
            out.append(_order(loc, phrasing, labels.order))
        else:
            raise KeyError(f"未知题族：{family!r}")
    return out
