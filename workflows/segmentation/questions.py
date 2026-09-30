"""文档包切分的题族：把切分计划翻成契约形态的题目。

输入：切分计划、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题面**自带材料**——种类表与切分规则都印在 state 里，题面只需要指明问的是哪一件事，
只读 state 与题面的考生能自己答（范式 §9.8）。四个题族：

- `flag`（noul，逐类一条，共 4 条）：这一包里有没有出现「某一类」的页。每类在 126 组组合里
  出现 57 次（45.2%），所以这一族的 True / False 按构造就接近对半；
- `count`（choice，3 档）：一共出现了几类（1~3）。选项表就是取值表，答案不退化；
- `combo`（choice，4 档）：实际出现的是哪几类。选项 = 真值 + 三个「同大小、差一处」的组合
  （`features.SegmentationPlan.distractors`）——大小一样，考生没法靠数数把它划掉，只能真去读页；
- `route`（choice，3 档）：按 state 里的切分规则该切成几份（1 份 / 2 份 / 3 份及以上）。

**为什么没有「哪一类没出现」那种反向问法**：它只在「出现 3 类」时有唯一解（14 个子集里只有
4 个），出题条件一苛刻，那一族在 10 条的小批量里就一条都出不来，审计会当场报「题族一条都没出」。
四族在**每一个** case 上都能出题，小批量也稳。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import features, locales
from .features import SegmentationPlan, label_for

#: 题族的固定顺序：先逐类问，再问类数、问组合、问份数。
FAMILIES: Final[tuple[str, ...]] = ("flag", "count", "combo", "route")


from jc.assemble import choice as _choice


from jc.assemble import noul as _noul


def count_text(loc: dict[str, JsonValue], count: int) -> str:
    """输入：语言字典、类数。输出：`count` 那一族的选项描述（审计也用它比对）。"""
    return str(loc["fam"]["count"]["opt"][f"n{count}"])


def combo_text(loc: dict[str, JsonValue], classes: tuple[str, ...]) -> str:
    """输入：语言字典、种类编号元组。输出：一组种类的可读描述（组合题的选项与审计共用）。

    预期行为：先按 `CLASS_IDS` 的规范顺序排好再取种类名拼起来——同一个集合只有一种写法，传进来的
    顺序不影响结果（选项描述要**两两不同**，`_choice` 会当场验这一点）。
    """
    spec = loc["fam"]["combo"]
    joiner = str(spec["joiner"])
    wanted = set(classes)
    return joiner.join(
        str(loc["class"][cls]["name"]) for cls in features.CLASS_IDS if cls in wanted
    )


def route_text(loc: dict[str, JsonValue], docs: int) -> str:
    """输入：语言字典、份数。输出：按份数算出来的那一条切分描述。"""
    return str(loc["fam"]["route"]["opt"][features.route_for(docs)])


def _flag(loc: dict[str, JsonValue], phrasing: int, cls: str, present: bool) -> LabeledQuestion:
    """输入：语言字典、问法编号、种类编号、这一包有没有出现它。输出：一道 noul 题。

    预期行为：题面里点名种类名与判据原文——只读 state 的考生要能自己按字面判，而不是靠猜这一族
    问的是哪条规则。
    """
    spec = loc["fam"]["flag"]
    entry = loc["class"][cls]
    instructions = locales.pick(spec["q"], phrasing).format(
        name=str(entry["name"]), rule=str(entry["rule"])
    )
    criteria = {
        "true": str(spec["criteria_true"]),
        "false": str(spec["criteria_false"]),
    }
    return _noul("flag", instructions, criteria, present)


def _count(loc: dict[str, JsonValue], phrasing: int, count: int) -> LabeledQuestion:
    """输入：语言字典、问法编号、类数。输出：一道 choice 题（1~3 档全出）。"""
    spec = loc["fam"]["count"]
    options = {
        f"n{value}": count_text(loc, value)
        for value in range(1, features.MAX_PACKET_CLASSES + 1)
    }
    return _choice("count", locales.pick(spec["q"], phrasing), options, f"n{count}")


def _combo(loc: dict[str, JsonValue], phrasing: int, plan: SegmentationPlan) -> LabeledQuestion:
    """输入：语言字典、问法编号、切分计划。输出：一道 choice 题（真值 + 三个同大小干扰）。"""
    spec = loc["fam"]["combo"]
    combos = [plan.classes, *plan.distractors]
    options = {f"k{index}": combo_text(loc, combo) for index, combo in enumerate(combos)}
    return _choice("combo", locales.pick(spec["q"], phrasing), options, "k0")


def _route(loc: dict[str, JsonValue], phrasing: int, docs: int) -> LabeledQuestion:
    """输入：语言字典、问法编号、份数。输出：一道 choice 题（三个切分档全出）。"""
    spec = loc["fam"]["route"]
    options = {level: str(spec["opt"][level]) for level in features.ROUTE_LEVELS}
    return _choice(
        "route", locales.pick(spec["q"], phrasing), options, features.route_for(docs)
    )


def build(plan: SegmentationPlan, loc: dict[str, JsonValue], phrasing: int) -> list[LabeledQuestion]:
    """输入：切分计划、语言字典、问法编号。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：choice 的 `answer` 是**选项名**（编号由组装器最后一步映射）；noul 的 `answer`
    是布尔。
    """
    labels = label_for(plan)
    out: list[LabeledQuestion] = [
        _flag(loc, phrasing, cls, labels.contains(cls)) for cls in features.CLASS_IDS
    ]
    out.append(_count(loc, phrasing, labels.count))
    out.append(_combo(loc, phrasing, plan))
    out.append(_route(loc, phrasing, labels.docs))
    return out
