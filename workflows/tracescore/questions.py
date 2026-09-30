"""代理轨迹自评的题族：把自评计划翻成契约形态的题目。

输入：自评计划、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题面**自带材料**——行为信号表与处置档位表都印在 state 里，题面只需要指明问的是哪一条，
只读 state 与题面的考生能自己答（范式 §9.8）。四个题族：

- `flag`（noul，逐信号一条，共 4 条）：这段轨迹有没有踩中「某一条」信号。每条信号在 16 组命中
  组合里各出现 8 次，所以这一族的 True / False 按构造就是 50% / 50%；
- `count`（choice，5 档）：一共踩中几条（0~4）。选项表就是取值表，答案不退化；
- `combo`（choice，4 档）：实际踩中的是哪一组。选项 = 真值 + 三个「只差一两处」的邻近组合
  （`features.TraceScorePlan.distractors`）——差得越少越难分辨，题目才有区分度；
- `route`（choice，3 档）：按 state 里的处置档位该走哪一档（继续推进 / 换方案 / 停下问人）。

**为什么没有「哪一条没踩」那种反向问法**：它只在「踩 3 条」时有唯一解（16 组里只有 4 组），
出题条件一苛刻，那一族在 10 条的小批量里就一条都出不来，审计会当场报「题族一条都没出」。四族
在**每一个** case 上都能出题，小批量也稳。

**为什么 `count` 与 `route` 不做「答案位置固定」的优化**：它们与 `flag` 共享同一个自由度
（命中条数），但问的是三件不同的事（几条 / 哪几条 / 怎么处置）。`route` 的三档阈值是
`4 分及以上 / 正好 3 分 / 2 分及以下`，在 16 组命中组合上占比 5:6:5——刻意选三档而不是四档，
是因为分档口径只有「继续 / 换方案 / 停下问人」这三种真实动作（见范式 §9.25）。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import features, locales
from .features import TraceScorePlan, label_for

#: 题族的固定顺序：先逐条问，再问条数、问组合、问处置。
FAMILIES: Final[tuple[str, ...]] = ("flag", "count", "combo", "route")


from jc.assemble import choice as _choice


from jc.assemble import noul as _noul


def count_text(loc: dict[str, JsonValue], count: int) -> str:
    """输入：语言字典、命中条数。输出：`count` 那一族的选项描述（审计也用它比对）。"""
    return str(loc["fam"]["count"]["opt"][f"c{count}"])


def combo_text(loc: dict[str, JsonValue], hits: tuple[str, ...]) -> str:
    """输入：语言字典、信号编号元组。输出：一组信号的可读描述（组合题的选项与审计共用）。

    预期行为：按 `TAG_IDS` 的规范顺序取信号名拼起来；空组合用该语言里那句「哪条都不算」。选项
    描述要**两两不同**——`_choice` 会当场验这一点，所以一组信号只有一种写法。
    """
    spec = loc["fam"]["combo"]
    if not hits:
        return str(spec["none"])
    joiner = str(spec["joiner"])
    return joiner.join(str(loc["tag"][tag]["name"]) for tag in hits)


def route_text(loc: dict[str, JsonValue], score: int) -> str:
    """输入：语言字典、推进分。输出：按档位阈值算出来的那一条处置描述。"""
    return str(loc["fam"]["route"]["opt"][features.route_for(score)])


def _flag(loc: dict[str, JsonValue], phrasing: int, tag: str, hit: bool) -> LabeledQuestion:
    """输入：语言字典、问法编号、信号编号、是否命中。输出：一道 noul 题。

    预期行为：题面里点名信号名与判据原文——只读 state 的考生要能自己按字面判，而不是靠猜这一族
    问的是哪条规则。
    """
    spec = loc["fam"]["flag"]
    entry = loc["tag"][tag]
    instructions = locales.pick(spec["q"], phrasing).format(
        name=str(entry["name"]), rule=str(entry["rule"])
    )
    criteria = {
        "true": str(spec["criteria_true"]),
        "false": str(spec["criteria_false"]),
    }
    return _noul("flag", instructions, criteria, hit)


def _count(loc: dict[str, JsonValue], phrasing: int, count: int) -> LabeledQuestion:
    """输入：语言字典、问法编号、命中条数。输出：一道 choice 题（0~4 档全出）。"""
    spec = loc["fam"]["count"]
    options = {f"c{value}": count_text(loc, value) for value in range(len(features.TAG_IDS) + 1)}
    return _choice("count", locales.pick(spec["q"], phrasing), options, f"c{count}")


def _combo(loc: dict[str, JsonValue], phrasing: int, plan: TraceScorePlan) -> LabeledQuestion:
    """输入：语言字典、问法编号、自评计划。输出：一道 choice 题（真值 + 三个邻近干扰）。"""
    spec = loc["fam"]["combo"]
    combos = [plan.hits, *plan.distractors]
    options = {f"k{index}": combo_text(loc, combo) for index, combo in enumerate(combos)}
    return _choice("combo", locales.pick(spec["q"], phrasing), options, "k0")


def _route(loc: dict[str, JsonValue], phrasing: int, score: int) -> LabeledQuestion:
    """输入：语言字典、问法编号、推进分。输出：一道 choice 题（三个处置档全出）。"""
    spec = loc["fam"]["route"]
    options = {level: str(spec["opt"][level]) for level in features.ROUTE_LEVELS}
    return _choice("route", locales.pick(spec["q"], phrasing), options, features.route_for(score))


def build(plan: TraceScorePlan, loc: dict[str, JsonValue], phrasing: int) -> list[LabeledQuestion]:
    """输入：自评计划、语言字典、问法编号。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：choice 的 `answer` 是**选项名**（编号由组装器最后一步映射）；noul 的 `answer`
    是布尔。
    """
    labels = label_for(plan)
    out: list[LabeledQuestion] = [
        _flag(loc, phrasing, tag, labels.contains(tag)) for tag in features.TAG_IDS
    ]
    out.append(_count(loc, phrasing, labels.count))
    out.append(_combo(loc, phrasing, plan))
    out.append(_route(loc, phrasing, labels.score))
    return out
