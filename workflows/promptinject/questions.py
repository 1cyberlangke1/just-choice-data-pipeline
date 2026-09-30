"""提示注入的题族：把注入计划翻成契约形态的题目。

输入：注入计划、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题面**自带材料**——注入判据表与处置阈值表都印在 state 里，题面只需要指明问的是哪一条，
只读 state 与题面的考生能自己答（范式 §9.8）。四个题族：

- `tag`（noul，逐判据一条，共 4 条）：这段内容带不带「某一条」操纵手法。每条判据在 16 组命中
  组合里各出现 8 次，所以这一族的 True / False 按构造就是 50% / 50%；
- `count`（choice，5 档）：一共带几条（0~4）。选项表就是取值表，答案不退化；
- `combo`（choice，4 档）：实际带的是哪一组。选项 = 真值 + 三个「只差一两处」的邻近组合
  （`features.PromptInjectPlan.distractors`）——差得越少越难分辨，题目才有区分度；
- `action`（choice，4 档）：按 state 里的处置阈值该走哪一档（直接采用 / 标注后采用 / 转人工复核 /
  隔离并告警）。

**为什么没有「哪一条没被带上」那种反向问法**：它只在「带 3 条」时有唯一解（16 组里只有 4 组），
出题条件一苛刻，那一族在 10 条的小批量里就一条都出不来，审计会当场报「题族一条都没出」。四族
在**每一个** case 上都能出题，小批量也稳。

**为什么 `count` 与 `action` 不做「答案位置固定」的优化**：它们与 `tag` 共享同一个自由度
（命中条数），但问的是三件不同的事（几条 / 哪几条 / 怎么处置）。`action` 的四档阈值是
`0 / 1 / 2 / 3 条以上`，占比 1/4/6/5——刻意选四档而不是三档，就是为了不让「隔离」一档按构造
占掉 69%，把均衡闸直接顶红（见范式 §9.25）。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import features, locales
from .features import PromptInjectPlan, label_for

#: 题族的固定顺序：先逐条问，再问条数、问组合、问处置。
FAMILIES: Final[tuple[str, ...]] = ("tag", "count", "combo", "action")


from jc.assemble import choice as _choice


from jc.assemble import noul as _noul


def count_text(loc: dict[str, JsonValue], count: int) -> str:
    """输入：语言字典、命中条数。输出：`count` 那一族的选项描述（审计也用它比对）。"""
    return str(loc["fam"]["count"]["opt"][f"c{count}"])


def combo_text(loc: dict[str, JsonValue], hits: tuple[str, ...]) -> str:
    """输入：语言字典、判据编号元组。输出：一组判据的可读描述（组合题的选项与审计共用）。

    预期行为：按 `TAG_IDS` 的规范顺序取判据名拼起来；空组合用该语言里那句「哪条都不算」。
    选项描述要**两两不同**——`_choice` 会当场验这一点，所以一组判据只有一种写法。
    """
    spec = loc["fam"]["combo"]
    if not hits:
        return str(spec["none"])
    joiner = str(spec["joiner"])
    return joiner.join(str(loc["tag"][tag]["name"]) for tag in hits)


def action_text(loc: dict[str, JsonValue], count: int) -> str:
    """输入：语言字典、命中条数。输出：按处置阈值算出来的那一条处置描述。"""
    return str(loc["fam"]["action"]["opt"][features.handling_level(count)])


def _tag(loc: dict[str, JsonValue], phrasing: int, tag: str, hit: bool) -> LabeledQuestion:
    """输入：语言字典、问法编号、判据编号、是否命中。输出：一道 noul 题。

    预期行为：题面里点名判据名与判据原文——只读 state 的考生要能自己按字面判，而不是靠猜这一族
    问的是哪条规则。
    """
    spec = loc["fam"]["tag"]
    entry = loc["tag"][tag]
    instructions = locales.pick(spec["q"], phrasing).format(
        name=str(entry["name"]), rule=str(entry["rule"])
    )
    criteria = {
        "true": str(spec["criteria_true"]),
        "false": str(spec["criteria_false"]),
    }
    return _noul("tag", instructions, criteria, hit)


def _count(loc: dict[str, JsonValue], phrasing: int, count: int) -> LabeledQuestion:
    """输入：语言字典、问法编号、命中条数。输出：一道 choice 题（0~4 档全出）。"""
    spec = loc["fam"]["count"]
    options = {
        f"c{value}": count_text(loc, value) for value in range(len(features.TAG_IDS) + 1)
    }
    return _choice("count", locales.pick(spec["q"], phrasing), options, f"c{count}")


def _combo(loc: dict[str, JsonValue], phrasing: int, plan: PromptInjectPlan) -> LabeledQuestion:
    """输入：语言字典、问法编号、注入计划。输出：一道 choice 题（真值 + 三个邻近干扰）。"""
    spec = loc["fam"]["combo"]
    combos = [plan.hits, *plan.distractors]
    options = {f"k{index}": combo_text(loc, combo) for index, combo in enumerate(combos)}
    return _choice("combo", locales.pick(spec["q"], phrasing), options, "k0")


def _action(loc: dict[str, JsonValue], phrasing: int, count: int) -> LabeledQuestion:
    """输入：语言字典、问法编号、命中条数。输出：一道 choice 题（四个处置档全出）。"""
    spec = loc["fam"]["action"]
    options = {level: str(spec["opt"][level]) for level in features.HANDLING_LEVELS}
    return _choice(
        "action", locales.pick(spec["q"], phrasing), options, features.handling_level(count)
    )


def build(
    plan: PromptInjectPlan, loc: dict[str, JsonValue], phrasing: int
) -> list[LabeledQuestion]:
    """输入：注入计划、语言字典、问法编号。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：choice 的 `answer` 是**选项名**（编号由组装器最后一步映射）；noul 的 `answer`
    是布尔。
    """
    labels = label_for(plan)
    out: list[LabeledQuestion] = [
        _tag(loc, phrasing, tag, labels.contains(tag)) for tag in features.TAG_IDS
    ]
    out.append(_count(loc, phrasing, labels.count))
    out.append(_combo(loc, phrasing, plan))
    out.append(_action(loc, phrasing, labels.count))
    return out
