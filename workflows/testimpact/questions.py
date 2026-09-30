"""测试影响选择的题族：把改动计划翻成契约形态的题目。

输入：改动计划、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题面**自带材料**——覆盖表与跑法档位表都印在 state 里，题面只需要指明问的是哪一条，
只读 state 与题面的考生能自己答（范式 §9.8）。四个题族：

- `flag`（noul，逐套件一条，共 3 条）：这次改动需不需要跑「某一套」。三条 noul 在 16 组触及
  组合里各出现 9 次（`unit` / `integration` / `e2e` 的覆盖路径集都落在 8 组里，再加「单独触及
  盲区」那一组要跑全套），所以 True 占 9/16 = 56%，落在均衡闸的噪声地板之内；
- `count`（choice，4 档）：一共要跑几套（0~3）。选项表就是取值表，答案不退化；
- `which`（choice，4 档）：实际要跑的是哪一组。选项 = 真值 + 三个「只差一两套」的邻近集合
  （`features.ImpactPlan.distractors`）——差得越少越难分辨，题目才有区分度；
- `route`（choice，4 档）：按 state 里的跑法档位表该走哪一档（只跑冒烟 / 跑一套 / 跑两套 / 全套）。

**为什么 `count` 与 `route` 都留着**：它们共享同一个自由度（需要几套），但问的是两件事——「几套」
是数出来的，「哪一档」要照着印在 state 里的档位表把数翻成跑法。这与 `logtriage` 的
`count` / `route` 是同一套分工（那条耦合在范式里已经记过：题族间耦合会在均衡报告里照实标出）。

**为什么 `flag` 问的是「套件」而不是「路径」**：问「这次改动碰到 `src/api/handlers.py` 了吗」
只是复述 diff 的头一行，考不出覆盖表怎么读；问「需不需要跑集成测试」才逼着考生把触及的路径
翻成套件（还要认出「触及盲区 = 全套都要跑」这条规则）。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import features, locales
from .features import ImpactPlan, label_for

#: 题族的固定顺序：先逐套件问，再问套数、问组合、问档位。
FAMILIES: Final[tuple[str, ...]] = ("flag", "count", "which", "route")


from jc.assemble import choice as _choice


from jc.assemble import noul as _noul


def count_text(loc: dict[str, JsonValue], count: int) -> str:
    """输入：语言字典、需要跑的套数。输出：`count` 那一族的选项描述（审计也用它比对）。"""
    return str(loc["fam"]["count"]["opt"][f"c{count}"])


def need_text(loc: dict[str, JsonValue], need: tuple[str, ...]) -> str:
    """输入：语言字典、套件集合。输出：一组套件的可读描述（组合题的选项与审计共用）。

    预期行为：按 `SUITES` 的规范顺序取套件显示名拼起来；空集合用该语言里那句「哪套都不用跑」。
    选项描述要**两两不同**——`_choice` 会当场验这一点，所以一组套件只有一种写法。
    """
    spec = loc["fam"]["which"]
    if not need:
        return str(spec["none"])
    joiner = str(spec["joiner"])
    return joiner.join(str(loc["suite"][suite]["name"]) for suite in need)


def route_text(loc: dict[str, JsonValue], level: str) -> str:
    """输入：语言字典、跑法档位。输出：档位表里那一条跑法动作（与 state 里那张表同一份文案）。"""
    return str(loc["fam"]["route"]["opt"][level])


def _flag(loc: dict[str, JsonValue], phrasing: int, suite: str, needed: bool) -> LabeledQuestion:
    """输入：语言字典、问法编号、套件名、需不需要跑。输出：一道 noul 题。

    预期行为：题面里点名套件显示名——只读 state 的考生要能自己按覆盖表判，而不是靠猜这一族
    问的是哪一套。
    """
    spec = loc["fam"]["flag"]
    instructions = locales.pick(spec["q"], phrasing).format(name=str(loc["suite"][suite]["name"]))
    criteria = {
        "true": str(spec["criteria_true"]),
        "false": str(spec["criteria_false"]),
    }
    return _noul("flag", instructions, criteria, needed)


def _count(loc: dict[str, JsonValue], phrasing: int, count: int) -> LabeledQuestion:
    """输入：语言字典、问法编号、需要跑的套数。输出：一道 choice 题（0~3 档全出）。"""
    spec = loc["fam"]["count"]
    options = {
        f"c{value}": count_text(loc, value) for value in range(len(features.SUITES) + 1)
    }
    return _choice("count", locales.pick(spec["q"], phrasing), options, f"c{count}")


def _which(loc: dict[str, JsonValue], phrasing: int, plan: ImpactPlan) -> LabeledQuestion:
    """输入：语言字典、问法编号、改动计划。输出：一道 choice 题（真值 + 三个邻近干扰）。"""
    spec = loc["fam"]["which"]
    combos = [plan.need, *plan.distractors]
    options = {f"k{index}": need_text(loc, combo) for index, combo in enumerate(combos)}
    return _choice("which", locales.pick(spec["q"], phrasing), options, "k0")


def _route(loc: dict[str, JsonValue], phrasing: int, level: str) -> LabeledQuestion:
    """输入：语言字典、问法编号、跑法档位。输出：一道 choice 题（四个档位全出）。"""
    spec = loc["fam"]["route"]
    options = {name: str(spec["opt"][name]) for name in features.LEVELS}
    return _choice("route", locales.pick(spec["q"], phrasing), options, level)


def build(
    plan: ImpactPlan, loc: dict[str, JsonValue], phrasing: int
) -> list[LabeledQuestion]:
    """输入：改动计划、语言字典、问法编号。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：choice 的 `answer` 是**选项名**（编号由组装器最后一步映射）；noul 的 `answer`
    是布尔。逐套件那一族的出题顺序就是 `features.SUITES` 的顺序（审计按同一个顺序比答案）。
    """
    labels = label_for(plan)
    out: list[LabeledQuestion] = [
        _flag(loc, phrasing, suite, labels.needs(suite)) for suite in features.SUITES
    ]
    out.append(_count(loc, phrasing, len(labels.need)))
    out.append(_which(loc, phrasing, plan))
    out.append(_route(loc, phrasing, labels.level))
    return out
