"""聊天回复门控的题族：把「消息 + 三条候选」翻成契约形态的题目。

输入：完整计划、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题面**自带材料**——判据表、毛病表、档位表都印在 state 里，题面只需要指明问的是哪一条，
只读 state 与题面的考生能自己答（范式 §9.8）。四个题族：

- `tag`（noul，逐判据一条，共 4 条）：这条消息踩不踩「某一条」要小心的形状。每条判据在 45 个
  case 里各出现约一半（见 `features.tags_for` 的说明），所以这一族按构造就是 50% / 50%；
- `count`（choice，5 档）：一共踩几条（0~4）。选项表就是取值表，答案不退化；
- `route`（choice，4 档）：按 state 里的处置档位该走哪一档（直接回 / 先确认 / 先搁置 / 转给能
  担责的人）；
- `reply`（choice，3 档）：三条候选里**该发哪一条**。选项是「第 1 / 2 / 3 条候选」——
  三条候选的正文按这个编号印在 state 里，所以考生读得到、答得出，而答案由候选自己的毛病算出来。

**为什么 `reply` 的选项是位置而不是候选正文**：选项文案必须在**不碰渲染结果**的前提下由计划
推出来（`ask` 拿不到 state，见 `jc.workflow.Workflow` 的三步生成）。位置是计划的一维
（`features.SLOTS`），所以它推得出来；而候选正文只有渲染完才知道。把位置当选项、正文印在
state 里，两者合起来仍然自足，而且答案位置与命中条数正交（见 `features` 开头那段「为什么」）。

**为什么 `tag` 那一族逐条问、不做反向问法**：反向（「哪一条没踩」）只在「踩 3 条」时有唯一解，
出题条件一苛刻，那一族在 10 条的小批量里就一条都出不来，审计会当场报「题族一条都没出」。
四族在**每一个** case 上都能出题，小批量也稳。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import features, locales
from .features import ChatPlan, label_for

#: 题族的固定顺序：先逐条问，再问条数、问处置、问该发哪条候选。
FAMILIES: Final[tuple[str, ...]] = ("tag", "count", "route", "reply")


from jc.assemble import choice as _choice


from jc.assemble import noul as _noul


def count_text(loc: dict[str, JsonValue], count: int) -> str:
    """输入：语言字典、命中条数。输出：`count` 那一族的选项描述（审计也用它比对）。"""
    return str(loc["fam"]["count"]["opt"][f"c{count}"])


def route_text(loc: dict[str, JsonValue], count: int) -> str:
    """输入：语言字典、命中条数。输出：按处置阈值算出来的那一条处置描述。"""
    return str(loc["fam"]["route"]["opt"][features.handling_level(count)])


def reply_text(loc: dict[str, JsonValue], index: int) -> str:
    """输入：语言字典、候选下标。输出：`reply` 那一族的选项描述（审计也用它比对）。"""
    return str(loc["fam"]["reply"]["opt"][f"r{index}"])


def _tag(loc: dict[str, JsonValue], phrasing: int, tag: str, hit: bool) -> LabeledQuestion:
    """输入：语言字典、问法编号、判据编号、是否命中。输出：一道 noul 题。

    预期行为：题面里点名判据名与判据原文——只读 state 的考生要能自己按字面判。
    """
    spec = loc["fam"]["tag"]
    entry = loc["tag"][tag]
    instructions = locales.pick(spec["q"], phrasing).format(
        name=str(entry["name"]), rule=str(entry["rule"])
    )
    criteria = {"true": str(spec["criteria_true"]), "false": str(spec["criteria_false"])}
    return _noul("tag", instructions, criteria, hit)


def _count(loc: dict[str, JsonValue], phrasing: int, count: int) -> LabeledQuestion:
    """输入：语言字典、问法编号、命中条数。输出：一道 choice 题（0~4 档全出）。"""
    spec = loc["fam"]["count"]
    options = {f"c{value}": count_text(loc, value) for value in range(len(features.TAG_IDS) + 1)}
    return _choice("count", locales.pick(spec["q"], phrasing), options, f"c{count}")


def _route(loc: dict[str, JsonValue], phrasing: int, count: int) -> LabeledQuestion:
    """输入：语言字典、问法编号、命中条数。输出：一道 choice 题（四个处置档全出）。"""
    spec = loc["fam"]["route"]
    options = {level: str(spec["opt"][level]) for level in features.HANDLING_LEVELS}
    return _choice(
        "route", locales.pick(spec["q"], phrasing), options, features.handling_level(count)
    )


def _reply(loc: dict[str, JsonValue], phrasing: int, clean: int) -> LabeledQuestion:
    """输入：语言字典、问法编号、该发的候选下标。输出：一道 choice 题（三条候选全出）。

    预期行为：选项名是 `r0` / `r1` / `r2`，描述是「第 N 条候选」——state 里三条候选就是按这个
    编号印的，所以选项与正文一一对得上。
    """
    spec = loc["fam"]["reply"]
    options = {f"r{index}": reply_text(loc, index) for index in features.CANDIDATE_SLOTS}
    return _choice("reply", locales.pick(spec["q"], phrasing), options, f"r{clean}")


def build(plan: ChatPlan, loc: dict[str, JsonValue], phrasing: int) -> list[LabeledQuestion]:
    """输入：完整计划、语言字典、问法编号。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：choice 的 `answer` 是**选项名**（编号由组装器最后一步映射）；noul 的 `answer`
    是布尔。
    """
    labels = label_for(plan)
    out: list[LabeledQuestion] = [
        _tag(loc, phrasing, tag, labels.contains(tag)) for tag in features.TAG_IDS
    ]
    out.append(_count(loc, phrasing, labels.count))
    out.append(_route(loc, phrasing, labels.count))
    out.append(_reply(loc, phrasing, plan.reply.clean_answer))
    return out
