"""内容审核的题族：把审核计划翻成契约形态的题目。

输入：审核计划、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题面**自带材料**——政策条款表与处置阈值表都印在 state 里，题面只需要指明问的是哪一条，
只读 state 与题面的考生能自己答（范式 §9.8）。四个题族：

- `clause`（noul，逐条款一条，共 4 条）：这条正文踩中「某一条」吗。每条条款在 16 组命中组合里
  各出现 8 次，所以这一族的 True / False 按构造就是 50% / 50%；
- `count`（choice，5 档）：一共踩中几条（0~4）。选项表就是取值表，答案不退化；
- `combo`（choice，4 档）：实际踩中的是哪一组。选项 = 真值 + 三个「只差一两处」的邻近组合
  （`features.GuardrailPlan.distractors`）——差得越少越难分辨，题目才有区分度；
- `action`（choice，4 档）：按 state 里的处置阈值该给什么处置（放行 / 提示 / 复核 / 拦截）。

**为什么没有「哪一条没被踩中」那种反向问法**：它只在「踩中 3 条」时有唯一解（16 组里只有 4 组），
出题条件一苛刻，那一族在 10 条的小批量里就一条都出不来，审计会当场报「题族一条都没出」。四族
在**每一个** case 上都能出题，小批量也稳。

**为什么 `count` 与 `action` 不做「答案位置固定」的优化**：它们与 `clause` 共享同一个自由度
（命中条数），但问的是三件不同的事（几条 / 哪几条 / 怎么处置）。`action` 的四档阈值是
`0 / 1 / 2 / 3 条以上`，占比 1/4/6/5——刻意选四档而不是三档，就是为了不让「拦截」一档按构造
占掉 69%，把均衡闸直接顶红（见范式 §9.25）。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import features, locales
from .features import GuardrailPlan, label_for

#: 题族的固定顺序：先逐条问，再问条数、问组合、问处置。
FAMILIES: Final[tuple[str, ...]] = ("clause", "count", "combo", "action")


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

    预期行为：判据也印进题目里——考生得知道「踩中」在本题里指什么（按字面算，不做语义联想）。
    """
    return LabeledQuestion(
        family,
        {"type": "noul", "instructions": instructions, "criteria": dict(criteria)},
        answer,
        shuffle=False,
    )


def count_text(loc: dict[str, JsonValue], count: int) -> str:
    """输入：语言字典、命中条数。输出：`count` 那一族的选项描述（审计也用它比对）。"""
    return str(loc["fam"]["count"]["opt"][f"c{count}"])


def combo_text(loc: dict[str, JsonValue], hits: tuple[str, ...]) -> str:
    """输入：语言字典、条款编号元组。输出：一组条款的可读描述（组合题的选项与审计共用）。

    预期行为：按 `CLAUSE_IDS` 的规范顺序取条款名拼起来；空组合用该语言里那句「哪条都不算」。
    选项描述要**两两不同**——`_choice` 会当场验这一点，所以一组条款只有一种写法。
    """
    spec = loc["fam"]["combo"]
    if not hits:
        return str(spec["none"])
    joiner = str(spec["joiner"])
    return joiner.join(str(loc["clause"][clause]["name"]) for clause in hits)


def action_text(loc: dict[str, JsonValue], count: int) -> str:
    """输入：语言字典、命中条数。输出：按处置阈值算出来的那一条处置描述。"""
    return str(loc["fam"]["action"]["opt"][features.action_level(count)])


def _clause(loc: dict[str, JsonValue], phrasing: int, clause: str, hit: bool) -> LabeledQuestion:
    """输入：语言字典、问法编号、条款编号、是否命中。输出：一道 noul 题。

    预期行为：题面里点名条款名与判据原文——只读 state 的考生要能自己按字面判，而不是靠猜这一族
    问的是哪条规则。
    """
    spec = loc["fam"]["clause"]
    entry = loc["clause"][clause]
    instructions = locales.pick(spec["q"], phrasing).format(
        name=str(entry["name"]), rule=str(entry["rule"])
    )
    criteria = {
        "true": str(spec["criteria_true"]),
        "false": str(spec["criteria_false"]),
    }
    return _noul("clause", instructions, criteria, hit)


def _count(loc: dict[str, JsonValue], phrasing: int, count: int) -> LabeledQuestion:
    """输入：语言字典、问法编号、命中条数。输出：一道 choice 题（0~4 档全出）。"""
    spec = loc["fam"]["count"]
    options = {
        f"c{value}": count_text(loc, value) for value in range(len(features.CLAUSE_IDS) + 1)
    }
    return _choice("count", locales.pick(spec["q"], phrasing), options, f"c{count}")


def _combo(loc: dict[str, JsonValue], phrasing: int, plan: GuardrailPlan) -> LabeledQuestion:
    """输入：语言字典、问法编号、审核计划。输出：一道 choice 题（真值 + 三个邻近干扰）。"""
    spec = loc["fam"]["combo"]
    combos = [plan.hits, *plan.distractors]
    options = {f"k{index}": combo_text(loc, combo) for index, combo in enumerate(combos)}
    return _choice("combo", locales.pick(spec["q"], phrasing), options, "k0")


def _action(loc: dict[str, JsonValue], phrasing: int, count: int) -> LabeledQuestion:
    """输入：语言字典、问法编号、命中条数。输出：一道 choice 题（四个处置档全出）。"""
    spec = loc["fam"]["action"]
    options = {level: str(spec["opt"][level]) for level in features.ACTION_LEVELS}
    return _choice("action", locales.pick(spec["q"], phrasing), options, features.action_level(count))


def build(
    plan: GuardrailPlan, loc: dict[str, JsonValue], phrasing: int
) -> list[LabeledQuestion]:
    """输入：审核计划、语言字典、问法编号。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：choice 的 `answer` 是**选项名**（编号由组装器最后一步映射）；noul 的 `answer`
    是布尔。
    """
    labels = label_for(plan)
    out: list[LabeledQuestion] = [
        _clause(loc, phrasing, clause, labels.contains(clause))
        for clause in features.CLAUSE_IDS
    ]
    out.append(_count(loc, phrasing, labels.count))
    out.append(_combo(loc, phrasing, plan))
    out.append(_action(loc, phrasing, labels.count))
    return out
