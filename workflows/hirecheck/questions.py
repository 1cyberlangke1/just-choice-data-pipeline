"""招聘初筛门禁的题族：把判定计划翻成契约形态的题目。

输入：判定计划、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题面**自带材料**——判据表、定级规则与处置档位表都印在 state 里，题面只需要指明问的是哪
一条，只读 state 与题面的考生能自己答（范式 §9.8）。四个题族：

- `flag`（noul，逐判据一条，共 4 条）：这份材料命中没命中「某条判据」。每条判据在 16 组命中组合里
  各出现 8 次，所以这一族的 True / False 按构造就是 50% / 50%；
- `grade`（**score**，5 档）：按定级规则算几级。选项表就是等级表，答案是**等级下标**（0 起）；
- `combo`（choice，4 档）：实际命中哪几条判据。选项 = 真值 + 三个「只差一两处」的邻近组合
  （`features.HireCheckPlan.distractors`）——差得越少越难分辨，题目才有区分度；
- `act`（choice，4 档）：按等级查档位表该走哪一档（排面试 / 电话初筛 / 人工复核 / 淘汰）。

**为什么这一族有 `score`**：初筛定级天生是有序的多档判断，硬塞进 choice 会把「3 级和 4 级差多少」
这个序信息丢掉。`grade` 的等级表**不打乱**（`shuffle=False`）——等级是有序的，打乱后「第 2 个选项」
不再比「第 1 个」高，那个序就没了（`rubric` 的 `band` 一族同款）。

**为什么没有「哪一条没命中」那种反向问法**：它只在「命中两条」时有唯一解（16 组里只有 6 组），出题
条件一苛刻，那一族在 10 条的小批量里就一条都出不来，审计会当场报「题族一条都没出」。四族在**每
一个** case 上都能出题，小批量也稳。

输入/输出：纯代码推导，禁止调模型。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import features, locales
from .features import HireCheckPlan

#: 题族的固定顺序（先逐判据判，再定级，再问组合与处置）。
FAMILIES: Final[tuple[str, ...]] = ("flag", "grade", "combo", "act")

#: 题型分组。
NOUL_FAMILIES: Final[tuple[str, ...]] = ("flag",)
SCORE_FAMILIES: Final[tuple[str, ...]] = ("grade",)
CHOICE_FAMILIES: Final[tuple[str, ...]] = ("combo", "act")


from jc.assemble import choice as _choice


from jc.assemble import noul as _noul


def _score(family: str, instructions: str, levels: list[str], answer: int) -> LabeledQuestion:
    """输入：题族、题面、等级表（有序）、金答案（**等级下标**，0 起）。输出：一道 score 题。

    预期行为：等级表原序、**不打乱**（`shuffle=False`）——`jc.assemble` 对 score 直接用下标当答案，
    所以下标必须与等级表同序；打乱会让「第几档」失去意义。
    """
    if len(levels) != len(features.LEVELS):
        raise ValueError(f"{family} 的等级表要有 {len(features.LEVELS)} 档，实际 {len(levels)}")
    if not 0 <= answer < len(levels):
        raise ValueError(f"{family} 的答案下标越界：{answer}")
    return LabeledQuestion(
        family, {"type": "score", "instructions": instructions, "criteria": levels}, answer,
        shuffle=False,
    )


def hits_text(loc: dict[str, JsonValue], hits: tuple[str, ...]) -> str:
    """输入：语言字典、命中组合。输出：一组判据的可读描述（组合题的选项与审计共用）。

    预期行为：按**传进来的顺序**取判据名拼起来（调用方给的都是规范顺序：计划里的命中组合与干扰组合
    都按 `TAG_IDS` 排过），空组合用该语言里那句「一条都没命中」。选项描述要**两两不同**——`_choice`
    会当场验这一点，所以一组判据只有一种写法。
    """
    spec = loc["fam"]["combo"]
    if not hits:
        return str(spec["none"])
    joiner = str(spec["joiner"])
    return joiner.join(str(loc["fact"][tag]["name"]) for tag in hits)


def level_option(loc: dict[str, JsonValue], level: int) -> str:
    """输入：语言字典、初筛等级。输出：`grade` 那一族里这一档的选项描述（审计也用它比对）。"""
    return str(loc["level"][f"L{level}"]["name"])


def act_option(loc: dict[str, JsonValue], act: str) -> str:
    """输入：语言字典、处置档位名。输出：`act` 那一族里这一档的选项描述（审计也用它比对）。"""
    return str(loc["act"][act]["name"])


def _flag(loc: dict[str, JsonValue], phrasing: int, tag: str, hit: bool) -> LabeledQuestion:
    """输入：语言字典、问法编号、判据编号、是否命中。输出：一道 noul 题。

    预期行为：题面里点名判据名与判据原文——只读 state 的考生要能自己按字面判，而不是靠猜这一族问的
    是哪条规则。
    """
    spec = loc["fam"]["flag"]
    entry = loc["fact"][tag]
    criteria = {"true": str(spec["criteria_true"]), "false": str(spec["criteria_false"])}
    return _noul(
        "flag",
        locales.pick(spec["q"], phrasing).format(name=entry["name"], rule=entry["rule"]),
        criteria,
        hit,
    )


def _grade(loc: dict[str, JsonValue], phrasing: int, level: int) -> LabeledQuestion:
    """输入：语言字典、问法编号、初筛等级。输出：一道 score 题（五档全出）。

    预期行为：等级表按 `features.LEVELS` 的顺序取，答案就是**等级下标**（等级 1 → 下标 0）。
    """
    levels = [level_option(loc, item) for item in features.LEVELS]
    return _score("grade", locales.pick(loc["fam"]["grade"]["q"], phrasing), levels, level - 1)


def _combo(loc: dict[str, JsonValue], phrasing: int, plan: HireCheckPlan) -> LabeledQuestion:
    """输入：语言字典、问法编号、判定计划。输出：一道 choice 题（真值 + 三个邻近干扰）。"""
    combos = [plan.hits, *plan.distractors]
    options = {f"k{index}": hits_text(loc, combo) for index, combo in enumerate(combos)}
    return _choice("combo", locales.pick(loc["fam"]["combo"]["q"], phrasing), options, "k0")


def _act(loc: dict[str, JsonValue], phrasing: int, level: int) -> LabeledQuestion:
    """输入：语言字典、问法编号、初筛等级。输出：一道 choice 题（四个档位全出）。"""
    options = {name: act_option(loc, name) for name in features.ACTS}
    return _choice(
        "act",
        locales.pick(loc["fam"]["act"]["q"], phrasing),
        options,
        features.act_for(level),
    )


def build(plan: HireCheckPlan, loc: dict[str, JsonValue], phrasing: int) -> list[LabeledQuestion]:
    """输入：判定计划、语言字典、问法编号。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：choice 的 `answer` 是**选项名**（编号由组装器最后一步映射）；score 的 `answer` 是
    **等级下标**；noul 的 `answer` 是布尔。
    """
    labels = features.label_for(plan)
    out: list[LabeledQuestion] = [
        _flag(loc, phrasing, tag, labels.contains(tag)) for tag in features.TAG_IDS
    ]
    out.append(_grade(loc, phrasing, labels.level))
    out.append(_combo(loc, phrasing, plan))
    out.append(_act(loc, phrasing, labels.level))
    return out
