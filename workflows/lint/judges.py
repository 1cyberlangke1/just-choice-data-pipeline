"""行级规则检查 workflow 的判别器：把产物与「文本本身」对一遍。

输入：样本（含 state / questions / answers）、真值规格。输出：`Verdict`。
预期行为：两个判别器走**两条不同的路**——
- `roundtrip`：从 state **文本**把规则与逐行原文读回来，用 `spec.matches` 重算命中，
  再用同一套出题逻辑重算答案，逐题比对。这条路能抓到「渲染把某一行写串味了（本该命中的
  行不再命中）」与「答案与文本不符」；
- `label`：拿真值规格重新出一次题，比对题号与答案。这条路能抓到组装器 / 选项打乱出错。

`recompute` 与 `compare` 都对外暴露：`audit` 要拿它们在**产物层面**重算同一套东西，
这样老产物也能事后查（不用重跑生成）。
"""

from __future__ import annotations

import random
from typing import Any

from jc import assemble
from jc.judge import Verdict
from jc.workflow import family_of

from . import locales, parse, questions, render
from .spec import LintSpec


def answer_value(question: dict[str, Any], answer: dict[str, Any]) -> Any:
    """输入：一道题、它的答案。输出：可比的值。

    预期行为：choice 给**选项名**（不是编号——编号只由组装器产生），noul 给布尔；
    编号越界或类型不对给 None（那是对不上，不是崩）。
    """
    if question["type"] == "choice":
        options = list(question["criteria"])
        index = answer.get("choice")
        if not isinstance(index, int) or isinstance(index, bool):
            return None
        if not 0 <= index < len(options):
            return None
        return options[index]
    if question["type"] == "noul":
        value = answer.get("noul")
        return value if isinstance(value, bool) else None
    return answer.get("score")


def recompute(reading: parse.LintReading, loc: dict[str, Any]) -> dict[str, list[Any]]:
    """输入：读法从文本读回来的片段、语言字典。输出：{题族: 期望答案列表}（按题序）。

    预期行为：走**同一套出题逻辑**（`questions.build`），只是输入换成从文本读回来的
    规则 / 逐行原文 / 逐行命中——所以「渲染写歪」会直接变成「答案对不上」。
    问法编号给 0：它只影响措辞，不影响答案。
    """
    out: dict[str, list[Any]] = {}
    for item in questions.build(reading.rule, reading.lines, reading.matched, loc, 0):
        out.setdefault(item.family, []).append(item.answer)
    return out


def compare(sample: dict[str, Any], expected: dict[str, list[Any]]) -> list[str]:
    """输入：一条样本、期望答案表。输出：差异列表（空 = 全对）。

    预期行为：按题号走一遍，同族按出现顺序对上第几条；题多了、题少了、答案不对、
    题族不认识都各记一笔——不静默取第一个。「少题」那一趟遍历**期望表**（不是产物）：
    遍历产物时整族缺席会一声不响，漏报比误报危险。
    """
    legal = set(questions.FAMILIES)
    seen: dict[str, int] = {}
    problems: list[str] = []
    for qid, question in sample["questions"].items():
        family = family_of(qid)
        if family not in legal:
            problems.append(f"未知题族 {family!r}（题号 {qid}）")
            continue
        index = seen.get(family, 0)
        seen[family] = index + 1
        want = expected.get(family, [])
        if index >= len(want):
            problems.append(f"{qid} 多出来了：文本里只算得出 {len(want)} 道 {family} 题")
            continue
        got = answer_value(question, sample["answers"].get(qid, {}))
        if got != want[index]:
            problems.append(f"{qid}：答案是 {got!r}，按文本重算应为 {want[index]!r}")
    for family, want in expected.items():
        count = seen.get(family, 0)
        if count < len(want):
            problems.append(f"{family} 少了 {len(want) - count} 道题")
    return problems


class RoundtripJudge:
    """从 state 文本把规则与片段读回来，重算命中后逐题比对。"""

    name = "roundtrip"

    def __init__(self, locale: str) -> None:
        """输入：语言代码。输出：判别器实例。"""
        self.locale = locale

    def judge(self, sample: dict[str, Any], spec: Any = None) -> Verdict:
        """输入：样本（与忽略的规格）。输出：Verdict。"""
        del spec
        loc = locales.load(self.locale)
        try:
            reading = parse.read_state(sample["state"], loc)
        except (ValueError, KeyError, TypeError) as exc:
            return Verdict(False, reason=f"state 读不出来：{exc}")
        problems = compare(sample, recompute(reading, loc))
        if problems:
            return Verdict(False, reason="；".join(problems[:3]), raw={"problems": problems})
        return Verdict(True, reason="文本读回来之后逐题都对上")


class LabelJudge:
    """拿真值规格重新出一次题，比对题号与答案。"""

    name = "label"

    def __init__(self, locale: str) -> None:
        """输入：语言代码。输出：判别器实例。"""
        self.locale = locale

    def judge(self, sample: dict[str, Any], spec: Any = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。"""
        if not isinstance(spec, LintSpec):
            return Verdict(False, reason=f"判别器拿到的规格类型不对：{type(spec).__name__}")
        loc = locales.load(self.locale)
        expected = questions.build(
            spec.plan.rule,
            render.render_lines(spec.plan, loc),
            spec.plan.matched(),
            loc,
            spec.plan.phrasing,
        )
        qids = assemble.assign_family_ids(expected)
        if list(sample["questions"]) != qids:
            return Verdict(
                False, reason=f"题号与规格不符：{list(sample['questions'])} vs {qids}"
            )
        for qid, item in zip(qids, expected, strict=True):
            got = answer_value(sample["questions"][qid], sample["answers"].get(qid, {}))
            if got != item.answer:
                return Verdict(
                    False, reason=f"{qid}：答案是 {got!r}，按规格应为 {item.answer!r}"
                )
        return Verdict(True, reason="题号与答案与规格逐题一致")
