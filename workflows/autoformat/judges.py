"""结构恢复 workflow 的判别器：把产物与「文本本身」对一遍。

输入：样本（含 state / questions / answers）、真值规格。输出：`Verdict`。
预期行为：两个判别器走**两条不同的路**——
- `roundtrip`：从 state **文本**反解出块结构，再重算四个题族的答案，逐题比对。
  这条路能抓到「渲染把信息写丢了」与「答案与文本不符」；
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

from . import locales, parse, questions
from .spec import DocSpec


def answer_value(question: dict[str, Any], answer: dict[str, Any]) -> Any:
    """输入：一道题、它的答案。输出：可比的值。

    预期行为：choice 给**选项名**（不是编号——编号只由组装器产生），noul 给布尔，
    score 给等级下标；编号越界或类型不对给 None（那是对不上，不是崩）。
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


def recompute(doc: parse.ParsedDoc, loc: dict[str, Any]) -> dict[str, list[Any]]:
    """输入：反解出来的文档、语言字典。输出：{题族: 期望答案列表}（按题序）。

    预期行为：完全从**文本**重算——并段由块切分给，块类型由块本身给，
    提示框种类由关键词给，列表有序性由时序词给。
    """
    commands = parse.command_tokens(loc)
    pairs = parse.asked_pairs(doc.lines, commands)
    callouts = doc.callouts()
    items = doc.list_items()
    return {
        "join": [doc.same_block(left, right) for left, right in pairs],
        "block_type": list(doc.kinds()),
        "callout_kind": [callouts[0].callout_kind] if callouts else [],
        "list_ordered": [items[0].list_ordered] if items else [],
    }


def compare(sample: dict[str, Any], expected: dict[str, list[Any]]) -> list[str]:
    """输入：一条样本、期望答案表。输出：差异列表（空 = 全对）。

    预期行为：按题号走一遍，同族按出现顺序对上第几条；题多了、题少了、答案不对
    都各记一笔——不静默取第一个。
    """
    seen: dict[str, int] = {family: 0 for family in questions.FAMILIES}
    problems: list[str] = []
    for qid, question in sample["questions"].items():
        family = family_of(qid)
        if family not in seen:
            problems.append(f"未知题族 {family!r}（题号 {qid}）")
            continue
        index = seen[family]
        seen[family] += 1
        want = expected.get(family, [])
        if index >= len(want):
            problems.append(f"{qid} 多出来了：文本里只算得出 {len(want)} 道 {family} 题")
            continue
        got = answer_value(question, sample["answers"].get(qid, {}))
        if got != want[index]:
            problems.append(f"{qid}：答案是 {got!r}，按文本重算应为 {want[index]!r}")
    for family, count in seen.items():
        want = expected.get(family, [])
        if count < len(want):
            problems.append(f"{family} 少了 {len(want) - count} 道题")
    return problems


class RoundtripJudge:
    """从 state 文本反解后重算四个题族，逐题比对。"""

    name = "roundtrip"

    def __init__(self, locale: str) -> None:
        """输入：语言代码。输出：判别器实例。"""
        self.locale = locale

    def judge(self, sample: dict[str, Any], spec: Any = None) -> Verdict:
        """输入：样本（与忽略的规格）。输出：Verdict。"""
        del spec
        loc = locales.load(self.locale)
        try:
            doc = parse.parse_state(sample["state"], loc)
        except ValueError as exc:
            return Verdict(False, reason=f"state 反解失败：{exc}")
        problems = compare(sample, recompute(doc, loc))
        if problems:
            return Verdict(False, reason="；".join(problems[:3]), raw={"problems": problems})
        return Verdict(True, reason="文本反解后四个题族全部对上")


class LabelJudge:
    """拿真值规格重新出一次题，比对题号与答案。"""

    name = "label"

    def __init__(self, locale: str) -> None:
        """输入：语言代码。输出：判别器实例。"""
        self.locale = locale

    def judge(self, sample: dict[str, Any], spec: Any = None) -> Verdict:
        """输入：样本、真值规格。输出：Verdict。"""
        if not isinstance(spec, DocSpec):
            return Verdict(False, reason=f"判别器拿到的规格类型不对：{type(spec).__name__}")
        loc = locales.load(self.locale)
        expected = questions.build(spec, loc, spec.phrasing, random.Random(0))
        qids = assemble.assign_family_ids(expected)
        if list(sample["questions"]) != qids:
            return Verdict(False, reason=f"题号与规格不符：{list(sample['questions'])} vs {qids}")
        for qid, item in zip(qids, expected, strict=True):
            got = answer_value(sample["questions"][qid], sample["answers"].get(qid, {}))
            if got != item.answer:
                return Verdict(False, reason=f"{qid}：答案是 {got!r}，按规格应为 {item.answer!r}")
        return Verdict(True, reason="四个题族与规格逐题一致")
