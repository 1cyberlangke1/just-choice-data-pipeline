"""工具调用门控的产物审计：只读 jsonl 本身能查的东西。

输入：run 目录、要审的切分、语言。
输出：(各切分计数, 缺失项列表)。
预期行为：不重新生成、不调模型——从 state 里**解析出状态行与策略表**，再用产物里记下的
答案做两条交叉校验：

- **`decision` 必须与「命中的规则 + 审批状态」自洽**：`none` 只能是放行、两条硬拦规则
  只能是拒绝、三条审批规则则看审批状态；
- **`overridden` 必须与「命中的规则 + 审批状态」自洽**：只有「命中审批规则」且
  「审批状态是 granted」才为真。

这两条纯代码就能重算，不需要 spec 就能事后查老产物。`rule` 与 `severity` 两族的答案
依赖那句意图承载的两维（工具 / 内容性质），纯代码读不出来——那正是它们由
`judges.TraceJudge` 反解校验的原因。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final

from jc import jsonio
from jc.jsontypes import JsonValue

from . import features, judges as judges_mod, locales, questions, render

#: 允许出现的题族（出现别的就说明版本串了）。
KNOWN_FAMILIES: Final[frozenset[str]] = frozenset(questions.FAMILIES)


def _family_of(qid: str) -> str:
    """输入：题号。输出：题族名（去掉 `_2` 这类去重后缀）。"""
    return "".join(ch for ch in qid if not ch.isdigit()).rstrip("_")


def _recorded(question: dict[str, JsonValue], answer: dict[str, JsonValue]) -> JsonValue:
    """输入：题面、答案。输出：把编号换回可读形态后的答案值。

    预期行为：choice 的编号是**产物里那个打乱顺序**里的下标（见 `jc.assemble`），
    所以先取回选项名再比；score / noul 原样返回。
    """
    kind = answer.get("type")
    if kind == "choice":
        names = judges_mod.option_names(question)
        index = answer.get("choice")
        if isinstance(index, int) and 0 <= index < len(names):
            return names[index]
        return None
    if kind == "score":
        return answer.get("score")
    return answer.get("noul")


def decision_from(rule: JsonValue, approval: str) -> str | None:
    """输入：产物里记下的规则名、状态行里的审批状态。输出：该自洽的处置（推不出就 None）。

    预期行为：只按 `features.RAW_DECISIONS` 与「审批把 needs_approval 抬成 allow」这两条
    规则推；规则名不认识就返回 None（说明产物串版本了，交给上层记一笔）。
    """
    if not isinstance(rule, str) or rule not in features.RAW_DECISIONS:
        return None
    raw = features.RAW_DECISIONS[rule]
    if raw == "needs_approval" and approval == "granted":
        return "allow"
    return raw


def overridden_from(rule: JsonValue, approval: str) -> bool | None:
    """输入：产物里记下的规则名、状态行里的审批状态。输出：该自洽的 `overridden`。

    预期行为：规则名不认识就返回 None（同上）。
    """
    if not isinstance(rule, str) or rule not in features.RAW_DECISIONS:
        return None
    return rule in features.APPROVAL_RULES and approval == "granted"


def audit_run(
    run_dir: Path, splits: tuple[str, ...], locale: str
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。"""
    loc = locales.load(locale)  # 语言不存在就当场报错，别拖到解析阶段
    canonical = render.canonical_rules(loc)
    report: dict[str, JsonValue] = {}
    missing: list[str] = []
    for split in splits:
        path = run_dir / f"{split}.jsonl"
        counts: dict[str, JsonValue] = {
            "samples": 0,
            "families": Counter(),
            "types": Counter(),
            "frame_bad": 0,
            "rules_bad": 0,
            "decision_bad": 0,
            "overridden_bad": 0,
            "unknown_families": Counter(),
        }
        if not path.exists():
            missing.append(f"{split}: 产物不存在（{path.name}）")
            report[split] = counts
            continue
        for _, row in jsonio.iter_jsonl(path):
            counts["samples"] += 1
            answers = row["answers"]
            recorded: dict[str, JsonValue] = {}
            for qid, question in row["questions"].items():
                family = _family_of(qid)
                counts["families"][family] += 1
                counts["types"][question["type"]] += 1
                if family not in KNOWN_FAMILIES:
                    counts["unknown_families"][family] += 1
                if family in {"rule", "decision", "overridden"}:
                    recorded[family] = _recorded(question, answers[qid])
            frame = render.parse_state(row["state"])
            if frame.call is None:
                counts["frame_bad"] += 1
                continue
            if frame.rules != canonical:
                counts["rules_bad"] += 1
            approval = frame.call.approval
            expected_decision = decision_from(recorded.get("rule"), approval)
            if recorded.get("decision") != expected_decision:
                counts["decision_bad"] += 1
            expected_overridden = overridden_from(recorded.get("rule"), approval)
            if recorded.get("overridden") != expected_overridden:
                counts["overridden_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        if counts["frame_bad"]:
            missing.append(f"{split}: {counts['frame_bad']} 条的状态行解析不出来")
        if counts["rules_bad"]:
            missing.append(f"{split}: {counts['rules_bad']} 条的策略表与规范表对不上")
        if counts["decision_bad"]:
            missing.append(f"{split}: {counts['decision_bad']} 条的处置与「规则 + 审批」不自洽")
        if counts["overridden_bad"]:
            missing.append(f"{split}: {counts['overridden_bad']} 条的 overridden 与「规则 + 审批」不自洽")
        report[split] = counts
    return report, missing
