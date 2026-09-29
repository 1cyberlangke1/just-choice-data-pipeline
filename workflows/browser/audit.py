"""浏览器 / GUI 操作的产物审计：读 JSONL 逐条核对纯代码能重算的那部分。

**能重算的**（元素表是代码渲染的，意图由反解判别器在合成期把关，所以下面这些都算得出来）：
- 门控表与意图表必须逐行等于代码写的那两张规范表；
- 由「记录下的意图 + 产物里那张元素表」能反推出 `target` / `action` / `done` / `blocked` /
  `confirm` / `risk` 六格（见 `expected_from`），它们必须与产物里记下的答案逐格相同。

**算不出来的**：那句话到底在不在说那个意图——那是 `TraceJudge` 在合成期反解出来的。
审计写不出强校验时**不硬编一个看起来像校验的东西**（见范式 §9.13）：老实写明
「`intent` 那一格靠合成期的反解把关」。

输入：run 目录、切分名、语言。输出：各切分计数 + 缺失项列表。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from jc import jsonio
from jc.jsontypes import JsonValue

from . import features, judges as judges_mod, locales, questions, render

#: 本 workflow 认得的题族（产物里出现别的就是串版本了）。
KNOWN_FAMILIES: frozenset[str] = frozenset(questions.FAMILIES)

#: 需要参与交叉校验的题族（choice 与 noul 与 score 都算）。
CHECKED_FAMILIES: frozenset[str] = frozenset(questions.FAMILIES)


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


def expected_from(
    rows: tuple[render.ElementRow, ...], intent: JsonValue
) -> tuple[str, str, bool, bool, bool, int] | None:
    """输入：解析出来的元素表、产物里记下的意图。输出：该自洽的六格（推不出就 None）。

    预期行为：**只看代码渲染的那张表**——
    - 取角色等于该意图要求的那几行（目标行，`decoy=same_role` 时还多一行干扰行）；
    - 干扰行的状态一定不可用，所以「有可用的」那一行就是目标行（唯一）；
    - 没有可用的但有「已是当前」那一行 → 已完成；两者都没有 → 被拦住；
    - 风险由该意图 × 目标行敏不敏感查表（两行同属一组词表，敏感与否一致）。

    推不出来（意图不认识、表里没有该角色的行、出现了两行可用）一律返回 None——
    审计宁可记一笔「算不出来」，也不给一个看着像校验的东西。
    """
    if not isinstance(intent, str) or intent not in features.ROLE_OF_INTENT:
        return None
    role = features.role_for_intent(intent)
    group = [row for row in rows if row[1] == role]
    if not group:
        return None
    sensitive = "yes" if any(features.is_sensitive(row[2]) for row in group) else "no"
    risk = features.RISK_TABLE[(intent, sensitive)]
    ready = [row for row in group if row[3] == "ready"]
    current = [row for row in group if row[3] == "current"]
    if len(ready) + len(current) > 1:
        return None  # 两行都可用 / 都是当前：这一格没有唯一答案，判不了
    if ready:
        label = ready[0][2]
        return (
            label,
            features.action_for_intent(intent),
            False,
            False,
            features.is_sensitive(label),
            risk,
        )
    if current:
        return (features.NONE_TARGET, "none", True, False, False, risk)
    return (features.NONE_TARGET, "none", False, True, False, risk)


def audit_run(
    run_dir: Path, splits: tuple[str, ...], locale: str
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。"""
    locales.load(locale)  # 语言不存在就当场报错，别拖到解析阶段（表按「任一语言」收）
    gates = render.gate_row_variants()
    criteria = render.criteria_rows()
    report: dict[str, JsonValue] = {}
    missing: list[str] = []
    for split in splits:
        path = run_dir / f"{split}.jsonl"
        counts: dict[str, JsonValue] = {
            "samples": 0,
            "families": Counter(),
            "types": Counter(),
            "frame_bad": 0,
            "gates_bad": 0,
            "criteria_bad": 0,
            "intent_bad": 0,
            "elements_bad": 0,
            "target_bad": 0,
            "action_bad": 0,
            "gate_bad": 0,
            "risk_bad": 0,
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
                if family in CHECKED_FAMILIES:
                    recorded[family] = _recorded(question, answers[qid])
            screen = render.parse_state(row["state"])
            if screen.elements is None:
                counts["frame_bad"] += 1
                continue
            if screen.gates not in gates:
                counts["gates_bad"] += 1
            if screen.criteria != criteria:
                counts["criteria_bad"] += 1
            intent = recorded.get("intent")
            if not isinstance(intent, str) or intent not in features.ROLE_OF_INTENT:
                counts["intent_bad"] += 1
                continue
            expected = expected_from(screen.elements, intent)
            if expected is None:
                counts["elements_bad"] += 1
                continue
            target, action, done, blocked, confirm, risk = expected
            if recorded.get("target") != target:
                counts["target_bad"] += 1
            if recorded.get("action") != action:
                counts["action_bad"] += 1
            if (
                recorded.get("done") is not done
                or recorded.get("blocked") is not blocked
                or recorded.get("confirm") is not confirm
            ):
                counts["gate_bad"] += 1
            if recorded.get("risk") != risk:
                counts["risk_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        if counts["frame_bad"]:
            missing.append(f"{split}: {counts['frame_bad']} 条的元素表解析不出来")
        if counts["gates_bad"]:
            missing.append(f"{split}: {counts['gates_bad']} 条的门控表与规范表对不上")
        if counts["criteria_bad"]:
            missing.append(f"{split}: {counts['criteria_bad']} 条的意图表与规范表对不上")
        if counts["intent_bad"]:
            missing.append(f"{split}: {counts['intent_bad']} 条的意图答案不是合法取值")
        if counts["elements_bad"]:
            missing.append(f"{split}: {counts['elements_bad']} 条反推不出该有的那六格")
        if counts["target_bad"]:
            missing.append(f"{split}: {counts['target_bad']} 条的「该操作哪个元素」与元素表不自洽")
        if counts["action_bad"]:
            missing.append(f"{split}: {counts['action_bad']} 条的「该做什么动作」与意图不自洽")
        if counts["gate_bad"]:
            missing.append(f"{split}: {counts['gate_bad']} 条的三档门控与元素表不自洽")
        if counts["risk_bad"]:
            missing.append(f"{split}: {counts['risk_bad']} 条的风险档与意图表不自洽")
        report[split] = counts
    return report, missing
