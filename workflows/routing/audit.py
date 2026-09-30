"""工单分诊的产物审计：读 JSONL 逐条核对纯代码能重算的那部分。

能重算的有三条：
- `queue` 必须与产物里记下的 `rule` 自洽（查 `features.RAW_QUEUES`）；
- `priority` 必须与产物里记下的 `urgency` 和属性行里的客户等级自洽（查
  `features.PRIORITY_TABLE`）；
- `needs_human` 必须与产物里记下的 `queue` 和属性行（产品域 / 客户等级）自洽。

`rule` 本身依赖正文承载的两维（是什么问题 / 有多急），纯代码读不出来——那正是它必须由
反解判别器在合成时把关的原因。审计写不出强校验时**不硬编一个看起来像校验的东西**
（见范式 §9.13）：老实写明「这一族靠合成期的反解把关」。

输入：run 目录、切分名、语言。输出：各切分计数 + 缺失项列表。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from jc.contract import recorded_value
from jc.workflow import family_of as _family_of
from jc import jsonio
from jc.jsontypes import JsonValue

from . import features, locales, questions, render

#: 本 workflow 认得的题族（产物里出现别的就是串版本了）。
KNOWN_FAMILIES: frozenset[str] = frozenset(questions.FAMILIES)

#: 只要命中这三条之一就该有人看的队列（与 `features.needs_human_for` 的第一条一致）。
HIGH_TOUCH_QUEUES: frozenset[str] = frozenset(
    {"incident_response", "security_incident", "privacy_ops"}
)


def queue_from(rule: JsonValue) -> str | None:
    """输入：产物里记下的规则名。输出：该自洽的队列（推不出就 None）。

    预期行为：只查 `features.RAW_QUEUES`；规则名不认识就返回 None（说明产物串版本了，
    交给上层记一笔）。
    """
    if not isinstance(rule, str) or rule not in features.RAW_QUEUES:
        return None
    return features.RAW_QUEUES[rule]


def priority_from(urgency: JsonValue, tier: str) -> int | None:
    """输入：产物里记下的紧急度、属性行里的客户等级。输出：该自洽的档位（推不出就 None）。"""
    if not isinstance(urgency, str) or (urgency, tier) not in features.PRIORITY_TABLE:
        return None
    return features.PRIORITY_TABLE[(urgency, tier)]


def needs_human_consistent(
    queue: JsonValue, area: str, tier: str, recorded: JsonValue
) -> bool:
    """输入：记下的队列、属性行里的产品域与客户等级、记下的 `needs_human`。输出：自洽吗。

    预期行为：记下的值不是布尔、或者队列词元不认识，就**不做要求**（缺的那一格由别的
    计数记）；否则按 `features.needs_human_for` 的口径重算一遍再比。
    """
    if not isinstance(recorded, bool) or not isinstance(queue, str) or queue not in features.QUEUES:
        return True
    expected = queue in HIGH_TOUCH_QUEUES or (
        area in features.CRITICAL_AREAS and tier == "enterprise"
    )
    return recorded == expected


def audit_run(
    run_dir: Path, splits: tuple[str, ...], locale: str
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。"""
    loc = locales.load(locale)  # 语言不存在就当场报错，别拖到解析阶段
    canonical_queue = render.queue_rule_row_variants()
    canonical_priority = render.priority_rows(loc)
    canonical_flags = render.flag_row_variants()
    report: dict[str, JsonValue] = {}
    missing: list[str] = []
    for split in splits:
        path = run_dir / f"{split}.jsonl"
        counts: dict[str, JsonValue] = {
            "samples": 0,
            "families": Counter(),
            "types": Counter(),
            "attrs_bad": 0,
            "queue_table_bad": 0,
            "priority_table_bad": 0,
            "flags_bad": 0,
            "queue_bad": 0,
            "priority_bad": 0,
            "human_bad": 0,
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
                if family in {"rule", "queue", "urgency", "priority", "needs_human"}:
                    recorded[family] = recorded_value(question, answers[qid])
            ticket = render.parse_state(row["state"])
            if ticket.attrs is None:
                counts["attrs_bad"] += 1
                continue
            if ticket.queue_rules not in canonical_queue:
                counts["queue_table_bad"] += 1
            if ticket.priority != canonical_priority:
                counts["priority_table_bad"] += 1
            if ticket.flags not in canonical_flags:
                counts["flags_bad"] += 1
            if recorded.get("queue") != queue_from(recorded.get("rule")):
                counts["queue_bad"] += 1
            if recorded.get("priority") != priority_from(
                recorded.get("urgency"), ticket.attrs.tier
            ):
                counts["priority_bad"] += 1
            if not needs_human_consistent(
                recorded.get("queue"),
                ticket.attrs.area,
                ticket.attrs.tier,
                recorded.get("needs_human"),
            ):
                counts["human_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        if counts["attrs_bad"]:
            missing.append(f"{split}: {counts['attrs_bad']} 条的属性行解析不出来")
        if counts["queue_table_bad"]:
            missing.append(f"{split}: {counts['queue_table_bad']} 条的派单表与规范表对不上")
        if counts["priority_table_bad"]:
            missing.append(f"{split}: {counts['priority_table_bad']} 条的优先级表与规范表对不上")
        if counts["flags_bad"]:
            missing.append(f"{split}: {counts['flags_bad']} 条的附加规则与规范表对不上")
        if counts["queue_bad"]:
            missing.append(f"{split}: {counts['queue_bad']} 条的队列与「命中规则」不自洽")
        if counts["priority_bad"]:
            missing.append(f"{split}: {counts['priority_bad']} 条的优先级与「紧急度 × 客户等级」不自洽")
        if counts["human_bad"]:
            missing.append(f"{split}: {counts['human_bad']} 条的「要不要人工」与队列 / 属性行不自洽")
        report[split] = counts
    return report, missing
