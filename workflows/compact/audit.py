"""会话上下文压缩的产物审计：读 JSONL 逐条核对纯代码能重算的那部分。

能重算的有两条：
- `disposition` 必须与产物里记下的 `rule` 自洽（查 `features.RAW_DISPOSITIONS`）；
- `pin_overrode` 为真时，命中规则必须是 `pinned_keep`、而且位置行里的 `pins` 必须是
  `pinned`——「钉住起了作用」这件事只能发生在钉住的那一条上。

`rule` 本身依赖正文承载的两维（这一块是什么 / 有没有承载持久事实），纯代码读不出来——
那正是它必须由反解判别器在合成时把关的原因。审计写不出强校验时**不硬编一个看起来像
校验的东西**（见范式 §9.13）：老实写明「这两族靠合成期的反解把关」。

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


def disposition_from(rule: JsonValue) -> str | None:
    """输入：产物里记下的规则名。输出：该自洽的处置（推不出就 None）。

    预期行为：只查 `features.RAW_DISPOSITIONS`；规则名不认识就返回 None（说明产物串版本了，
    交给上层记一笔）。
    """
    if not isinstance(rule, str) or rule not in features.RAW_DISPOSITIONS:
        return None
    return features.RAW_DISPOSITIONS[rule]


def pin_overrode_consistent(rule: JsonValue, pins: str, recorded: JsonValue) -> bool:
    """输入：记下的规则名、位置行里的钉住状态、记下的 `pin_overrode`。输出：自洽吗。

    预期行为：`pin_overrode` 为真时，规则必须是 `pinned_keep`、钉住状态必须是 `pinned`；
    为假时不做额外要求（不钉住、或者钉了但本来也是 `keep`，两种都讲得通）。
    """
    if recorded is not True:
        return True
    return rule == "pinned_keep" and pins == "pinned"


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
            "disposition_bad": 0,
            "pin_bad": 0,
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
                if family in {"rule", "disposition", "pin_overrode"}:
                    recorded[family] = recorded_value(question, answers[qid])
            frame = render.parse_state(row["state"])
            if frame.where is None:
                counts["frame_bad"] += 1
                continue
            if frame.rules != canonical:
                counts["rules_bad"] += 1
            expected = disposition_from(recorded.get("rule"))
            if recorded.get("disposition") != expected:
                counts["disposition_bad"] += 1
            if not pin_overrode_consistent(
                recorded.get("rule"), frame.where.pins, recorded.get("pin_overrode")
            ):
                counts["pin_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        if counts["frame_bad"]:
            missing.append(f"{split}: {counts['frame_bad']} 条的位置行解析不出来")
        if counts["rules_bad"]:
            missing.append(f"{split}: {counts['rules_bad']} 条的处置表与规范表对不上")
        if counts["disposition_bad"]:
            missing.append(f"{split}: {counts['disposition_bad']} 条的处置与「命中规则」不自洽")
        if counts["pin_bad"]:
            missing.append(f"{split}: {counts['pin_bad']} 条的「钉住起作用」与规则 / 位置行不自洽")
        report[split] = counts
    return report, missing
