"""检索结果判读的产物审计：读 JSONL 逐条核对纯代码能重算的那部分。

能重算的只有一条：`verdict` 必须与产物里记下的 `rule` 自洽（查 `features.RAW_VERDICTS`）。
`rule` 本身依赖文本承载的三维（查询意图 / 是否切题 / 是不是垃圾），纯代码读不出来——
那正是它必须由反解判别器在合成时把关的原因。审计写不出强校验时**不硬编一个看起来像
校验的东西**（见范式 §9.13）：老实写明「这三族靠合成期的反解把关」。

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


def verdict_from(rule: JsonValue) -> str | None:
    """输入：产物里记下的规则名。输出：该自洽的处置（推不出就 None）。

    预期行为：只查 `features.RAW_VERDICTS`；规则名不认识就返回 None（说明产物串版本了，
    交给上层记一笔）。
    """
    if not isinstance(rule, str) or rule not in features.RAW_VERDICTS:
        return None
    return features.RAW_VERDICTS[rule]


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
            "verdict_bad": 0,
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
                if family in {"rule", "verdict"}:
                    recorded[family] = recorded_value(question, answers[qid])
            frame = render.parse_state(row["state"])
            if frame.source is None:
                counts["frame_bad"] += 1
                continue
            if frame.rules != canonical:
                counts["rules_bad"] += 1
            expected = verdict_from(recorded.get("rule"))
            if recorded.get("verdict") != expected:
                counts["verdict_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        if counts["frame_bad"]:
            missing.append(f"{split}: {counts['frame_bad']} 条的来源行解析不出来")
        if counts["rules_bad"]:
            missing.append(f"{split}: {counts['rules_bad']} 条的规则表与规范表对不上")
        if counts["verdict_bad"]:
            missing.append(f"{split}: {counts['verdict_bad']} 条的处置与「命中规则」不自洽")
        report[split] = counts
    return report, missing
