"""改动引入性判定的产物审计：读 JSONL 逐条核对纯代码能重算的那部分。

能重算的有五件，全都在产物里的改动记录 + 那两行文本上机械可判：

- `cite` 必须等于「发现里写的那个行号在记录里是第几行」（查不到就是 `none`）；
- `introduced` / `supported` 必须等于按口径在记录表上重推的结果；
- `actionable` 必须等于「修法里有没有反引号记号」——这是提示词与 `check_draft` 一起
  钉死的可判形式，审计照同一条重推；
- `triage` 必须等于把上面三件代进口径表算出来的处置档；
- `decoy` 必须等于「表里有没有一行是证据符号的近形」。

证据符号本身依赖那条发现的措辞，审计按「先取表内出现过的记号、没有再取表外第一个」重认
——认不出来（一个反引号记号都没有）时**不硬编一个看起来像校验的东西**（见范式 §9.13），
老实记成解析不出来。

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

#: 纯代码能重算答案的题族。
CHECKED_FAMILIES: frozenset[str] = frozenset(questions.FAMILIES)


def evidence_symbol_of(rows: tuple[features.RecordRow, ...], finding: str) -> str:
    """输入：改动记录、发现原文。输出：这条发现引用的证据符号（认不出来就是空串）。

    预期行为：先看反引号记号里有没有**表内出现过**的符号（那才是「引用记录」）；
    一个都没有再取表外的第一个（`evidence=none` 的形态）。
    """
    mentioned = render.backticked(finding)
    inside = {row.symbol for row in rows}
    for mark in mentioned:
        if mark in inside:
            return mark
    return mentioned[0] if mentioned else ""


def audit_run(
    run_dir: Path, splits: tuple[str, ...], locale: str
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。

    预期行为：`locale` 只用来**校验调用方给的语言存在**，判定本身**与它无关**——一次 run 里
    两种语言的产物混在同一个文件里（管线按 `locales[seq % len]` 轮转），拿调用方给的那一种
    去认标记，另一种语言的产物会被平白报成「表对不上」。所以解析走
    `render.parse_state` 与 `render.cite_line_any`（它们把各语言的段落名与行号写法都试一遍）。
    """
    locales.load(locale)  # 语言不存在就当场报错，别拖到解析阶段
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
            "cite_bad": 0,
            "introduced_bad": 0,
            "supported_bad": 0,
            "actionable_bad": 0,
            "triage_bad": 0,
            "decoy_bad": 0,
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
                    recorded[family] = recorded_value(question, answers[qid])
            frame = render.parse_state(row["state"])
            if not frame.rows:
                counts["frame_bad"] += 1
                continue
            if frame.rules != tuple(features.RULES):
                counts["rules_bad"] += 1
                continue
            rows = frame.rows
            cited_line = render.cite_line_any(frame.finding)
            if cited_line is None:
                counts["cite_bad"] += 1
                continue
            index = next((i for i, item in enumerate(rows) if item.line == cited_line), None)
            if recorded.get("cite") != features.cite_key(None if index is None else index + 1):
                counts["cite_bad"] += 1
            symbol = evidence_symbol_of(rows, frame.finding)
            introduced = features.introduced_for(rows, cited_line)
            supported = features.supported_for(rows, cited_line, symbol)
            actionable = bool(render.backticked(frame.fix))
            if recorded.get("introduced") != introduced:
                counts["introduced_bad"] += 1
            if recorded.get("supported") != supported:
                counts["supported_bad"] += 1
            if recorded.get("actionable") is not actionable:
                counts["actionable_bad"] += 1
            if recorded.get("triage") != features.triage_for(introduced, supported, actionable):
                counts["triage_bad"] += 1
            decoy = any(item.symbol == features.near_form(symbol) for item in rows)
            if recorded.get("decoy") is not decoy:
                counts["decoy_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        if counts["frame_bad"]:
            missing.append(f"{split}: {counts['frame_bad']} 条的改动记录解析不出来")
        if counts["rules_bad"]:
            missing.append(f"{split}: {counts['rules_bad']} 条的判定口径与规范表对不上")
        if counts["cite_bad"]:
            missing.append(f"{split}: {counts['cite_bad']} 条的「指向哪一行」与发现原文对不上")
        if counts["introduced_bad"]:
            missing.append(f"{split}: {counts['introduced_bad']} 条的引入性与记录表重推的对不上")
        if counts["supported_bad"]:
            missing.append(f"{split}: {counts['supported_bad']} 条的证据强度与记录表重推的对不上")
        if counts["actionable_bad"]:
            missing.append(f"{split}: {counts['actionable_bad']} 条的「修法落不落地」与修法原文对不上")
        if counts["triage_bad"]:
            missing.append(f"{split}: {counts['triage_bad']} 条的处置档与口径表重推的对不上")
        if counts["decoy_bad"]:
            missing.append(f"{split}: {counts['decoy_bad']} 条的诱饵判断与记录表重推的对不上")
        report[split] = counts
    return report, missing
