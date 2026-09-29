"""命令审批 workflow 的产物审计：只读 jsonl 本身能查的东西。

输入：run 目录、要审的切分、语言。
输出：(各切分计数, 缺失项列表)。
预期行为：不重新生成、不调模型——只查**题族是否齐全、题型分布**，以及
**`destructive` / `elevated` 两个答案与命令原文是否自洽**（这两条纯代码就能重算，
不需要 spec 就能事后查老产物）。

读命令那一行用 `render.command_of_any`（逐语言试），**与判别器同一个口径**。用调用方给的
`locale` 去读会出事：产物是 en、调用方按 zh 读时那一行认不出来，重算出来的两个布尔全是
False，于是报一堆**假的**「答案与原文矛盾」（实测 9 条里 4 条）。同一个字段的两个读者不该有
两种健壮度——判别器早就用 `command_of_any` 了，审计跟上。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final

from jc import jsonio
from jc.jsontypes import JsonValue

from . import judges as judges_mod, locales, questions, render

#: 允许出现的题族（出现别的就说明版本串了）。
KNOWN_FAMILIES: Final[frozenset[str]] = frozenset(questions.FAMILIES)


def audit_run(
    run_dir: Path, splits: tuple[str, ...], locale: str
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。"""
    locales.load(locale)  # 语言不存在就当场报错，别拖到解析阶段才报
    report: dict[str, JsonValue] = {}
    missing: list[str] = []
    for split in splits:
        path = run_dir / f"{split}.jsonl"
        counts: dict[str, JsonValue] = {
            "samples": 0,
            "families": Counter(),
            "types": Counter(),
            "destructive_ok": 0,
            "destructive_bad": 0,
            "elevated_ok": 0,
            "elevated_bad": 0,
            "unknown_families": Counter(),
        }
        if not path.exists():
            missing.append(f"{split}: 产物不存在（{path.name}）")
            report[split] = counts
            continue
        for _, row in jsonio.iter_jsonl(path):
            counts["samples"] += 1
            command = render.command_of_any(row["state"])
            answers = row["answers"]
            for qid, question in row["questions"].items():
                family = "".join(ch for ch in qid if not ch.isdigit()).rstrip("_")
                counts["families"][family] += 1
                counts["types"][question["type"]] += 1
                if family not in KNOWN_FAMILIES:
                    counts["unknown_families"][family] += 1
            # 两条「纯代码能重算」的自洽检查：答案必须与命令原文对得上
            if "destructive" in answers:
                expected = bool(answers["destructive"]["noul"])
                found = judges_mod.destructive_in(command)
                counts["destructive_ok" if found == expected else "destructive_bad"] += 1
            if "elevated" in answers:
                expected = bool(answers["elevated"]["noul"])
                found = render.elevated_in(command)
                counts["elevated_ok" if found == expected else "elevated_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        if counts["destructive_bad"]:
            missing.append(f"{split}: {counts['destructive_bad']} 条 destructive 答案与命令原文矛盾")
        if counts["elevated_bad"]:
            missing.append(f"{split}: {counts['elevated_bad']} 条 elevated 答案与命令原文矛盾")
        if counts["unknown_families"]:
            missing.append(f"{split}: 出现未知题族 {dict(counts['unknown_families'])}")
        report[split] = counts
    return report, missing
