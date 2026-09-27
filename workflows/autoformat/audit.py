"""产物审计：只看产物本身，重算四个题族的答案并与产物里的答案比对。

输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。
预期行为：把 `judges.compare` 的判据搬到产物级，这样**老产物**也能事后查（不用重跑生成）。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from jc import jsonio
from jc.workflow import family_of

from . import judges, locales, parse, questions


def audit_run(
    run_dir: Path, splits: tuple[str, ...], locale: str
) -> tuple[dict[str, Any], list[str]]:
    """输入：run 目录、切分、语言。输出：(计数, 缺失项)。

    预期行为：逐行反解 state、重算答案、比对；题族是否齐全、编号是否越界、有没有
    未知题族都各记一笔。反解不出来直接记一条「对不上」，不静默跳过。
    """
    loc = locales.load(locale)
    known = set(questions.FAMILIES)
    report: dict[str, Any] = {}
    missing: list[str] = []
    for split in splits:
        path = run_dir / f"{split}.jsonl"
        counts: dict[str, Any] = {
            "samples": 0,
            "families": Counter(),
            "types": Counter(),
            "kinds": Counter(),
            "index_bad": 0,
            "label_mismatch": 0,
            "unknown_families": Counter(),
        }
        if not path.exists():
            missing.append(f"{split}: 产物不存在（{path.name}）")
            report[split] = counts
            continue
        for _, row in jsonio.iter_jsonl(path):
            counts["samples"] += 1
            try:
                doc = parse.parse_state(row["state"], loc)
            except (ValueError, KeyError) as exc:
                counts["label_mismatch"] += 1
                missing.append(f"{split}: 有一条 state 反解不出来（{exc}）")
                continue
            counts["kinds"].update(doc.kinds())
            counts["label_mismatch"] += len(judges.compare(row, judges.recompute(doc, loc)))
            for qid, question in row["questions"].items():
                family = family_of(qid)
                counts["families"][family] += 1
                counts["types"][question["type"]] += 1
                if family not in known:
                    counts["unknown_families"][family] += 1
                if question["type"] == "choice":
                    index = row["answers"][qid]["choice"]
                    if not 0 <= index < len(question["criteria"]):
                        counts["index_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        if counts["index_bad"]:
            missing.append(f"{split}: {counts['index_bad']} 处编号越界")
        if counts["label_mismatch"]:
            missing.append(f"{split}: {counts['label_mismatch']} 处答案与文本重算不一致")
        if counts["unknown_families"]:
            missing.append(f"{split}: 出现未知题族 {dict(counts['unknown_families'])}")
        report[split] = counts
    return report, missing
