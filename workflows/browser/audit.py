"""产物审计：只看产物本身，重算四个题族的答案并与产物里的答案比对。

输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。
预期行为：把 `judges` 的判据搬到产物级，这样**老产物**也能事后查（不用重跑生成）。
「题族齐全」按**产物里真的读出来的元素行与任务**点名——任务已经完成时定位 / 动作 / 风险
三族本来就不该出，按「题族全都要有」点名会把它们误报成缺题。
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

    预期行为：逐行反解界面、重算答案、比对；题族是否齐全、编号是否越界、有没有未知题族
    都各记一笔。反解不出来直接记一条「对不上」，不静默跳过。
    """
    loc = locales.load(locale)
    labels = parse.labels_of(loc)
    known = set(questions.FAMILIES)
    report: dict[str, Any] = {}
    missing: list[str] = []
    for split in splits:
        path = run_dir / f"{split}.jsonl"
        required: set[str] = set()
        counts: dict[str, Any] = {
            "samples": 0,
            "families": Counter(),
            "types": Counter(),
            "dones": Counter(),
            "tasks": Counter(),
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
                screen = parse.parse_screen(row["state"], labels)
                expected = judges.recompute(screen, loc)
            except (ValueError, KeyError, TypeError) as exc:
                counts["label_mismatch"] += 1
                missing.append(f"{split}: 有一条界面反解不出来（{exc}）")
                continue
            required |= set(
                questions.families_for(judges.rows_from(screen), screen.task, screen.object)
            )
            counts["dones"][str(expected["done"][0])] += 1
            counts["tasks"][screen.task] += 1
            problems = judges.compare(row, expected)
            if problems:
                counts["label_mismatch"] += len(problems)
            for qid, question in row["questions"].items():
                family = family_of(qid)
                counts["families"][family] += 1
                counts["types"][question["type"]] += 1
                if family not in known:
                    counts["unknown_families"][family] += 1
                if question["type"] != "choice":
                    continue
                options = list(question["criteria"])
                index = row["answers"].get(qid, {}).get("choice")
                if not isinstance(index, int) or isinstance(index, bool):
                    counts["index_bad"] += 1
                elif not 0 <= index < len(options):
                    counts["index_bad"] += 1
        for family in sorted(required):
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
