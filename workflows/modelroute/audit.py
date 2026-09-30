"""模型路由与降级的产物审计：只读 jsonl 本身能查的东西。

输入：run 目录、要审的切分、语言。
输出：(各切分计数, 缺失项列表)。
预期行为：不重新生成、不调模型——从 state 里**解析出候选表与门槛行**，再拿同一套纯函数
重算 `route` / `blocked` / `cost` / `slack` 四族，与产物里记的答案比对。这四族的答案
本来就该能从文本重算出来；重算不出来或对不上，说明渲染、解析或组装有一处脱节。

`kind` 那一族**不在这里重算**：它由模型写的那句请求承载，纯代码读不出来（那正是它由
`judges.TraceJudge` 反解校验的原因）。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final

from jc.contract import recorded_value
from jc.workflow import family_of as _family_of
from jc import jsonio
from jc.jsontypes import JsonValue

from . import features, locales, questions, render

#: 允许出现的题族（出现别的就说明版本串了）。
KNOWN_FAMILIES: Final[frozenset[str]] = frozenset(questions.FAMILIES)

#: 能从 state 纯代码重算的题族 → 该族在 `features` 里的重算函数。
RECOMPUTABLE: Final[dict[str, str]] = {
    "route": "route_from",
    "blocked": "blocked_from",
    "cost": "cost_from",
    "slack": "slack_from",
}


def audit_run(
    run_dir: Path, splits: tuple[str, ...], locale: str
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。"""
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
            "unknown_families": Counter(),
        }
        for family in RECOMPUTABLE:
            counts[f"{family}_ok"] = 0
            counts[f"{family}_bad"] = 0
        if not path.exists():
            missing.append(f"{split}: 产物不存在（{path.name}）")
            report[split] = counts
            continue
        for _, row in jsonio.iter_jsonl(path):
            counts["samples"] += 1
            for qid, question in row["questions"].items():
                family = _family_of(qid)
                counts["families"][family] += 1
                counts["types"][question["type"]] += 1
                if family not in KNOWN_FAMILIES:
                    counts["unknown_families"][family] += 1
            frame = render.parse_state(row["state"])
            if frame.needs is None or frame.catalog != features.CATALOG:
                counts["frame_bad"] += 1
                continue
            needs = frame.needs
            for qid, question in row["questions"].items():
                family = _family_of(qid)
                recompute = RECOMPUTABLE.get(family)
                if recompute is None:
                    continue
                expected = getattr(features, recompute)(needs)
                found = recorded_value(question, row["answers"][qid])
                counts[f"{family}_{'ok' if found == expected else 'bad'}"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        if counts["frame_bad"]:
            missing.append(f"{split}: {counts['frame_bad']} 条的门槛行或候选表解析不出来")
        for family in RECOMPUTABLE:
            if counts[f"{family}_bad"]:
                missing.append(
                    f"{split}: {counts[f'{family}_bad']} 条 {family} 答案与文本重算的结果矛盾"
                )
        if counts["unknown_families"]:
            missing.append(f"{split}: 出现未知题族 {dict(counts['unknown_families'])}")
        report[split] = counts
    return report, missing
