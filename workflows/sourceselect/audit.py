"""来源选择的产物审计：读 JSONL 逐条核对纯代码能重算的那部分。

能重算的有四件，全都在产物里的候选表上机械可判：

- `pick` 必须等于「按产物里记下的 `need` 在产物里的候选表上筛一遍」的结果；
- `enough` 必须等于「筛出东西了吗」；
- `cover` 必须等于选中那一行的覆盖档；
- `near_miss` 必须等于「表里有没有一条只差首要要求那一项的行」。

`need` 本身依赖那句查询的措辞，纯代码读不出来——那正是它必须由反解判别器在合成时把关的
原因。审计写不出强校验时**不硬编一个看起来像校验的东西**（见范式 §9.13）：老实写明
「这一族靠合成期的反解把关」。

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


def expected_pick(rows: tuple[features.SourceRow, ...], need: str) -> str:
    """输入：候选表、首要要求。输出：按选源口径该选哪一行的选项名。

    预期行为：与 `features.winner_index` 用的是同一份实现——审计不重写第二套口径，
    否则两套一旦分叉，审计自己就成了新的错误来源。
    """
    winner = features.winner_index(rows, need)
    return features.pick_key(None if winner is None else winner + 1)


def audit_run(
    run_dir: Path, splits: tuple[str, ...], locale: str
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。

    预期行为：`locale` 只用来**校验调用方给的语言存在**，判定本身**与它无关**——一次 run 里
    两种语言的产物混在同一个文件里（管线按 `locales[seq % len]` 轮转），拿调用方给的那一种
    去认标记，另一种语言的产物会被平白报成「表对不上」。所以解析走
    `render.parse_state`（它把各语言的段落名都试一遍）。
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
            "pick_bad": 0,
            "enough_bad": 0,
            "cover_bad": 0,
            "near_miss_bad": 0,
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
            state = row["state"]
            frame = render.parse_state(state)
            if not frame.rows:
                counts["frame_bad"] += 1
                continue
            if frame.rules != tuple(features.RULES):
                counts["rules_bad"] += 1
                continue
            rows = frame.rows
            need = recorded.get("need")
            if not isinstance(need, str) or need not in features.NEEDS:
                counts["pick_bad"] += 1
                continue
            if recorded.get("pick") != expected_pick(rows, need):
                counts["pick_bad"] += 1
            winner = features.winner_index(rows, need)
            if recorded.get("enough") is not (winner is not None):
                counts["enough_bad"] += 1
            cover = 0 if winner is None else features.AXIS_RANK["coverage"][rows[winner].coverage]
            if recorded.get("cover") != cover:
                counts["cover_bad"] += 1
            near = any(features.is_near_miss(item, need) for item in rows)
            if recorded.get("near_miss") is not near:
                counts["near_miss_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        if counts["frame_bad"]:
            missing.append(f"{split}: {counts['frame_bad']} 条的候选表解析不出来")
        if counts["rules_bad"]:
            missing.append(f"{split}: {counts['rules_bad']} 条的选源口径与规范表对不上")
        if counts["pick_bad"]:
            missing.append(f"{split}: {counts['pick_bad']} 条的「选哪一行」与表上筛出来的对不上")
        if counts["enough_bad"]:
            missing.append(f"{split}: {counts['enough_bad']} 条的「有没有达标行」与表上筛出来的对不上")
        if counts["cover_bad"]:
            missing.append(f"{split}: {counts['cover_bad']} 条的覆盖档与选中那一行对不上")
        if counts["near_miss_bad"]:
            missing.append(f"{split}: {counts['near_miss_bad']} 条的诱饵判断与表上重算的对不上")
        report[split] = counts
    return report, missing
