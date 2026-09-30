"""文档包切分的产物审计：把落盘的每一行按**文本**重推一遍，与记录下来的答案对。

这是「标签与文本必须自洽」的第二道闸（第一道在渲染期与判别器里）：判别器只看得到当次样本，
审计看得到**已经落盘的产物**，所以它能抓住「判别器被跳过」「跑完之后文件被改过」这类漏网。

输入：run 目录、要审的切分、语言。输出：(各切分的计数, 缺失项列表)。预期行为：全部用纯代码
重算，零 token；`missing` 为空才算这一批干净。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final

from jc.contract import recorded_text
from jc.workflow import family_of as _family_of
from jc import jsonio
from jc.jsontypes import JsonValue

from . import features, locales, questions, render

#: 本 workflow 认识的题族。
KNOWN_FAMILIES: Final[frozenset[str]] = frozenset(questions.FAMILIES)


def audit_run(
    run_dir: Path, splits: tuple[str, ...], locale: str
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。

    预期行为：`locale` 只用来**当场验一下这个语言存不存在**（别拖到解析阶段才报）；规范文案则按
    **每条 state 自己的语言**取——一次 run 里混了 zh / en 时，用调用方给的那一种去比会平白报一堆
    「文案对不上」。
    """
    locales.load(locale)
    report: dict[str, JsonValue] = {}
    missing: list[str] = []
    for split in splits:
        path = run_dir / f"{split}.jsonl"
        counts: dict[str, JsonValue] = {
            "samples": 0,
            "families": Counter(),
            "types": Counter(),
            "content_bad": 0,
            "flag_bad": 0,
            "count_bad": 0,
            "combo_bad": 0,
            "route_bad": 0,
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
                recorded[qid] = recorded_text(question, answers[qid])
            parsed = render.parse_state(row["state"])
            if parsed.classes != features.CLASS_IDS:
                counts["content_bad"] += 1
                continue
            if not parsed.pages or any(not text.strip() for text in parsed.pages):
                counts["content_bad"] += 1
                continue
            state_loc = locales.load(render.detect_locale(row["state"]) or locale)
            labels = features.observed_labels(parsed.pages)
            flag_qids = [qid for qid in recorded if _family_of(qid) == "flag"]
            if len(flag_qids) != len(features.CLASS_IDS):
                counts["flag_bad"] += 1
            else:
                # 逐类那一族的出题顺序就是 `CLASS_IDS` 的顺序（见 `questions.build`）。
                for index, cls in enumerate(features.CLASS_IDS):
                    if recorded[flag_qids[index]] is not labels.contains(cls):
                        counts["flag_bad"] += 1
            if recorded.get("count") != questions.count_text(state_loc, labels.count):
                counts["count_bad"] += 1
            if recorded.get("combo") != questions.combo_text(state_loc, labels.classes):
                counts["combo_bad"] += 1
            if recorded.get("route") != questions.route_text(state_loc, labels.docs):
                counts["route_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        for label, key, why in (
            ("种类表或页序认不出来", "content_bad", "种类表不是四个编号，或有一页是空的"),
            ("逐类答案与页面不自洽", "flag_bad", "flag 那一族的答案与按文本重推的对不上"),
            ("类数与页面不自洽", "count_bad", "count 的答案与按文本重推的对不上"),
            ("种类组合与页面不自洽", "combo_bad", "combo 的答案与按文本重推的对不上"),
            ("切分档位与页面不自洽", "route_bad", "route 的答案与按文本重推的对不上"),
        ):
            if counts[key]:
                missing.append(f"{split}: {counts[key]} 条的{label}（{why}）")
        report[split] = counts
    return report, missing
