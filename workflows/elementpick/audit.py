"""产物审计：只看产物本身，重算五个题族的答案并与产物里的答案比对。

输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。
预期行为：把 `judges` 的判据搬到产物级，这样**老产物**也能事后查（不用重跑生成）。
「题族齐全」按**产物里真的读出来的动作标签与元素表**点名——没有落点的屏本来就只有
`found` 与 `shadow` 两族，按「题族全都要有」点名会把这种样本误报成缺题。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from collections import Counter
from pathlib import Path

from jc import jsonio
from jc.workflow import family_of

from . import judges, locales, parse, questions


def _index_bad(question: dict[str, JsonValue], answer: dict[str, JsonValue]) -> bool:
    """输入：一道题、它的答案。输出：答案下标越界或类型不对。

    预期行为：choice 查选项个数、score 查档位数（noul 是布尔，没有下标可查）；越界说明
    组装器或判别器错了，必须在产物层面点名。
    """
    if question["type"] == "noul":
        return False
    field = "choice" if question["type"] == "choice" else "score"
    index = answer.get(field)
    if not isinstance(index, int) or isinstance(index, bool):
        return True
    return not 0 <= index < len(question["criteria"])


def audit_run(
    run_dir: Path, splits: tuple[str, ...], locale: str
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分、语言。输出：(计数, 缺失项)。

    预期行为：逐行反解清单、重算答案、比对；题族是否齐全、编号是否越界、有没有未知题族
    都各记一笔。反解不出来直接记一条「对不上」，不静默跳过。
    """
    loc = locales.load(locale)
    # 用**合并标签表**反解：产物按 `locales[seq % len]` 轮转渲染，调用方给的语言未必是
    # 产物那一种；拿单一语言的标签去认，另一种语言的清单会整屏「反解不出来」。
    labels = parse.merged()
    known = set(questions.FAMILIES)
    report: dict[str, JsonValue] = {}
    missing: list[str] = []
    for split in splits:
        path = run_dir / f"{split}.jsonl"
        required: set[str] = set()
        counts: dict[str, JsonValue] = {
            "samples": 0,
            "families": Counter(),
            "types": Counter(),
            "found_true": 0,
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
                missing.append(f"{split}: 有一条清单反解不出来（{exc}）")
                continue
            required |= set(questions.families_for(screen.rows, screen.action))
            # 有落点的样本数（这些样本才有 `target` / `role` / `decoy`）。
            counts["found_true"] += sum(1 for value in expected["found"] if value)
            problems = judges.compare(row, expected)
            if problems:
                counts["label_mismatch"] += len(problems)
            for qid, question in row["questions"].items():
                family = family_of(qid)
                counts["families"][family] += 1
                counts["types"][question["type"]] += 1
                if family not in known:
                    counts["unknown_families"][family] += 1
                if _index_bad(question, row["answers"].get(qid, {})):
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
