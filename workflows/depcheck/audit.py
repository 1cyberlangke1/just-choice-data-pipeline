"""依赖与许可证风险门控的产物审计：拿**产物自己的文本**重推标签，再与记录下来的答案逐条比。

输入：run 目录、要审的切分、语言。
输出：`(各切分的计数, 缺失项列表)`。
预期行为：这一层**不看生成时的计划**——它只读 jsonl 里的 state 与 answers，按 `features.PATTERNS`
重算「命中了哪几个标记」，再由权重公式算出等级与处置档位，然后逐族比对：

- `flag` 那一族的四道 noul 与重算的命中组合逐条一致；
- `grade` 的答案（等级下标）指向的**等级描述**与公式算出来的那一档一致；
- `combo` 的答案（选项下标）指向的组合描述与重算的命中组合一致；
- `act` 的答案指向的动作描述与重算的处置档位一致。

为什么比对的是**描述文字**而不是编号：编号是组装器洗过的（choice 打乱、score 按下标），比编号
只能证明「组装器自洽」；比文字才能抓住「选项文案抄错」「权重与题面各说各的」这类真错。

计数全零且每个题族都出过题才算这一批合格；有一项非零就进 `missing`，让 `jc audit` 当场报出来。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Final

from jc.contract import recorded_text
from jc import jsonio
from jc.jsontypes import JsonValue
from jc.workflow import family_of as _family_of

from . import features, locales, questions, render

#: 题族的固定顺序（与 `questions.FAMILIES` 同源；这里显式列出来是为了让审计的期望不依赖别处）。
KNOWN_FAMILIES: Final[tuple[str, ...]] = questions.FAMILIES


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
            "grade_bad": 0,
            "combo_bad": 0,
            "act_bad": 0,
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
            if parsed.facts != features.TAG_IDS:
                counts["content_bad"] += 1
                continue
            if parsed.text is None or not parsed.text.strip():
                counts["content_bad"] += 1
                continue
            if features.missing_keys(parsed.text):
                counts["content_bad"] += 1
                continue
            state_loc = locales.load(render.detect_locale(row["state"]) or locale)
            labels = features.observed_labels(parsed.text)
            flag_qids = [qid for qid in recorded if _family_of(qid) == "flag"]
            if len(flag_qids) != len(features.TAG_IDS):
                counts["flag_bad"] += 1
            else:
                # 逐标记那一族的出题顺序就是 `TAG_IDS` 的顺序（见 `questions.build`）。
                for index, tag in enumerate(features.TAG_IDS):
                    if recorded[flag_qids[index]] is not labels.contains(tag):
                        counts["flag_bad"] += 1
            if recorded.get("grade") != questions.level_option(state_loc, labels.level):
                counts["grade_bad"] += 1
            if recorded.get("combo") != questions.hits_text(state_loc, labels.hits):
                counts["combo_bad"] += 1
            if recorded.get("act") != questions.act_option(state_loc, labels.act):
                counts["act_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        for label, key, why in (
            ("标记表或依赖记录认不出来", "content_bad", "标记表不是 D1~D4，或依赖记录那一块缺行 / 是空的"),
            ("逐标记答案与依赖记录不自洽", "flag_bad", "flag 那一族的答案与按文本重推的对不上"),
            ("风险等级与依赖记录不自洽", "grade_bad", "grade 的答案与按权重公式算出来的对不上"),
            ("命中组合与依赖记录不自洽", "combo_bad", "combo 的答案与按文本重推的对不上"),
            ("处置档位与依赖记录不自洽", "act_bad", "act 的答案与按文本重推的对不上"),
        ):
            if counts[key]:
                missing.append(f"{split}: {counts[key]} 条的{label}（{why}）")
        report[split] = counts
    return report, missing
