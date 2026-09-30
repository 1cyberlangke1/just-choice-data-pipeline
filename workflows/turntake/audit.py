"""回合结束判定的产物审计：读 JSONL 逐条核对纯代码能重算的那部分。

能重算的有三条：

- **判据表与用户输入**必须从 state 里解析得出（判据表里 T1…T4 齐全且顺序规范、输入非空）；
- 拿输入走 `features.observed_labels`，**命中组合是纯代码从文本重推的**——所以答案对不对得上，
  靠的是「构造侧从计划推」与「审计侧从文本重推」两条独立路径互相印证：渲染要是没兑现计划，
  这两条就会打架，当场暴露；
- 由文本重推出来的**命中组合**必须与四族题的答案一致：逐判据的 noul、`count` 的档位描述、
  `combo` 的组合描述、`act` 的处置描述。比的是**选项描述**（不是选项名），所以「选项文案抄错」
  也在这里暴露。

唯一纯代码读不出来的是「这段输入**读起来**像不像一个真人在打字/说话」——机械口径能算出命中
组合，但输入要是写成自指句（「本段踩的判据是：……」），读起来就完全是另一回事。那正是
`judges.TraceJudge` 在合成时把关的事，审计这里老实写明「这一条靠合成期的反解把关」（范式 §9.13：
写不出强校验时不要硬编一个看起来像校验的东西）。

输入：run 目录、切分名、语言。输出：各切分计数 + 缺失项列表。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from jc.contract import recorded_text
from jc.workflow import family_of as _family_of
from jc import jsonio
from jc.jsontypes import JsonValue

from . import features, locales, questions, render

#: 本 workflow 认得的题族（产物里出现别的就是串版本了）。
KNOWN_FAMILIES: frozenset[str] = frozenset(questions.FAMILIES)


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
            if parsed.tags != features.TAG_IDS:
                counts["content_bad"] += 1
                continue
            if parsed.text is None or not parsed.text.strip():
                counts["content_bad"] += 1
                continue
            state_loc = locales.load(render.detect_locale(row["state"]) or locale)
            labels = features.observed_labels(parsed.text)
            flag_qids = [qid for qid in recorded if _family_of(qid) == "flag"]
            if len(flag_qids) != len(features.TAG_IDS):
                counts["flag_bad"] += 1
            else:
                # 逐判据那一族的出题顺序就是 `TAG_IDS` 的顺序（见 `questions.build`）。
                for index, tag in enumerate(features.TAG_IDS):
                    if recorded[flag_qids[index]] is not labels.contains(tag):
                        counts["flag_bad"] += 1
            if recorded.get("count") != questions.count_text(state_loc, labels.count):
                counts["count_bad"] += 1
            if recorded.get("combo") != questions.combo_text(state_loc, labels.hits):
                counts["combo_bad"] += 1
            if recorded.get("act") != questions.act_text(state_loc, labels.count):
                counts["act_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        for label, key, why in (
            ("判据表或输入认不出来", "content_bad", "判据表不是 T1~T4，或输入那一块是空的"),
            ("逐判据答案与输入不自洽", "flag_bad", "flag 那一族的答案与按文本重推的对不上"),
            ("命中条数与输入不自洽", "count_bad", "count 的答案与按文本重推的对不上"),
            ("命中组合与输入不自洽", "combo_bad", "combo 的答案与按文本重推的对不上"),
            ("处置档位与输入不自洽", "act_bad", "act 的答案与按文本重推的对不上"),
        ):
            if counts[key]:
                missing.append(f"{split}: {counts[key]} 条的{label}（{why}）")
        report[split] = counts
    return report, missing
