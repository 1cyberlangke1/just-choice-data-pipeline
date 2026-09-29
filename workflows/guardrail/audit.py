"""内容审核的产物审计：读 JSONL 逐条核对纯代码能重算的那部分。

能重算的有三条：

- **条款表与正文**必须从 state 里解析得出（条款表里 P1…P4 齐全且顺序规范、正文非空）；
- 拿正文走 `features.observed_labels`，**命中组合是纯代码从文本重推的**——所以答案对不对得上，
  靠的是「构造侧从计划推」与「审计侧从文本重推」两条独立路径互相印证：渲染要是没兑现计划，
  这两条就会打架，当场暴露；
- 由文本重推出来的**命中组合**必须与四族题的答案一致：逐条款的 noul、`count` 的档位描述、
  `combo` 的组合描述、`action` 的处置描述。比的是**选项描述**（不是选项名），所以「选项文案抄错」
  也在这里暴露。

唯一纯代码读不出来的是「这段正文**读起来**是不是一段像样的内容」——机械口径能算出命中组合，但
正文要是写成自指句（「本段包含的号码是……」），读起来就完全是另一回事。那正是
`judges.TraceJudge` 在合成时把关的事，审计这里老实写明「这一条靠合成期的反解把关」（范式 §9.13：
写不出强校验时不要硬编一个看起来像校验的东西）。

输入：run 目录、切分名、语言。输出：各切分计数 + 缺失项列表。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from jc import jsonio
from jc.jsontypes import JsonValue

from . import features, judges as judges_mod, locales, questions, render

#: 本 workflow 认得的题族（产物里出现别的就是串版本了）。
KNOWN_FAMILIES: frozenset[str] = frozenset(questions.FAMILIES)


def _family_of(qid: str) -> str:
    """输入：题号。输出：题族名（去掉 `_17` 这类序号后缀）。"""
    return "".join(ch for ch in qid if not ch.isdigit()).rstrip("_")


def _recorded(question: dict[str, JsonValue], answer: dict[str, JsonValue]) -> JsonValue:
    """输入：题面、答案。输出：把编号换回**选项描述**（choice）或布尔（noul）后的答案值。

    预期行为：choice 的编号是**产物里那个打乱顺序**里的下标（见 `jc.assemble`），所以先取回选项
    描述再比——这样「选项文案抄错」也会被审计抓住，而只比选项名是抓不住的。
    """
    if answer.get("type") == "choice":
        names = judges_mod.option_names(question)
        index = answer.get("choice")
        criteria = question.get("criteria")
        if isinstance(index, int) and 0 <= index < len(names) and isinstance(criteria, dict):
            return criteria.get(names[index])
        return None
    return answer.get("noul")


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
            "clause_bad": 0,
            "count_bad": 0,
            "combo_bad": 0,
            "action_bad": 0,
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
                recorded[qid] = _recorded(question, answers[qid])
            parsed = render.parse_state(row["state"])
            if parsed.clauses != features.CLAUSE_IDS:
                counts["content_bad"] += 1
                continue
            if parsed.text is None or not parsed.text.strip():
                counts["content_bad"] += 1
                continue
            state_loc = locales.load(render.detect_locale(row["state"]) or locale)
            labels = features.observed_labels(parsed.text)
            clause_qids = [qid for qid in recorded if _family_of(qid) == "clause"]
            if len(clause_qids) != len(features.CLAUSE_IDS):
                counts["clause_bad"] += 1
            else:
                # 逐条款那一族的出题顺序就是 `CLAUSE_IDS` 的顺序（见 `questions.build`）。
                for index, clause in enumerate(features.CLAUSE_IDS):
                    if recorded[clause_qids[index]] is not labels.contains(clause):
                        counts["clause_bad"] += 1
            if recorded.get("count") != questions.count_text(state_loc, labels.count):
                counts["count_bad"] += 1
            if recorded.get("combo") != questions.combo_text(state_loc, labels.hits):
                counts["combo_bad"] += 1
            if recorded.get("action") != questions.action_text(state_loc, labels.count):
                counts["action_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        for label, key, why in (
            ("条款表或正文认不出来", "content_bad", "条款表不是 P1~P4，或正文那一块是空的"),
            ("逐条款答案与正文不自洽", "clause_bad", "clause 那一族的答案与按文本重推的对不上"),
            ("命中条数与正文不自洽", "count_bad", "count 的答案与按文本重推的对不上"),
            ("命中组合与正文不自洽", "combo_bad", "combo 的答案与按文本重推的对不上"),
            ("处置档位与正文不自洽", "action_bad", "action 的答案与按文本重推的对不上"),
        ):
            if counts[key]:
                missing.append(f"{split}: {counts[key]} 条的{label}（{why}）")
        report[split] = counts
    return report, missing
