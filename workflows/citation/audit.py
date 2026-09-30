"""引文核查的产物审计：读 JSONL 逐条核对纯代码能重算的那部分。

能重算的有四条：

- **文档、引文、声称出处**必须从 state 里解析得出：3 节、每节 2 句、引文非空、声称出处是个
  合法节号；
- 拿这三样走 `features.observed_labels`，**结论是纯代码从文本重推的**——所以答案对不对得上，
  靠的是「构造侧从计划推」与「审计侧从文本重推」两条独立路径互相印证：渲染要是没兑现计划，
  这两条就会打架，当场暴露；
- 由文本重推出来的**结论 / 在不在文档里 / 出自第几节 / 声称的那一节里有没有**必须与四族
  题的答案一致；比的是**选项描述**（不是选项名），所以「选项文案抄错」也在这里暴露；
- **查不到的引文不许出 `section` 那一族**——那是构造侧的两阶段依赖，产物里出现了就是坏数据。

唯一纯代码读不出来的是「这篇短文**读起来**是不是真的一篇短文」——机械口径能认出句子与数字，
但正文要是写成自指句（「本文引用的那句话是……」），读起来就完全是另一回事。那正是
`judges.TraceJudge` 在合成时把关的事，审计这里老实写明「这一条靠合成期的反解把关」
（范式 §9.13：写不出强校验时不要硬编一个看起来像校验的东西）。

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

    预期行为：`locale` 只用来**当场验一下这个语言存不存在**（别拖到解析阶段才报）；规范文案则
    按**每条 state 自己的语言**取——一次 run 里混了 zh / en 时，用调用方给的那一种去比会平白
    报一堆「文案对不上」。
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
            "doc_bad": 0,
            "quote_bad": 0,
            "claim_bad": 0,
            "verdict_bad": 0,
            "in_doc_bad": 0,
            "section_bad": 0,
            "in_claimed_bad": 0,
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
            if (
                parsed.title is None
                or len(parsed.sections) != features.SECTIONS
                or any(
                    len(group) != features.SENTENCES_PER_SECTION
                    for group in parsed.sections
                )
            ):
                counts["doc_bad"] += 1
                continue
            if parsed.quote is None or not parsed.quote.strip():
                counts["quote_bad"] += 1
                continue
            if parsed.claimed is None or not 0 <= parsed.claimed < features.SECTIONS:
                counts["claim_bad"] += 1
                continue
            state_loc = locales.load(render.detect_locale(row["state"]) or locale)
            labels = features.observed_labels(parsed.sections, parsed.quote, parsed.claimed)
            if recorded.get("verdict") != str(state_loc["fam"]["verdict"]["opt"][labels.verdict]):
                counts["verdict_bad"] += 1
            if recorded.get("in_doc") is not labels.in_doc:
                counts["in_doc_bad"] += 1
            if labels.in_doc:
                assert labels.section is not None
                if recorded.get("section") != render.sec_ref(state_loc, labels.section):
                    counts["section_bad"] += 1
            elif "section" in recorded:
                # 查不到的引文没有「真正出自第几节」这回事，出了这一族就是构造侧写歪了。
                counts["section_bad"] += 1
            if recorded.get("in_claimed") is not labels.at_claimed:
                counts["in_claimed_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        for label, key, why in (
            ("文档结构认不出来", "doc_bad", "认不出文档抬头，或节数/每节句数不是 3/2"),
            ("引文认不出来", "quote_bad", "认不出引文抬头，或引文是空的"),
            ("声称出处认不出来", "claim_bad", "认不出声称出处抬头，或节号不在 1~3"),
            ("结论与文档不自洽", "verdict_bad", "verdict 的答案与按文本重推的对不上"),
            ("在不在文档里与文档不自洽", "in_doc_bad", "in_doc 的答案与按文本重推的对不上"),
            ("出自第几节与文档不自洽", "section_bad", "section 的答案或出题条件与文本重推的对不上"),
            ("声称那一节里有没有与文档不自洽", "in_claimed_bad", "in_claimed 的答案与文本重推的对不上"),
        ):
            if counts[key]:
                missing.append(f"{split}: {counts[key]} 条的{label}（{why}）")
        report[split] = counts
    return report, missing
