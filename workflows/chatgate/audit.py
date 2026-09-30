"""聊天回复门控的产物审计：读 JSONL 逐条核对纯代码能重算的那部分。

能重算的有三条：

- **三张表、消息与三条候选**必须从 state 里解析得出（判据表里 C1…C4 齐全且顺序规范、毛病表里
  F1…F3 齐全且顺序规范、消息非空、恰好三条候选）；
- 拿消息与三条候选走 `features.observed_labels`，**命中组合与「哪条候选干净」是纯代码从文本
  重推的**——所以答案对不对得上，靠的是「构造侧从计划推」与「审计侧从文本重推」两条独立路径
  互相印证：渲染要是没兑现计划，这两条就会打架，当场暴露；
- 由文本重推出来的**命中组合**与**干净候选**必须与四族题的答案一致：逐判据的 noul、`count` 的
  档位描述、`route` 的处置描述、`reply` 的「第 N 条候选」。比的是**选项描述**（不是选项名），
  所以「选项文案抄错」也在这里暴露。

唯一纯代码读不出来的是「这条消息与这三条候选**读起来**像不像真的」——机械口径能算出命中组合与
毛病，但候选要是写成自指句（「本候选踩的毛病是：……」），读起来就完全是另一回事。那正是
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
            "tag_bad": 0,
            "count_bad": 0,
            "route_bad": 0,
            "reply_bad": 0,
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
            if parsed.tags != features.TAG_IDS or parsed.flaws != features.FLAW_IDS:
                counts["content_bad"] += 1
                continue
            if parsed.message is None or not parsed.message.strip():
                counts["content_bad"] += 1
                continue
            if len(parsed.replies) != len(features.CANDIDATE_SLOTS):
                counts["content_bad"] += 1
                continue
            state_loc = locales.load(render.detect_locale(row["state"]) or locale)
            labels = features.observed_labels(parsed.message, parsed.replies)
            tag_qids = [qid for qid in recorded if _family_of(qid) == "tag"]
            if len(tag_qids) != len(features.TAG_IDS):
                counts["tag_bad"] += 1
            else:
                # 逐判据那一族的出题顺序就是 `TAG_IDS` 的顺序（见 `questions.build`）。
                for index, tag in enumerate(features.TAG_IDS):
                    if recorded[tag_qids[index]] is not labels.contains(tag):
                        counts["tag_bad"] += 1
            if recorded.get("count") != questions.count_text(state_loc, labels.count):
                counts["count_bad"] += 1
            if recorded.get("route") != questions.route_text(state_loc, labels.count):
                counts["route_bad"] += 1
            if len(labels.clean) != 1:
                counts["reply_bad"] += 1
            elif recorded.get("reply") != questions.reply_text(state_loc, labels.clean[0]):
                counts["reply_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        for label, key, why in (
            ("判据表 / 毛病表 / 消息 / 候选认不出来", "content_bad", "三张表不是 C1~C4 与 F1~F3，或消息为空、候选不是三条"),
            ("逐判据答案与消息不自洽", "tag_bad", "tag 那一族的答案与按文本重推的对不上"),
            ("命中条数与消息不自洽", "count_bad", "count 的答案与按文本重推的对不上"),
            ("处置档位与消息不自洽", "route_bad", "route 的答案与按文本重推的对不上"),
            ("干净候选与候选不自洽", "reply_bad", "reply 的答案与按文本重推的对不上，或干净的不是恰好一条"),
        ):
            if counts[key]:
                missing.append(f"{split}: {counts[key]} 条的{label}（{why}）")
        report[split] = counts
    return report, missing
