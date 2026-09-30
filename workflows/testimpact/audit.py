"""测试影响选择的产物审计：读 JSONL 逐条核对纯代码能重算的那部分。

能重算的有三条：

- **覆盖表、档位表与改动**必须从 state 里解析得出（覆盖表里 `P1`…`P5` 齐全且顺序规范、档位表里
  四档齐全且顺序规范、提交标题与改动都非空，改动里出现的文件也必须在覆盖表里——出现表外路径
  等于落点算不出来）；
- 拿改动走 `features.observed_labels`，**落点是纯代码从文本重推的**——所以答案对不对得上，靠的
  是「构造侧从计划推」与「审计侧从文本重推」两条独立路径互相印证：渲染要是没兑现计划，这两条就
  会打架，当场暴露；
- 由文本重推出来的**落点**必须与四族题的答案一致：逐套件的 noul、`count` 的套数描述、`which` 的
  集合描述、`route` 的跑法描述。比的是**选项描述**（不是选项名），所以「选项文案抄错」也在这里
  暴露。

唯一纯代码读不出来的是「这次改动**读起来**是不是一次像样的改动」——机械口径能算出落点，但改动
要是写成自指句（「本次改动调整了……里的边界判断」），读起来就完全是另一回事。那正是
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
            "table_bad": 0,
            "flag_bad": 0,
            "count_bad": 0,
            "which_bad": 0,
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
            if parsed.paths != features.PATH_IDS or parsed.levels != features.LEVELS:
                counts["table_bad"] += 1
                continue
            if parsed.diff is None or not parsed.diff.strip():
                counts["table_bad"] += 1
                continue
            state_loc = locales.load(render.detect_locale(row["state"]) or locale)
            labels = features.observed_labels(parsed.diff)
            outside = [item for item in labels.touched if item not in features.PATH_IDS]
            if outside:
                # 改动里出现了覆盖表之外的路径（模型自己编了个文件）：落点按表算不出来，
                # 记成「改动认不出来」——别混进「答案对不上」，那是两回事。
                counts["table_bad"] += 1
                continue
            flag_qids = [qid for qid in recorded if _family_of(qid) == "flag"]
            if len(flag_qids) != len(features.SUITES):
                counts["flag_bad"] += 1
            else:
                # 逐套件那一族的出题顺序就是 `SUITES` 的顺序（见 `questions.build`）。
                for index, suite in enumerate(features.SUITES):
                    if recorded[flag_qids[index]] is not labels.needs(suite):
                        counts["flag_bad"] += 1
            if recorded.get("count") != questions.count_text(state_loc, len(labels.need)):
                counts["count_bad"] += 1
            if recorded.get("which") != questions.need_text(state_loc, labels.need):
                counts["which_bad"] += 1
            if recorded.get("route") != questions.route_text(state_loc, labels.level):
                counts["route_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        for label, key, why in (
            (
                "覆盖表 / 档位表 / 改动认不出来",
                "table_bad",
                "覆盖表不是 P1~P5、档位表不是四档、改动那一块是空的，或改动里出现了覆盖表之外的路径",
            ),
            ("逐套件答案与改动不自洽", "flag_bad", "flag 那一族的答案与按文本重推的对不上"),
            ("套数与改动不自洽", "count_bad", "count 的答案与按文本重推的对不上"),
            ("套件集合与改动不自洽", "which_bad", "which 的答案与按文本重推的对不上"),
            ("跑法档位与改动不自洽", "route_bad", "route 的答案与按文本重推的对不上"),
        ):
            if counts[key]:
                missing.append(f"{split}: {counts[key]} 条的{label}（{why}）")
        report[split] = counts
    return report, missing
