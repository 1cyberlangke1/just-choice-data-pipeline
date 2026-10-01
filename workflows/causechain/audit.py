"""根因定位的产物审计：读 JSONL 逐条核对纯代码能重算的那部分。

能重算的有三条：

- **依赖图与现场日志**必须从 state 里解析得出（边表要还原成四张图里的某一张、六个服务一条不少）；
- 拿现场日志走 `features.observed_labels`，**出错集合与根因是纯代码从文本重推的**——所以答案
  对不对得上，靠的是「构造侧从计划推」与「审计侧从文本重推」两条独立路径互相印证：渲染要是
  没兑现计划，这两条就会打架，当场暴露；
- 由文本重推出来的**根因 / 跳数 / 受影响集合 / 影响面**必须与五族题的答案一致：`root` 的
  选项描述、`hop` 的档位描述、`span` 的集合描述、`isroot` 的布尔、`count` 的档位描述。
  比的是**选项描述**（不是选项名），所以「选项文案抄错」也在这里暴露。

`hop` 与 `isroot` 两族「这次问的是谁」不写在答案里，而是写在**题面**里——审计用
`questions.named_service` 把服务编号从题面还原出来，再按重推出的跳数比档位。题面里一个服务
都没点名、或点了多个，都算这一族对不上（那说明题面模板被改坏了）。

唯一纯代码读不出来的是「这六行**读起来**像不像真的现场日志」——机械口径能算出事件类型，
但日志要是写成自指句（「本行属于超时事件」），读起来就完全是另一回事。那正是
`judges.TraceJudge` 在合成时把关的事，审计这里老实写明「这一条靠合成期的反解把关」
（范式 §9.13：写不出强校验时不要硬编一个看起来像校验的东西）。

输入：run 目录、切分名、语言。输出：各切分计数 + 缺失项列表。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from jc import jsonio
from jc.contract import recorded_text
from jc.jsontypes import JsonValue
from jc.workflow import family_of as _family_of

from . import features, locales, questions, render

#: 本 workflow 认得的题族（产物里出现别的就是串版本了）。
KNOWN_FAMILIES: frozenset[str] = frozenset(questions.FAMILIES)


def _check_row(
    row: dict[str, JsonValue], counts: dict[str, JsonValue], locale: str
) -> None:
    """输入：一条产物、计数器、兜底语言。输出：无。预期行为：只加计数，不改产物。"""
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
    graph = features.graph_of(render.parse_edges(parsed.tags))
    if graph is None:
        counts["graph_bad"] += 1
        return
    state_loc = locales.load(render.detect_locale(row["state"]) or locale)
    logs = render.parse_logs(parsed.text, state_loc)
    labels = features.observed_labels(graph, logs)
    if labels is None:
        counts["log_bad"] += 1
        return
    root_question = row["questions"].get("root")
    if not isinstance(root_question, dict):
        counts["root_bad"] += 1
    else:
        want = questions.root_text(state_loc, graph, labels.root)
        if recorded.get("root") != want:
            counts["root_bad"] += 1
    hop_question = row["questions"].get("hop")
    if isinstance(hop_question, dict):
        probed = questions.named_service(str(hop_question.get("instructions", "")))
        if probed is None or probed not in labels.distances:
            counts["hop_bad"] += 1
        else:
            want_hop = questions.hop_text(
                state_loc, features.hop_class(labels.distances[probed])
            )
            if recorded.get("hop") != want_hop:
                counts["hop_bad"] += 1
    if recorded.get("span") != questions.span_text(state_loc, labels.errored):
        counts["span_bad"] += 1
    isroot_question = row["questions"].get("isroot")
    if isinstance(isroot_question, dict):
        probed = questions.named_service(str(isroot_question.get("instructions", "")))
        if probed is None or recorded.get("isroot") != (probed == labels.root):
            counts["isroot_bad"] += 1
    if recorded.get("count") != questions.count_text(state_loc, labels.count_index):
        counts["count_bad"] += 1


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
            "graph_bad": 0,
            "log_bad": 0,
            "root_bad": 0,
            "hop_bad": 0,
            "span_bad": 0,
            "isroot_bad": 0,
            "count_bad": 0,
            "unknown_families": Counter(),
        }
        if not path.exists():
            missing.append(f"{split}: 产物不存在（{path.name}）")
            report[split] = counts
            continue
        for _, row in jsonio.iter_jsonl(path):
            counts["samples"] += 1
            _check_row(row, counts, locale)
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        for label, key, why in (
            ("依赖图认不出来", "graph_bad", "边表不是四张图里的任何一张"),
            ("现场日志认不出来", "log_bad", "六个服务凑不齐，或某一行按字面算不出事件类型"),
            ("根因与日志不自洽", "root_bad", "root 的答案与按文本重推的对不上"),
            ("跳数与日志不自洽", "hop_bad", "hop 的档位与按文本重推的对不上"),
            ("受影响集合与日志不自洽", "span_bad", "span 的集合与按文本重推的对不上"),
            ("根因判定与日志不自洽", "isroot_bad", "isroot 的布尔与按文本重推的对不上"),
            ("影响面与日志不自洽", "count_bad", "count 的档位与按文本重推的对不上"),
        ):
            if counts[key]:
                missing.append(f"{split}: {counts[key]} 条的{label}（{why}）")
        report[split] = counts
    return report, missing
