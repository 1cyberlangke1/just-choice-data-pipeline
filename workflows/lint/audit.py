"""行级规则检查的产物审计：读 JSONL 逐条核对纯代码能重算的那部分。

能重算的有四条：

- **片段行**必须从 state 里解析得出、且正好 8 行；
- 拿规则名与那 8 行原文走 `features.matches`，**逐行命中标记**是纯代码算的——所以答案对不
  对得上，靠的是「构造侧从行计划推」与「审计侧从文本重推」两条独立路径互相印证：模型要是
  写出的行没兑现计划，这两条就会打架，当场暴露；
- 由逐行标记算出的**命中行数 / 处置档位 / 唯一命中行**必须与 `count` / `gate` / `spot` 三族
  的答案一致；
- **处置办法表**必须逐行等于规范表（渲染抄错表也在这里暴露）。

唯一纯代码读不出来的是「那段片段**读起来**是不是真的命中」——机械口径能认出 `API_KEY =
"sk-..."`，但片段要是把凭据写在引用的示例里、读起来像另一回事。那正是 `judges.TraceJudge`
在合成时把关的事，审计这里老实写明「这一条靠合成期的反解把关」（范式 §9.13：写不出强校验
时不要硬编一个看起来像校验的东西）。

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
    """输入：题号。输出：题族名（去掉 `_1` 这类序号后缀）。"""
    return "".join(ch for ch in qid if not ch.isdigit()).rstrip("_")


def line_qid(row: int) -> str:
    """输入：行下标（0 起）。输出：那一行对应 `line` 题的题号。

    预期行为：组装器给同族题加后缀，第一道是 `line`、之后是 `line_2`…`line_8`，所以行下标
    与题号是固定对应——审计拿它把重算出来的逐行标记对回答案。
    """
    return "line" if row == 0 else f"line_{row + 1}"


def _recorded(question: dict[str, JsonValue], answer: dict[str, JsonValue]) -> JsonValue:
    """输入：题面、答案。输出：把编号换回可读形态后的答案值。

    预期行为：choice 的编号是**产物里那个打乱顺序**里的下标（见 `jc.assemble`），所以先取回
    选项名再比；noul 原样返回。
    """
    if answer.get("type") == "choice":
        names = judges_mod.option_names(question)
        index = answer.get("choice")
        if isinstance(index, int) and 0 <= index < len(names):
            return names[index]
        return None
    return answer.get("noul")


def audit_run(
    run_dir: Path, splits: tuple[str, ...], locale: str
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。

    预期行为：`locale` 只用来**当场验一下这个语言存不存在**（别拖到解析阶段才报）；规范表
    则按**每条 state 自己的语言**取——一次 run 里混了 zh / en 时，用调用方给的那一种去比
    会平白报一堆「处置办法表对不上」。
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
            "rule_bad": 0,
            "genre_bad": 0,
            "lines_bad": 0,
            "gate_table_bad": 0,
            "line_bad": 0,
            "count_bad": 0,
            "gate_bad": 0,
            "spot_bad": 0,
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
            if parsed.rule is None:
                counts["rule_bad"] += 1
                continue
            if parsed.genre is None:
                counts["genre_bad"] += 1
                continue
            if len(parsed.lines) != features.SIZE:
                counts["lines_bad"] += 1
                continue
            state_loc = locales.load(render.detect_locale(row["state"]) or locale)
            if parsed.gate != render.gate_rows(state_loc):
                counts["gate_table_bad"] += 1
            matched = tuple(features.matches(parsed.rule, text) for text in parsed.lines)
            hits = sum(matched)
            for row_index in range(features.SIZE):
                if recorded.get(line_qid(row_index)) is not matched[row_index]:
                    counts["line_bad"] += 1
            if recorded.get("count") != f"c{hits}":
                counts["count_bad"] += 1
            if recorded.get("gate") != features.gate_of(hits):
                counts["gate_bad"] += 1
            wanted_spot = f"row{matched.index(True) + 1}" if hits == 1 else None
            if recorded.get("spot") != wanted_spot:
                counts["spot_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        for label, key, why in (
            ("片段抬头里认不出规则名", "rule_bad", "规则名不在规则表里"),
            ("片段抬头里认不出体裁", "genre_bad", "抬头与三条体裁都对不上"),
            ("片段行数不是 8", "lines_bad", "编号行少一行或多一行"),
            ("处置办法表与规范表对不上", "gate_table_bad", "处置表不是规范形态"),
            ("逐行命中标记与片段不自洽", "line_bad", "line 族的答案与按文本重算的对不上"),
            ("命中行数与片段不自洽", "count_bad", "count 的答案与按文本重算的对不上"),
            ("处置与片段不自洽", "gate_bad", "gate 的答案与按文本重算的对不上"),
            ("定位与片段不自洽", "spot_bad", "spot 的出不出或取值与按文本重算的对不上"),
        ):
            if counts[key]:
                missing.append(f"{split}: {counts[key]} 条的{label}（{why}）")
        report[split] = counts
    return report, missing
