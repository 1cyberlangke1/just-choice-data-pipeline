"""检索重排的产物审计：读 JSONL 逐条核对纯代码能重算的那部分。

能重算的有四条：

- **查询、关键词表、候选原文**必须从 state 里解析得出，候选正好 5 条、关键词个数在取值表
  里、每个关键词都出现在查询里；
- 拿关键词表与那 5 条候选原文走 `features.coverage`，**逐条覆盖率**是纯代码算的——所以答案
  对不对得上，靠的是「构造侧从计划推」与「审计侧从文本重推」两条独立路径互相印证：模型要是
  写出的候选没兑现计划，这两条就会打架，当场暴露；
- 由覆盖率算出的**名次 / 相关条数 / 第一名**必须与 `top` / `count` / `pair` / `order` 四族
  的答案一致；比的是**选项描述**（不是选项名），所以「选项文案抄错」也在这里暴露；
- **排序口径**必须逐行等于规范表（渲染抄错口径也在这里暴露）。

唯一纯代码读不出来的是「这段 state **读起来**是不是真的按口径排序」——机械口径能认出
关键词，但候选要是把关键词写进引用的反例里、读起来像另一回事。那正是 `judges.TraceJudge`
在合成时把关的事，审计这里老实写明「这一条靠合成期的反解把关」（范式 §9.13：写不出强校验
时不要硬编一个看起来像校验的东西）。

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


def pair_index(qid: str) -> int:
    """输入：`pair` 一族的题号。输出：它在 `features.PAIRS` 里的下标。

    预期行为：组装器给同族题加后缀，第一道是 `pair`、之后是 `pair_2`…`pair_20`，所以题号
    与无序对是固定对应——审计拿它把重算出来的先后关系对回答案。
    """
    if qid == "pair":
        return 0
    return int(qid.split("_")[1]) - 1


def _expected_order_text(loc: dict[str, JsonValue], order: tuple[int, ...]) -> str:
    """输入：语言字典、重算出来的名次。输出：前三名那一项的**规范文案**。"""
    head = order[:3]
    return str(loc["fam"]["order"]["opt"]["row"]).format(
        a=render.cand_ref(loc, head[0]),
        b=render.cand_ref(loc, head[1]),
        c=render.cand_ref(loc, head[2]),
    )


def audit_run(
    run_dir: Path, splits: tuple[str, ...], locale: str
) -> tuple[dict[str, JsonValue], list[str]]:
    """输入：run 目录、切分、语言。输出：(各切分计数, 缺失项列表)。

    预期行为：`locale` 只用来**当场验一下这个语言存不存在**（别拖到解析阶段才报）；规范表则
    按**每条 state 自己的语言**取——一次 run 里混了 zh / en 时，用调用方给的那一种去比会平白
    报一堆「排序口径对不上」。
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
            "terms_bad": 0,
            "candidates_bad": 0,
            "rule_bad": 0,
            "top_bad": 0,
            "count_bad": 0,
            "pair_bad": 0,
            "order_bad": 0,
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
            if parsed.query is None:
                counts["terms_bad"] += 1
                counts["candidates_bad"] += 1
                continue
            state_loc = locales.load(render.detect_locale(row["state"]) or locale)
            if len(parsed.terms) not in features.TERM_COUNTS:
                counts["terms_bad"] += 1
                continue
            if len(parsed.candidates) != features.CANDIDATES:
                counts["candidates_bad"] += 1
                continue
            if parsed.rule != render.rule_rows(state_loc):
                counts["rule_bad"] += 1
            scores = tuple(
                features.coverage(text, list(parsed.terms)) for text in parsed.candidates
            )
            order = tuple(
                sorted(range(features.CANDIDATES), key=lambda index: (-scores[index], index))
            )
            relevant = sum(1 for value in scores if value > 0)
            top = order[0] if relevant > 0 else None
            # `cand_ref` 收的是**0 起**的下标（与 `questions._top` 的用法一致）——这里再 +1
            # 就会把「候选 4」写成「候选 5」，正好差一格。
            wanted_top = render.cand_ref(state_loc, top) if top is not None else None
            if recorded.get("top") != wanted_top:
                counts["top_bad"] += 1
            if recorded.get("count") != str(state_loc["fam"]["count"]["opt"][f"n{relevant}"]):
                counts["count_bad"] += 1
            for index, (left, right) in enumerate(features.PAIRS):
                qid = "pair" if index == 0 else f"pair_{index + 1}"
                above = order.index(left) < order.index(right)
                if recorded.get(qid) is not above:
                    counts["pair_bad"] += 1
            if recorded.get("order") != _expected_order_text(state_loc, order):
                counts["order_bad"] += 1
        for family in questions.FAMILIES:
            if counts["families"][family] == 0:
                missing.append(f"{split}: 题族 {family} 一条都没出")
        for label, key, why in (
            ("关键词个数不在取值表里", "terms_bad", "认不出关键词行，或个数不是 2/3"),
            ("候选条数不是 5", "candidates_bad", "编号行少一条或多一条"),
            ("排序口径与规范表对不上", "rule_bad", "口径不是规范形态"),
            ("第一名与候选不自洽", "top_bad", "top 的出不出或取值与按文本重算的对不上"),
            ("相关条数与候选不自洽", "count_bad", "count 的答案与按文本重算的对不上"),
            ("先后关系与候选不自洽", "pair_bad", "pair 的答案与按文本重算的对不上"),
            ("前三名顺序与候选不自洽", "order_bad", "order 的选项文案与按文本重算的对不上"),
        ):
            if counts[key]:
                missing.append(f"{split}: {counts[key]} 条的{label}（{why}）")
        report[split] = counts
    return report, missing
