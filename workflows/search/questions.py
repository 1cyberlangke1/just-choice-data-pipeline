"""检索 workflow 的题族：把真值规格翻成契约形态的题目。

输入：`SearchSpec`、语言字典、问法编号、RNG。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（`best_result` 等），每族一题；
- 判据（相关度量表、排序规则、覆盖度的算法、矛盾与同源的定义）**全部写在题面或等级表里**
  （范式 §9.8）：只读 state 与题面的读者必须能自己算出来；
- 题面里点名的编号是**题面的一部分**——判别器靠从题面读编号来重算，不回头问生成器。
"""

from __future__ import annotations

import random
from typing import Any, Final

from jc.workflow import LabeledQuestion

from . import locales, render
from . import spec as spec_mod
from .spec import SearchSpec, TruthRow

CHOICE_FAMILIES: Final[tuple[str, ...]] = ("best_result", "next_action")
NOUL_FAMILIES: Final[tuple[str, ...]] = ("answerable", "conflict", "same_source", "probe_fresh")
SCORE_FAMILIES: Final[tuple[str, ...]] = ("relevance", "covered_points")
#: 题族顺序：先选（排序 / 路由），再问单项事实，最后给等级。
FAMILIES: Final[tuple[str, ...]] = CHOICE_FAMILIES + NOUL_FAMILIES + SCORE_FAMILIES

#: 题面里点名「结果 N」的那几族（判别器从题面读编号）。
PROBE_FAMILIES: Final[tuple[str, ...]] = ("relevance", "covered_points", "probe_fresh")
#: 题面里点名「结果 i 与结果 j」的那两族。
PAIR_FAMILIES: Final[tuple[str, ...]] = ("conflict", "same_source")


def _choice(
    family: str, instructions: str, options: dict[str, str], answer: str, shuffle: bool = True
) -> LabeledQuestion:
    """输入：题族、题面、选项、金答案（选项名）、是否打乱。输出：一道 choice 题。

    预期行为：选项描述重复直接报错——那意味着两个选项看起来一样，考生没法分辨，
    是构造 bug，必须在生成时炸掉而不是产出一条坏数据。
    """
    if len(set(options.values())) != len(options):
        raise ValueError(f"{family} 的选项描述重复：{options}")
    if answer not in options:
        raise ValueError(f"{family} 的答案 {answer!r} 不在选项里：{sorted(options)}")
    question = {"type": "choice", "instructions": instructions, "criteria": options}
    return LabeledQuestion(family, question, answer, shuffle=shuffle)


def _result_options(spec: SearchSpec, loc: dict[str, Any]) -> dict[str, str]:
    """输入：规格、语言字典。输出：{选项名: 描述}——选项名是 `result_<编号>`。

    预期行为：描述带文档号与更新日期，保证**互不相同**：`both` 那一档里被点名的两条是
    同一份文档、同一个标题，只靠日期区分得开。
    """
    options: dict[str, str] = {}
    for row in spec.results:
        fields = render.topic_fields(row.topic, loc)
        title = loc["title"][row.title]["text"].format(**fields)
        options[f"result_{row.index}"] = loc["option"]["result"]["text"].format(
            doc_id=row.doc_id, title=title, updated=row.updated
        )
    return options


def _truth_rows(spec: SearchSpec, loc: dict[str, Any]) -> tuple[TruthRow, ...]:
    """输入：规格、语言字典。输出：每条的**可判定事实**（构造侧）。

    预期行为：摘要文本用语言文件的模板现渲染——和 `render` 是同一批模板，所以
    「覆盖了几个要点」在构造侧与判别侧算出来的是同一件事。判别侧另有一遍**从文本重算**
    的实现，两边对不上由 `tests/workflows/search/test_judges.py` 当场抓。
    """
    rows: list[TruthRow] = []
    for row in spec.results:
        fields = render.topic_fields(row.topic, loc)
        rows.append(
            TruthRow(
                index=row.index,
                doc_id=row.doc_id,
                topic_match=row.topic == spec.topic.name,
                kind_match=row.kind == spec.want_kind,
                fresh=row.updated >= spec.since,
                authority=row.authority,
                updated=row.updated,
                snippet=loc["snippet"][row.snippet]["text"].format(**fields),
            )
        )
    return tuple(rows)


def build(
    spec: SearchSpec, loc: dict[str, Any], phrasing: int, rng: random.Random
) -> list[LabeledQuestion]:
    """输入：规格、语言字典、问法编号、RNG。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：`best_result` **不打乱**选项——选项名是 `result_<编号>`，有自然顺序，
    而「答案落在第几个选项」等于被点名的位置本身（由 case 决定，分布均匀），
    打乱只会让题面难读（范式 §5.8 的例外：有自然顺序的集合不打乱）。
    `next_action` 打乱：三档由**描述**承载，不靠位置承载。
    """
    del rng  # 选项顺序由组装器打乱，这里不掷骰子
    fam = loc["fam"]
    topic = loc["topic"][spec.topic.name]
    points = tuple(topic["points"])
    rows = _truth_rows(spec, loc)
    by_index = {row.index: row for row in rows}
    top = spec_mod.best_of(rows)
    probe = by_index[spec.case.probe]
    left, right = (by_index[index] for index in spec.case.pair)
    noul_answers: dict[str, bool] = {
        "answerable": spec_mod.covers_of(top, points) == len(points),
        "conflict": spec_mod.conflict_of(left, right, topic["metric"], tuple(topic["values"])),
        "same_source": left.doc_id == right.doc_id,
        "probe_fresh": probe.fresh,
    }
    out: list[LabeledQuestion] = [
        _choice(
            "best_result",
            locales.pick(fam["best_result"]["q"], phrasing),
            _result_options(spec, loc),
            f"result_{top.index}",
            shuffle=False,
        ),
        _choice(
            "next_action",
            locales.pick(fam["next_action"]["q"], phrasing),
            dict(loc["option"]["next_action"]),
            spec_mod.action_of(rows, points),
        ),
    ]
    for family in NOUL_FAMILIES:
        entry = fam[family]
        instructions = locales.pick(entry["q"], phrasing)
        if family in PROBE_FAMILIES:
            instructions = instructions.format(n=spec.case.probe)
        elif family in PAIR_FAMILIES:
            instructions = instructions.format(i=spec.case.pair[0], j=spec.case.pair[1])
        out.append(
            LabeledQuestion(
                family,
                {
                    "type": "noul",
                    "instructions": instructions,
                    "criteria": {
                        "true": entry["criteria"]["true"],
                        "false": entry["criteria"]["false"],
                    },
                },
                noul_answers[family],
                shuffle=False,
            )
        )
    for family in SCORE_FAMILIES:
        entry = fam[family]
        answer = (
            spec_mod.relevance_of(probe)
            if family == "relevance"
            else spec_mod.covers_of(probe, points)
        )
        out.append(
            LabeledQuestion(
                family,
                {
                    "type": "score",
                    "instructions": locales.pick(entry["q"], phrasing).format(n=spec.case.probe),
                    "criteria": list(entry["levels"]),
                },
                answer,
                shuffle=False,
            )
        )
    return out
