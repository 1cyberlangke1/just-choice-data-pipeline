"""检索重排（rerank）workflow 的题族：把真值规格翻成契约形态的题目。

输入：`RerankSpec`、语言字典、问法编号、RNG。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（`first` / `ahead` / `relevance` 等），每族一题；
- 判据（四条必要条件、排序四条规则、来源权威度次序、时效线）**全部写在题面里**
  （范式 §9.8）：只读 state 与题面的读者必须能自己算出来；
- 点名了候选编号 / 位次的那几族，题面里的编号是**题面的一部分**——判别器靠从题面
  读编号来重算，不回头问生成器；
- choice 的选项**全部打乱**：四条候选在 state 里本来就有编号，选项顺序没有语义，
  打乱能压掉位置捷径；选项描述里带着编号，打乱不影响可读性。
"""

from __future__ import annotations

import random
from typing import Any, Final

from jc.workflow import LabeledQuestion

from . import locales, render
from . import spec as spec_mod
from .spec import RerankSpec

CHOICE_FAMILIES: Final[tuple[str, ...]] = ("first", "nth", "last")
NOUL_FAMILIES: Final[tuple[str, ...]] = ("ahead", "same_relevance", "on_topic")
SCORE_FAMILIES: Final[tuple[str, ...]] = ("relevance",)
#: 题族顺序：先给名次（第一 / 第 j / 最后），再问逐对与「是不是在讲这件事」，最后给档位。
FAMILIES: Final[tuple[str, ...]] = CHOICE_FAMILIES + NOUL_FAMILIES + SCORE_FAMILIES

#: 题面里点名了候选编号的那几族（判别器从题面读编号）。
NAMED_FAMILIES: Final[tuple[str, ...]] = (
    "ahead",
    "same_relevance",
    "on_topic",
    "relevance",
)
#: 题面里点名了位次的那一族（判别器从题面读位次）。
POSITION_FAMILIES: Final[tuple[str, ...]] = ("nth",)


def rules_block(spec: RerankSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：题面末尾那段判据原文。

    预期行为：四条必要条件里的主题 / 方面按**这一条样本**的查询填；时效线取
    `spec_mod.DEADLINE`（不是语言文件里另写一份，避免两处对不上）；来源次序按
    `SOURCE_RANK` 生成，同理。
    """
    rules = loc["rules"]
    case = spec.case
    ordered = sorted(spec_mod.SOURCE_RANK, key=lambda key: -spec_mod.SOURCE_RANK[key])
    sources = " > ".join(loc["source"][key] for key in ordered)
    return "\n".join(
        (
            rules["conditions_head"],
            rules["condition_topic"].format(topic=loc["topic"][case.topic]),
            rules["condition_aspect"].format(aspect=loc["aspect"][case.aspect]),
            rules["condition_fresh"].format(deadline=spec_mod.DEADLINE),
            rules["condition_example"],
            rules["order_head"],
            rules["order_relevance"],
            rules["order_source"].format(sources=sources),
            rules["order_day"],
            rules["order_index"],
        )
    )


def options_of(spec: RerankSpec, loc: dict[str, Any]) -> dict[str, str]:
    """输入：规格、语言字典。输出：{选项名: 选项描述}（四条候选各一个）。

    预期行为：选项名是编号的小写形式（`c1`），描述是「编号 · 来源 · 更新日期」——
    描述里带着编号，打乱之后仍然能对上；四条描述必然互不相同（编号就不同）。
    """
    case = spec.case
    out: dict[str, str] = {}
    for index, cid in enumerate(render.CAND_IDS):
        cand = case.shape.cands[index]
        out[cid.lower()] = loc["option"]["cand"].format(
            cid=cid, source=loc["source"][cand.source], day=cand.day
        )
    return out


def _choice(family: str, instructions: str, options: dict[str, str], answer: str) -> LabeledQuestion:
    """输入：题族、题面、选项、金答案（选项名）。输出：一道 choice 题。

    预期行为：选项描述重复直接报错——两个选项看起来一样，考生没法分辨，是构造 bug。
    """
    if len(set(options.values())) != len(options):
        raise ValueError(f"{family} 的选项描述重复：{options}")
    if answer not in options:
        raise ValueError(f"{family} 的答案 {answer!r} 不在选项里：{sorted(options)}")
    return LabeledQuestion(
        family, {"type": "choice", "instructions": instructions, "criteria": options}, answer
    )


def _noul(
    family: str, entry: dict[str, Any], instructions: str, fill: dict[str, str], answer: bool
) -> LabeledQuestion:
    """输入：题族、词条、题面、填充表、金答案。输出：一道 noul 题。

    预期行为：判据（true / false 两行文案）一并写进 `criteria`——官方支持 noul 带描述，
    带上它模型才知道「真」与「假」分别意味着什么。
    """
    question = {
        "type": "noul",
        "instructions": instructions,
        "criteria": {key: text.format(**fill) for key, text in entry["criteria"].items()},
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def _score(family: str, loc: dict[str, Any], instructions: str, answer: int) -> LabeledQuestion:
    """输入：题族、语言字典、题面、金答案（档位下标）。输出：一道 score 题。

    预期行为：`criteria` 是档位描述数组，下标从 0 起、与 `spec` 算出来的档位一一对应。
    """
    question = {
        "type": "score",
        "instructions": instructions,
        "criteria": list(loc["fam"][family]["levels"]),
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def build(
    spec: RerankSpec, loc: dict[str, Any], phrasing: int, rng: random.Random
) -> list[LabeledQuestion]:
    """输入：规格、语言字典、问法编号、RNG。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：这里不掷骰子（`rng` 是协议的一部分，选项打乱由组装器做），
    所以同一份规格 + 同一份语言文件必得同一批题目。
    """
    del rng  # 题目完全由规格与语言文件决定
    case = spec.case
    shape = case.shape
    rules = rules_block(spec, loc)
    options = options_of(spec, loc)

    def instructions(family: str, **fill: str) -> str:
        """输入：题族、题面占位符。输出：题面 + 判据原文。"""
        head = locales.pick(loc["fam"][family]["q"], phrasing).format(**fill)
        return f"{head}\n\n{rules}"

    out: list[LabeledQuestion] = [
        _choice(
            "first",
            instructions("first"),
            options,
            render.cand_id(spec.first()).lower(),
        ),
        _choice(
            "nth",
            instructions("nth", nth=str(shape.nth)),
            options,
            render.cand_id(spec.nth()).lower(),
        ),
        _choice(
            "last",
            instructions("last"),
            options,
            render.cand_id(spec.last()).lower(),
        ),
    ]
    left, right = shape.ahead
    pair = {"left": render.cand_id(left), "right": render.cand_id(right)}
    out.append(
        _noul(
            "ahead",
            loc["fam"]["ahead"],
            instructions("ahead", **pair),
            pair,
            spec.ahead(),
        )
    )
    left, right = shape.tie
    pair = {"left": render.cand_id(left), "right": render.cand_id(right)}
    out.append(
        _noul(
            "same_relevance",
            loc["fam"]["same_relevance"],
            instructions("same_relevance", **pair),
            pair,
            spec.same_relevance(),
        )
    )
    probe = {"cand": render.cand_id(shape.probe)}
    out.append(
        _noul(
            "on_topic",
            loc["fam"]["on_topic"],
            instructions("on_topic", **probe),
            probe,
            spec.on_topic(),
        )
    )
    out.append(
        _score(
            "relevance",
            loc,
            instructions("relevance", **probe),
            spec.probe_relevance(),
        )
    )
    return out
