"""模型路由与降级 workflow 的题族：把真值规格翻成契约形态的题目。

输入：`RouteSpec`、语言字典、问法编号、RNG。输出：`LabeledQuestion` 列表。
预期行为：题号就是题族名（`tier` / `next` / `effort` / `is_first` / `degrade` / `cache`），
每族一题；**判据（档位表 + 首选档表 + 升级链规则 + 推理深度挪档表 + 成本与缓存规则）
整段写进每道题的题面里**（范式 §9.8）：只读 state 与题面的读者必须能自己算出来。

`next` 是**两阶段依赖**：`max` 之上没有档，所以 `reason` 这类请求根本不出这一族——
「出不出」这件事写进 `families_for`，审计拿它算「应有题族」，不能只写在 `build` 里。
"""

from __future__ import annotations

import random
from typing import Any, Final

from jc.workflow import LabeledQuestion

from . import locales
from .spec import RouteSpec, has_next

CHOICE_FAMILIES: Final[tuple[str, ...]] = ("effort", "next", "tier")
NOUL_FAMILIES: Final[tuple[str, ...]] = ("cache", "degrade", "is_first")
#: 全部合法题族（判别器用它认「未知题族」）。
FAMILIES: Final[tuple[str, ...]] = tuple(sorted({*CHOICE_FAMILIES, *NOUL_FAMILIES}))


def families_for(task: str) -> tuple[str, ...]:
    """输入：任务类型。输出：这种请求应该出的题族（排序后）。

    预期行为：与 `build` 的出题规则一一对应——`has_next` 为假时不出 `next`，
    这条必须写在这里，不能只写在 `build` 里。
    """
    out = ["cache", "degrade", "effort", "is_first", "tier"]
    if has_next(task):
        out.append("next")
    return tuple(sorted(out))


def _instructions(loc: dict[str, Any], family: str, phrasing: int) -> str:
    """输入：语言字典、题族、问法编号。输出：题面（问句 + 判据原文）。

    预期行为：判据（`loc["policy"]["text"]`）整段拼在问句后面——题面自带判据，
    考生不需要任何外部知识。
    """
    return f"{locales.pick(loc['fam'][family]['q'], phrasing)} {loc['policy']['text']}"


def _choice(
    family: str, instructions: str, options: dict[str, str], answer: str
) -> LabeledQuestion:
    """输入：题族、题面、选项、金答案（选项名）。输出：一道 choice 题。

    预期行为：选项描述重复直接报错——两个选项看起来一样，考生没法分辨，是构造 bug，
    必须在生成时炸掉而不是产出一条坏数据。
    """
    if len(set(options.values())) != len(options):
        raise ValueError(f"{family} 的选项描述重复：{options}")
    if answer not in options:
        raise ValueError(f"{family} 的答案 {answer!r} 不在选项里：{sorted(options)}")
    question = {"type": "choice", "instructions": instructions, "criteria": options}
    return LabeledQuestion(family, question, answer)


def _noul(family: str, loc: dict[str, Any], phrasing: int, answer: bool) -> LabeledQuestion:
    """输入：题族、语言字典、问法编号、金答案。输出：一道 noul 题。

    预期行为：判据（true / false 两行文案）一并写进 `criteria`——官方支持 noul 带描述，
    带上它模型才知道「真」与「假」分别意味着什么。
    """
    entry = loc["fam"][family]
    question = {
        "type": "noul",
        "instructions": _instructions(loc, family, phrasing),
        "criteria": {"true": entry["criteria"]["true"], "false": entry["criteria"]["false"]},
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def build(
    spec: RouteSpec, loc: dict[str, Any], phrasing: int, rng: random.Random
) -> list[LabeledQuestion]:
    """输入：规格、语言字典、问法编号、RNG。输出：题目 + 金答案（顺序稳定）。

    预期行为：这里不掷骰子（`rng` 是协议的一部分，选项打乱由组装器做），所以同一份规格
    + 同一份语言文件必得同一批题目。顺序：先选档与升级链，再推理深度，最后三个判断。
    """
    del rng  # 题目完全由规格与语言文件决定
    out: list[LabeledQuestion] = [
        _choice(
            "tier",
            _instructions(loc, "tier", phrasing),
            dict(loc["option"]["tier"]),
            spec.tier,
        )
    ]
    if spec.next_tier is not None:
        out.append(
            _choice(
                "next",
                _instructions(loc, "next", phrasing),
                dict(loc["option"]["next"]),
                spec.next_tier,
            )
        )
    out.append(
        _choice(
            "effort",
            _instructions(loc, "effort", phrasing),
            dict(loc["option"]["effort"]),
            spec.effort,
        )
    )
    out.extend(
        _noul(family, loc, phrasing, getattr(spec, family)) for family in NOUL_FAMILIES
    )
    return out
