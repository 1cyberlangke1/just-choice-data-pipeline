"""记录匹配（match）workflow 的题族：把真值规格翻成契约形态的题目。

输入：`MatchSpec`、语言字典、问法编号、RNG。输出：`LabeledQuestion` 列表。
预期行为：
- 题号就是题族名（`verdict` / `same_name` / `band` 等），每族一题；
- 判据（归一化对照表、三种关系、处置三条规则、档位定义）**全部写在题面里**（范式 §9.8）：
  只读 state 与题面的读者必须能自己算出来；
- choice 的选项**全部打乱**（组装器做）：三个处置结论在题面里的展示顺序没有语义，
  打乱能压掉位置捷径；选项描述自带含义，打乱不影响可读性。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import random
from typing import Final

from jc.workflow import LabeledQuestion

from . import locales
from . import spec as spec_mod
from .spec import MatchSpec

CHOICE_FAMILIES: Final[tuple[str, ...]] = ("verdict", "variant_count")
#: 五个「这一栏一致吗」的题族 → 栏名（顺序即题面里的提问顺序）。
NOUL_FIELDS: Final[dict[str, str]] = {
    "same_name": "name",
    "same_brand": "brand",
    "same_size": "size",
    "same_abv": "abv",
    "same_pack": "pack",
}
SCORE_FAMILIES: Final[tuple[str, ...]] = ("band",)
#: 题族顺序：先给处置与「几栏只是写法不同」，再逐栏问一致，最后给档位。
FAMILIES: Final[tuple[str, ...]] = CHOICE_FAMILIES + tuple(NOUL_FIELDS) + SCORE_FAMILIES
#: `variant_count` 的选项：十二种字段关系实际会产生的栏数（按构造就是 0~3 各 3 条）。
VARIANT_CHOICES: Final[tuple[int, ...]] = tuple(
    sorted({spec_mod.variant_count_of(pattern) for pattern in spec_mod.PATTERNS})
)


def rules_block(loc: dict[str, JsonValue]) -> str:
    """输入：语言字典。输出：题面末尾那段判据原文。

    预期行为：整段从语言文件的 `[rules].lines` 原样取——判据必须让只读 state 与题面的
    读者自己算得出来，所以它不能藏在判别器或生成器的代码里。
    """
    return "\n".join(loc["rules"]["lines"])


def options_of(family: str, spec: MatchSpec, loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：题族、规格、语言字典。输出：{选项名: 选项描述}。

    预期行为：`verdict` 三个选项固定（同一个商品 / 不同商品 / 人工复核），
    `variant_count` 四个选项固定（0~3 栏）；选项名与 `spec` 算出来的答案是同一套键，
    答案不在选项里会当场报错（构造 bug，不能让它进产物）。
    """
    if family == "verdict":
        return dict(loc["fam"]["verdict"]["criteria"])
    if family == "variant_count":
        return {
            f"v{count}": loc["fam"]["variant_count"]["criteria"][f"v{count}"]
            for count in VARIANT_CHOICES
        }
    raise ValueError(f"未知题族：{family!r}（选项只有 verdict 与 variant_count 两族）")


from jc.assemble import choice as _choice


def _noul(
    family: str, entry: dict[str, JsonValue], instructions: str, fill: dict[str, str], answer: bool
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


def _score(family: str, loc: dict[str, JsonValue], instructions: str, answer: int) -> LabeledQuestion:
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
    spec: MatchSpec, loc: dict[str, JsonValue], phrasing: int, rng: random.Random
) -> list[LabeledQuestion]:
    """输入：规格、语言字典、问法编号、RNG。输出：题目 + 金答案（顺序按 `FAMILIES`）。

    预期行为：这里不掷骰子（`rng` 是协议的一部分，选项打乱由组装器做），
    所以同一份规格 + 同一份语言文件必得同一批题目。
    """
    del rng  # 题目完全由规格与语言文件决定
    rules = rules_block(loc)

    def instructions(family: str, **fill: str) -> str:
        """输入：题族、题面占位符。输出：题面 + 判据原文。"""
        head = locales.pick(loc["fam"][family]["q"], phrasing).format(**fill)
        return f"{head}\n\n{rules}"

    out: list[LabeledQuestion] = [
        _choice("verdict", instructions("verdict"), options_of("verdict", spec, loc), spec.verdict),
        _choice(
            "variant_count",
            instructions("variant_count"),
            options_of("variant_count", spec, loc),
            f"v{spec.variant_count}",
        ),
    ]
    for family, field in NOUL_FIELDS.items():
        fill = {"field": loc["field"][field]}
        out.append(
            _noul(
                family,
                loc["fam"][family],
                instructions(family, **fill),
                fill,
                spec.same[spec_mod.FIELDS.index(field)],
            )
        )
    out.append(_score("band", loc, instructions("band"), spec.band))
    return out
