"""报销收据归类的题族：把注入特征翻成契约形态的题目。

输入：注入特征、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题面**自带材料**——归类表、上限表与档位表都印在 state 里，题面只需要指明问的是
哪一件事，只读 state 与题面的考生能自己答（范式 §9.8）。四个题族：

- `cat`（choice，4 档）：这条收据行该归到归类表里的哪一类。选项 = 四个类别（描述就是各自的
  关键词清单），答案 = 真值那一类；选项由组装器打乱，所以答案位置均匀；
- `over`（noul）：金额有没有超过**它所属类别**的单笔上限。24 组里 8 组未超、16 组超，
  真值按构造是 1/3 与 2/3；
- `tier`（score，4 档）：按档位表该走哪一档。档位下标就是答案（从 0 开始）；
- `near`（noul）：收据行里有没有出现「看着像某一类、但不在归类表里」的说法。诱饵是
  `test_ood` 的留出取值，所以同分布侧一律 False、留出侧一律 True。

**为什么 `tier` 用 score 而不是 choice**：它是**有序**的四档（自动通过 < 补票据 < 主管复核 <
财务复核），有序档位的自然形态就是评分——用 choice 会把「顺序」这个信息丢掉。三原语里
`score` 的答案本来就是等级下标（见 `jc.assemble`），这里正好用上。

**为什么没有「哪一条超限」那种反向问法**：一条样本只有一个类别与一个金额，「哪一条」在单行
收据上没有意义；四族在**每一个** case 上都能出题，10 条的小批量也稳。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue
from jc.workflow import LabeledQuestion

from . import features, locales, render
from .features import ReceiptCase, label_for

#: 题族的固定顺序：先归类，再问超限、问档位、问诱饵。
FAMILIES: Final[tuple[str, ...]] = ("cat", "over", "tier", "near")


def _instructions(loc: dict[str, JsonValue], family: str, phrasing: int) -> str:
    """输入：语言字典、题族名、问法编号。输出：这一条题面。预期行为：问法按编号轮转。"""
    templates = loc["fam"][family]["q"]
    if not isinstance(templates, list) or not templates:
        raise ValueError(f"题族 {family} 缺问法模板")
    return str(locales.pick([str(text) for text in templates], phrasing))


from jc.assemble import choice as _choice


from jc.assemble import noul as _noul


def category_text(loc: dict[str, JsonValue], cid: str) -> str:
    """输入：语言字典、类别编号。输出：归类题里该类别的**选项描述**（审计也用它比对）。

    预期行为：就是该类的关键词清单按 `word_sep` 拼起来——选项自己说清「这一类收哪些说法」，
    考生不必回头翻 state 也能分辨四个选项；审计拿同一份函数重算，两边不可能漂移。
    """
    joiner = str(loc["state"]["word_sep"])
    return joiner.join(features.keyword_table(loc)[cid])


def tier_levels(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：档位表四个档位的名字（按 `TIERS` 顺序）。"""
    return [render.tier_name(loc, tier) for tier in features.TIERS]


def tier_index(tier: str) -> int:
    """输入：档位键。输出：它在档位表里的下标（score 的答案就是它）。"""
    return features.TIERS.index(tier)


def _cat(loc: dict[str, JsonValue], case: ReceiptCase, phrasing: int) -> LabeledQuestion:
    """输入：语言字典、注入特征、问法编号。输出：`cat` 那一族的一道题。"""
    options = {
        render.category_name(loc, cid): category_text(loc, cid)
        for cid in features.CATEGORY_IDS
    }
    return _choice(
        "cat",
        _instructions(loc, "cat", phrasing),
        options,
        render.category_name(loc, case.category),
    )


def _over(loc: dict[str, JsonValue], case: ReceiptCase, phrasing: int) -> LabeledQuestion:
    """输入：语言字典、注入特征、问法编号。输出：`over` 那一族的一道题。"""
    spec = loc["fam"]["over"]
    return _noul(
        "over",
        _instructions(loc, "over", phrasing),
        {"true": str(spec["true"]), "false": str(spec["false"])},
        label_for(case).over,
    )


def _tier(loc: dict[str, JsonValue], case: ReceiptCase, phrasing: int) -> LabeledQuestion:
    """输入：语言字典、注入特征、问法编号。输出：`tier` 那一族的一道题（score）。"""
    labels = label_for(case)
    return LabeledQuestion(
        "tier",
        {
            "type": "score",
            "instructions": _instructions(loc, "tier", phrasing),
            "criteria": tier_levels(loc),
        },
        tier_index(labels.tier),
        shuffle=False,
    )


def _near(loc: dict[str, JsonValue], case: ReceiptCase, phrasing: int) -> LabeledQuestion:
    """输入：语言字典、注入特征、问法编号。输出：`near` 那一族的一道题。"""
    spec = loc["fam"]["near"]
    return _noul(
        "near",
        _instructions(loc, "near", phrasing),
        {"true": str(spec["true"]), "false": str(spec["false"])},
        label_for(case).near,
    )


def build(
    case: ReceiptCase, loc: dict[str, JsonValue], phrasing: int
) -> list[LabeledQuestion]:
    """输入：注入特征、语言字典、问法编号。输出：这一条样本的全部题目（顺序固定）。

    预期行为：题族顺序就是 `FAMILIES`，答案全部由 `label_for` 推出来——纯代码、不碰模型。
    """
    return [
        _cat(loc, case, phrasing),
        _over(loc, case, phrasing),
        _tier(loc, case, phrasing),
        _near(loc, case, phrasing),
    ]
