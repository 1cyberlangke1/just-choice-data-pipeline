"""行级规则检查 workflow 的题族：把片段与命中标记翻成契约形态的题目。

输入：规则名、逐行原文、逐行命中标记、语言字典、问法编号。
输出：`LabeledQuestion` 列表。预期行为：题面**自带判据**——`gate` 题把处置办法（哪几档命中
数走哪一档）写在题面里，`line` / `count` / `spot` 题说清「命中这条规则」是什么意思，
只读 state 与题面的考生能自己答（范式 §9.8）。

四个题族：
- `line`（noul，逐行）：第 k 行命中这条规则吗——同族多题，题号由组装器加 `_2`、`_3` 后缀；
- `count`（choice，9 选项）：这段片段里有几行命中；
- `gate`（choice，3 选项）：按处置办法走哪一档（放行 / 人工复核 / 拦下）；
- `spot`（choice，8 选项，**只在恰好一行命中时出**）：是哪一行。

`spot` 只在「恰好一行命中」时出，是为了让答案唯一——多行命中时「哪一行命中」没有唯一解，
出这种题就是坏数据。它同时是一个**两阶段依赖**：出不出这一族由命中行数决定。
"""

from __future__ import annotations

from typing import Any, Final

from jc.workflow import LabeledQuestion

from . import locales
from .spec import SIZE, gate_of


def families_for(hits: int) -> tuple[str, ...]:
    """输入：命中行数。输出：这种片段应该出的题族（排序后）。

    预期行为：与 `build` 的出题规则一一对应——审计拿它算「应有题族」，所以「恰好一行命中
    才出 `spot`」这条必须写在这里，不能只写在 `build` 里。
    """
    out = ["count", "gate", "line"]
    if hits == 1:
        out.append("spot")
    return tuple(sorted(out))


#: 全部合法题族。判别器用它认「未知题族」。
FAMILIES: Final[tuple[str, ...]] = tuple(sorted({*families_for(0), *families_for(1)}))
CHOICE_FAMILIES: Final[tuple[str, ...]] = ("count", "gate", "spot")
NOUL_FAMILIES: Final[tuple[str, ...]] = ("line",)


def _choice(
    family: str, instructions: str, options: dict[str, str], answer: str
) -> LabeledQuestion:
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
    family: str, instructions: str, criteria: dict[str, str], answer: bool
) -> LabeledQuestion:
    """输入：题族、题面、判据（true / false 两行文案）、金答案。输出：一道 noul 题。"""
    return LabeledQuestion(
        family,
        {"type": "noul", "instructions": instructions, "criteria": dict(criteria)},
        answer,
        shuffle=False,
    )


def _fill(text: str, **slots: str) -> str:
    """输入：模板、槽位取值。输出：填好的文本。

    预期行为：填完还剩 `{xxx}` 直接报错——题面少填一个槽会变成一道读不懂的题，
    而答案照样发出去，这种样本比直接报错危险得多。
    """
    out = text
    for name, value in slots.items():
        out = out.replace("{" + name + "}", value)
    if "{" in out and "}" in out:
        raise ValueError(f"题面渲染后还剩占位符：{out!r}")
    return out


def _line_question(
    index: int, matched: bool, loc: dict[str, Any], phrasing: int
) -> LabeledQuestion:
    """输入：行号（从 1 起）、这一行是否命中、语言字典、问法编号。输出：`line` 题。"""
    entry = loc["fam"]["line"]
    criteria = {
        side: _fill(str(entry["criteria"][side]), n=str(index)) for side in ("true", "false")
    }
    return _noul(
        "line",
        _fill(locales.pick(entry["q"], phrasing), n=str(index)),
        criteria,
        matched,
    )


def _count_question(hits: int, loc: dict[str, Any], phrasing: int) -> LabeledQuestion:
    """输入：命中行数、语言字典、问法编号。输出：`count` 题（选项 0~8）。"""
    entry = loc["fam"]["count"]
    options = {f"c{value}": str(entry["opt"][f"c{value}"]) for value in range(SIZE + 1)}
    return _choice("count", locales.pick(entry["q"], phrasing), options, f"c{hits}")


def _gate_question(hits: int, loc: dict[str, Any], phrasing: int) -> LabeledQuestion:
    """输入：命中行数、语言字典、问法编号。输出：`gate` 题（放行 / 人工复核 / 拦下）。"""
    entry = loc["fam"]["gate"]
    options = {name: str(entry["opt"][name]) for name in ("pass", "review", "block")}
    return _choice("gate", locales.pick(entry["q"], phrasing), options, gate_of(hits))


def _spot_question(
    lines: tuple[str, ...], matched: tuple[bool, ...], loc: dict[str, Any], phrasing: int
) -> LabeledQuestion:
    """输入：逐行原文、逐行命中标记、语言字典、问法编号。输出：`spot` 题。

    预期行为：只在**恰好一行命中**时被调用（调用方保证），选项就是片段里的每一行
    （描述写成「第 n 行：原文」，所以选项之间天然互不相同）。
    """
    entry = loc["fam"]["spot"]
    options = {
        f"L{index}": _fill(str(entry["opt"]["row"]), n=str(index), line=line)
        for index, line in enumerate(lines, start=1)
    }
    return _choice(
        "spot",
        locales.pick(entry["q"], phrasing),
        options,
        f"L{matched.index(True) + 1}",
    )


def build(
    rule: str,
    lines: tuple[str, ...],
    matched: tuple[bool, ...],
    loc: dict[str, Any],
    phrasing: int,
) -> list[LabeledQuestion]:
    """输入：规则名、逐行原文、逐行命中标记、语言字典、问法编号。输出：题目 + 金答案。

    预期行为：顺序稳定（逐行题按行号，然后 count、gate、最后 spot）；`rule` 只用来校验
    （题面里的规则名由 state 承担），长度对不上直接报错。
    """
    if len(lines) != SIZE or len(matched) != SIZE:
        raise ValueError(f"片段必须是 {SIZE} 行，收到 {len(lines)} 行 / {len(matched)} 个标记")
    if rule not in loc["rule"]:
        raise KeyError(f"语言文件里没有规则 {rule!r}")
    out = [
        _line_question(index, flag, loc, phrasing)
        for index, flag in enumerate(matched, start=1)
    ]
    hits = sum(1 for flag in matched if flag)
    out.append(_count_question(hits, loc, phrasing))
    out.append(_gate_question(hits, loc, phrasing))
    if hits == 1:
        out.append(_spot_question(lines, matched, loc, phrasing))
    return out
