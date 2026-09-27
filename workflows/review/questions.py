"""改动审查 workflow 的题族：把一屏改动翻成契约形态的题目。

输入：六处改动、语言字典、问法编号。输出：`LabeledQuestion` 列表。
预期行为：题号就是题族名（`flag` / `severity` / `gate` / `top`）；**判据（三条规则 +
该不该提的二维表 + 严重度表 + 门控表）整段写进每道题的题面里**（范式 §9.8）：
只读 state 与题面的读者必须能自己算出来。

四个题族：
- `flag`（noul，逐处 6 道，总是出）：这一处该不该提意见；
- `severity`（score 3 档，**只对该提的处**）：这一处的严重度是第几档；
- `gate`（choice 3，**只对该提的处**）：这一处该怎么处置；
- `top`（choice 6，**只在最严重的那一处唯一时出**）：最严重的是第几处。

「出不出」写进 `families_for`，审计拿它算「应有题族」，不能只写在 `build` 里。
"""

from __future__ import annotations

from typing import Any, Final

from jc.workflow import LabeledQuestion

from . import locales
from .spec import GATES, SLOTS, Finding, gate_of, raises_, severity_of, top_position

CHOICE_FAMILIES: Final[tuple[str, ...]] = ("gate", "top")
NOUL_FAMILIES: Final[tuple[str, ...]] = ("flag",)
SCORE_FAMILIES: Final[tuple[str, ...]] = ("severity",)
#: 全部合法题族（判别器用它认「未知题族」）。
FAMILIES: Final[tuple[str, ...]] = tuple(
    sorted({*CHOICE_FAMILIES, *NOUL_FAMILIES, *SCORE_FAMILIES})
)


def families_for(rows: tuple[Finding, ...]) -> tuple[str, ...]:
    """输入：六处改动。输出：这一屏应该出的题族（排序后）。

    预期行为：与 `build` 的出题规则一一对应——`flag` 总是出；`severity` 与 `gate`
    只要有一处该提就出；`top` 只在最严重的那一处唯一时出（并列时「最严重的是哪一处」
    没有唯一解，出这种题就是坏数据，所以它是两阶段依赖）。这条必须写在这里，
    不能只写在 `build` 里。
    """
    out = ["flag"]
    if any(raises_(row) for row in rows):
        out += ["severity", "gate"]
    if top_position(rows) is not None:
        out.append("top")
    return tuple(sorted(out))


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


def _instructions(loc: dict[str, Any], family: str, phrasing: int, **slots: str) -> str:
    """输入：语言字典、题族、问法编号、槽位取值。输出：题面（问句 + 判据原文）。

    预期行为：判据（`loc["policy"]["text"]`）整段拼在问句后面——题面自带判据，
    考生不需要任何外部知识。
    """
    question = _fill(locales.pick(loc["fam"][family]["q"], phrasing), **slots)
    return f"{question} {loc['policy']['text']}"


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


def _noul(
    family: str, loc: dict[str, Any], phrasing: int, answer: bool, position: int
) -> LabeledQuestion:
    """输入：题族、语言字典、问法编号、金答案、处号。输出：一道 noul 题。

    预期行为：判据（true / false 两行文案）一并写进 `criteria`——官方支持 noul 带描述，
    带上它模型才知道「真」与「假」分别意味着什么。
    """
    entry = loc["fam"][family]
    question = {
        "type": "noul",
        "instructions": _instructions(loc, family, phrasing, n=str(position)),
        "criteria": {"true": entry["criteria"]["true"], "false": entry["criteria"]["false"]},
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def _score(
    family: str, loc: dict[str, Any], phrasing: int, answer: int, position: int
) -> LabeledQuestion:
    """输入：题族、语言字典、问法编号、金答案（档位下标）、处号。输出：一道 score 题。

    预期行为：`criteria` 是档位描述数组，下标从 0 起、与 `spec` 算出来的档位一一对应
    （轻 = 0、中 = 1、重 = 2），所以等级数组**不能打乱**。
    """
    entry = loc["fam"][family]
    question = {
        "type": "score",
        "instructions": _instructions(loc, family, phrasing, n=str(position)),
        "criteria": list(entry["levels"]),
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def _position_options(rows: tuple[Finding, ...], loc: dict[str, Any]) -> dict[str, str]:
    """输入：六处改动、语言字典。输出：位置选项（`p1`..`p6`）。

    预期行为：描述里带上那一处的抬头三格（文件 · 变更 · 规模），并且带上处号——处号让
    六个选项天然互不相同（两处改动可能抬头三格完全一样，只靠三格会撞车）。
    """
    template = str(loc["option"]["position"]["element"])
    out: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        out[f"p{index}"] = _fill(
            template,
            n=str(index),
            file=str(loc["file"][row.file]),
            change=str(loc["change"][row.change]),
            size=str(loc["size"][row.size]),
        )
    return out


def build(
    rows: tuple[Finding, ...], loc: dict[str, Any], phrasing: int
) -> list[LabeledQuestion]:
    """输入：六处改动、语言字典、问法编号。输出：题目 + 金答案（顺序稳定）。

    预期行为：这里不掷骰子（选项打乱由组装器做），所以同一屏改动 + 同一份语言文件必得
    同一批题目。顺序：逐处该不该提（6 道）、逐处严重度（只对该提的处）、逐处门控
    （只对该提的处）、最严重的是哪一处（唯一时）。
    """
    if len(rows) != len(SLOTS):
        raise ValueError(f"改动应该是 {len(SLOTS)} 处，收到 {len(rows)} 处")
    out: list[LabeledQuestion] = [
        _noul("flag", loc, phrasing, raises_(row), index)
        for index, row in enumerate(rows, start=1)
    ]
    raised = [
        (index, row) for index, row in enumerate(rows, start=1) if raises_(row)
    ]
    for index, row in raised:
        severity = severity_of(row)
        assert severity is not None  # 该提的处一定有严重度
        out.append(_score("severity", loc, phrasing, severity - 1, index))
    gate_options = {name: str(loc["option"]["gate"][name]) for name in GATES}
    for index, row in raised:
        gate = gate_of(row)
        assert gate is not None  # 该提的处一定有门控
        out.append(
            _choice(
                "gate",
                _instructions(loc, "gate", phrasing, n=str(index)),
                gate_options,
                gate,
            )
        )
    top = top_position(rows)
    if top is not None:
        out.append(
            _choice(
                "top",
                _instructions(loc, "top", phrasing),
                _position_options(rows, loc),
                f"p{top}",
            )
        )
    return out
