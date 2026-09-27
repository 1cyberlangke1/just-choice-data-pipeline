"""技能选择 workflow 的题族：把一屏选择单翻成契约形态的题目。

输入：请求标签 / 授权 / 已占用标签 / 六行技能表 / 语言字典 / 问法编号。输出：`LabeledQuestion` 列表。
预期行为：题号就是题族名（`fits` / `pick` / `scope` / `allow` / `conflict`）；**判据
（完全一致的匹配、权限档位表、资源冲突表）整段写进每道题的题面里**（范式 §9.8）：
只读 state 与题面的读者必须能自己算出来。

五个题族：
- `fits`（noul，**总是出**）：技能表里有没有一个技能能处理这条请求；
- `pick`（choice 6，**只在有时**）：该用第几个技能；
- `scope`（choice 3，**只在有时**）：那个技能的权限范围是哪一档；
- `allow`（noul，**只在有时**）：按这份授权，能不能用它；
- `conflict`（noul，**只在有时**）：它与已占用的技能是不是碰同一类资源。

「出不出」写进 `families_for`，审计拿它算「应有题族」，不能只写在 `build` 里。
答案一律**只吃技能表里的那几格**（不是从条目键反查表），所以判别器把文本读回来重算时，
「渲染把某一格写歪」会直接变成「答案对不上」。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, Protocol

from jc.workflow import LabeledQuestion

from . import locales
from .spec import CATALOG, SCOPES, allowed, share_resource


class RowLike(Protocol):
    """技能表的一行（`spec.CatalogRow` 与 `parse.ParsedRow` 都满足它）。"""

    tag: str
    scope: str
    resource: str


CHOICE_FAMILIES: Final[tuple[str, ...]] = ("pick", "scope")
NOUL_FAMILIES: Final[tuple[str, ...]] = ("allow", "conflict", "fits")
#: 全部合法题族（判别器用它认「未知题族」）。
FAMILIES: Final[tuple[str, ...]] = tuple(sorted({*CHOICE_FAMILIES, *NOUL_FAMILIES}))


def _index_of(rows: Sequence[RowLike], tag: str, field: str) -> int:
    """输入：技能表、标签、字段名。输出：那一行的位置（从 1 起）。

    预期行为：找不到直接报人话——已占用的技能不在表里说明渲染或构造漏了一行，
    静默取第 0 行会让冲突判据跟着掷骰子。
    """
    for index, row in enumerate(rows, start=1):
        if row.tag == tag:
            return index
    raise ValueError(f"{field} 那一格认不出技能表里的行：{tag!r}")


def target_slot(rows: Sequence[RowLike], tag: str) -> int | None:
    """输入：技能表、请求标签。输出：适用场景**完全一致**的那一行（没有给 None）。

    预期行为：完全一致才匹配——干扰标签把技能标签当子串含在里面，按子串匹配会选错行。
    """
    for index, row in enumerate(rows, start=1):
        if row.tag == tag:
            return index
    return None


def families_for(tag: str, rows: Sequence[RowLike]) -> tuple[str, ...]:
    """输入：请求标签、技能表。输出：这一屏应该出的题族（排序后）。

    预期行为：与 `build` 的出题规则一一对应——`fits` 总是出；另外四族只在真的匹配到
    技能时才出（没有能用的技能时「选哪个 / 什么权限 / 给不给 / 撞不撞」都无从谈起）。
    """
    out = ["fits"]
    if target_slot(rows, tag) is not None:
        out += ["pick", "scope", "allow", "conflict"]
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

    `shuffle=False`：两个 choice 题族的选项描述都自带**自然顺序**（位置族按「第 n 个」
    编号、权限族是三档阶梯），按契约「有自然顺序的集合不许打乱」。附带的好处是答案编号
    就等于选项名的分布（构造出来就是均匀的），不再带一层打乱的随机抖动——打乱时
    `test_ood` 的 `pick` 曾偏到 0.0952，而噪声地板只有 0.0863，那是种子运气，不是构造。
    """
    if len(set(options.values())) != len(options):
        raise ValueError(f"{family} 的选项描述重复：{options}")
    if answer not in options:
        raise ValueError(f"{family} 的答案 {answer!r} 不在选项里：{sorted(options)}")
    question = {"type": "choice", "instructions": instructions, "criteria": options}
    return LabeledQuestion(family, question, answer, shuffle=False)


def _noul(
    family: str, loc: dict[str, Any], phrasing: int, answer: bool, position: int | None = None
) -> LabeledQuestion:
    """输入：题族、语言字典、问法编号、金答案、位置（可选）。输出：一道 noul 题。

    预期行为：判据（true / false 两行文案）一并写进 `criteria`——官方支持 noul 带描述，
    带上它模型才知道「真」与「假」分别意味着什么。
    """
    entry = loc["fam"][family]
    slots = {} if position is None else {"n": str(position)}
    question = {
        "type": "noul",
        "instructions": _instructions(loc, family, phrasing, **slots),
        "criteria": {"true": entry["criteria"]["true"], "false": entry["criteria"]["false"]},
    }
    return LabeledQuestion(family, question, answer, shuffle=False)


def _slot_options(rows: Sequence[RowLike], loc: dict[str, Any]) -> dict[str, str]:
    """输入：技能表、语言字典。输出：位置选项（`s1`..`s6`）。

    预期行为：描述里带上那一行的三格（适用场景 · 权限范围 · 资源），并且带上位置号——
    位置号让六个选项天然互不相同。
    """
    template = str(loc["option"]["slot"]["element"])
    out: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        out[f"s{index}"] = _fill(
            template,
            n=str(index),
            tag=str(loc["tag"][row.tag]),
            scope=str(loc["scope"][row.scope]),
            resource=str(loc["resource"][row.resource]),
        )
    return out


def build(
    tag: str,
    grant: str,
    inuse: str,
    rows: Sequence[RowLike],
    loc: dict[str, Any],
    phrasing: int,
) -> list[LabeledQuestion]:
    """输入：请求标签 / 授权 / 已占用标签 / 技能表 / 语言字典 / 问法编号。输出：题目 + 金答案。

    预期行为：这里不掷骰子（选项打乱由组装器做），所以同一屏选择单 + 同一份语言文件必得
    同一批题目。顺序：有没有能用的、选哪个、什么权限、给不给用、撞不撞。
    """
    if len(rows) != len(CATALOG):
        raise ValueError(f"技能表应该是 {len(CATALOG)} 行，收到 {len(rows)} 行")
    slot = target_slot(rows, tag)
    out: list[LabeledQuestion] = [_noul("fits", loc, phrasing, slot is not None)]
    if slot is None:
        return out
    target = rows[slot - 1]
    out.append(
        _choice(
            "pick",
            _instructions(loc, "pick", phrasing),
            _slot_options(rows, loc),
            f"s{slot}",
        )
    )
    out.append(
        _choice(
            "scope",
            _instructions(loc, "scope", phrasing, n=str(slot)),
            {name: str(loc["option"]["scope"][name]) for name in SCOPES},
            target.scope,
        )
    )
    out.append(_noul("allow", loc, phrasing, allowed(target.scope, grant), slot))
    occupied = rows[_index_of(rows, inuse, "已占用") - 1]
    out.append(
        _noul(
            "conflict",
            loc,
            phrasing,
            share_resource(target.resource, occupied.resource),
            slot,
        )
    )
    return out
