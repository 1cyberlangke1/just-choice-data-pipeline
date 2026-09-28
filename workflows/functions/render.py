"""函数调用 workflow 的渲染：把绑定计划写成一句自然语言请求。

输入：`CallSpec`、语言字典。输出：契约允许的 state（一句话）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一句话（逐字节），
所以「两次生成逐字节一致」这条能直接查。渲染规则（也是 `parse` 的读法）：
- `lead` 里放「意图 + 主体」（`{subject}` 槽）；
- 被提到的可选参数按各自的**分段模板**接在后面，模板里的 `{参数名}` 槽填该参数的**值短语**；
- **没被提到的参数在文本里一个词都不出现**——这正是 `stated_*` 题的 False 形态。

短语表按**参数**组织（`[val.<键>]`）：股票 / 基准这类参数按别名分 `ticker` / `name` 两张子表，
其余参数只有一张表。别名只影响短语，不影响答案。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import re
from typing import Final

from .spec import KIND_FLAG, SUBJECT_ARGS, Binding, CallPlan, CallSpec, phrase_key_of

_LEFTOVER: Final[re.Pattern[str]] = re.compile(r"\{\w+\}")


def is_aliased(table: dict[str, JsonValue]) -> bool:
    """输入：`[val.<键>]` 那张表。输出：它是不是「按别名分两张」的表。

    预期行为：股票 / 基准这类参数按别名分 `ticker` / `name` 两张子表，其余参数只有一张表。
    """
    return "ticker" in table or "name" in table


def as_phrases(value: object) -> tuple[str, ...]:
    """输入：语言文件里的一格。输出：它的短语。

    预期行为：单语言表里一格是字符串；`locales.merged()` 的合并表里一格是列表（各语言的写法），
    两种都要吃得下——`reparse` 读的是合并表。
    """
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def value_phrases(loc: dict[str, JsonValue], key: str, value: str) -> tuple[str, ...]:
    """输入：语言字典（单语言或合并表）、短语表键、取值。输出：该取值**全部写法**的短语。

    预期行为：单语言表里股票 / 基准按别名分两张子表（两种写法都要认）；合并表里每格已经是
    一个短语列表。两种形状都支持，所以 `reparse` 没有语言参数也读得动。
    """
    table = loc["val"][key]
    if is_aliased(table):
        return tuple(
            phrase
            for alias in ("ticker", "name")
            for phrase in as_phrases(table[alias][value])
        )
    return as_phrases(table[value])


def flag_phrases(loc: dict[str, JsonValue], name: str) -> tuple[str, ...]:
    """输入：语言字典（单语言或合并表）、参数名。输出：该 `flag` 参数的出现短语（全部写法）。"""
    return as_phrases(loc["flag"][name])


def value_phrase(loc: dict[str, JsonValue], key: str, value: str, alias: str) -> str:
    """输入：单语言字典、短语表键、取值、别名写法。输出：该取值在请求里要用的那条短语。

    预期行为：**渲染只接受单语言表**——合并表里一格是多种语言的写法，渲染出来会中英混排，
    所以这里一旦发现不止一条写法就直接报错。
    """
    table = loc["val"][key]
    phrases = as_phrases(table[alias][value]) if is_aliased(table) else as_phrases(table[value])
    if len(phrases) != 1:
        raise ValueError(f"渲染只接受单语言短语表：{key}/{value} 有 {len(phrases)} 条写法")
    return phrases[0]


def _fill(text: str, **slots: str) -> str:
    """输入：模板、槽位取值。输出：填好的文本。预期行为：填完还剩 `{xxx}` 直接报错。"""
    out = text
    for name, value in slots.items():
        out = out.replace("{" + name + "}", value)
    leftover = _LEFTOVER.search(out)
    if leftover:
        raise ValueError(f"渲染后还剩占位符 {leftover.group(0)!r}：{out!r}")
    return out


def join_items(parts: list[str], loc: dict[str, JsonValue]) -> str:
    """输入：若干短语、语言字典。输出：列举形态（中文顿号 + 「和」，英文逗号 + and）。"""
    marks = loc["mark"]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return str(marks["list_sep"]).join(parts[:-1]) + str(marks["list_last"]) + parts[-1]


def subject_of(plan: CallPlan, loc: dict[str, JsonValue]) -> str:
    """输入：计划、语言字典。输出：`lead` 的 `{subject}` 槽取值。

    预期行为：单主体直接写；一对主体用连接词；一组主体列举。`top_movers` 没有主体槽
    （它的参数都是可选参数，各走自己的分段模板）。
    """
    if plan.tool == "plot_price":
        return value_phrase(loc, "symbol", plan.binding("symbol").value, plan.alias)
    if plan.tool == "compare_returns":
        return join_items(
            [
                value_phrase(loc, "symbol", member, plan.alias)
                for member in plan.binding("symbols").members
            ],
            loc,
        )
    if plan.tool == "rolling_correlation":
        return join_items(
            [
                value_phrase(loc, "symbol", plan.binding("symbol").value, plan.alias),
                value_phrase(loc, "benchmark", plan.binding("benchmark").value, plan.alias),
            ],
            loc,
        )
    return ""


def segment_slots(binding: Binding, alias: str, loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：一个绑定、别名写法、语言字典。输出：该参数分段模板要填的槽。

    预期行为：`flag` 参数没有取值，填的是它自己的出现短语（`[flag]` 表）。
    """
    if binding.kind == KIND_FLAG:
        return {binding.name: str(loc["flag"][binding.name])}
    return {binding.name: value_phrase(loc, phrase_key_of(binding.name), binding.value, alias)}


def render_request(spec: CallSpec, loc: dict[str, JsonValue]) -> str:
    """输入：规格、语言字典。输出：请求文本（一句话）。

    预期行为：主体参数（`SUBJECT_ARGS`）的内容写在 `lead` 里，其余被提到的参数按
    `[req.<工具>]` 的分段模板依次接上；某个被提到的参数没有分段模板就直接报错——否则
    那句话会悄悄丢掉一个参数，而 `stated_*` 的答案就跟着错。
    """
    plan = spec.plan
    table = loc["req"][plan.tool]
    lead = table["lead"]
    pieces = [_fill(str(lead[plan.phrasing % len(lead)]), subject=subject_of(plan, loc))]
    for binding in plan.args:
        if not binding.stated or binding.name in SUBJECT_ARGS[plan.tool]:
            continue
        segment = table.get(binding.name)
        if segment is None:
            raise KeyError(f"{plan.tool} 的参数 {binding.name} 没有分段模板")
        pieces.append(_fill(str(segment), **segment_slots(binding, plan.alias, loc)))
    return "".join(pieces) + str(loc["mark"]["close"])


def render_state(spec: CallSpec, loc: dict[str, JsonValue]) -> str:
    """输入：规格、语言字典。输出：state 字符串。预期行为：就是那句请求。"""
    return render_request(spec, loc)
