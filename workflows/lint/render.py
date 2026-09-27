"""行级规则检查 workflow 的渲染：把行计划写成「一句规则 + 一段编号片段」。

输入：`LintSpec`、语言字典。输出：契约允许的 state（多行字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本（逐字节），
所以「两次生成逐字节一致」这条能直接查。渲染规则（也是 `parse` 的读法）：
- 第一行是查询句（`[q]` 按问法编号取，`{rule}` 槽填规则短语）；
- 第二行是体裁抬头（`[lead]` 按体裁取，`{n}` 槽填行数）；
- 之后每行是 `LINE_TEMPLATE`（`{n}. {line}`），行号从 1 起。

行内容按**来源**取：命中行来自 `[hit.<规则>]` 的模板 + 槽，背景行来自 `[bg.<体裁>]`。
`long` 规则的模板就是 `{payload}`，长句写在槽里。
"""

from __future__ import annotations

from typing import Any

from .spec import LINE_TEMPLATE, LintPlan, LintSpec


def as_phrases(value: Any) -> tuple[str, ...]:
    """输入：语言文件里的一格。输出：它的短语。

    预期行为：单语言表里一格是字符串；`locales.merged()` 的合并表里一格是列表（各语言的写法），
    两种都要吃得下——`reparse` 读的是合并表。
    """
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _fill(text: str, **slots: str) -> str:
    """输入：模板、槽位取值。输出：填好的文本。预期行为：填完还剩 `{xxx}` 直接报错。"""
    out = text
    for name, value in slots.items():
        out = out.replace("{" + name + "}", value)
    if "{" in out and "}" in out:
        raise ValueError(f"渲染后还剩占位符：{out!r}")
    return out


def hit_line(loc: dict[str, Any], rule: str, genre: str, slot: int) -> str:
    """输入：语言字典、规则、体裁、槽号。输出：这一条命中行的原文。

    预期行为：模板取自 `[hit.<规则>.<体裁>].tpl`，槽值取自 `[hit.<规则>.slots].s<槽号>`；
    模板要的槽槽值里没有就直接报错（少填一个槽会让这一行悄悄变成非命中行，答案跟着错）。
    """
    table = loc["hit"][rule]
    template = str(table[genre]["tpl"])
    # `ts` 是**所有**日志体裁模板共用的槽：命中行与背景行的时刻取自同一张表，
    # 所以「带不带时刻」不携带任何命中信息（不这么写，日志体裁的命中行会全是同一个
    # 时刻，而背景行时刻各异——考生一眼就能看出哪几行是命中行）。
    values = {"ts": str(loc["ts"][slot])}
    values.update({key: str(value) for key, value in table["slots"][f"s{slot}"].items()})
    return _fill(template, **values)


def bg_line(loc: dict[str, Any], genre: str, slot: int) -> str:
    """输入：语言字典、体裁、槽号。输出：这一条背景行的原文。"""
    return str(loc["bg"][genre][slot])


def render_lines(plan: LintPlan, loc: dict[str, Any]) -> tuple[str, ...]:
    """输入：行计划、语言字典。输出：片段每一行的原文（不含行号前缀，按片段顺序）。"""
    return tuple(
        hit_line(loc, plan.rule, plan.genre, entry.slot)
        if entry.kind == "hit"
        else bg_line(loc, plan.genre, entry.slot)
        for entry in plan.entries
    )


def render_state(spec: LintSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：state 字符串（查询句 + 抬头 + 编号片段）。"""
    plan = spec.plan
    lines = render_lines(plan, loc)
    query = _fill(
        str(loc["q"][plan.phrasing % len(loc["q"])]), rule=str(loc["rule"][plan.rule])
    )
    lead = _fill(str(loc["lead"][plan.genre]), n=str(len(lines)))
    body = "\n".join(
        LINE_TEMPLATE.format(n=index + 1, line=line) for index, line in enumerate(lines)
    )
    return f"{query}\n{lead}\n{body}"
