"""行情读数判读（quoteread）的渲染：把真值规格写成一屏「读数表」。

输入：`QuoteSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。版式固定成
「抬头（请求号）→ 读数表区块抬头 → 若干带编号的读数行」，每行是
`N. <标的名> <竖线> <读数>`：标的名与单位都查语言表，**竖线是结构**（反解按它切分，
所以它进语言文件而不是硬编码在反解里）。

**档位表 / 警戒线 / 换算规则不写进 state，写进题面**：那些是判据原文，题面必须自带
（范式 §9.8），而 state 只描述「这一屏里有什么」——读者拿 state + 题面就能自己把四个答案
算出来。

读数一律用**整数算术**拼出来（`abs(bp) // 100` 与 `abs(bp) % 100`），不经过浮点：浮点会把
`-0.45` 变成 `-0.44999999999999996`，反解再乘回去就成了另一个数。
"""

from __future__ import annotations

from jc.jsontypes import JsonValue

from .features import BP_PER_PCT, UNITS
from .spec import QuoteSpec


def name_text(loc: dict[str, JsonValue], name_id: str) -> str:
    """输入：语言字典、标的名键（`I1`~`I3` / `A1`~`A3`）。输出：这一标的的可读名。

    预期行为：键不在表里直接报错——悄悄退回键名会把「标的名没翻译」这件事藏进产物里。
    """
    table = loc["name"]
    if not isinstance(table, dict) or name_id not in table:
        raise ValueError(f"标的名键 {name_id!r} 不在语言表的 [name] 里")
    return str(table[name_id])


def unit_text(loc: dict[str, JsonValue], unit_key: str) -> str:
    """输入：语言字典、单位键（`pct` / `bp`）。输出：单位的写法原文（如 `%` / `bp`）。"""
    table = loc["unit"]
    if not isinstance(table, dict) or unit_key not in table:
        raise ValueError(f"单位键 {unit_key!r} 不在语言表的 [unit] 里（可用：{UNITS}）")
    return str(table[unit_key])


def format_value(loc: dict[str, JsonValue], bp: int, unit_key: str) -> str:
    """输入：语言字典、基准点数、单位键。输出：读数原文（带符号，如 `-1.50%` / `-150bp`）。

    预期行为：正数带 `+`、负数带 `-`、零不带符号；百分比按 `1% = 100 基准点` 展开成两位小数；
    全程整数算术，绝不经过浮点。
    """
    unit = unit_text(loc, unit_key)
    sign = "+" if bp > 0 else ("-" if bp < 0 else "")
    size = abs(bp)
    if unit_key == "pct":
        body = f"{size // BP_PER_PCT}.{size % BP_PER_PCT:02d}"
    elif unit_key == "bp":
        body = str(size)
    else:
        raise ValueError(f"未知单位键：{unit_key!r}（可用：{UNITS}）")
    return f"{sign}{body}{unit}"


def row_line(
    loc: dict[str, JsonValue], index: int, name_id: str, bp: int, unit_key: str
) -> str:
    """输入：语言字典、行号（1 起）、标的名键、基准点数、单位键。输出：这一行的原文。

    预期行为：编号从 1 起、点号与竖线是**结构**（中英共用），缩进两格只是给人看的——
    反解按 `strip()` 后的内容认编号。
    """
    meta = loc["meta"]
    return (
        f"  {index}. {name_text(loc, name_id)}"
        f" {meta['sep']} {format_value(loc, bp, unit_key)}"
    )


def build(spec: QuoteSpec, loc: dict[str, JsonValue]) -> str:
    """输入：规格、语言字典。输出：`state` 全文。

    预期行为：区块抬头用方括号包住（与其它 workflow 一致）；每一行一条读数、编号连续。
    """
    meta = loc["meta"]
    case = spec.case
    lines = [f"{meta['request']} {spec.query_id}", f"[{meta['table']}]"]
    lines.extend(
        row_line(loc, index, name_id, bp, unit_key)
        for index, (name_id, bp, unit_key) in enumerate(
            zip(case.name_ids, case.bp_values, case.unit_seq, strict=True), start=1
        )
    )
    return "\n".join(lines) + "\n"
