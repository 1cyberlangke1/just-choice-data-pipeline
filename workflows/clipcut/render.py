"""片段筛选与去冗余（clipcut）的渲染：把真值规格写成一屏「片段表」。

输入：`ClipSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。版式固定成
「抬头（请求号）→ 片段表区块抬头 → 若干带编号的片段行」，每行是
`N. <主题名>（<高亮词>）`：主题名与高亮词都查语言表，**圆括号是结构**（反解按它切分，
所以它进语言文件而不是硬编码在反解里）。

**保留规则不写进 state，写进题面**：规则是判据原文，题面必须自带（范式 §9.8），而 state
只描述「这一屏里有什么」——读者拿 state + 题面就能自己把四个答案算出来。
"""

from __future__ import annotations

from jc.jsontypes import JsonValue

from .spec import ClipSpec


def topic_text(loc: dict[str, JsonValue], topic: str) -> str:
    """输入：语言字典、主题键（`T1`~`T5`）。输出：这一主题的可读名。

    预期行为：键不在表里直接报错——悄悄退回键名会把「主题名没翻译」这件事藏进产物里。
    """
    table = loc["topic"]
    if not isinstance(table, dict) or topic not in table:
        raise ValueError(f"主题 {topic!r} 不在语言表的 [topic] 里")
    return str(table[topic])


def flag_text(loc: dict[str, JsonValue], flag: bool) -> str:
    """输入：语言字典、这一段有没有高亮点。输出：高亮词的原文。"""
    table = loc["flag"]
    if not isinstance(table, dict):
        raise ValueError("语言表的 [flag] 必须是对象（yes / no 两个词）")
    return str(table["yes" if flag else "no"])


def segment_line(
    loc: dict[str, JsonValue], index: int, topic: str, flag: bool
) -> str:
    """输入：语言字典、片段号（1 起）、主题键、有没有高亮点。输出：这一行的原文。

    预期行为：编号从 1 起、点号是**结构**（中英共用 `N.`），缩进两格只是给人看的——
    反解按 `strip()` 后的内容认编号。
    """
    meta = loc["meta"]
    return (
        f"  {index}. {topic_text(loc, topic)}"
        f"{meta['paren_open']}{flag_text(loc, flag)}{meta['paren_close']}"
    )


def build(spec: ClipSpec, loc: dict[str, JsonValue]) -> str:
    """输入：规格、语言字典。输出：`state` 全文。

    预期行为：区块抬头用方括号包住（与其它 workflow 一致）；每一段一行、编号连续。
    """
    meta = loc["meta"]
    case = spec.case
    lines = [f"{meta['request']} {spec.query_id}", f"[{meta['segments']}]"]
    lines.extend(
        segment_line(loc, index, topic, flag)
        for index, (topic, flag) in enumerate(
            zip(case.topic_seq, case.flag_seq, strict=True), start=1
        )
    )
    return "\n".join(lines) + "\n"
