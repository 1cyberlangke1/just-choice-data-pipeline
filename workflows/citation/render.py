"""引文核查 workflow 的渲染：把真值规格写成一份「来源文档 + 待核查的引文」。

输入：`CitationSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
版式固定成「抬头 → 来源文档一行 → 若干小节 → 主张 / 引文 / 声称出处」：
小节行以 `[编号]` 或 `编号.` 开头，句子行缩进两格（缩进就是「命令从哪开始」那种界碑）。
"""

from __future__ import annotations

from typing import Any

from .spec import CitationSpec, Fact, Section


def fact_text(fact: Fact | None, loc: dict[str, Any]) -> str:
    """输入：一句话、语言字典。输出：它在文档里的文本。

    预期行为：`negated` 取反面的写法；`None` 直接报错——渲染阶段拿到空句子说明构造漏了，
    静默渲染成空串会产出一条「引文是空字符串」的坏数据。
    """
    if fact is None:
        raise ValueError("要渲染的句子是空的")
    entry = loc["topic"][fact.topic]["sentence"][fact.sentence]
    return entry["neg"] if fact.negated else entry["doc"]


def claim_text(spec: CitationSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：主张的**转述**文本（不含框架）。

    预期行为：取 `claim` / `claim_neg`——它必须与文档原文不同（由 `test_locales.py` 锁住），
    否则「判引文是否支持主张」就退化成字符串匹配了。
    """
    entry = loc["topic"][spec.claim.topic]["sentence"][spec.claim.sentence]
    return entry["claim_neg"] if spec.claim.negated else entry["claim"]


def truncate(text: str) -> str:
    """输入：一句原文。输出：它的**前半句**。

    预期行为：有空格的语言在词边界切（英文切出半个词很难看），没有空格的按字符数切。
    切出来必须是**真前缀**——所以它一定不等于文档里的任何整句（由 `test_locales.py` 锁住
    「没有哪一句是另一句的真前缀」）。
    """
    half = len(text) // 2
    if " " in text:
        cut = text.rfind(" ", 0, half + 1)
        if cut > 0:
            return text[:cut].rstrip()
    return text[:half].rstrip()


def quote_text(spec: CitationSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：引文那一行写出来的文本。

    预期行为：按 case 的引文写法分三种——`verbatim` 照抄原文；`truncated` 是原文的前半句；
    `reworded` 写成了**主张的转述**（模型引用自己的复述，而不是文档原文）。
    后两种都保证不等于文档里的任何整句，所以「引文在不在文档里」的答案不是靠记的。
    """
    form = spec.case.quote_form
    if form == "verbatim":
        return fact_text(spec.quote_source, loc)
    if form == "truncated":
        return truncate(fact_text(spec.quote_source, loc))
    return claim_text(spec, loc)


def heading_text(section: Section, loc: dict[str, Any]) -> str:
    """输入：小节、语言字典。输出：小节标题文本（主题 + 下标）。"""
    return loc["topic"][section.heading_topic]["headings"][section.heading]


def head_marker(index: int, doc_style: str) -> str:
    """输入：小节编号、编号写法。输出：小节行开头的标记（`[1]` 或 `1.`）。"""
    return f"[{index}]" if doc_style == "bracketed" else f"{index}."


def doc_line(spec: CitationSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：「来源文档」那一行的值（标题 + 小节数后缀）。"""
    meta = loc["meta"]
    label = loc["topic"][spec.case.topic]["label"]
    return f"{label}{meta['doc_suffix'].format(n=len(spec.sections))}"


def claim_line(spec: CitationSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：「主张」那一行的值（框架 + 转述）。"""
    frames = loc["meta"]["claim_frames"]
    return frames[spec.case.claim_style % len(frames)].format(claim=claim_text(spec, loc))


def source_line(spec: CitationSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：「声称出处」那一行的值。"""
    return f"{loc['meta']['section_word']} {spec.claimed}"


def build(spec: CitationSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：整段文本。

    预期行为：抬头 → 来源文档 → 每节（空行 + 小节行 + 两句）→ 空行 → 主张 / 引文 / 声称出处。
    标点取自语言文件（英文文档不该出现「：」）。
    """
    meta = loc["meta"]
    colon = meta["colon"]
    lines = [meta["title"], f"{meta['doc_label']}{colon}{doc_line(spec, loc)}"]
    for section in spec.sections:
        lines.append("")
        lines.append(f"{head_marker(section.index, spec.case.doc_style)} {heading_text(section, loc)}")
        for fact in section.facts:
            lines.append(f"  {fact_text(fact, loc)}")
    lines.append("")
    lines.append(f"{meta['claim_label']}{colon}{claim_line(spec, loc)}")
    lines.append(f"{meta['quote_label']}{colon}{quote_text(spec, loc)}")
    lines.append(f"{meta['source_label']}{colon}{source_line(spec, loc)}")
    return "\n".join(lines)
