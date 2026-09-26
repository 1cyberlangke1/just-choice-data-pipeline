"""工单分诊 workflow 的渲染：把真值规格写成一屏工单。

输入：`TicketSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
版式固定成「标题行 → 八个字段行 → 正文行」，每行都是「标签 + 分隔符 + 值」；
正文那一行**是判据的一部分**：问题类型只出现在正文里（工单没有「问题类型」字段），
判别器要靠读正文把问题类型认回来。
"""

from __future__ import annotations

from typing import Any, Final

from .spec import TicketSpec

#: 字段行的渲染顺序（标签键，值从规格里取）。
FIELD_ORDER: Final[tuple[str, ...]] = (
    "ticket_label",
    "area_label",
    "tier_label",
    "impact_label",
    "urgency_label",
    "compliance_label",
    "created_label",
    "current_label",
)


def body_text(spec: TicketSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：正文那一段话。

    预期行为：三句拼起来——问题句（嵌产品域名）+ 影响面句 + 紧急度句。
    **不复述**工单号、时间、合规标记：那些在字段行里；正文只讲「发生了什么」。
    """
    area = loc["area"][spec.case.area]
    return loc["meta"]["sent_sep"].join(
        (
            loc["body"]["issue"][spec.case.issue].format(area=area),
            loc["body"]["impact"][spec.case.impact],
            loc["body"]["urgency"][spec.case.urgency],
        )
    )


def build(spec: TicketSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：整屏工单文本。

    预期行为：字段值取「语言文件里的原文」（产品域 / 等级 / 影响面 / 紧急度 / 合规标记）
    或规格里算好的原文（工单号 / 两个时间）——渲染器不做任何判断，判断全在 spec 的规则里。
    """
    meta = loc["meta"]
    colon = meta["colon"]
    values: dict[str, str] = {
        "ticket_label": spec.ticket_id,
        "area_label": loc["area"][spec.case.area],
        "tier_label": loc["tier"][spec.case.tier],
        "impact_label": loc["impact"][spec.case.impact],
        "urgency_label": loc["urgency"][spec.case.urgency],
        "compliance_label": loc["compliance"][spec.case.compliance],
        "created_label": spec.created,
        "current_label": spec.current,
    }
    lines = [meta["title"]]
    lines.extend(f"{meta[key]}{colon}{values[key]}" for key in FIELD_ORDER)
    lines.append(f"{meta['body_label']}{colon}{body_text(spec, loc)}")
    return "\n".join(lines)
