"""内容审核（护栏）workflow 的渲染：把真值规格写成一屏待审内容。

输入：`GuardSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
版式固定成「抬头 → 五个字段行 → 正文若干句（每句一行）」；正文的句序由
`spec.slot_order`（素材指纹派生）决定，所以「哪句先出现」判别器能独立复算，
而它又不是条款名的函数——答案不会顺着排版漏出去。
"""

from __future__ import annotations

from typing import Any, Final

from . import spec as spec_mod
from .spec import GuardSpec

#: 字段行的渲染顺序（标签键，值从规格里取）。
FIELD_ORDER: Final[tuple[str, ...]] = (
    "content_label",
    "reach_label",
    "direction_label",
    "audience_label",
    "history_label",
)


def sentence_text(kind: str, key: str, loc: dict[str, Any], wording: str) -> str:
    """输入：槽位类型、条款或编号、语言字典、措辞。输出：这一句的文本。

    预期行为：违规句按 `wording` 包一层——判据里写明三种写法都算违规，所以包裹
    **不改答案**，只改语气；擦边句与填充句不包裹（它们本来就按判据不算违规）。
    """
    if kind == "violation":
        raw = loc["sentence"]["violation"][key]
        return loc["sentence"]["wrapper"][wording].format(s=raw)
    if kind == "mention":
        return loc["sentence"]["mention"][key]
    return loc["sentence"]["filler"][int(key)]


def body_lines(spec: GuardSpec, loc: dict[str, Any]) -> tuple[str, ...]:
    """输入：规格、语言字典。输出：正文每一句（顺序 = `spec.slot_order`）。"""
    return tuple(
        sentence_text(kind, key, loc, spec.case.wording)
        for kind, key in spec_mod.slot_order(spec.case)
    )


def build(spec: GuardSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：整屏待审内容。

    预期行为：字段值取语言文件里的原文；渲染器不做任何判断，判断全在 spec 的规则里。
    """
    meta = loc["meta"]
    colon = meta["colon"]
    values: dict[str, str] = {
        "content_label": spec.content_id,
        "reach_label": loc["reach"][spec.case.reach],
        "direction_label": loc["direction"][spec.case.direction],
        "audience_label": loc["audience"][spec.case.audience],
        "history_label": loc["history"][spec.case.history],
    }
    lines = [meta["title"]]
    lines.extend(f"{meta[key]}{colon}{values[key]}" for key in FIELD_ORDER)
    lines.append(f"{meta['body_label']}{colon}")
    lines.extend(body_lines(spec, loc))
    return "\n".join(lines)
