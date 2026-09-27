"""模型路由与降级 workflow 的渲染：把真值规格写成一屏请求画像。

输入：`RouteSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
版式固定成「标题行 → 六个字段行 → 正文行」，每行都是「标签 + 分隔符 + 值」；
**正文那一行是判据的一部分**：任务类型只出现在正文里（画像没有「任务类型」字段），
判别器要靠读正文把任务认回来（范式 §2.4：一个维度只活在正文里，逼考生真的读题）。
"""

from __future__ import annotations

from typing import Any, Final

from .spec import RouteSpec

#: 字段行的渲染顺序（标签键，值从规格里取）。正文不在其中——它单独追加在最后。
FIELD_ORDER: Final[tuple[str, ...]] = (
    "request_label",
    "complexity_label",
    "latency_label",
    "cost_label",
    "retry_label",
    "data_label",
)


def body_text(spec: RouteSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：正文那一句话。

    预期行为：按任务类型取模板——**只讲「要做什么」**，不复述任何字段（复杂度 / 延迟 /
    成本 / 数据分级都在字段行里）。任务类型认得出，靠的是模板里那个**任务短语**。
    """
    return loc["body"][spec.case.task]


def build(spec: RouteSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：整屏画像文本。

    预期行为：字段值取「语言文件里的原文」（复杂度 / 延迟 / 成本 / 尝试记录 / 数据分级）
    或规格里算好的原文（请求号）——渲染器不做任何判断，判断全在 `spec` 的规则函数里。
    """
    meta = loc["meta"]
    colon = meta["colon"]
    values: dict[str, str] = {
        "request_label": spec.request_id,
        "complexity_label": loc["complexity"][spec.case.complexity],
        "latency_label": loc["latency"][spec.case.latency],
        "cost_label": loc["cost"][spec.case.cost],
        "retry_label": loc["retry"][spec.case.retry],
        "data_label": loc["data"][spec.case.data],
    }
    lines = [meta["title"]]
    lines.extend(f"{meta[key]}{colon}{values[key]}" for key in FIELD_ORDER)
    lines.append(f"{meta['body_label']}{colon}{body_text(spec, loc)}")
    return "\n".join(lines)
