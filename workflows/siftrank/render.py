"""多批次排名聚合（siftrank）workflow 的渲染：把真值规格写成一屏「排序提示 + 清单 + 批次」。

输入：`RankSpec`、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本。
版式固定成「排序请求号 → 排序提示 → 条目清单 → 四个批次」，每个批次一行、
条目之间用箭头分隔（箭头是**结构**，反解按它切分，所以它进语言文件）。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from . import spec as spec_mod
from .spec import RankSpec


def item_text(loc: dict[str, JsonValue], key: str) -> str:
    """输入：语言字典、条目键。输出：这个条目的名字（该语言的写法）。"""
    return str(loc["item"][key])


def batch_text(loc: dict[str, JsonValue], batch: tuple[str, ...]) -> str:
    """输入：语言字典、一个批次（条目键序列）。输出：`A > B > C > D` 这样一行。

    预期行为：箭头取语言文件里的写法，**它同时是反解的分隔符**——两边差一个空格，
    就会把好产物整片判成坏数据（范式 §9.71 四）。
    """
    return str(loc["meta"]["arrow"]).join(item_text(loc, key) for key in batch)


def listing_text(loc: dict[str, JsonValue]) -> str:
    """输入：语言字典。输出：条目清单那一行的值（按 `spec.ITEMS` 顺序）。

    预期行为：清单顺序是**并列时展示先后的依据**，所以它必须由代码按 `ITEMS` 定，
    不能跟着批次里出现的顺序走。
    """
    return str(loc["meta"]["list_sep"]).join(item_text(loc, key) for key in spec_mod.ITEMS)


def build(spec: RankSpec, loc: dict[str, JsonValue]) -> str:
    """输入：规格、语言字典。输出：`state` 全文。

    预期行为：批次的编号从 1 起、连续；条目清单恒按 `ITEMS` 顺序写（它是并列的判据之一）。
    """
    meta = loc["meta"]
    lines = [
        f"{meta['request']}{meta['sep']}{spec.request_id}",
        f"{meta['prompt']}{meta['sep']}{loc['flavor'][spec.case.flavor]}",
        f"{meta['items']}{meta['sep']}{listing_text(loc)}",
    ]
    for index, batch in enumerate(spec.batches, start=1):
        lines.append(f"{meta['batch']} {index}{meta['sep']}{batch_text(loc, batch)}")
    return "\n".join(lines) + "\n"
