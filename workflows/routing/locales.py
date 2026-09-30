"""工单分诊的多语言文案。

输入：语言代码（zh / en）。
输出：解析后的 TOML 字典（提示词 + 题族文案 + 三张表 + 反解提示词）。
预期行为：结果带缓存；缺语言直接报错并列出可用语言。
"""

from __future__ import annotations
from pathlib import Path
from typing import Final

from jc import locale_files
from jc.jsontypes import JsonValue


LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales")

#: 格子之间的分隔符。选竖线是因为它在中英里写法相同，反解不用跨语言试。
CELL_SEP: Final[str] = "|"


#: 三个函数由通用层按本包目录绑好（读取 / 缓存 / 报错 / 回绕只在那里有一份），见 `jc.locale_files.make`。
available, load, pick = locale_files.make(LOCALE_DIR)


def header(loc: dict[str, JsonValue], title: str) -> str:
    """输入：语言字典、段落标题。输出：形如「【工单正文】」的段落头。"""
    return str(loc["meta"]["header"]).format(title=title)


def row_line(loc: dict[str, JsonValue], cells: tuple[str, ...]) -> str:
    """输入：语言字典、一行格子。输出：形如 `- outage | 故障 | ...` 的一行。

    预期行为：渲染与解析共用同一个前缀与分隔符，改一处两处都变——不然「写出来的行
    解不回来」这种错会两边各写一份才暴露。
    """
    return f"{loc['meta']['row_prefix']}{f' {CELL_SEP} '.join(cells)}"


def feature_line(loc: dict[str, JsonValue], dim: str, value: str) -> str:
    """把一条注入特征渲染成提示词里的一行。

    输入：语言字典、维度名、取值。
    输出：形如「- 这条工单是什么问题：故障（写成「这条线现在完全用不了」那种话）」的一行。
    预期行为：短语缺失直接抛 KeyError——静默给空字符串会让模型自由发挥，
    那就等于标签不再由特征决定。
    """
    spec = loc["feature"][dim][value]
    return str(loc["prompt"]["bullet"]).format(
        dim=loc["dims"][dim], brief=spec["brief"], hint=spec["hint"]
    )
