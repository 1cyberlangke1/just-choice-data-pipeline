"""模型路由与降级的多语言文案。

输入：语言代码（zh / en）。
输出：解析后的 TOML 字典（提示词 + 题族文案 + 反解提示词）。
预期行为：结果带缓存；缺语言直接报错并列出可用语言。
"""

from __future__ import annotations
from pathlib import Path
from typing import Final

from jc import locale_files
from jc.jsontypes import JsonValue


LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales")


#: 三个函数由通用层按本包目录绑好（读取 / 缓存 / 报错 / 回绕只在那里有一份），见 `jc.locale_files.make`。
available, load, pick = locale_files.make(LOCALE_DIR)


def header(loc: dict[str, JsonValue], title: str) -> str:
    """输入：语言字典、段落标题。输出：形如「【请求】」的段落头。"""
    return str(loc["meta"]["header"]).format(title=title)


def feature_line(loc: dict[str, JsonValue], dim: str, value: str) -> str:
    """把一条注入特征渲染成提示词里的一行。

    输入：语言字典、维度名、取值。
    输出：形如「- 任务类型：写或改代码（材料里要出现具体的代码片段或改动要求）」的一行。
    预期行为：短语缺失直接抛 KeyError——静默给空字符串会让模型自由发挥，
    那就等于标签不再由特征决定。
    """
    spec = loc["feature"][dim][value]
    return str(loc["prompt"]["bullet"]).format(
        dim=loc["dims"][dim], brief=spec["brief"], hint=spec["hint"]
    )
