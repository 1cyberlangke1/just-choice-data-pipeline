"""迷宫场景的多语言文案。

输入：语言代码（zh / en）。
输出：解析后的 TOML 字典（含方向词、图例词、题族问法与选项）。
预期行为：结果带缓存；缺语言直接报错并列出可用语言。
"""

from __future__ import annotations
from pathlib import Path
from typing import Final

from jc import locale_files
from jc.jsontypes import JsonValue


LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales")


#: 三个函数由通用层按本包目录绑好（读取 / 缓存 / 报错 / 回绕只在那里有一份），见 `jc.locale_files.make`。
_api = locale_files.make(LOCALE_DIR)
available, load = _api.available, _api.load


def phrasing(loc: dict[str, JsonValue], family: str, index: int, **kwargs: str) -> str:
    """取某题族的第 index 种问法并填充占位符。

    输入：语言字典、题族名、问法编号、占位符键值。
    输出：渲染好的问法字符串。
    预期行为：编号超出范围时取模回绕；未知占位符保持原样。
    """
    templates: list[str] = loc["fam"][family]["q"]
    template = templates[index % len(templates)]
    out = template
    for key, value in kwargs.items():
        out = out.replace("{" + key + "}", value)
    return out
