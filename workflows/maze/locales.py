"""迷宫场景的多语言文案。

输入：语言代码（zh / en）。
输出：解析后的 TOML 字典（含方向词、图例词、题族问法与选项）。
预期行为：结果带缓存；缺语言直接报错并列出可用语言。
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales")


def available() -> list[str]:
    """列出可用语言（按文件名）。"""
    return sorted(p.stem for p in LOCALE_DIR.glob("*.toml"))


@lru_cache(maxsize=8)
def load(lang: str) -> dict[str, Any]:
    """读取并缓存一个语言文件。

    输入：语言代码。
    输出：字典。预期行为：文件不存在时抛 FileNotFoundError 并给出可用列表。
    """
    path = LOCALE_DIR / f"{lang}.toml"
    if not path.exists():
        raise FileNotFoundError(f"没有语言文件 {path.name}；可用：{', '.join(available())}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def phrasing(loc: dict[str, Any], family: str, index: int, **kwargs: str) -> str:
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


def phrasing_count(loc: dict[str, Any], family: str) -> int:
    """某题族有多少种问法。"""
    return len(loc["fam"][family]["q"])
