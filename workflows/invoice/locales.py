"""单据 workflow 的多语言文案。

输入：语言代码（zh / en）。输出：解析后的 TOML 字典（字段标签 + 词表 + 问法 + 选项文案）。
预期行为：结果带缓存；缺语言直接报错并列出可用语言。
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales")


def available() -> list[str]:
    """输入：无。输出：可用语言列表（按文件名）。"""
    return sorted(path.stem for path in LOCALE_DIR.glob("*.toml"))


@lru_cache(maxsize=8)
def load(lang: str) -> dict[str, Any]:
    """读取并缓存一个语言文件。

    输入：语言代码。输出：字典。预期行为：文件不存在时抛 FileNotFoundError 并列出可用语言。
    """
    path = LOCALE_DIR / f"{lang}.toml"
    if not path.exists():
        raise FileNotFoundError(f"没有语言文件 {path.name}；可用：{', '.join(available())}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def pick(seq: list[str], index: int) -> str:
    """输入：候选列表、编号。输出：第 index 个（越界回绕）。"""
    return seq[index % len(seq)]


def label_of(loc: dict[str, Any], slot: str, style: str) -> str:
    """输入：语言字典、槽名、标签风格。输出：这个槽在本次渲染里用的标签文本。

    预期行为：`canonical` 用第一个标签，其余风格用最后一个（别名）。
    「同一个字段换种说法」是**语义**而不是字符串匹配，也是本 workflow 的值留出之一；
    标签列表长度不足 2 时直接报错，避免别名与正名撞成同一个字符串。
    """
    labels = loc["field"][slot]["labels"]
    if len(labels) < 2:
        raise KeyError(f"字段 {slot!r} 的标签至少要两个（正名 + 别名），实际 {labels!r}")
    return labels[0] if style == "canonical" else labels[-1]


def word_of(loc: dict[str, Any], group: str, index: int) -> str:
    """输入：语言字典、词表名、下标。输出：词条。

    预期行为：下标越界直接报错——spec 存的是下标，词表短了就是数据错位，不能静默回绕。
    """
    items = loc["word"][group]["items"]
    if not 0 <= index < len(items):
        raise IndexError(f"词表 {group!r} 只有 {len(items)} 条，取不到下标 {index}")
    return items[index]
