"""记录匹配（match）workflow 的多语言文案。

输入：语言代码（zh / en）。输出：解析后的 TOML 字典（抬头标签 / 五栏栏名 / 额外栏 /
每栏每个值的三种写法 / 判据原文 / 八个题族的题面与判据）。
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
    """输入：候选列表、编号。输出：第 index 个（越界回绕）。

    预期行为：问法池比题族数量少时也照常工作——回绕而不是报错（题面措辞不是判据）。
    """
    return seq[index % len(seq)]


def reverse(table: dict[str, Any]) -> dict[str, Any]:
    """输入：{键: 写法表或词}。输出：{写法: 键}。

    预期行为：写法重复时直接报错——反解会把两个键认成一个，是构造 bug，不能静默取其一。
    写法表是字符串数组时逐个展开；值是字符串时按单个词收。
    """
    out: dict[str, Any] = {}
    for key, value in table.items():
        words = value if isinstance(value, list) else [value]
        for word in words:
            if word in out:
                raise ValueError(f"写法 {word!r} 同时属于 {out[word]!r} 与 {key!r}")
            out[word] = key
    return out
