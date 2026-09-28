"""完成声明核验的多语言文案。

输入：语言代码（zh / en）。
输出：解析后的 TOML 字典（提示词 + 题族文案 + 反解提示词）。
预期行为：结果带缓存；缺语言直接报错并列出可用语言。
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Final

from jc.jsontypes import JsonValue

LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales")


def available() -> list[str]:
    """输入：无。输出：可用语言列表（按文件名）。"""
    return sorted(path.stem for path in LOCALE_DIR.glob("*.toml"))


@lru_cache(maxsize=8)
def load(lang: str) -> dict[str, JsonValue]:
    """读取并缓存一个语言文件。

    输入：语言代码。输出：字典。
    预期行为：文件不存在时抛 FileNotFoundError 并列出可用语言。
    """
    path = LOCALE_DIR / f"{lang}.toml"
    if not path.exists():
        raise FileNotFoundError(f"没有语言文件 {path.name}；可用：{', '.join(available())}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def pick(seq: list[str], index: int) -> str:
    """输入：候选列表、编号。输出：第 index 个（越界回绕）。"""
    return seq[index % len(seq)]


def header(loc: dict[str, JsonValue], title: str) -> str:
    """输入：语言字典、段落标题。输出：形如「【声明】」的段落头。"""
    return str(loc["meta"]["header"]).format(title=title)


def feature_line(loc: dict[str, JsonValue], dim: str, value: str) -> str:
    """把一条注入特征渲染成提示词里的一行。

    输入：语言字典、维度名、取值。
    输出：形如「- 口气：留有余地（大概、应该、看上去）」的一行。
    预期行为：短语缺失直接抛 KeyError——静默给空字符串会让模型自由发挥，
    那就等于标签不再由特征决定。
    """
    spec = loc["feature"][dim][value]
    return str(loc["prompt"]["bullet"]).format(
        dim=loc["dims"][dim], brief=spec["brief"], hint=spec["hint"]
    )