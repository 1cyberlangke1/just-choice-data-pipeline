"""打分与评分的多语言文案。

输入：语言代码（zh / en）。
输出：解析后的 TOML 字典（提示词 + 题族文案 + 判据名 + 反解提示词）。
预期行为：结果带缓存；缺语言直接报错并列出可用语言。
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Final

from jc.jsontypes import JsonValue

LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales")

#: 格子之间的分隔符。选竖线是因为它在中英里写法相同，反解不用跨语言试。
CELL_SEP: Final[str] = "|"


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
    """输入：语言字典、段落标题。输出：形如「【待评材料】」的段落头。"""
    return str(loc["meta"]["header"]).format(title=title)


def row_line(loc: dict[str, JsonValue], cells: tuple[str, ...]) -> str:
    """输入：语言字典、一行格子。输出：形如 `- complete | 3 | 3 | 4` 的一行。

    预期行为：渲染与解析共用同一个前缀与分隔符，改一处两处都变——不然「写出来的行
    解不回来」这种错会两边各写一份才暴露。
    """
    return f"{loc['meta']['row_prefix']}{f' {CELL_SEP} '.join(cells)}"


def feature_line(loc: dict[str, JsonValue], dim: str, value: str) -> str:
    """把一条注入特征渲染成提示词里的一行。

    输入：语言字典、维度名、取值。
    输出：形如「- 材料篇幅：短（3~6 行）的一行」。
    预期行为：短语缺失直接抛 KeyError——静默给空字符串会让模型自由发挥，
    那就等于标签不再由特征决定。
    """
    spec = loc["feature"][dim][value]
    return str(loc["prompt"]["bullet"]).format(
        dim=loc["dims"][dim], brief=spec["brief"], hint=spec["hint"]
    )
