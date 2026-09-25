"""混合式 workflow 的多语言文案。

输入：语言代码（zh / en）。输出：解析后的 TOML 字典（骨架模板 + 题面 + 词表）。
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


def word_of(loc: dict[str, Any], group: str, index: int) -> str:
    """输入：语言字典、词表名、下标。输出：词条。

    预期行为：下标越界直接报错——spec 存的是下标，词表短了就是数据错位，不能静默回绕。
    """
    items = loc["word"][group]["items"]
    if not 0 <= index < len(items):
        raise IndexError(f"词表 {group!r} 只有 {len(items)} 条，取不到下标 {index}")
    return items[index]


@lru_cache(maxsize=1)
def merged_labels() -> dict[str, str]:
    """输入：无。输出：{标签文本: 'title' / 'agenda'}（各语言合并）。

    预期行为：`Workflow.reparse` 没有语言参数，只能靠结构 + 标签文本反解；
    各语言的标签互不重叠（由 test_locales.py 锁住），所以合并表不会撞车。
    """
    out: dict[str, str] = {}
    for lang in available():
        loc = load(lang)
        for slot in ("title", "agenda"):
            label = loc["bullet"][f"{slot}_label"]
            if out.setdefault(label, slot) != slot:
                raise ValueError(f"标签 {label!r} 在两种语言里指的不是同一件事")
    return out


@lru_cache(maxsize=1)
def room_words() -> frozenset[str]:
    """输入：无。输出：所有语言里出现过的会议室名（文本卫生检查用）。

    预期行为：检查的是**全部**会议室名而不是日程里用到的那几间——
    标题里提到别的会议室同样是「夹带骨架事实」，而且顺带省掉了「本场用了哪几间」这一步。
    """
    return frozenset(
        name for lang in available() for name in load(lang)["word"]["room"]["items"]
    )


@lru_cache(maxsize=1)
def weekday_words() -> frozenset[str]:
    """输入：无。输出：所有语言里出现过的星期名（文本卫生检查用）。

    预期行为：星期几是**隐藏标签**（state 里只写日期、不写星期），
    文本里提到星期就可能与骨架矛盾，也把 `meeting_day` 那道题提前泄底，一律拒。
    """
    return frozenset(
        name for lang in available() for name in load(lang)["word"]["weekday"]["names"]
    )
