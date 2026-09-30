"""workflow 语言包（`workflows/<name>/locales/*.toml`）的通用读写。

输入：一个 workflow 的 `locales` 目录。
输出：`LocaleApi`——`available()` / `load(lang)` / `pick(seq, index)` 三个已经绑好该目录的函数。
预期行为：读取、缓存、缺文件报错、编号回绕这些机制**只在这里有一份**；各 workflow 的
`locales.py` 只声明「目录在哪」，其余由 `make()` 现绑——48 个包各抄一份是纯重复。

`make()` 返回的是三个闭包（NamedTuple 装着），这是本模块唯一的「手法」：把目录参数提前绑死，
调用方才能继续写 `locales.load("zh")` 这种不带目录的写法。
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Callable, Final, NamedTuple

from .jsontypes import JsonValue

#: 语言文件缓存条数：48 个 workflow × 2 种语言还有富余。
CACHE_SIZE: Final[int] = 256


class LocaleApi(NamedTuple):
    """一个 workflow 的语言包接口。

    字段：三个函数——`available()` 列语言、`load(lang)` 取字典、`pick(seq, index)` 回绕取项。
    """

    available: Callable[[], list[str]]
    load: Callable[[str], dict[str, JsonValue]]
    pick: Callable[[list[str], int], str]


def available(locale_dir: Path) -> list[str]:
    """输入：语言目录。输出：可用语言列表（按文件名）。"""
    return sorted(path.stem for path in locale_dir.glob("*.toml"))


@lru_cache(maxsize=CACHE_SIZE)
def _read(locale_dir: Path, lang: str) -> dict[str, JsonValue]:
    """输入：语言目录、语言代码。输出：解析后的字典。

    预期行为：结果带缓存（键是「目录 + 语言」，所以不同 workflow 的同名语言互不串味）；
    文件不存在时抛 FileNotFoundError 并列出该目录下可用的语言。
    """
    path = locale_dir / f"{lang}.toml"
    if not path.exists():
        raise FileNotFoundError(f"没有语言文件 {path.name}；可用：{', '.join(available(locale_dir))}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def pick(seq: list[str], index: int) -> str:
    """输入：候选列表、编号。输出：第 index 个（越界回绕）。

    预期行为：问法池比题族数量少时也照常工作——回绕而不是报错（题面措辞不是判据）。
    """
    return seq[index % len(seq)]


def make(locale_dir: Path) -> LocaleApi:
    """输入：一个 workflow 的 locales 目录。输出：绑好该目录的三个函数。

    预期行为：返回的 `load` 是闭包，调用方不必知道目录在哪；`pick` 与目录无关，直接共用。
    """

    def list_langs() -> list[str]:
        """输入：无。输出：该目录下的可用语言。"""
        return available(locale_dir)

    def load(lang: str) -> dict[str, JsonValue]:
        """输入：语言代码。输出：解析后的字典。预期行为：缺文件直接报错并列出可用语言。"""
        return _read(locale_dir, lang)

    return LocaleApi(available=list_langs, load=load, pick=pick)
