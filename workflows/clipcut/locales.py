"""片段筛选与去冗余（clipcut）的多语言文案。

输入：语言代码（zh / en）。输出：解析后的 TOML 字典（抬头 / 主题名表 / 高亮词表 / 判据原文 /
      四个题族的题面与选项描述）。
预期行为：结果带缓存；缺语言直接报错并列出可用语言。
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Final

from jc import locale_files
from jc.jsontypes import JsonValue


LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales")


#: 三个函数由通用层按本包目录绑好（读取 / 缓存 / 报错 / 回绕只在那里有一份），见 `jc.locale_files.make`。
available, load, pick = locale_files.make(LOCALE_DIR)


def reverse(table: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """输入：{键: 词或词表}。输出：{词: 键}。

    预期行为：同一个词落在两个键上直接报错——反解会把两个键认成一个，是构造 bug，
    不能静默取其一。词表是数组时逐个展开。
    """
    out: dict[str, JsonValue] = {}
    for key, value in table.items():
        words = value if isinstance(value, list) else [value]
        for word in words:
            if word in out:
                raise ValueError(f"写法 {word!r} 同时属于 {out[word]!r} 与 {key!r}")
            out[word] = key
    return out


@lru_cache(maxsize=1)
def words() -> dict[str, JsonValue]:
    """输入：无。输出：**合并各语言**之后的两张反查表（主题 / 高亮）。

    预期行为：键 → 原文的翻译在判别器里做，产物按 `locales[seq % len]` 轮转渲染，审计拿到的
    语言未必是产物那一种；只认一种语言的写法会把另一种语言的产物整片读成「表外写法」。

    跨语言指向同一个键是**允许**的（两种语言可能撞词），指向不同键则直接报错。
    """
    tables: dict[str, dict[str, str]] = {"topic": {}, "flag": {}}
    for lang in available():
        loc = load(lang)
        for name in ("topic", "flag"):
            for word, key in reverse(loc[name]).items():
                known = tables[name].get(str(word))
                if known is not None and known != key:
                    raise ValueError(f"写法 {word!r} 在多种语言里指向不同键（{known} / {key}）")
                tables[name][str(word)] = str(key)
    return {"topic": dict(tables["topic"]), "flag": dict(tables["flag"])}
