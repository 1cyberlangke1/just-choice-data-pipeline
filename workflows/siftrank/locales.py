"""多批次排名聚合（siftrank）workflow 的多语言文案。

输入：语言代码（zh / en）。输出：解析后的 TOML 字典（四个抬头 / 排序提示 / 条目名 /
判据原文 / 四个题族的题面与选项）。
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
    """输入：{键: 名字}。输出：{名字: 键}。

    预期行为：名字重复时直接报错——反解会把两个条目认成一个，是构造 bug，不能静默取其一。
    """
    out: dict[str, JsonValue] = {}
    for key, value in table.items():
        text = str(value)
        if text in out:
            raise ValueError(f"名字 {text!r} 同时属于 {out[text]!r} 与 {key!r}")
        out[text] = key
    return out


@lru_cache(maxsize=1)
def words() -> dict[str, JsonValue]:
    """输入：无。输出：**合并各语言**之后的条目反查表（`{条目名: 条目键}`）。

    预期行为：键 → 名字的翻译在判别器里做，产物按 `locales[seq % len]` 轮转渲染，审计拿到的
    语言未必是产物那一种；只认一种语言的名字会把另一种语言的产物整片读成「表外名字」。

    跨语言指向同一个键是**允许**的（两种语言可能撞词），指向不同键则直接报错——那说明有一
    处名字写重了，反解会认错人。
    """
    table: dict[str, str] = {}
    for lang in available():
        for word, key in reverse(load(lang)["item"]).items():
            known = table.get(str(word))
            if known is not None and known != key:
                raise ValueError(f"名字 {word!r} 在多种语言里指向不同条目（{known} / {key}）")
            table[str(word)] = str(key)
    return {"item": dict(table)}
