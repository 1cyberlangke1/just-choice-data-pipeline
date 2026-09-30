"""记录匹配（match）workflow 的多语言文案。

输入：语言代码（zh / en）。输出：解析后的 TOML 字典（抬头标签 / 五栏栏名 / 额外栏 /
每栏每个值的三种写法 / 判据原文 / 八个题族的题面与判据）。
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
    """输入：{键: 写法表或词}。输出：{写法: 键}。

    预期行为：写法重复时直接报错——反解会把两个键认成一个，是构造 bug，不能静默取其一。
    写法表是字符串数组时逐个展开；值是字符串时按单个词收。
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
def merged() -> dict[str, JsonValue]:
    """输入：无。输出：只够**归一化**用的合并语言字典（形状与单个语言文件不同）。

    预期行为：`pack` 是各语言 `[pack]` 的并集（键 → 各语言写法），其余键照抄第一份
    ——`judges.recompute` 从 `loc` 里只取 `pack`（栏名与侧标题是 `parse.merged_tables()` 的事）。

    为什么需要它：产物按 `locales[seq % len]` 轮转渲染，审计拿到的语言未必是产物那一种；
    拿单一语言的同义词表去归一化，另一种语言的产物会整片读成「包装不在同义词表里」。
    判别器不受影响：它判的那条样本语言是确定的，另一种语言的写法本来就该算渲染写歪。
    """
    pack: dict[str, list[str]] = {}
    for lang in available():
        table = load(lang)["pack"]
        for key, value in table.items():
            bucket = pack.setdefault(str(key), [])
            for word in value if isinstance(value, list) else [value]:
                if str(word) not in bucket:
                    bucket.append(str(word))
    return {"pack": pack}
