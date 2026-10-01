"""有界任务推进（planstep）的多语言文案。

输入：语言代码（zh / en）。输出：解析后的 TOML 字典（抬头 / 区块抬头 / 任务名表 / 状态词 /
      前置写法 / 四条推进规则 / 四个题族的题面与选项描述）。
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
    """输入：{键: 词}。输出：{词: 键}。

    预期行为：同一个词落在两个键上直接报错——反解会把两个键认成一个，是构造 bug，
    不能静默取其一。
    """
    out: dict[str, JsonValue] = {}
    for key, value in table.items():
        word = str(value)
        if word in out:
            raise ValueError(f"写法 {word!r} 同时属于 {out[word]!r} 与 {key!r}")
        out[word] = key
    return out


@lru_cache(maxsize=1)
def merged() -> dict[str, JsonValue]:
    """输入：无。输出：**合并各语言**之后的结构词元表。

    预期行为：反解拿不到「产物是哪一种语言」，所以状态词、前置写法、任务名都合并成集合；
    两个语言撞词是**允许**的（合并后是同一个集合），但同一格里的写法不能互相为前缀
    ——那会让反解把「待开始」认成「待开」这类更短的标签。
    """
    status: set[str] = set()
    deps_none: set[str] = set()
    deps_prefix: set[str] = set()
    tasks: dict[str, str] = {}
    for lang in available():
        loc = load(lang)
        status.update(str(value) for value in loc["status"].values())
        deps_none.add(str(loc["meta"]["deps_none"]))
        deps_prefix.add(str(loc["meta"]["deps_prefix"]))
        for word, key in reverse(loc["task"]).items():
            known = tasks.get(str(word))
            if known is not None and known != key:
                raise ValueError(f"任务名 {word!r} 在多种语言里指向不同键（{known} / {key}）")
            tasks[str(word)] = str(key)
    return {
        "status": sorted(status),
        "deps_none": sorted(deps_none),
        "deps_prefix": sorted(deps_prefix),
        "task": dict(tasks),
    }
