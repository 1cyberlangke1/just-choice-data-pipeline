"""结构恢复的多语言文案。

输入：语言代码（zh / en）。
输出：解析后的 TOML 字典（提示词 + 块类型表 + 提示框关键词 + 时序词 + 命令工具名 + 题族文案）。
预期行为：结果带缓存；缺语言直接报错并列出可用语言。
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Final

from jc import locale_files
from jc.jsontypes import JsonValue


LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales")

#: 要**按子键**合并的那几张表（键路径）：每个子键的值都是列表。命令族 / 提示框种类 / 时序词
#: 分组在两种语言里用的是同一套键名，所以按子键拼起来即可（`code_tool.build` 与
#: `code_tool.build` 并成一个池）。其余的表（纯文案）不参与合并。
_MERGE_TABLES: Final[tuple[tuple[str, ...], ...]] = (
    ("code_tool",),
    ("callout_word",),
    ("seq_word",),
)
#: 要**直接拼接**的那几张列表（键路径）：引用署名这种「两种语言各写一个前缀」的表。
_MERGE_LISTS: Final[tuple[tuple[str, ...], ...]] = (("mark", "attribution"),)


def _dig(node: JsonValue, path: tuple[str, ...], lang: str) -> JsonValue:
    """输入：语言字典、键路径、语言代码。输出：路径指向的值。

    预期行为：中途缺键或上一级不是表就报人话（带上语言与完整路径，便于定位是哪份 toml 写漏了）。
    """
    cur: JsonValue = node
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise ValueError(f"{lang} 的 {'.'.join(path)} 缺了一节 {key!r}")
        cur = cur[key]
    return cur


def _extend(target: list[JsonValue], pool: list[JsonValue]) -> None:
    """输入：目标列表、来源列表。输出：无（原地追加）。

    预期行为：按顺序追加**去重**后的项——两份语言文件里重复出现的工具名（`code_tool` 两边一样）
    只留一份，否则反解判据会拿到一堆重复项。
    """
    target.extend(item for item in pool if item not in target)


#: 三个函数由通用层按本包目录绑好（读取 / 缓存 / 报错 / 回绕只在那里有一份），见 `jc.locale_files.make`。
available, load, pick = locale_files.make(LOCALE_DIR)


@lru_cache(maxsize=1)
def merged() -> dict[str, JsonValue]:
    """输入：无。输出：所有语言的结构表**合并**成的一份（给 `reparse` 用）。

    预期行为：只合 `_MERGE_TABLES` 与 `_MERGE_LISTS` 里那几张表，去重保序；`code_tool` /
    `callout_word` / `seq_word` 是**按子键**合（命令族 / 提示框种类 / 时序词分组），
    `mark.attribution` 直接拼。

    为什么需要它：`Workflow.reparse` 没有语言参数，而 state 的行号与块结构是**语言中立**的
    （没有抬头可以认语言），所以反解只能拿「所有语言的标记一起认」的读法。合并的前提是
    两边的标记互不重叠——由测试钉住。
    """
    out: dict[str, JsonValue] = {}
    for lang in available():
        loc = load(lang)
        for path in _MERGE_TABLES:
            source = _dig(loc, path, lang)
            if not isinstance(source, dict):
                raise ValueError(f"{lang}.{'.'.join(path)} 不是一个表：{source!r}")
            head = path[0]
            nested = out.setdefault(head, {})
            if not isinstance(nested, dict):
                raise ValueError(f"合并表 {head} 已经不是表了：{nested!r}")
            for key, pool in source.items():
                if not isinstance(pool, list):
                    raise ValueError(f"{lang}.{'.'.join(path)}.{key} 必须是列表：{pool!r}")
                target = nested.setdefault(key, [])
                if not isinstance(target, list):
                    raise ValueError(f"合并表 {head}.{key} 已经不是列表了：{target!r}")
                _extend(target, pool)
        for path in _MERGE_LISTS:
            pool = _dig(loc, path, lang)
            if not isinstance(pool, list):
                raise ValueError(f"{lang}.{'.'.join(path)} 必须是列表：{pool!r}")
            node = out
            for key in path[:-1]:
                nxt = node.setdefault(key, {})
                if not isinstance(nxt, dict):
                    raise ValueError(f"合并表 {key} 已经不是表了：{nxt!r}")
                node = nxt
            target = node.setdefault(path[-1], [])
            if not isinstance(target, list):
                raise ValueError(f"合并表 {'.'.join(path)} 已经不是列表了：{target!r}")
            _extend(target, pool)
    return out
