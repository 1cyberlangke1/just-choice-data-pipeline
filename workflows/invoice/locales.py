"""单据 workflow 的多语言文案。

输入：语言代码（zh / en）。输出：解析后的 TOML 字典（字段标签 + 词表 + 问法 + 选项文案）。
预期行为：结果带缓存；缺语言直接报错并列出可用语言。
"""

from __future__ import annotations
from pathlib import Path
from typing import Final

from jc import locale_files
from jc.jsontypes import JsonValue


LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales")


#: 三个函数由通用层按本包目录绑好（读取 / 缓存 / 报错 / 回绕只在那里有一份），见 `jc.locale_files.make`。
available, load, pick = locale_files.make(LOCALE_DIR)


def label_of(loc: dict[str, JsonValue], slot: str, style: str) -> str:
    """输入：语言字典、槽名、标签风格。输出：这个槽在本次渲染里用的标签文本。

    预期行为：`canonical` 用第一个标签，其余风格用最后一个（别名）。
    「同一个字段换种说法」是**语义**而不是字符串匹配，也是本 workflow 的值留出之一；
    标签列表长度不足 2 时直接报错，避免别名与正名撞成同一个字符串。
    """
    labels = loc["field"][slot]["labels"]
    if len(labels) < 2:
        raise KeyError(f"字段 {slot!r} 的标签至少要两个（正名 + 别名），实际 {labels!r}")
    return labels[0] if style == "canonical" else labels[-1]


def word_of(loc: dict[str, JsonValue], group: str, index: int) -> str:
    """输入：语言字典、词表名、下标。输出：词条。

    预期行为：下标越界直接报错——spec 存的是下标，词表短了就是数据错位，不能静默回绕。
    """
    items = loc["word"][group]["items"]
    if not 0 <= index < len(items):
        raise IndexError(f"词表 {group!r} 只有 {len(items)} 条，取不到下标 {index}")
    return items[index]
