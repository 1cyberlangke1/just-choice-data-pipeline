"""改动审查的多语言文案。

输入：语言代码（zh / en）。
输出：解析后的 TOML 字典（提示词 + 题族文案 + 三张表 + 反解提示词）。
预期行为：结果带缓存；缺语言直接报错并列出可用语言。
"""

from __future__ import annotations
from pathlib import Path
from typing import Final

from jc import locale_files
from jc.jsontypes import JsonValue


LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales")

#: 格子之间的分隔符。选竖线是因为它在中英里写法相同，反解不用跨语言试。
CELL_SEP: Final[str] = "|"


#: 三个函数由通用层按本包目录绑好（读取 / 缓存 / 报错 / 回绕只在那里有一份），见 `jc.locale_files.make`。
available, load, pick = locale_files.make(LOCALE_DIR)



#: 转出去：实现只有一份，住在 `jc.locale_files`（11 家逐字同构，只留那一份）。
header = locale_files.header


def row_line(loc: dict[str, JsonValue], cells: tuple[str, ...]) -> str:
    """输入：语言字典、一行格子。输出：形如 `- credential | …` 的一行。

    预期行为：渲染与解析共用同一个前缀与分隔符，改一处两处都变——不然「写出来的行
    解不回来」这种错会两边各写一份才暴露。
    """
    return f"{loc['meta']['row_prefix']}{f' {CELL_SEP} '.join(cells)}"


#: 转出去：实现只有一份，住在 `jc.locale_files`（11 家逐字同构，只留那一份）。
feature_line = locale_files.feature_line
tagged_prompt = locale_files.tagged_prompt
