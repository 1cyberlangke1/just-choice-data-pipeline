"""检索结果判读的多语言文案。

输入：语言代码（zh / en）。
输出：解析后的 TOML 字典（提示词 + 题族文案 + 规则表 + 反解提示词）。
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



#: 转出去：实现只有一份，住在 `jc.locale_files`（11 家逐字同构，只留那一份）。
header = locale_files.header


#: 转出去：实现只有一份，住在 `jc.locale_files`（11 家逐字同构，只留那一份）。
feature_line = locale_files.feature_line
tagged_prompt = locale_files.tagged_prompt
