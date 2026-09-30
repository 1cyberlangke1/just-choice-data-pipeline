"""评测与基准 workflow 的多语言文案。

输入：语言代码（zh / en）。输出：解析后的 TOML 字典（抬头两个字段名 + 参考格标签 +
候选词表 + 判据原文 + 五族问法 + 两族等级数组 + 选项描述）。
预期行为：结果带缓存；缺语言直接报错并列出可用语言。
"""

from __future__ import annotations
from pathlib import Path
from typing import Final

from jc import locale_files


LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales")


#: 三个函数由通用层按本包目录绑好（读取 / 缓存 / 报错 / 回绕只在那里有一份），见 `jc.locale_files.make`。
available, load, pick = locale_files.make(LOCALE_DIR)
