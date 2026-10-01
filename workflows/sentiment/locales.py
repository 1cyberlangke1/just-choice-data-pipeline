"""情感与情绪（sentiment）的多语言文案。

输入：语言代码（zh / en）。输出：解析后的 TOML 字典（抬头 / 区块抬头 / 场景表 / 说话人称呼 /
      情绪词表 / 五条判读规则 / 四个题族的题面与选项描述）。
预期行为：结果带缓存；缺语言直接报错并列出可用语言。

这里**没有**「合并各语言」的读法：词表与场景都印在 state 里，反解直接读 state 自己那一份
（`parse.parse_state`），不需要拿语言表去猜。语言无关的部分只有结构词元与区块抬头，
那一份合并表在 `parse.merged_tables()`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from jc import locale_files

LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales")


#: 三个函数由通用层按本包目录绑好（读取 / 缓存 / 报错 / 回绕只在那里有一份），见 `jc.locale_files.make`。
available, load, pick = locale_files.make(LOCALE_DIR)
