"""浏览器 / GUI 操作（browser）的多语言文案。

输入：语言代码（zh / en）。
输出：解析后的 TOML 字典（提示词 + 题族文案 + 角色 / 状态 / 意图词表 + 反解提示词）。
预期行为：结果带缓存；缺语言直接报错并列出可用语言。

**元素名（收件地址 / 删除 / 概览……）不在这里**——它们取自特征层 `features.LABELS` 的固定
词表，两种语言共用：那是界面快照里原样照抄的名字（真实界面上的按钮就是这几个字），
而 `wanted_label` 这条金标签必须与语言无关（反解判别器要拿它比对）。这里只管
角色名、状态名与所有说明性文字。
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


def target_line(loc: dict[str, JsonValue], label: str) -> str:
    """把「用户指的是哪个元素」渲染成提示词里的一行。

    输入：语言字典、元素名。
    输出：形如「- 用户指的是哪个元素：收件地址（那句话里必须原样写出这个名字）」的一行。
    预期行为：元素名由代码选定（`features.pick_label`），模型只能照抄，不能自己挑。
    """
    return str(loc["prompt"]["target_bullet"]).format(label=label)


def role_word(loc: dict[str, JsonValue], role: str) -> str:
    """输入：语言字典、角色词元。输出：该角色的中文 / 英文名。预期行为：未知角色抛 KeyError。"""
    return str(loc["role"][role])


def state_word(loc: dict[str, JsonValue], state: str) -> str:
    """输入：语言字典、状态词元。输出：该状态的中文 / 英文名。预期行为：未知状态抛 KeyError。"""
    return str(loc["state"][state])
