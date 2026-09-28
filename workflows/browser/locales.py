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

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Final

from jc.jsontypes import JsonValue

LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales")


def available() -> list[str]:
    """输入：无。输出：可用语言列表（按文件名）。"""
    return sorted(path.stem for path in LOCALE_DIR.glob("*.toml"))


@lru_cache(maxsize=8)
def load(lang: str) -> dict[str, JsonValue]:
    """读取并缓存一个语言文件。

    输入：语言代码。输出：字典。
    预期行为：文件不存在时抛 FileNotFoundError 并列出可用语言。
    """
    path = LOCALE_DIR / f"{lang}.toml"
    if not path.exists():
        raise FileNotFoundError(f"没有语言文件 {path.name}；可用：{', '.join(available())}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def pick(seq: list[str], index: int) -> str:
    """输入：候选列表、编号。输出：第 index 个（越界回绕）。"""
    return seq[index % len(seq)]


def header(loc: dict[str, JsonValue], title: str) -> str:
    """输入：语言字典、段落标题。输出：形如「【用户请求】」的段落头。"""
    return str(loc["meta"]["header"]).format(title=title)


def feature_line(loc: dict[str, JsonValue], dim: str, value: str) -> str:
    """把一条注入特征渲染成提示词里的一行。

    输入：语言字典、维度名、取值。
    输出：形如「- 用户这一轮想干什么：往一个输入框里填东西（写成像……）」的一行。
    预期行为：短语缺失直接抛 KeyError——静默给空字符串会让模型自由发挥，
    那就等于标签不再由特征决定。
    """
    spec = loc["feature"][dim][value]
    return str(loc["prompt"]["bullet"]).format(
        dim=loc["dims"][dim], brief=spec["brief"], hint=spec["hint"]
    )


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