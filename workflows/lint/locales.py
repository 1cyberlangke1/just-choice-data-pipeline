"""行级规则检查（lint）workflow 的多语言文案。

输入：语言代码（zh / en）。输出：解析后的 TOML 字典（规则短语 + 命中行 / 背景行 + 题面）。
预期行为：结果带缓存；缺语言直接报错并列出可用语言。
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales")
#: 默认语言：构造**与语言无关**的随机计划时固定用它（规则名两种语言一致）。
DEFAULT: Final[str] = "zh"


def available() -> list[str]:
    """输入：无。输出：可用语言列表（按文件名）。"""
    return sorted(path.stem for path in LOCALE_DIR.glob("*.toml"))


@lru_cache(maxsize=8)
def load(lang: str) -> dict[str, Any]:
    """读取并缓存一个语言文件。

    输入：语言代码。输出：字典。预期行为：文件不存在时抛 FileNotFoundError 并列出可用语言。
    """
    path = LOCALE_DIR / f"{lang}.toml"
    if not path.exists():
        raise FileNotFoundError(f"没有语言文件 {path.name}；可用：{', '.join(available())}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


@lru_cache(maxsize=1)
def merged() -> dict[str, Any]:
    """输入：无。输出：全部语言合并后的字典（**形状与单个语言文件不同**，只够反解用）。

    预期行为：供**没有语言参数**的 `Workflow.reparse` 用——`rule` → 规则名 → 全部语言的
    规则短语（列表）。片段里的行内容不需要合并表：读法直接拿 state 里的原文重算命中。
    """
    rules: dict[str, list[str]] = {}
    for lang in available():
        loc = load(lang)
        for name, phrase in loc["rule"].items():
            bucket = rules.setdefault(name, [])
            if str(phrase) not in bucket:
                bucket.append(str(phrase))
    return {"rule": rules}


def pick(seq: list[str], index: int) -> str:
    """输入：候选列表、编号。输出：第 index 个（越界回绕）。

    预期行为：问法池比题族数量少时也照常工作——回绕而不是报错（题面措辞不是判据）。
    """
    return seq[index % len(seq)]
