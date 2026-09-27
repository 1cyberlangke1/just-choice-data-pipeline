"""结构恢复（autoformat）workflow 的多语言文案。

输入：语言代码（zh / en）。输出：解析后的 TOML 字典（槽位 + 模板 + 问法 + 选项描述）。
预期行为：结果带缓存；缺语言直接报错并列出可用语言。
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any, Final

LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales")
#: 默认语言：构造**与语言无关**的随机计划时固定用它（槽位名两种语言一致）。
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
    """输入：无。输出：全部语言合并后的字典（形状与单个语言文件一致，只含反解要用的表）。

    预期行为：供**没有语言参数**的 `Workflow.reparse` 用。关键词 / 命令 / 时序词取并集；
    同一个关键词落在两个种类里，`parse.callout_index` 会直接报错——那会让反解出现歧义，
    宁可炸也不要静默取一个。
    """
    callout: dict[str, list[str]] = {}
    code: dict[str, list[str]] = {}
    seq: list[str] = []
    attribution = ""
    for lang in available():
        loc = load(lang)
        for kind, words in loc["callout_word"].items():
            bucket = callout.setdefault(kind, [])
            bucket.extend(word for word in words if word not in bucket)
        for family, pool in loc["code"].items():
            bucket = code.setdefault(family, [])
            bucket.extend(template for template in pool if template not in bucket)
        seq.extend(word for word in loc["seq_word"]["step"] if word not in seq)
        attribution = str(loc["mark"]["attribution"])
    return {
        "callout_word": callout,
        "code": code,
        "seq_word": {"step": seq},
        "mark": {"attribution": attribution},
    }


def pick(seq: list[str], index: int) -> str:
    """输入：候选列表、编号。输出：第 index 个（越界回绕）。

    预期行为：问法池比题族数量少时也照常工作——回绕而不是报错（题面措辞不是判据）。
    """
    return seq[index % len(seq)]
