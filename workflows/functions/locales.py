"""函数调用（functions）workflow 的多语言文案。

输入：语言代码（zh / en）。输出：解析后的 TOML 字典（意图 + 取值短语 + 工具 / 参数说明 + 题面）。
预期行为：结果带缓存；缺语言直接报错并列出可用语言。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Final

LOCALE_DIR: Final[Path] = Path(__file__).with_name("locales")
#: 默认语言：构造**与语言无关**的随机计划时固定用它（取值名两种语言一致）。
DEFAULT: Final[str] = "zh"


def available() -> list[str]:
    """输入：无。输出：可用语言列表（按文件名）。"""
    return sorted(path.stem for path in LOCALE_DIR.glob("*.toml"))


@lru_cache(maxsize=8)
def load(lang: str) -> dict[str, JsonValue]:
    """读取并缓存一个语言文件。

    输入：语言代码。输出：字典。预期行为：文件不存在时抛 FileNotFoundError 并列出可用语言。
    """
    path = LOCALE_DIR / f"{lang}.toml"
    if not path.exists():
        raise FileNotFoundError(f"没有语言文件 {path.name}；可用：{', '.join(available())}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


@lru_cache(maxsize=1)
def merged() -> dict[str, JsonValue]:
    """输入：无。输出：全部语言合并后的字典（**形状与单个语言文件不同**，只够反解用）。

    预期行为：供**没有语言参数**的 `Workflow.reparse` 用。
    - `intent`：工具 → 全部语言的意图短语（列表）；
    - `val`：短语表键 → 取值 → 全部语言的写法（列表，带别名的表先摊平）；
    - `flag`：参数名 → 全部语言的出现短语（列表）。

    所以读法（`render.value_phrases` / `parse`）要同时支持「单语言表（字符串 / 别名两张子表）」
    与「合并表（每格是列表）」两种形状——`reparse` 读的是后者。
    """
    intent: dict[str, list[str]] = {}
    val: dict[str, dict[str, list[str]]] = {}
    flags: dict[str, list[str]] = {}
    for lang in available():
        loc = load(lang)
        for tool, phrase in loc["intent"].items():
            bucket = intent.setdefault(tool, [])
            if str(phrase) not in bucket:
                bucket.append(str(phrase))
        for key, table in loc["val"].items():
            bucket = val.setdefault(key, {})
            for value, phrases in _iter_phrases(table):
                slot = bucket.setdefault(value, [])
                slot.extend(phrase for phrase in phrases if phrase not in slot)
        for name, phrase in loc["flag"].items():
            bucket = flags.setdefault(name, [])
            if str(phrase) not in bucket:
                bucket.append(str(phrase))
    return {"intent": intent, "val": val, "flag": flags}


def _iter_phrases(table: dict[str, JsonValue]) -> list[tuple[str, tuple[str, ...]]]:
    """输入：单语言的 `[val.<键>]` 表。输出：[(取值, 该取值的全部写法)]。

    预期行为：带别名的表把两张子表按取值合起来；不带别名的表一格就是一条写法。
    放在这里（而不是 `render`）是因为只有 `merged` 用它，而 `render` 要避免依赖本模块。
    """
    if "ticker" in table or "name" in table:
        names = sorted({value for alias in ("ticker", "name") for value in table[alias]})
        return [
            (value, tuple(str(table[alias][value]) for alias in ("ticker", "name")))
            for value in names
        ]
    return [
        (value, (phrases,) if isinstance(phrases, str) else tuple(str(p) for p in phrases))
        for value, phrases in table.items()
    ]


def pick(seq: list[str], index: int) -> str:
    """输入：候选列表、编号。输出：第 index 个（越界回绕）。

    预期行为：问法池比题族数量少时也照常工作——回绕而不是报错（题面措辞不是判据）。
    """
    return seq[index % len(seq)]
