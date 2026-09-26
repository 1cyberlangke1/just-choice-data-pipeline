"""命令审批 workflow 的反解：把请求文本读回结构化字段。

输入：`state` 文本 + 标签表。输出：`ParsedRequest`。
预期行为：**不复用渲染时的任何计算**——只按文本本身解析，所以能抓到渲染错位、
字段漏写、命令被截断这类硬伤。认不出的行（标题、空行）直接忽略。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

from . import locales
from .spec import DIALECTS, READ_FILES

#: 标签键（语言文件 `[meta]` 里那几个）。
META_KEYS: Final[tuple[str, ...]] = (
    "cwd_label",
    "env_label",
    "goal_label",
    "reads_label",
    "cmd_label",
)

#: 已读文件列表的分隔（中文用顿号、英文用逗号；反解时两种都认）。
_SEP: Final[re.Pattern[str]] = re.compile(r"[、,;]\s*")


@dataclass(frozen=True)
class ParsedRequest:
    """反解出来的请求：全是**原始文本**（外加从执行环境认出的方言）。"""

    dialect: str
    cwd: str
    goal: str
    reads: tuple[str, ...]
    command: str


def label_index(loc: dict[str, Any]) -> dict[str, str]:
    """输入：语言字典。输出：{标签文本: 槽名}。"""
    return {loc["meta"][key]: key for key in META_KEYS}


def merged_labels() -> dict[str, str]:
    """输入：无。输出：合并全部语言的 {标签文本: 槽名}。

    预期行为：各语言的标签**互不重叠**——重叠会让语言无关的反解把两种语言的标题
    串成一行（由 `tests/workflows/shell/test_locales.py` 锁住）。
    """
    merged: dict[str, str] = {}
    for lang in locales.available():
        for text, slot in label_index(locales.load(lang)).items():
            if text in merged:
                raise ValueError(f"标签 {text!r} 在 {lang} 与别的语言里重复了")
            merged[text] = slot
    return merged


def _match_label(line: str, labels: dict[str, str]) -> tuple[str, str] | None:
    """输入：一行文本、标签表。输出：(槽名, 值)，或 None（这行不是字段行）。

    预期行为：取「行首能匹配上的**最长**标签」——只按前缀匹配时，一个标签是另一个的
    前缀就会读错槽位。标签后面的冒号（中文全角 / 英文半角）一并去掉。
    """
    best: tuple[str, str] | None = None
    for text, slot in labels.items():
        if not line.startswith(text):
            continue
        if best is None or len(text) > len(best[0]):
            best = (text, slot)
    if best is None:
        return None
    value = line[len(best[0]) :].strip().lstrip("：:").strip()
    return best[1], value


def dialect_of(value: str) -> str:
    """输入：执行环境那一行的值。输出：方言名。

    预期行为：只看**首个词元**（`bash（Linux）` / `powershell（Windows）` 都认），
    所以文案怎么翻译都不影响反解。认不出来直接报错——静默退回默认方言会让
    反解用错模板、把好数据判成坏的。
    """
    found = re.match(r"[A-Za-z]+", value.strip())
    token = found.group(0) if found else ""
    for dialect in DIALECTS:
        if token == dialect:
            return dialect
    raise ValueError(f"执行环境 {value!r} 里认不出方言（可用：{list(DIALECTS)}）")


def parse_request(state: str, labels: dict[str, str]) -> ParsedRequest:
    """输入：state 文本、标签表。输出：反解结果。

    预期行为：逐行认标签；遇到 `cmd_label` 之后的行全部拼成命令（渲染时命令单独占一行）。
    已读文件按**该方言的**已知文件池过滤——占位符与噪声自然被丢掉，
    不需要给每种语言写一条分支。
    """
    fields: dict[str, str] = {}
    command_lines: list[str] = []
    in_command = False
    for raw in state.splitlines():
        line = raw.strip()
        if not line:
            continue
        if in_command:
            command_lines.append(line)
            continue
        found = _match_label(line, labels)
        if found is None:
            continue
        slot, value = found
        if slot == "cmd_label":
            in_command = True
            if value:
                command_lines.append(value)
        else:
            fields[slot] = value
    dialect = dialect_of(fields.get("env_label", ""))
    pool = frozenset(READ_FILES[dialect])
    reads = tuple(tok for tok in _SEP.split(fields.get("reads_label", "")) if tok in pool)
    return ParsedRequest(
        dialect=dialect,
        cwd=fields.get("cwd_label", ""),
        goal=fields.get("goal_label", ""),
        reads=reads,
        command=" ".join(command_lines),
    )
