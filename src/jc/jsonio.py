"""严格 JSON 读写。

与标准库 json 的差别（全部显式拒绝，不静默放行）：
- 重复 key：默认只保留最后一个，会让 choice 选项数与编号一起错位 → object_pairs_hook 检测；
- NaN / Infinity / -Infinity：不是合法 JSON → parse_constant 拒绝；
- UTF-8 BOM：Windows 工具常写，会污染首个字段名 → 读原始字节检查；
- 写入统一 UTF-8 + LF：不加 newline="\\n" 时 Windows 文本模式会把 \\n 写成 \\r\\n。

输入/输出：loads_* 返回解析后的 Python 对象或抛 StrictJsonError；write_jsonl 落盘。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Final

from . import contract

_BOM: Final[bytes] = b"\xef\xbb\xbf"


class StrictJsonError(Exception):
    """严格解析失败。code 为 contract 里的错误码，msg 为人类可读说明。"""

    def __init__(self, code: str, msg: str) -> None:
        super().__init__(f"{code}: {msg}")
        self.code = code
        self.msg = msg


class _DupKey(Exception):
    """内部信号：JSON 对象里出现重复 key。"""

    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key


def _pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """构造 dict 时顺带检测重复 key。

    输入：同一层对象的 (key, value) 列表（json 模块按出现顺序给出）。
    输出：dict。预期行为：出现重复 key 立刻抛 _DupKey，不做「保留最后一个」。
    """
    seen: set[str] = set()
    for key, _ in pairs:
        if key in seen:
            raise _DupKey(key)
        seen.add(key)
    return dict(pairs)


def _reject_constant(name: str) -> Any:
    """拒绝 NaN / Infinity / -Infinity（它们不是合法 JSON）。

    输入：json 模块识别到的常量字面量名。
    输出：永不返回，直接抛 StrictJsonError。
    """
    raise StrictJsonError(contract.E_JSON_NAN, f"不是合法 JSON 的字面量：{name}")


def loads(text: str) -> Any:
    """严格解析一段 JSON 文本。

    输入：str（不含 BOM；含 BOM 请用 loads_bytes）。
    输出：解析后的 Python 对象。
    预期行为：重复 key → E_JSON_DUP_KEY；NaN/Infinity → E_JSON_NAN；语法错 → E_JSON_PARSE。
    """
    try:
        return json.loads(text, object_pairs_hook=_pairs_hook, parse_constant=_reject_constant)
    except _DupKey as exc:
        raise StrictJsonError(contract.E_JSON_DUP_KEY, f"重复的 key：{exc.key}") from exc
    except StrictJsonError:
        raise
    except json.JSONDecodeError as exc:
        raise StrictJsonError(contract.E_JSON_PARSE, f"第 {exc.lineno} 列 {exc.colno}：{exc.msg}") from exc


def loads_bytes(raw: bytes) -> Any:
    """严格解析字节流，先查 BOM 再按 UTF-8 解码。

    输入：文件原始字节。
    输出：解析后的 Python 对象。
    预期行为：带 UTF-8 BOM → E_BOM；非 UTF-8 → E_JSON_PARSE。
    """
    if raw.startswith(_BOM):
        raise StrictJsonError(contract.E_BOM, "文件带 UTF-8 BOM，必须去掉")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StrictJsonError(contract.E_JSON_PARSE, f"不是合法 UTF-8：{exc}") from exc
    return loads(text)


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, Any]]:
    """逐行读取 JSONL，产出 (行号, 对象)；空行跳过。

    输入：文件路径。
    输出：迭代器，行号从 1 开始。
    预期行为：任何解析失败抛 StrictJsonError（调用方负责带上行号），不静默跳过坏行。
    """
    raw = Path(path).read_bytes()
    if raw.startswith(_BOM):
        raise StrictJsonError(contract.E_BOM, "文件带 UTF-8 BOM，必须去掉")
    text = raw.decode("utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield lineno, loads(line)
        except StrictJsonError as exc:
            raise StrictJsonError(exc.code, f"第 {lineno} 行：{exc.msg}") from exc


def dumps_line(obj: Any) -> str:
    """把一行样本序列化成紧凑单行 JSON。

    输入：可序列化对象。
    输出：不含换行的 str。
    预期行为：ensure_ascii=False（中文原样），分隔符无空格，保留 dict 插入顺序（不 sort_keys）。
    """
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    """写出 JSONL（UTF-8、LF 换行）。

    输入：目标路径与可迭代的样本对象。
    输出：写出的行数。
    预期行为：显式 newline="\\n"，避免 Windows 文本模式写成 CRLF。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(dumps_line(row))
            handle.write("\n")
            count += 1
    return count


def dumps_pretty(obj: Any) -> str:
    """格式化输出（只用于报告与调试，不用于数据文件）。"""
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False)
