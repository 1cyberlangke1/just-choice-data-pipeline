"""检索 workflow 的反解：把结果页文本读回结构化字段。

输入：`state` 文本 + 标签表 + 分隔符表。输出：`ParsedPage`。
预期行为：**不复用渲染时的任何计算**——只按文本本身解析，所以能抓到字段漏写、
分隔符写错、结果被截断这类硬伤。认不出的行（标题、结果条数行）直接忽略。

这里只做**结构**反解（各格都是原文），把「原文 → 键」的翻译留给判别器——
判别器手上有语言文件，而 `Workflow.reparse` 没有语言参数。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

from . import locales

#: 语言文件 `[meta]` 里那几个标签。
META_KEYS: Final[tuple[str, ...]] = (
    "query_label",
    "kind_label",
    "since_label",
    "points_label",
    "results_label",
    "snippet_label",
    "updated_label",
)

#: 要点列表的分隔（中文用顿号、英文用逗号；反解时两种都认）。
_SEP: Final[re.Pattern[str]] = re.compile(r"[、,;]\s*")


@dataclass(frozen=True)
class ParsedResult:
    """反解出来的一条结果：全是**原始文本**。"""

    index: int
    doc_id: str
    title: str
    kind: str
    authority: str
    updated: str
    snippet: str


@dataclass(frozen=True)
class ParsedPage:
    """反解出来的一页：请求头四个字段 + 整页结果。"""

    query: str
    want_kind: str
    since: str
    points: tuple[str, ...]
    results: tuple[ParsedResult, ...]


def label_index(loc: dict[str, Any]) -> dict[str, str]:
    """输入：语言字典。输出：{标签文本: 槽名}。"""
    return {loc["meta"][key]: key for key in META_KEYS}


def merged_labels() -> dict[str, str]:
    """输入：无。输出：合并全部语言的 {标签文本: 槽名}。

    预期行为：各语言的标签**互不重叠**——重叠会让语言无关的反解把两种语言的字段串成一行
    （由 `tests/workflows/search/test_locales.py` 锁住）。
    """
    merged: dict[str, str] = {}
    for lang in locales.available():
        for text, slot in label_index(locales.load(lang)).items():
            if text in merged:
                raise ValueError(f"标签 {text!r} 在 {lang} 与别的语言里重复了")
            merged[text] = slot
    return merged


def merged_seps() -> tuple[str, ...]:
    """输入：无。输出：全部语言的标题行分隔符（去重、排序）。"""
    return tuple(sorted({locales.load(lang)["meta"]["field_sep"] for lang in locales.available()}))


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


def _strip_updated(value: str, labels: dict[str, str]) -> str:
    """输入：标题行最后一格、标签表。输出：剥掉「更新」标签后的日期文本。

    预期行为：认得出任意语言的更新标签；都认不出就原样返回（判别器会拿它去比日期，
    比不上的那种格式自然会暴露成解析错误）。
    """
    for text, slot in labels.items():
        if slot == "updated_label" and value.startswith(text):
            return value[len(text) :].strip()
    return value


def _split_head(line: str, seps: tuple[str, ...]) -> dict[str, str]:
    """输入：一条结果标题行、全部分隔符。输出：{格名: 原文}。

    预期行为：格数必须恰好五格（文档号 / 标题 / 类型 / 来源 / 更新），否则报人话——
    标题里混进分隔符会立刻在这里炸掉，而不是静默错位。
    """
    found = re.match(r"\[(\d+)\]\s*", line)
    if found is None:
        raise ValueError(f"结果行开头不是 [编号]：{line!r}")
    pattern = "|".join(re.escape(sep) for sep in seps)
    parts = re.split(pattern, line[found.end() :], maxsplit=4)
    if len(parts) != 5:
        raise ValueError(f"结果行切不出五格（拿到 {len(parts)} 格）：{line!r}")
    return {
        "index": found.group(1),
        "doc_id": parts[0].strip(),
        "title": parts[1].strip(),
        "kind": parts[2].strip(),
        "authority": parts[3].strip(),
        "updated": parts[4].strip(),
    }


def parse_page(state: str, labels: dict[str, str], seps: tuple[str, ...]) -> ParsedPage:
    """输入：state 文本、标签表、分隔符表。输出：反解结果。

    预期行为：逐行认标签；`[编号]` 开头的是结果标题行，摘要标签开头的是上一条的摘要，
    其余认得出的标签进请求头。摘要行出现在任何结果之前直接报错——那是渲染错位。
    """
    fields: dict[str, str] = {}
    heads: list[dict[str, str]] = []
    snippets: list[str] = []
    for raw in state.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("["):
            heads.append(_split_head(line, seps))
            snippets.append("")
            continue
        found = _match_label(line, labels)
        if found is None:
            continue
        slot, value = found
        if slot == "snippet_label":
            if not heads:
                raise ValueError(f"摘要行出现在任何结果之前：{line!r}")
            snippets[-1] = value
            continue
        fields[slot] = value
    results = tuple(
        ParsedResult(
            index=int(head["index"]),
            doc_id=head["doc_id"],
            title=head["title"],
            kind=head["kind"],
            authority=head["authority"],
            updated=_strip_updated(head["updated"], labels),
            snippet=snippet,
        )
        for head, snippet in zip(heads, snippets, strict=True)
    )
    return ParsedPage(
        query=fields.get("query_label", ""),
        want_kind=fields.get("kind_label", ""),
        since=fields.get("since_label", ""),
        points=tuple(tok for tok in _SEP.split(fields.get("points_label", "")) if tok),
        results=results,
    )
