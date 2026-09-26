"""检索重排（rerank）workflow 的反解：把一屏重排请求读回结构化字段。

输入：`state` 文本 + 标签表。输出：`ParsedList`。
预期行为：**不复用渲染时的任何计算**——只按文本本身解析，所以能抓到查询漏写、
候选块缺行、标签写错这类硬伤。抬头那行认不出，直接忽略。
这里只做**结构**反解（查询原文、每条候选的来源 / 日期 / 正文各行），
把「原文 → 键」的翻译与模板反解留给判别器——判别器手上有语言文件，
而 `Workflow.reparse` 没有语言参数。

候选表头的识别是**语言无关**的：一行里含 `|` 且第一段形如「字母 + 数字」就算表头，
所以中英两版共用一套反解逻辑，不必为每种语言写一条正则。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

from . import locales

#: 语言文件 `[meta]` 里那几个标签（反解要认全，缺一格就报错）。
META_KEYS: Final[tuple[str, ...]] = (
    "query_label",
    "cand_label",
    "source_label",
    "day_label",
)

#: 分隔符：中英两版各用一种，反解两种都认。
COLONS: Final[tuple[str, ...]] = ("：", ":")

#: 候选表头的字段分隔符。
SEPARATOR: Final[str] = "|"

#: 表头第一段的形状：`C1` / `C2`……（语言无关）。
HEADER_ID: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z]+\d+$")


@dataclass(frozen=True)
class ParsedCand:
    """反解出来的一条候选：来源与日期是**原始文本**，正文是逐行的元组。"""

    cid: str
    source: str
    day: str
    body: tuple[str, ...]


@dataclass(frozen=True)
class ParsedList:
    """反解出来的一屏重排请求：查询原文 + 若干候选。"""

    query: str
    cands: tuple[ParsedCand, ...]


def label_index(loc: dict[str, Any]) -> dict[str, str]:
    """输入：语言字典。输出：{标签原文: 槽位名}（只收 `META_KEYS` 里那几个）。

    预期行为：两个标签文案相同时报错——否则反解会把两格认成一格。
    """
    out: dict[str, str] = {}
    for key in META_KEYS:
        text = loc["meta"][key]
        if text in out:
            raise ValueError(f"标签文案重复：{text!r}（{out[text]} 与 {key}）")
        out[text] = key
    return out


def merged_labels() -> dict[str, str]:
    """输入：无。输出：全部语言的标签合并表。

    预期行为：供语言无关的 `reparse` 用；各语言的标签互不重叠（由 `test_locales` 锁住），
    重叠了这里会报错而不是静默取其一。
    """
    out: dict[str, str] = {}
    for lang in locales.available():
        for text, slot in label_index(locales.load(lang)).items():
            if text in out:
                raise ValueError(f"标签 {text!r} 在多种语言里都出现（{out[text]} / {slot}）")
            out[text] = slot
    return out


def _match_label(line: str, labels: dict[str, str]) -> tuple[str, str] | None:
    """输入：一行文本、标签表。输出：(槽位, 值) 或 None。

    预期行为：取**最长**的匹配标签（标签之间有前缀关系时才不会认错），
    并且要求标签后面紧跟分隔符——否则抬头「重排请求」会被认成「查询」那一格。
    """
    best: tuple[str, str] | None = None
    for text, slot in labels.items():
        if not line.startswith(text):
            continue
        rest = line[len(text) :].lstrip()
        if rest[:1] not in COLONS:
            continue
        if best is None or len(text) > len(best[0]):
            best = (text, slot)
    if best is None:
        return None
    return best[1], line[len(best[0]) :].strip().lstrip("：:").strip()


def _header(line: str, labels: dict[str, str]) -> ParsedCand | None:
    """输入：一行文本、标签表。输出：候选（正文暂空）或 None。

    预期行为：按 `|` 切段，第一段必须是编号，其余每段都必须是「标签 + 分隔符 + 值」；
    任何一段认不出来就返回 None（说明这行不是候选表头，交给正文逻辑处理）。
    """
    if SEPARATOR not in line:
        return None
    parts = [part.strip() for part in line.split(SEPARATOR)]
    if not HEADER_ID.match(parts[0]):
        return None
    fields: dict[str, str] = {}
    for part in parts[1:]:
        found = _match_label(part, labels)
        if found is None:
            return None
        fields[found[0]] = found[1]
    missing = [key for key in ("source_label", "day_label") if key not in fields]
    if missing:
        raise ValueError(f"候选 {parts[0]} 的表头缺这几格：{missing}")
    return ParsedCand(parts[0], fields["source_label"], fields["day_label"], ())


def parse_list(state: str, labels: dict[str, str]) -> ParsedList:
    """输入：state 文本、标签表。输出：反解结果。

    预期行为：逐行认标签；`cand_label` 之后的行按「表头开新块、其余行进当前块的正文」
    处理；查询缺了、候选一条都没有、或者有候选一行正文都没有，都报人话——
    缺格说明渲染漏写了，是硬伤，不能当空值蒙过去。
    """
    query: str | None = None
    cands: list[ParsedCand] = []
    current: ParsedCand | None = None
    body: list[str] = []
    in_cands = False

    def flush() -> None:
        """输入：无。输出：无。预期行为：把当前块收进 `cands`（正文可能为空，后面再查）。"""
        nonlocal current, body
        if current is not None:
            cands.append(ParsedCand(current.cid, current.source, current.day, tuple(body)))
        current, body = None, []

    for raw in state.splitlines():
        line = raw.strip()
        if not line:
            continue
        if in_cands:
            header = _header(line, labels)
            if header is not None:
                flush()
                current = header
                continue
        found = _match_label(line, labels)
        if found is not None:
            if found[0] == "query_label":
                query = found[1]
            elif found[0] == "cand_label":
                in_cands = True
            continue
        if in_cands and current is not None:
            body.append(line)
    flush()

    if query is None:
        raise ValueError("重排请求缺「查询」那一行")
    if not cands:
        raise ValueError("重排请求里一条候选都没有")
    empty = [cand.cid for cand in cands if not cand.body]
    if empty:
        raise ValueError(f"这几条候选一行正文都没有：{empty}")
    return ParsedList(query, tuple(cands))
