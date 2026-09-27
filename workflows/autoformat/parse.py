"""结构恢复 workflow 的读法：把渲染文本读回「行 + 内容块」。

输入：`state` 文本 + 语言字典。输出：`ParsedDoc`。
预期行为：**只按文本本身读**（空行切段、短行认标题 / 列表项、命令行认代码块、关键词认
提示框、署名前缀认引用），不复用构造时的任何坐标计算，所以能抓到折行写歪、块之间忘了
空行、列表项写太长这类硬伤。构造侧与判别侧**共用这一套读法**。

`Workflow.reparse` 没有语言参数，所以这里显式收一个语言字典：判别器给本语言的文件，
`reparse` 给 `locales.merged()` 的合并表（各语言的标签与关键词互不重叠，由测试钉住）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

from .spec import (
    KIND_CALLOUT,
    KIND_CODE,
    KIND_HEADING,
    KIND_LIST_ITEM,
    KIND_PARAGRAPH,
    KIND_QUOTE,
)

#: 句末（或分句）标点：出现在行末就说明这一行**不是**「被硬换行折断的半句话」。
TERMINAL: Final[str] = "。！？：；.!?:;"
#: 「短行」上限（字符）。折行后的**非末行**一定比它长（由测试钉住），所以短行只可能是
#: 标题或列表项——这就是「标题 / 列表项」与「段落中间行」的判据。
SHORT_LIMIT: Final[int] = 36
#: 行号格式：`L007| 正文`（空行是 `L007|`）。
LINE_RE: Final[re.Pattern[str]] = re.compile(r"^(L\d{3})\|(?: (.*))?$")


def line_id(index: int) -> str:
    """输入：1 起的行号。输出：行标识（`L007`）。预期行为：三位零填充。"""
    return f"L{index:03d}"


@dataclass(frozen=True)
class ParsedLine:
    """反解出来的一行：行标识 + 正文（空行正文为空串）。"""

    id: str
    text: str


@dataclass(frozen=True)
class ParsedBlock:
    """反解出来的一个内容块。

    `text` 是**按空格拼回来**的（英文正确、中文可读）；判关键词时同时看「按空格拼」与
    「直接拼」两种形态——中文折行处本来没有空格，只按一种拼会漏掉跨行的关键词。
    """

    kind: str
    line_ids: tuple[str, ...]
    text: str
    callout_kind: str = ""
    list_ordered: bool = False


@dataclass(frozen=True)
class ParsedDoc:
    """反解出来的一份文档：全部行（含空行）+ 全部块（按从上到下）。"""

    lines: tuple[ParsedLine, ...]
    blocks: tuple[ParsedBlock, ...]

    def block_index(self, ident: str) -> int | None:
        """输入：行标识。输出：该行所属块的序号，或 None（空行 / 认不出来）。"""
        for index, block in enumerate(self.blocks):
            if ident in block.line_ids:
                return index
        return None

    def same_block(self, left: str, right: str) -> bool:
        """输入：两个行标识。输出：它们是不是同一个块。预期行为：任一行认不出来即 False。"""
        index = self.block_index(left)
        return index is not None and index == self.block_index(right)

    def kinds(self) -> tuple[str, ...]:
        """输入：无。输出：各块的类型（按块序）。"""
        return tuple(block.kind for block in self.blocks)

    def callouts(self) -> tuple[ParsedBlock, ...]:
        """输入：无。输出：全部提示框块（按块序）。"""
        return tuple(block for block in self.blocks if block.kind == KIND_CALLOUT)

    def list_items(self) -> tuple[ParsedBlock, ...]:
        """输入：无。输出：全部列表项块（按块序）。"""
        return tuple(block for block in self.blocks if block.kind == KIND_LIST_ITEM)


# ---------- 文本判据（构造侧与判别侧共用） ----------


def ends_terminal(text: str) -> bool:
    """输入：一行正文。输出：它是不是以句末标点收尾。预期行为：空串算 False。"""
    return bool(text) and text[-1] in TERMINAL


def is_short_unpunctuated(text: str) -> bool:
    """输入：一行正文。输出：它是不是「短且没有句末标点」。预期行为：空串算 False。"""
    return bool(text) and len(text) <= SHORT_LIMIT and not ends_terminal(text)


def command_tokens(loc: dict[str, Any]) -> frozenset[str]:
    """输入：语言字典。输出：全部命令模板的首词集合（判「这一行是不是命令」）。"""
    return frozenset(
        template.split(" ", 1)[0] for pool in loc["code"].values() for template in pool
    )


def is_code_line(text: str, commands: frozenset[str]) -> bool:
    """输入：一行正文、命令首词集合。输出：这一行是不是命令行。

    预期行为：只看**首词**——折行后的散文行也可能以小写词开头，所以不能用「像英文」来判。
    """
    return bool(text) and text.split(" ", 1)[0] in commands


def is_short_line(text: str, commands: frozenset[str]) -> bool:
    """输入：一行正文、命令首词集合。输出：它是不是「短行」（标题 / 列表项候选）。

    预期行为：命令行即使很短也不算短行——否则命令块会被拆成一堆「列表项」。
    """
    return is_short_unpunctuated(text) and not is_code_line(text, commands)


def callout_index(loc: dict[str, Any]) -> dict[str, str]:
    """输入：语言字典。输出：{关键词: 种类}。

    预期行为：同一个关键词出现在两个种类里直接报错——否则反推种类会有歧义。
    """
    out: dict[str, str] = {}
    for kind, words in loc["callout_word"].items():
        for word in words:
            if word in out:
                raise ValueError(f"提示框关键词 {word!r} 同时属于 {out[word]} 与 {kind}")
            out[word] = kind
    return out


def seq_words(loc: dict[str, Any]) -> tuple[str, ...]:
    """输入：语言字典。输出：时序词表（判列表是不是「步骤列表」）。"""
    return tuple(loc["seq_word"]["step"])


def attribution_of(loc: dict[str, Any]) -> str:
    """输入：语言字典。输出：引用块的行末署名前缀（`——`）。"""
    return str(loc["mark"]["attribution"])


# ---------- 反解 ----------


def parse_lines(state: str) -> tuple[ParsedLine, ...]:
    """输入：state 文本。输出：全部行。预期行为：行号格式不对直接报人话。"""
    lines: list[ParsedLine] = []
    for raw in state.split("\n"):
        matched = LINE_RE.match(raw)
        if matched is None:
            raise ValueError(f"行号格式不对：{raw!r}")
        lines.append(ParsedLine(matched.group(1), matched.group(2) or ""))
    return tuple(lines)


def _prose_block(
    line_ids: tuple[str, ...], parts: list[str], words: dict[str, str], mark: str
) -> ParsedBlock:
    """输入：行标识、各行的正文、关键词表、署名前缀。输出：散文块（提示框 / 引用 / 段落）。

    预期行为：关键词在「按空格拼」与「直接拼」两种形态里任一命中就算命中——中文折行处
    没有空格，只按空格拼会漏掉被折行拆开的关键词。
    """
    spaced = " ".join(parts)
    joined = "".join(parts)
    for word, kind in words.items():
        if word in spaced or word in joined:
            return ParsedBlock(KIND_CALLOUT, line_ids, spaced, callout_kind=kind)
    if mark in joined:
        return ParsedBlock(KIND_QUOTE, line_ids, spaced)
    return ParsedBlock(KIND_PARAGRAPH, line_ids, spaced)


def _split_region(
    region: tuple[ParsedLine, ...],
    commands: frozenset[str],
    words: dict[str, str],
    seq: tuple[str, ...],
    mark: str,
) -> list[ParsedBlock]:
    """输入：一段连续非空行、判据表。输出：这一段切出来的块。

    预期行为（判据顺序固定，构造侧保证互斥）：
    1. 全是命令行 → 一个代码块；
    2. 全是短行 → 两项以上是**每行一个列表项**，只有一行是**标题**（或引用）；
    3. 其余 → 开头那行若是短行就当标题，剩下的拼成一段散文。
    """
    texts = [line.text for line in region]
    if all(is_code_line(text, commands) for text in texts):
        return [ParsedBlock(KIND_CODE, tuple(line.id for line in region), "\n".join(texts))]
    if all(is_short_line(text, commands) for text in texts):
        if len(region) >= 2:
            ordered = any(word in text for text in texts for word in seq)
            return [
                ParsedBlock(KIND_LIST_ITEM, (line.id,), line.text, list_ordered=ordered)
                for line in region
            ]
        text = texts[0]
        kind = KIND_QUOTE if mark in text else KIND_HEADING
        return [ParsedBlock(kind, (region[0].id,), text)]
    out: list[ParsedBlock] = []
    start = 0
    if len(region) >= 2 and is_short_line(texts[0], commands):
        kind = KIND_QUOTE if mark in texts[0] else KIND_HEADING
        out.append(ParsedBlock(kind, (region[0].id,), texts[0]))
        start = 1
    rest = region[start:]
    if rest:
        out.append(
            _prose_block(tuple(line.id for line in rest), [line.text for line in rest], words, mark)
        )
    return out


def parse_state(state: str, loc: dict[str, Any]) -> ParsedDoc:
    """输入：state 文本、语言字典。输出：反解结果。

    预期行为：先用空行切段，再逐段按 `_split_region` 的判据切块；空行本身不属于任何块。
    """
    lines = parse_lines(state)
    commands = command_tokens(loc)
    words = callout_index(loc)
    seq = seq_words(loc)
    mark = attribution_of(loc)
    blocks: list[ParsedBlock] = []
    index = 0
    total = len(lines)
    while index < total:
        if not lines[index].text:
            index += 1
            continue
        end = index
        while end + 1 < total and lines[end + 1].text:
            end += 1
        blocks.extend(_split_region(tuple(lines[index : end + 1]), commands, words, seq, mark))
        index = end + 1
    return ParsedDoc(lines, tuple(blocks))


def asked_pairs(lines: tuple[ParsedLine, ...], commands: frozenset[str]) -> list[tuple[str, str]]:
    """输入：全部行、命令首词集合。输出：需要问「是不是接着上一行」的相邻行对。

    预期行为：只问**两个非空行**且**都不是命令行**的相邻对——命令行块的上下都被空行包着，
    所以代码块天然不出这一族题；这样「并段」这一族问的都是散文的折行与块的边界。
    """
    out: list[tuple[str, str]] = []
    for index in range(len(lines) - 1):
        left, right = lines[index], lines[index + 1]
        if not left.text or not right.text:
            continue
        if is_code_line(left.text, commands) or is_code_line(right.text, commands):
            continue
        out.append((left.id, right.id))
    return out
