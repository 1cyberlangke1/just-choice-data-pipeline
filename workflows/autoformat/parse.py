"""结构恢复（autoformat）的读法：把渲染出来的纯文本读回「行 + 内容块」。

输入：`state` 文本 + 语言字典。输出：`ParsedDoc`。
预期行为：**只按文本本身读**（空行切段、短行认标题 / 列表项、命令行认代码块、关键词认提示框、
署名前缀认引用），不复用构造时的任何坐标计算，所以能抓到折行写歪、块之间忘了空行、列表项写太长
这类硬伤。构造侧与判别侧**共用这一套读法**——两边各写一套的话，「构造写歪」与「读法写歪」会各错
一半还互相印证。

`Workflow.reparse` 没有语言参数，所以这里显式收一个语言字典：判别器给本语言的文件，`reparse` 给
`locales.merged()` 的合并表（各语言的标签与关键词互不重叠，由测试钉住）。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from jc.jsontypes import JsonValue

from .features import (
    KIND_CALLOUT,
    KIND_CODE,
    KIND_HEADING,
    KIND_LIST_ITEM,
    KIND_PARAGRAPH,
    KIND_QUOTE,
    SHORT_LIMIT,
    WRAPS,
)

#: 句末（或分句）标点：出现在行末就说明这一行**不是**「被硬换行折断的半句话」。
TERMINAL: Final[str] = "。！？：；.!?:;"
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

    `text` 是**按空格拼回来**的（英文正确、中文可读）；判关键词时同时看「按空格拼」与「直接拼」
    两种形态——中文折行处本来没有空格，只按一种拼会漏掉跨行的关键词。
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


# ---------- 字符预算 ----------


@dataclass(frozen=True)
class Limits:
    """一份语言里的字符区间（含端点）。字段含义见 `limits()`。"""

    heading: tuple[int, int]
    item: tuple[int, int]
    prose: tuple[int, int]
    quote: tuple[int, int]
    callout: tuple[int, int]

    def range_of(self, kind: str) -> tuple[int, int]:
        """输入：块类型。输出：这一类块的字符区间。预期行为：没有预算的类型直接报错。"""
        table = {
            KIND_HEADING: self.heading,
            KIND_LIST_ITEM: self.item,
            KIND_PARAGRAPH: self.prose,
            KIND_QUOTE: self.quote,
            KIND_CALLOUT: self.callout,
        }
        if kind not in table:
            raise ValueError(f"这一类块没有字符预算：{kind!r}")
        return table[kind]


#: `[limit]` 表里必须有的键，值一律是 `[下限, 上限]`（字符数）。
LIMIT_KEYS: Final[tuple[str, ...]] = ("heading", "item", "prose", "quote", "callout")


def _limit_pair(key: str, value: JsonValue) -> tuple[int, int]:
    """输入：键名、`[limit]` 里那一项。输出：(下限, 上限)。

    预期行为：必须是两个整数、下限为正、上限严格大于下限；写歪了报人话，别把预算悄悄跑偏。
    """
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"[limit.{key}] 必须是 [下限, 上限]：{value!r}")
    low, high = value
    if isinstance(low, bool) or isinstance(high, bool):
        raise ValueError(f"[limit.{key}] 的两端必须是整数：{value!r}")
    if not isinstance(low, int) or not isinstance(high, int):
        raise ValueError(f"[limit.{key}] 的两端必须是整数：{value!r}")
    if low <= 0 or high <= low:
        raise ValueError(f"[limit.{key}] 要求 0 < 下限 < 上限，实际 {low} / {high}")
    return (low, high)


def limits(loc: dict[str, JsonValue]) -> Limits:
    """输入：语言字典。输出：这份语言的字符预算（标题 / 列表项 / 正文 / 引用 / 提示框）。

    预期行为：把 `[limit]` 读成五组区间，并当场验三条与反解判据挂钩的硬约束——
    标题与列表项的**上限不许超过 `SHORT_LIMIT`**（超了就不是短行，那一块会被读成正文）；
    正文的**下限必须长过最大行宽**（否则折不成两行以上，「并段」那一族没有真话可问）；
    提示框的**下限必须长过 `SHORT_LIMIT`**（只有一行的提示框会被读成标题）。

    正文下限为什么钉在**最大行宽**而不是「两倍短行阈值」：反解只需要「正文不是短行」，
    折成两行是为了「并段」那一族有样本，而真正决定折几行的是行宽。写成两倍阈值的话，
    阈值一抬，正文下限就跟着虚高，段落被拉长、「并段」那一族的答案跟着往一边倒。

    为什么预算按语言给：中文一个字顶英文好几个字符，同一个「短标题」在两种语言里的字符数差一截
    ——真机上英文标题按中文的 6~24 写，10 条里 6 条当场被长度判据拒掉。而 `SHORT_LIMIT` 不能跟着
    分家：反解拿的是合并读法（state 里没有语言标记），阈值只能是一个。
    """
    table = loc.get("limit")
    if not isinstance(table, dict):
        raise ValueError(f"[limit] 不是一个表：{table!r}")
    missing = [key for key in LIMIT_KEYS if key not in table]
    if missing:
        raise ValueError(f"[limit] 缺了这几节：{missing}")
    extra = sorted(set(table) - set(LIMIT_KEYS))
    if extra:
        raise ValueError(f"[limit] 里多了这几节（没人读）：{extra}")
    out = {key: _limit_pair(key, table[key]) for key in LIMIT_KEYS}
    for key in ("heading", "item"):
        if out[key][1] > SHORT_LIMIT:
            raise ValueError(
                f"[limit.{key}] 的上限 {out[key][1]} 超过了短行阈值 {SHORT_LIMIT}，"
                f"这一块会被读成正文"
            )
    widest = max(WRAPS)
    if out["prose"][0] <= widest:
        raise ValueError(
            f"[limit.prose] 的下限 {out['prose'][0]} 不够长（要 > 最大行宽 {widest}），"
            f"正文折不成两行以上"
        )
    if out["callout"][0] <= SHORT_LIMIT:
        raise ValueError(
            f"[limit.callout] 的下限 {out['callout'][0]} 不高于短行阈值 {SHORT_LIMIT}，"
            f"只有一行且是短行的提示框会被读成标题"
        )
    return Limits(
        heading=out["heading"],
        item=out["item"],
        prose=out["prose"],
        quote=out["quote"],
        callout=out["callout"],
    )


def command_tokens(loc: dict[str, JsonValue]) -> frozenset[str]:
    """输入：语言字典。输出：全部命令工具名（判「这一行是不是命令」）。

    预期行为：取 `[code_tool]` 里各命令族的工具名并集——**只认首词**，所以工具名必须是那行的第一个
    词。折行后的散文行也可能以小写词开头，所以不能用「像英文」来判。
    """
    table = loc["code_tool"]
    if not isinstance(table, dict):
        raise ValueError(f"[code_tool] 不是一个表：{table!r}")
    tokens: set[str] = set()
    for pool in table.values():
        if not isinstance(pool, list):
            raise ValueError(f"[code_tool] 的值必须是列表：{pool!r}")
        tokens.update(str(item) for item in pool)
    return frozenset(tokens)


def is_code_line(text: str, commands: frozenset[str]) -> bool:
    """输入：一行正文、命令工具名集合。输出：这一行是不是命令行。"""
    return bool(text) and text.split(" ", 1)[0] in commands


def is_short_line(text: str, commands: frozenset[str]) -> bool:
    """输入：一行正文、命令工具名集合。输出：它算不算「短行」。

    预期行为：命令行**不算**短行（它是代码块的一部分），其余按 `is_short_unpunctuated`。
    """
    if is_code_line(text, commands):
        return False
    return is_short_unpunctuated(text)


def callout_index(loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：语言字典。输出：{关键词: 提示框种类}。预期行为：关键词跨种类不重复，重复就报错。"""
    table = loc["callout_word"]
    if not isinstance(table, dict):
        raise ValueError(f"[callout_word] 不是一个表：{table!r}")
    out: dict[str, str] = {}
    for kind, pool in table.items():
        if not isinstance(pool, list):
            raise ValueError(f"[callout_word.{kind}] 必须是列表：{pool!r}")
        for word in pool:
            text = str(word)
            if text in out:
                raise ValueError(f"提示框关键词 {text!r} 同时属于 {out[text]!r} 与 {kind!r}")
            out[text] = str(kind)
    return out


def callout_words(loc: dict[str, JsonValue]) -> tuple[str, ...]:
    """输入：语言字典。输出：全部提示框关键词（按字典序，便于稳定报错）。"""
    return tuple(sorted(callout_index(loc)))


def seq_words(loc: dict[str, JsonValue]) -> tuple[str, ...]:
    """输入：语言字典。输出：时序词表（判列表是不是「步骤列表」）。"""
    pool = loc["seq_word"]["step"]
    if not isinstance(pool, list):
        raise ValueError(f"[seq_word.step] 必须是列表：{pool!r}")
    return tuple(str(word) for word in pool)


def attribution_marks(loc: dict[str, JsonValue]) -> tuple[str, ...]:
    """输入：语言字典。输出：引用块的署名前缀（**列表**：合并读法下两种语言各有一个）。

    预期行为：至少一个，且两两不同；空表或非列表直接报人话。
    """
    pool = loc["mark"]["attribution"]
    if not isinstance(pool, list) or not pool:
        raise ValueError(f"[mark.attribution] 必须是非空列表：{pool!r}")
    marks = tuple(str(item) for item in pool)
    if len(set(marks)) != len(marks):
        raise ValueError(f"[mark.attribution] 里有重复：{marks}")
    return marks


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
    line_ids: tuple[str, ...], parts: list[str], words: dict[str, str], marks: tuple[str, ...]
) -> ParsedBlock:
    """输入：行标识、各行的正文、关键词表、署名前缀。输出：散文块（提示框 / 引用 / 段落）。

    预期行为：关键词在「按空格拼」与「直接拼」两种形态里任一命中就算命中——中文折行处没有空格，
    只按空格拼会漏掉被折行拆开的关键词。
    """
    spaced = " ".join(parts)
    joined = "".join(parts)
    for word, kind in words.items():
        if word in spaced or word in joined:
            return ParsedBlock(KIND_CALLOUT, line_ids, spaced, callout_kind=kind)
    if any(mark in joined for mark in marks):
        return ParsedBlock(KIND_QUOTE, line_ids, spaced)
    return ParsedBlock(KIND_PARAGRAPH, line_ids, spaced)


def _split_region(
    region: tuple[ParsedLine, ...],
    commands: frozenset[str],
    words: dict[str, str],
    seq: tuple[str, ...],
    marks: tuple[str, ...],
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
        kind = KIND_QUOTE if any(mark in text for mark in marks) else KIND_HEADING
        return [ParsedBlock(kind, (region[0].id,), text)]
    out: list[ParsedBlock] = []
    start = 0
    if len(region) >= 2 and is_short_line(texts[0], commands):
        kind = KIND_QUOTE if any(mark in texts[0] for mark in marks) else KIND_HEADING
        out.append(ParsedBlock(kind, (region[0].id,), texts[0]))
        start = 1
    rest = region[start:]
    if rest:
        out.append(
            _prose_block(tuple(line.id for line in rest), [line.text for line in rest], words, marks)
        )
    return out


def parse_state(state: str, loc: dict[str, JsonValue]) -> ParsedDoc:
    """输入：state 文本、语言字典。输出：反解结果。

    预期行为：先用空行切段，再逐段按 `_split_region` 的判据切块；空行本身不属于任何块。
    """
    lines = parse_lines(state)
    commands = command_tokens(loc)
    words = callout_index(loc)
    seq = seq_words(loc)
    marks = attribution_marks(loc)
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
        blocks.extend(_split_region(tuple(lines[index : end + 1]), commands, words, seq, marks))
        index = end + 1
    return ParsedDoc(lines, tuple(blocks))


def asked_pairs(lines: tuple[ParsedLine, ...], commands: frozenset[str]) -> list[tuple[str, str]]:
    """输入：全部行、命令工具名集合。输出：需要问「是不是接着上一行」的相邻行对。

    预期行为：只问**两个非空行**且**都不是命令行**的相邻对——命令行块的上下都被空行包着，所以代码块
    天然不出这一族题；这样「并段」这一族问的都是散文的折行与块的边界。

    这里给的是**候选**（全部相邻对）；真正出题的是 `balanced_pairs` 配平之后的那一批。
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


#: 某一边一对都没有时，另一边最多留几对（没有另一端可配，留多了就是白送捷径）。
LONE_PAIRS: Final[int] = 2


def _spread(pairs: list[tuple[str, str]], keep: int) -> list[tuple[str, str]]:
    """输入：一串相邻行对、要留几对。输出：抽稀后的子集（保持原顺序）。

    预期行为：`keep >= len(pairs)` 时原样返回；否则按 `i * n // keep` 取下标——**等距**取，
    免得留下的题全挤在文档开头。
    """
    if keep >= len(pairs):
        return list(pairs)
    total = len(pairs)
    return [pairs[index * total // keep] for index in range(keep)]


def balanced_pairs(
    pairs: list[tuple[str, str]], same_of: Callable[[str, str], bool]
) -> list[tuple[str, str]]:
    """输入：候选相邻行对、判「两行是不是同一块」的函数。输出：真正出题的那些（保持文档顺序）。

    预期行为：把「是」与「不是」**配平**——少的那一边一条不丢，多的那一边等距抽稀到同样条数；
    某一边一对都没有（整篇没有块边界、或整篇没有跨行折行）时，另一边最多留 `LONE_PAIRS` 对。

    为什么非配平不可：段落一折就是好几对「是」，而块边界一篇只有几个——train 全量实测 505 对里
    「是」占 **79%**，模型永远答「是」就有 79 分，这一族直接退化成捷径题。均衡闸（同时也是捷径闸）
    就是这么把它标出来的。

    构造侧（`render.Rendered.asked_pairs`）与审计侧（`audit.recompute`）**共用这一个实现**，
    两边各写一套的话，题号与重算结果会对不上。
    """
    same = [pair for pair in pairs if same_of(*pair)]
    diff = [pair for pair in pairs if not same_of(*pair)]
    if not same or not diff:
        keep = _spread(same or diff, LONE_PAIRS)
        return [pair for pair in pairs if pair in keep]
    cap = min(len(same), len(diff))
    kept = set(_spread(same, cap)) | set(_spread(diff, cap))
    return [pair for pair in pairs if pair in kept]
