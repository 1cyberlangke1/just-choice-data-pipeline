"""结构恢复 workflow 的渲染：把块计划写成「丢掉了标记、硬换行折过」的纯文本。

输入：`DocSpec`、语言字典。输出：契约允许的 `state`（纯字符串）+ 行 / 块的坐标。
预期行为：**纯函数**——同一份 spec + 同一份语言文件必得同一段文本（逐字节），
所以「两次生成逐字节一致」这条能直接查。渲染规则（也是 `parse` 的读法）：
- 每个块里的散文按行宽折行；**块与块之间空一行**；
- 两个例外：同一个列表里相邻的列表项之间**不空行**；标题**紧贴**它后面那段正文；
- 标记一律剥掉（没有 `#`、没有列表符号、没有引用符号）；
- 代码块整块不折行。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from . import parse
from .spec import KIND_CODE, KIND_HEADING, KIND_LIST_ITEM, KIND_PARAGRAPH, DocSpec

_LEFTOVER: re.Pattern[str] = re.compile(r"\{\w+\}")


@dataclass(frozen=True)
class Block:
    """渲染时解出来的块：类型 + 文本（散文整段、代码多行用 `\\n` 连）+ 伴随信息。"""

    kind: str
    text: str
    callout_kind: str = ""
    list_ordered: bool = False
    list_id: int = 0


def _template_text(loc: dict[str, Any], region: str, key: str, index: int) -> str:
    """输入：语言字典、模板位置。输出：模板原文。

    预期行为：`quote` 那张表的值是字符串（不是列表），其余表的值是列表——两种都支持。
    """
    value = loc[region][key]
    if isinstance(value, str):
        return value
    return value[index]


def resolve_blocks(spec: DocSpec, loc: dict[str, Any]) -> tuple[Block, ...]:
    """输入：规格、语言字典。输出：填好文案的块（顺序不变）。

    预期行为：槽位**全部**被替换掉；一段正文里的多个片段按 `[mark] sentence_join` 拼起来
    （中文不空格、英文空格）。渲染后还剩 `{xxx}` 说明语言文件与计划对不上，直接报错。
    """
    glue = str(loc["mark"]["sentence_join"])
    out: list[Block] = []
    for plan in spec.plan:
        if plan.kind == KIND_CODE:
            text = "\n".join(plan.lines)
        else:
            pieces: list[str] = []
            for part in plan.parts:
                piece = _template_text(loc, part.region, part.key, part.index)
                for name, value in part.slots:
                    piece = piece.replace("{" + name + "}", value)
                leftover = _LEFTOVER.search(piece)
                if leftover:
                    raise ValueError(f"渲染后还剩占位符 {leftover.group(0)!r}：{piece!r}")
                pieces.append(piece)
            text = glue.join(pieces)
        out.append(
            Block(
                kind=plan.kind,
                text=text,
                callout_kind=plan.callout_kind,
                list_ordered=plan.list_ordered,
                list_id=plan.list_id,
            )
        )
    return tuple(out)


def wrap_text(text: str, width: int) -> list[str]:
    """输入：一段散文、行宽。输出：折行后的行列表。

    预期行为：贪心——每行尽量塞满 `width` 个字符；窗口里最后一个空格够靠后就在空格处断
    （英文），否则直接按字符数断（中文）。两种语言用同一条规则，所以折行不依赖语言。
    **非末行**一定比 `parse.SHORT_LIMIT` 长（由测试钉住），否则它会被读成标题。
    """
    out: list[str] = []
    rest = text.strip()
    while len(rest) > width:
        window = rest[: width + 1]
        cut = window.rfind(" ")
        # 断点太靠前（含「断出来的行还是短行」）就改成硬断到行宽——
        # 非末行一旦短过 SHORT_LIMIT，反解会把它读成标题，整块的切分就错了。
        if cut <= parse.SHORT_LIMIT:
            cut = width
        line = rest[:cut].rstrip()
        while len(line) <= parse.SHORT_LIMIT and cut < len(rest):
            cut += 1
            line = rest[:cut].rstrip()
        if not line or cut >= len(rest):
            break
        out.append(line)
        rest = rest[cut:].lstrip()
    if rest:
        out.append(rest)
    return out


def needs_blank(blocks: tuple[Block, ...], index: int) -> bool:
    """输入：块序列、当前块下标。输出：当前块前面要不要空一行。

    预期行为：默认要；两个例外——同一个列表里相邻的列表项不空行；标题紧贴后面的正文。
    """
    if index == 0:
        return False
    prev, cur = blocks[index - 1], blocks[index]
    if prev.kind == KIND_LIST_ITEM and cur.kind == KIND_LIST_ITEM and prev.list_id == cur.list_id:
        return False
    return not (prev.kind == KIND_HEADING and cur.kind == KIND_PARAGRAPH)


@dataclass(frozen=True)
class Rendered:
    """渲染结果：行标识、行正文（空行是空串）、各块的行区间（1 起、闭区间）、块序列。"""

    ids: tuple[str, ...]
    texts: tuple[str, ...]
    spans: tuple[tuple[int, int], ...]
    blocks: tuple[Block, ...]

    def state(self) -> str:
        """输入：无。输出：契约形态的 state 字符串（每行 `L007| 正文`）。"""
        return "\n".join(
            f"{ident}|" + (f" {text}" if text else "")
            for ident, text in zip(self.ids, self.texts, strict=True)
        )

    def line_number(self, ident: str) -> int | None:
        """输入：行标识。输出：1 起的行号，或 None（没有这一行）。"""
        for index, value in enumerate(self.ids, start=1):
            if value == ident:
                return index
        return None

    def block_index_of(self, ident: str) -> int | None:
        """输入：行标识。输出：该行所属块的序号，或 None（空行 / 没有这一行）。"""
        number = self.line_number(ident)
        if number is None:
            return None
        for index, (start, end) in enumerate(self.spans):
            if start <= number <= end:
                return index
        return None

    def same_block(self, left: str, right: str) -> bool:
        """输入：两个行标识。输出：它们是不是同一个块。"""
        index = self.block_index_of(left)
        return index is not None and index == self.block_index_of(right)

    def asked_pairs(self, loc: dict[str, Any]) -> list[tuple[str, str]]:
        """输入：语言字典。输出：需要问「是不是接着上一行」的相邻行对。

        预期行为：**直接用 `parse.asked_pairs`**——题目侧与判别侧必须是同一套判据，
        否则两边会各自算出不同的题号。
        """
        lines = tuple(
            parse.ParsedLine(ident, text)
            for ident, text in zip(self.ids, self.texts, strict=True)
        )
        return parse.asked_pairs(lines, parse.command_tokens(loc))


def layout(spec: DocSpec, loc: dict[str, Any]) -> Rendered:
    """输入：规格、语言字典。输出：渲染结果（行 / 块坐标一起算出来）。

    预期行为：块按顺序落成行；块前按 `needs_blank` 决定要不要空行；代码块整块不折行。
    """
    blocks = resolve_blocks(spec, loc)
    texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for index, block in enumerate(blocks):
        if needs_blank(blocks, index):
            texts.append("")
        start = len(texts) + 1
        if block.kind == KIND_CODE:
            texts.extend(block.text.split("\n"))
        else:
            texts.extend(wrap_text(block.text, spec.case.wrap))
        spans.append((start, len(texts)))
    ids = tuple(parse.line_id(number) for number in range(1, len(texts) + 1))
    return Rendered(ids, tuple(texts), tuple(spans), blocks)


def render_state(spec: DocSpec, loc: dict[str, Any]) -> str:
    """输入：规格、语言字典。输出：state 字符串。预期行为：`layout` 的文本形态。"""
    return layout(spec, loc).state()
