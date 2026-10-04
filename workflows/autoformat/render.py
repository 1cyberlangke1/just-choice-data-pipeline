"""结构恢复的渲染：把块计划 + 后端写的句子排成「丢掉了标记、硬换行折过」的纯文本。

输入：`DocSpec` / 块计划、后端交回的槽位文本、语言字典。输出：契约允许的 `state`（纯字符串）。
预期行为：**结构由代码定、句子由模型写**——

- 块的顺序、类型、列表有没有序、提示框是哪一类，全在 `features.build_plan` 里定死；
- 后端只写每一块的文字（`SlotDraft.texts`，一块一项），写歪了在 `check_draft` 里当场拦住。

渲染规则（也是 `parse` 的读法）：块里的散文按行宽折行；**块与块之间空一行**；两个例外——同一个
列表里相邻的列表项之间**不空行**，标题**紧贴**它后面那段正文；标记一律剥掉（没有井号、没有列表
符号、没有引用符号）；代码块整块不折行。
"""

from __future__ import annotations

from dataclasses import dataclass

from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.workflow import DraftError
from pydantic import BaseModel

from . import locales, parse
from .features import (
    KIND_CALLOUT,
    KIND_CODE,
    KIND_HEADING,
    KIND_LIST_ITEM,
    KIND_PARAGRAPH,
    KIND_QUOTE,
    MAX_CODE_LINES,
    MIN_CODE_LINES,
    DocPlan,
    Slot,
)
from .spec import DocSpec


class SlotDraft(BaseModel):
    """后端要交回来的草稿：**按块顺序**一项一个字符串。

    预期行为：只有这一个字段，且是数组——让模型一次把整份文档写完，省掉一轮往返；块与文本的对齐
    由位置决定，所以 `check_draft` 第一步就是数长度。
    """

    texts: list[str]


@dataclass(frozen=True)
class Block:
    """渲染时解出来的块：类型 + 文本（代码块多行用 `\\n` 连）+ 伴随信息。"""

    kind: str
    text: str
    callout_kind: str = ""
    list_ordered: bool = False
    list_id: int = 0


def split_rules_block(loc: dict[str, JsonValue]) -> str:
    """输入：语言字典。输出：切块规则（表头一行 + 逐条一行）。

    预期行为：**规则只写一处**（`[doc].split_rules`），考生题面（`questions.build`）与读者题面
    （`judges.build_trace_prompt`）都渲染这一份——写两份就会漂，而漂的那次正是 2026-10-04 真机
    36 条被拒 17 条（47%）的成因：渲染规则里有两处「不空行也算两块」（标题紧贴着它后面那段正文、
    同一个列表里相邻的列表项），提示词却只写了「块与块之间用空行隔开」，于是读者把标题并进了正文，
    而确定性读法 `parse.parse_state` 读同一份文本是对的——错的是读者手里的规则少了两条。
    """
    pool = loc["doc"]["split_rules"]
    if not isinstance(pool, list) or not pool:
        raise ValueError(f"[doc.split_rules] 必须是非空列表：{pool!r}")
    header = str(loc["doc"]["split_header"])
    return "\n".join([header, *(f"- {rule}" for rule in pool)])


# ---------- 逐槽位的字面校验 ----------


def _lines_of(text: str) -> tuple[str, ...]:
    """输入：一段文本。输出：去掉空行、逐行 strip 之后的行。"""
    return tuple(line.strip() for line in text.split("\n") if line.strip())


def forbidden_words(plan: DocPlan, loc: dict[str, JsonValue]) -> tuple[str, ...]:
    """输入：块计划、语言字典。输出：**这份文档里不该出现**的标记词（提示词逐条列出）。

    预期行为：把「别的内容块的标记」全列出来，再减掉这份计划**确实要用**的那些——
    只减提示框的关键词（按种类减）。
    """
    words: set[str] = set(parse.callout_words(loc))
    for kind in plan.callout_kinds():
        words -= {str(word) for word in loc["callout_word"][kind]}
    # 步骤词只在**这份计划一条有序列表都没有**时列进「不许出现」：它是「有序列表项」的判据，
    # 而无序列表项带上它，整张列表会被读成有序的。别的块（标题 / 正文 / 引用 / 提示框）**不**列——
    # 反解根本不拿步骤词判那几类，禁了只会把「First Week」这种正常说法一起禁掉。
    if not plan.list_flags() or not any(plan.list_flags()):
        words.update(parse.seq_words(loc))
    # 署名前缀**一律**列进「不许出现」：英文的 `--` 与中文的 `——` 都是散文里很常见的破折号，
    # 一旦写进别的块，那一块当场被读成引用。有引用块时也不省这一条——提示词里那句
    # 「除了上面明确要求它们的那一块」已经把它豁免掉了（见 `doc.forbidden_header`）。
    words.update(parse.attribution_marks(loc))
    return tuple(sorted(words))


def _check_slot(
    index: int,
    slot: Slot,
    text: str,
    loc: dict[str, JsonValue],
    lim: parse.Limits,
    marks: tuple[str, ...],
    steps: tuple[str, ...],
) -> None:
    """输入：块序号、槽位、文本、语言字典、字符预算、署名前缀表、时序词表。输出：无（不合格抛 ValueError）。

    预期行为：逐条查——字符数区间、结尾标点、不许出现的标记词、以及「这块必须带什么」。
    这里查的全是**字面**要求，判据与 `parse` 的读法一一对应；查完还会整体反解一遍（见 `check_draft`）。

    步骤词**只对列表项**是判据（有 = 有序、没有 = 无序）：反解只在「一整段全是短行」时才拿它判有序性，
    而标题 / 正文 / 引用 / 提示框都不会落进那一条分支。所以别在别的块上禁步骤词——英文的 `First`
    和「First Week」这种日常说法撞车，真机上 10 条里白扔 3 条。
    """
    where = f"第 {index} 块（{slot.kind}）"
    words = parse.callout_words(loc)
    # 命令行块**豁免**标记词检查：它的识别判据是**词法**的（每行以工具名开头），在读法的判据顺序里
    # 排在标记词之前，所以块里出现标记词也不会被读错。豁免是必须的——英文的署名前缀是 `--`，
    # 而命令行天然带长选项（`git --version`），不豁免的话每一个英文代码块都会被假拒稿。
    if any(word in text for word in words) and slot.kind not in (KIND_CALLOUT, KIND_CODE):
        raise DraftError(f"{where} 里出现了提示框关键词，那一块会被读成提示框：{text[:60]!r}")
    if slot.kind not in (KIND_QUOTE, KIND_CODE) and any(mark in text for mark in marks):
        raise DraftError(f"{where} 里出现了署名前缀，那一块会被读成引用：{text[:60]!r}")
    if slot.kind == KIND_HEADING:
        low, high = lim.heading
        if not low <= len(text) <= high:
            raise DraftError(f"{where} 长度 {len(text)} 不在 {low}~{high}")
        if parse.ends_terminal(text):
            raise DraftError(f"{where} 不该以标点结尾（那会被读成正文）：{text!r}")
        return
    if slot.kind == KIND_LIST_ITEM:
        low, high = lim.item
        if not low <= len(text) <= high:
            raise DraftError(f"{where} 长度 {len(text)} 不在 {low}~{high}")
        if parse.ends_terminal(text):
            raise DraftError(f"{where} 不该以标点结尾：{text!r}")
        has_step = any(word in text for word in steps)
        if slot.list_ordered and not has_step:
            raise DraftError(f"{where} 是有序列表的一项，但一个步骤词都没有：{text!r}")
        if not slot.list_ordered and has_step:
            raise DraftError(f"{where} 是无序列表的一项，却带了步骤词：{text!r}")
        return
    if slot.kind == KIND_CODE:
        lines = _lines_of(text)
        if not MIN_CODE_LINES <= len(lines) <= MAX_CODE_LINES:
            raise DraftError(f"{where} 有 {len(lines)} 行，要求 {MIN_CODE_LINES}~{MAX_CODE_LINES} 行")
        tools = {str(item) for item in loc["code_tool"][slot.code_family]}
        for line in lines:
            head = line.split(" ", 1)[0]
            if head not in tools:
                raise DraftError(f"{where} 有一行不以 {sorted(tools)} 里的词开头：{line!r}")
        return
    if slot.kind == KIND_QUOTE:
        low, high = lim.quote
        if not low <= len(text) <= high:
            raise DraftError(f"{where} 长度 {len(text)} 不在 {low}~{high}")
        used = [mark for mark in marks if mark in text]
        if len(used) != 1:
            raise DraftError(f"{where} 要恰好带一个署名前缀 {marks}，实际 {used}")
        # 提示词要的是「话 + 署名前缀 + 署名」，所以署名前缀**不在**结尾，后面必须真跟一个名字；
        # 名字里再出现一次前缀会让反解分不清署名从哪开始，所以前缀只许出现一次（上面已查）。
        name = text.split(used[0], 1)[1].strip()
        if not name:
            raise DraftError(f"{where} 的署名前缀后面没跟署名：{text[-20:]!r}")
        if parse.ends_terminal(name):
            raise DraftError(f"{where} 的署名不该以标点结尾：{name!r}")
        return
    if slot.kind == KIND_CALLOUT:
        pool = [str(word) for word in loc["callout_word"][slot.callout_kind]]
        if not any(text.startswith(word) for word in pool):
            raise DraftError(f"{where} 没以 {pool} 里的词开头：{text[:20]!r}")
        others = [word for word in words if word not in pool]
        if any(word in text for word in others):
            raise DraftError(f"{where} 里混进了别的种类的关键词：{text[:40]!r}")
        low, high = lim.callout
        if not low <= len(text) <= high:
            raise DraftError(f"{where} 长度 {len(text)} 不在 {low}~{high}")
        if not parse.ends_terminal(text):
            raise DraftError(f"{where} 必须以句末标点结尾：{text[-20:]!r}")
        return
    if slot.kind == KIND_PARAGRAPH:
        low, high = lim.prose
        if not low <= len(text) <= high:
            raise DraftError(f"{where} 长度 {len(text)} 不在 {low}~{high}")
        if not parse.ends_terminal(text):
            raise DraftError(f"{where} 必须以句末标点结尾：{text[-20:]!r}")
        return
    raise DraftError(f"认不出的块类型：{slot.kind!r}")


def check_draft(plan: DocPlan, texts: list[str], loc: dict[str, JsonValue]) -> None:
    """校验后端写的那一份文本是否兑现了块计划。

    输入：块计划、后端交回的槽位文本、语言字典。输出：无（不合格抛 `ValueError`，管线据此重渲染）。
    预期行为：分两步——

    1. **逐块查字面要求**（长度、结尾标点、不许出现的标记词、必须出现的标记词）；
    2. **整体反解一遍**：把文本按渲染规则排成纯文本再读回来，块的类型 / 提示框种类 / 列表有序性
       必须与计划**逐项相同**。第 2 步是这一族的核心——它把「渲染没兑现计划」变成渲染期就能拦住
       的错误，不用等判别器。
    """
    if len(texts) != len(plan.slots):
        raise DraftError(f"后端给了 {len(texts)} 段文字，计划要 {len(plan.slots)} 块")
    marks = parse.attribution_marks(loc)
    steps = parse.seq_words(loc)
    lim = parse.limits(loc)
    cleaned = [text.strip() for text in texts]
    for index, (slot, text) in enumerate(zip(plan.slots, cleaned, strict=True), start=1):
        if not text:
            raise DraftError(f"第 {index} 块是空的")
        if slot.kind != KIND_CODE and "\n" in text:
            raise DraftError(f"第 {index} 块（{slot.kind}）里带了换行，散文必须写成一段")
        _check_slot(index, slot, text, loc, lim, marks, steps)
    rendered = layout(plan, cleaned, loc)
    doc = parse.parse_state(rendered.state(), loc)
    if doc.kinds() != plan.kinds():
        raise DraftError(f"反解出来的块类型是 {doc.kinds()}，计划要的是 {plan.kinds()}")
    got_callouts = tuple(block.callout_kind for block in doc.callouts())
    if got_callouts != plan.callout_kinds():
        raise DraftError(f"反解出来的提示框种类是 {got_callouts}，计划要的是 {plan.callout_kinds()}")
    got_flags = tuple(block.list_ordered for block in doc.list_items())
    if got_flags != plan.list_flags():
        raise DraftError(f"反解出来的列表有序性是 {got_flags}，计划要的是 {plan.list_flags()}")


# ---------- 渲染 ----------


def blocks_from(plan: DocPlan, texts: list[str]) -> tuple[Block, ...]:
    """输入：块计划、槽位文本。输出：填好文字的块（顺序不变）。

    预期行为：长度对不上直接报错——静默截断会让「模型少写一块」变成一个看不出来的错位。
    """
    if len(texts) != len(plan.slots):
        raise DraftError(f"槽位文本有 {len(texts)} 段，计划有 {len(plan.slots)} 块")
    return tuple(
        Block(
            kind=slot.kind,
            text=text.strip(),
            callout_kind=slot.callout_kind,
            list_ordered=slot.list_ordered,
            list_id=slot.list_id,
        )
        for slot, text in zip(plan.slots, texts, strict=True)
    )


def wrap_text(text: str, width: int) -> list[str]:
    """输入：一段散文、行宽。输出：折行后的行列表。

    预期行为：贪心——每行尽量塞满 `width` 个字符；窗口里最后一个空格够靠后就在空格处断（英文），
    否则直接按字符数断（中文）。两种语言用同一条规则，所以折行不依赖语言。
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

    def asked_pairs(self, loc: dict[str, JsonValue]) -> list[tuple[str, str]]:
        """输入：语言字典。输出：真正出题的相邻行对（已按「是 / 不是」配平）。

        预期行为：**直接用 `parse.asked_pairs` + `parse.balanced_pairs`**——题目侧与审计侧重算用的是
        同一套判据，否则两边会各自算出不同的题号。配平的理由见 `parse.balanced_pairs`。
        """
        lines = tuple(
            parse.ParsedLine(ident, text)
            for ident, text in zip(self.ids, self.texts, strict=True)
        )
        candidates = parse.asked_pairs(lines, parse.command_tokens(loc))
        return parse.balanced_pairs(candidates, self.same_block)


def layout(plan: DocPlan, texts: list[str], loc: dict[str, JsonValue]) -> Rendered:
    """输入：块计划、槽位文本、语言字典。输出：渲染结果（行 / 块坐标一起算出来）。

    预期行为：块按顺序落成行；块前按 `needs_blank` 决定要不要空行；代码块整块不折行。
    """
    del loc
    blocks = blocks_from(plan, texts)
    out: list[str] = []
    spans: list[tuple[int, int]] = []
    for index, block in enumerate(blocks):
        if needs_blank(blocks, index):
            out.append("")
        start = len(out) + 1
        if block.kind == KIND_CODE:
            out.extend(_lines_of(block.text))
        else:
            out.extend(wrap_text(block.text, plan.case.wrap))
        spans.append((start, len(out)))
    ids = tuple(parse.line_id(number) for number in range(1, len(out) + 1))
    return Rendered(ids, tuple(out), tuple(spans), blocks)


def render_state(plan: DocPlan, texts: list[str], loc: dict[str, JsonValue]) -> str:
    """输入：块计划、槽位文本、语言字典。输出：state 字符串。预期行为：`layout` 的文本形态。"""
    return layout(plan, texts, loc).state()


def parse_state(state: str) -> parse.ParsedDoc:
    """输入：渲染后的 state。输出：纯代码能反解的部分。

    预期行为：用 `locales.merged()` 的**合并读法**——state 的行号与块结构是语言中立的（没有抬头
    可以认语言），所以反解只能拿「所有语言的标记一起认」。
    """
    return parse.parse_state(state, locales.merged())


# ---------- 提示词 ----------


def _role_of(loc: dict[str, JsonValue], slot: Slot) -> str:
    """输入：语言字典、槽位。输出：这一块的**类型名**（提示框带上种类）。"""
    name = str(loc["type"][slot.kind])
    if slot.kind != KIND_CALLOUT:
        return name
    kind = str(loc["callout_kind"][slot.callout_kind])
    return str(loc["doc"]["callout_role"]).format(kind=kind)


def _rule_of(loc: dict[str, JsonValue], slot: Slot) -> str:
    """输入：语言字典、槽位。输出：这一块的字面要求（字符区间、标点、必须带 / 不许带什么）。"""
    sep = str(loc["doc"]["list_sep"])
    rules = loc["rule"]
    lim = parse.limits(loc)
    if slot.kind == KIND_HEADING:
        low, high = lim.heading
        return str(rules["heading"]).format(min=low, max=high)
    if slot.kind == KIND_PARAGRAPH:
        low, high = lim.prose
        return str(rules["paragraph"]).format(min=low, max=high)
    if slot.kind == KIND_LIST_ITEM:
        key = "list_item_ordered" if slot.list_ordered else "list_item_bullet"
        low, high = lim.item
        return str(rules[key]).format(
            min=low, max=high, steps=sep.join(parse.seq_words(loc))
        )
    if slot.kind == KIND_CODE:
        tools = [str(item) for item in loc["code_tool"][slot.code_family]]
        return str(rules["code"]).format(
            min=MIN_CODE_LINES,
            max=MAX_CODE_LINES,
            family=str(loc["code_family"][slot.code_family]),
            tools=sep.join(tools),
        )
    if slot.kind == KIND_QUOTE:
        low, high = lim.quote
        return str(rules["quote"]).format(
            min=low, max=high, mark=parse.attribution_marks(loc)[0]
        )
    if slot.kind == KIND_CALLOUT:
        words = [str(word) for word in loc["callout_word"][slot.callout_kind]]
        low, high = lim.callout
        return str(rules["callout"]).format(
            min=low, max=high, words=sep.join(words)
        )
    raise DraftError(f"认不出的块类型：{slot.kind!r}")


def build_prompt(loc: dict[str, JsonValue], plan: DocPlan, feedback: str | None = None) -> str:
    """组装提示词。

    输入：语言字典、块计划、可选的上一次校验失败理由。输出：完整提示词（不含答案）。
    预期行为：**每一条硬性要求都写清楚**——逐块给类型与字面要求，再逐条列出「不许出现」的标记词
    （并说明命令行块例外），最后是自查与输出格式。这些就是「受控合成」的规格，判据全在
    `check_draft` 里机械复核，模型写歪了当场重渲染。

    为什么把「不许出现」也逐条列出来：提示框关键词与署名前缀是**反解的判据**——模型顺手在一段
    正文里写个「注意：」或「——」，那一块当场被读成提示框或引用，整份文档的块切分就错了。

    `feedback` 非空时附在**末尾**（`check_draft` 的原话）：字数对模型是**计数任务**，
    盲重采样修不掉，而原话里带着「哪一块、差多少」这个确定性信号。
    """
    doc = loc["doc"]
    lines = [str(doc["intro"]).format(topic=str(loc["topic_name"][plan.case.topic]))]
    lines.append(str(doc["lang_rule"]).format(language=loc["language_name"]))
    lines.append(str(doc["mark_rule"]))
    lines.append(str(doc["slot_header"]))
    for index, slot in enumerate(plan.slots, start=1):
        lines.append(
            str(doc["slot_line"]).format(
                index=index, role=_role_of(loc, slot), rule=_rule_of(loc, slot)
            )
        )
    forbidden = forbidden_words(plan, loc)
    if forbidden:
        lines.append(str(doc["forbidden_header"]))
        lines.append(
            str(doc["forbidden_line"]).format(words=str(doc["list_sep"]).join(forbidden))
        )
    lines.append(str(doc["self_check"]))
    lines.append(str(doc["output"]))
    if feedback is not None:
        lines.append(str(doc["fix_header"]))
        lines.append(feedback)
    return "\n".join(lines)


def build_task(
    spec: DocSpec, loc: dict[str, JsonValue], feedback: str | None = None
) -> GenerationTask:
    """输入：真值规格、语言字典、可选的上一次校验失败理由。输出：渲染任务。

    预期行为：任务 id 由规格算（断点续跑按它查进度），`spec` 里带上块计划——后端实现与假后端
    若想「照着计划答」，不必再解析提示词。**重写沿用同一个任务 id**：那是同一个 case 的同一次
    渲染，换 id 会让断点续跑把重写当成新任务再跑一遍。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="autoformat",
        prompt=build_prompt(loc, spec.plan, feedback),
        response_model=SlotDraft,
        seeds=(),
        spec={
            "case": spec.case.as_dict(),
            "kinds": list(spec.plan.kinds()),
            "callout_kinds": list(spec.plan.callout_kinds()),
            "list_flags": list(spec.plan.list_flags()),
        },
    )
