"""情感与情绪（sentiment）的结构反解：把 `state` 读回场景、词表与对话。

输入：`state` 文本 + 一张表（抬头 / 区块抬头 / 场景抬头 / 格分隔符 / 词分隔符 / 规则前缀）。
输出：`ParsedArc`。预期行为：只做结构反解，**不做归一化**——情绪词直接映射成档位，因为
词表就在 state 里印着。反解不复用渲染器的任何计算：区块靠抬头认、对话行靠行首编号认、
格靠分隔符切，所以渲染把某个词写漏了、写重了、编号跳号了，这里都会当场报人话。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from jc.jsontypes import JsonValue

from . import locales
from .features import LEVELS, TURNS

#: 对话行：行首编号 + 后面的格子。编号后的点号是结构，中英共用。
_ROW: Final[re.Pattern[str]] = re.compile(r"^(\d+)\.\s*(.+)$")
#: 词表行：档位标记 + 格分隔符 + 该档的词。档位标记允许带正号。
_LEX: Final[re.Pattern[str]] = re.compile(r"^([+-]?\d+)\s*(.*)$")

#: 一行对话必须有几格（说话人 / 情绪词）。
ROW_CELLS: Final[int] = 2


@dataclass(frozen=True)
class Tables:
    """反解用的表：抬头标签的几种写法、场景抬头、{区块抬头原文: slot}、两种分隔符与规则前缀。"""

    request: tuple[str, ...]
    topic_labels: tuple[str, ...]
    sections: dict[str, str]
    cell_sep: str
    word_sep: str
    rules_prefix: str


@dataclass(frozen=True)
class ParsedTurn:
    """反解出来的一个回合：说话人称呼原文 + 情绪档位。"""

    speaker: str
    level: int


@dataclass(frozen=True)
class ParsedArc:
    """反解出来的一整段：请求号 + 场景键 + 词表 + 每个回合 + 判读规则原文。"""

    request_id: str
    topic: str
    words: dict[int, tuple[str, ...]]
    turns: tuple[ParsedTurn, ...]
    rules: tuple[str, ...]

    @property
    def levels(self) -> tuple[int, ...]:
        """输入：无。输出：逐回合的分值序列（与 `features` 的入参同形）。"""
        return tuple(turn.level for turn in self.turns)

    @property
    def count(self) -> int:
        """输入：无。输出：这段对话有几个回合。"""
        return len(self.turns)


def tables_of(loc: dict[str, JsonValue]) -> Tables:
    """输入：语言字典。输出：`Tables`（只收这一种语言的写法）。"""
    meta = loc["meta"]
    request = str(meta["request"])
    if not request.strip():
        raise ValueError("抬头不能是空串")
    topic_label = str(meta["topic"])
    if not topic_label.strip():
        raise ValueError("场景抬头不能是空串")
    sections = {
        f"[{meta['words']}]": "words",
        f"[{meta['turns']}]": "turns",
        f"[{meta['rules']}]": "rules",
    }
    cell_sep = str(meta["cell_sep"])
    if not cell_sep.strip():
        raise ValueError("格分隔符不能是空白")
    word_sep = str(meta["word_sep"])
    if not word_sep.strip():
        raise ValueError("词分隔符不能是空白")
    return Tables(
        request=(request,),
        topic_labels=(topic_label,),
        sections=sections,
        cell_sep=cell_sep,
        word_sep=word_sep,
        rules_prefix=str(meta["rules_prefix"]),
    )


def merged_tables() -> Tables:
    """输入：无。输出：全部语言合并后的表。

    预期行为：`Workflow.reparse` 没有语言参数，只能靠合并表；各语言的标签与结构词元互不
    重叠（由 `tests/workflows/sentiment/test_locales.py` 锁住），重叠了这里报错而不是静默取其一。
    """
    sections: dict[str, str] = {}
    requests: list[str] = []
    topic_labels: list[str] = []
    cell_seps: set[str] = set()
    word_seps: set[str] = set()
    rules_prefixes: set[str] = set()
    for lang in locales.available():
        tables = tables_of(locales.load(lang))
        for text, slot in tables.sections.items():
            if text in sections:
                raise ValueError(f"区块抬头 {text!r} 在多种语言里都出现（{sections[text]} / {slot}）")
            sections[text] = slot
        requests.extend(tables.request)
        topic_labels.extend(tables.topic_labels)
        cell_seps.add(tables.cell_sep)
        word_seps.add(tables.word_sep)
        rules_prefixes.add(tables.rules_prefix)
    if len(set(requests)) != len(requests):
        raise ValueError(f"抬头标签在多种语言里重复：{requests}")
    if len(set(topic_labels)) != len(topic_labels):
        raise ValueError(f"场景抬头在多种语言里重复：{topic_labels}")
    if len(cell_seps) != 1:
        raise ValueError(f"格分隔符各语言不一致：{sorted(cell_seps)}——反解按它切格，必须唯一")
    if len(word_seps) != 1:
        raise ValueError(f"词分隔符各语言不一致：{sorted(word_seps)}")
    if len(rules_prefixes) != 1:
        raise ValueError(f"规则行前缀各语言不一致：{sorted(rules_prefixes)}")
    labels = [*requests, *topic_labels, *sections]
    for left in labels:
        for right in labels:
            if left != right and right.startswith(left):
                raise ValueError(f"标签 {left!r} 是 {right!r} 的前缀，反解会认错行")
    return Tables(
        request=tuple(requests),
        topic_labels=tuple(topic_labels),
        sections=sections,
        cell_sep=next(iter(cell_seps)),
        word_sep=next(iter(word_seps)),
        rules_prefix=next(iter(rules_prefixes)),
    )


def _header(line: str, labels: tuple[str, ...], what: str) -> str:
    """输入：一行文本、抬头标签的几种写法、报错用的名字。输出：抬头后面的值。

    预期行为：**最长前缀优先**（标签之间有前缀关系时才不会认错）；后面必须是空白，
    否则更长的抬头会被认成标签本身。
    """
    best = ""
    for label in labels:
        if line.startswith(label) and len(label) > len(best):
            rest = line[len(label) :].lstrip(" \t")
            if rest:
                best = label
    if not best:
        raise ValueError(f"{what}抬头认不出来：{line!r}")
    return line[len(best) :].lstrip(" \t")


def _lexicon_row(line: str, tables: Tables) -> tuple[int, tuple[str, ...]]:
    """输入：词表行原文、表。输出：(档位, 该档的词)。

    预期行为：档位标记后必须紧跟格分隔符，词按词分隔符切；空行、缺词、档位不在 -2~2 里
    都当场报人话——词表是判据的一部分，写歪了四个答案全跟着歪。
    """
    matched = _LEX.match(line.strip())
    if matched is None:
        raise ValueError(f"认不出的词表行：{line!r}")
    marker, rest = matched.group(1), matched.group(2).strip()
    level = int(marker)
    if level not in LEVELS:
        raise ValueError(f"词表档位不在 {list(LEVELS)} 里：{marker!r}")
    if not rest.startswith(tables.cell_sep):
        raise ValueError(f"词表行的档位后面缺少格分隔符 {tables.cell_sep!r}：{line!r}")
    parts = [part.strip() for part in rest[len(tables.cell_sep) :].split(tables.word_sep)]
    if not parts or any(not part for part in parts):
        raise ValueError(f"词表行的词为空：{line!r}")
    return level, tuple(parts)


def parse_state(state: object, tables: Tables) -> ParsedArc:
    """输入：渲染出来的 `state`、反解用的表。输出：`ParsedArc`。

    预期行为：抬头 → 场景 → 区块。词表区块内只认档位行，对话区块内只认**带编号的行**，
    编号必须从 1 连续，跳号说明渲染漏写了回合，当场报错。规则区块内的非空行收进 `rules`。
    """
    if not isinstance(state, str):
        raise ValueError(f"state 不是字符串：{type(state).__name__}")
    lines = state.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) < 4:
        raise ValueError(f"state 太短，认不出对话：{state!r}")
    request_id = _header(lines[0], tables.request, "请求号")
    topic = ""
    words: dict[int, tuple[str, ...]] = {}
    raw: list[tuple[str, str]] = []
    rules: list[str] = []
    section = ""
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        topic_label = next(
            (label for label in tables.topic_labels if stripped.startswith(f"[{label}]")), ""
        )
        if topic_label:
            topic = stripped[len(topic_label) + 2 :].strip()
            if not topic:
                raise ValueError(f"场景抬头后面没有场景名：{line!r}")
            continue
        if stripped in tables.sections:
            section = tables.sections[stripped]
            continue
        if not section:
            raise ValueError(f"内容出现在任何区块抬头之前：{line!r}")
        if section == "rules":
            rules.append(stripped[len(tables.rules_prefix) :].strip())
            continue
        if section == "words":
            level, texts = _lexicon_row(stripped, tables)
            if level in words:
                raise ValueError(f"词表里档位 {level} 出现了两次：{line!r}")
            words[level] = texts
            continue
        matched = _ROW.match(stripped)
        if matched is None:
            raise ValueError(f"认不出的对话行：{line!r}")
        number = int(matched.group(1))
        expected = len(raw) + 1
        if number != expected:
            raise ValueError(f"回合号跳号：期望 {expected}，实际 {number}")
        cells = [cell.strip() for cell in matched.group(2).split(tables.cell_sep)]
        if len(cells) != ROW_CELLS:
            raise ValueError(f"对话行有 {len(cells)} 格，应为 {ROW_CELLS} 格：{line!r}")
        raw.append((cells[0], cells[1]))
    if not topic:
        raise ValueError("state 里没有场景抬头")
    if set(words) != set(LEVELS):
        raise ValueError(f"词表缺档位：{sorted(set(LEVELS) - set(words))}")
    lookup = {word: level for level, texts in words.items() for word in texts}
    if len(lookup) != sum(len(texts) for texts in words.values()):
        raise ValueError("词表里有重复的词——反解会把两个档位认成一个")
    if not raw:
        raise ValueError("对话里一个回合都没有")
    if len(raw) != TURNS:
        raise ValueError(f"这段对话有 {len(raw)} 个回合，应为 {TURNS} 个")
    turns: list[ParsedTurn] = []
    for speaker, word in raw:
        if word not in lookup:
            raise ValueError(f"情绪词不在词表里：{word!r}——渲染写出了表外的词")
        turns.append(ParsedTurn(speaker=speaker, level=lookup[word]))
    return ParsedArc(
        request_id=request_id,
        topic=topic,
        words=dict(words),
        turns=tuple(turns),
        rules=tuple(rules),
    )
