"""情感与情绪（sentiment）的 state 渲染：场景、情绪词表、对话与四条判读规则。

**四条规则写进 state，不写进判别器**：规则是判据原文，题面必须自带（范式 §9.8），而 state
只描述「这段对话长什么样」——读者拿 state + 题面就能自己把四个答案算出来。

输入/输出：纯函数，只有字符串拼装；不碰 RNG、不碰后端、不碰文件。
"""

from __future__ import annotations

from typing import Final

from jc.jsontypes import JsonValue

from .features import LEVELS, word_slot
from .spec import SentimentSpec

#: 词表区块的档位顺序：从最好到最差（渲染顺序固定，反解不依赖它）。
LEXICON_LEVELS: Final[tuple[int, ...]] = tuple(reversed(LEVELS))


def topic_text(loc: dict[str, JsonValue], topic: str) -> tuple[str, str, str]:
    """输入：语言字典、场景键。输出：(场景名, 甲称呼, 乙称呼)。

    预期行为：键不在表里直接报错——悄悄退回键名会把「场景没翻译」这件事藏进产物里。
    """
    table = loc["topic"]
    if not isinstance(table, dict) or topic not in table:
        raise ValueError(f"场景 {topic!r} 不在语言表的 [topic] 里")
    entry = table[topic]
    if not isinstance(entry, dict):
        raise ValueError(f"[topic.{topic}] 必须是对象（title / role_a / role_b）")
    return str(entry["title"]), str(entry["role_a"]), str(entry["role_b"])


def speaker_text(loc: dict[str, JsonValue], topic: str, turn: int) -> str:
    """输入：语言字典、场景键、回合下标（0 起）。输出：这一回合的说话人称呼。

    预期行为：两个称呼按回合号交替——**只影响字面**，与四个答案都无关。
    """
    _title, role_a, role_b = topic_text(loc, topic)
    return role_a if turn % 2 == 0 else role_b


def word_text(loc: dict[str, JsonValue], level: int, slot: int) -> str:
    """输入：语言字典、档位、取词下标。输出：这一回合用的情绪词。

    预期行为：越界当场报错——词表每档的词数必须与 `features.WORD_SLOTS` 对齐，
    对不上就是渲染会取到表外词，早点炸更好定位。
    """
    table = loc["word"]
    if not isinstance(table, dict) or str(level) not in table:
        raise ValueError(f"档位 {level} 不在语言表的 [word] 里")
    slot_table = table[str(level)]
    if not isinstance(slot_table, list) or not slot_table:
        raise ValueError(f"[word.{level}] 必须是非空数组")
    return str(slot_table[slot % len(slot_table)])


def level_text(level: int) -> str:
    """输入：档位。输出：词表那一行的档位标记（正数带加号）。"""
    return f"+{level}" if level > 0 else str(level)


def lexicon_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：情绪词表区块的每一行（含行首缩进）。

    预期行为：五档各一行、档位从 +2 到 -2，每行的词按 `word_sep` 连——**整张表都印出来**，
    只读 state 的读者才能把每个词对回分值。
    """
    meta = loc["meta"]
    sep = str(meta["word_sep"])
    lines: list[str] = []
    for level in LEXICON_LEVELS:
        words = loc["word"][str(level)]
        if not isinstance(words, list):
            raise ValueError(f"[word.{level}] 必须是非空数组")
        joined = f" {sep} ".join(str(word) for word in words)
        lines.append(f"  {level_text(level)} {meta['cell_sep']} {joined}")
    return lines


def turn_line(loc: dict[str, JsonValue], topic: str, turn: int, level: int, slot: int) -> str:
    """输入：语言字典、场景键、回合下标、档位、取词下标。输出：对话区块里的一行。

    预期行为：两格按「说话人 | 情绪词」固定顺序拼；编号后的点号与格分隔符都是**结构**
    （中英共用），缩进两格只是给人看的——反解按 `strip()` 后的内容认编号。
    """
    cells = [speaker_text(loc, topic, turn), word_text(loc, level, slot)]
    sep = str(loc["meta"]["cell_sep"])
    return f"  {turn + 1}. {(' ' + sep + ' ').join(cells)}"


def rules_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：判读规则那一段的每一行（含行首前缀）。

    预期行为：整段从语言文件的 `[rules].lines` 原样取——判据必须让只读 state 与题面的读者
    自己算得出来，所以它不能藏在生成器的代码里。
    """
    prefix = str(loc["meta"]["rules_prefix"])
    return [f"{prefix}{line}" for line in loc["rules"]["lines"]]


def build(spec: SentimentSpec, loc: dict[str, JsonValue]) -> str:
    """输入：规格、语言字典。输出：`state` 全文。

    预期行为：区块抬头用方括号包住（与其它 workflow 一致）；场景跟在它自己的抬头后面；
    词表五档、对话五个回合、四条规则依次排开。
    """
    meta = loc["meta"]
    case = spec.case
    title, _role_a, _role_b = topic_text(loc, case.topic)
    levels = case.levels
    lines = [
        f"{meta['request']} {spec.query_id}",
        f"[{meta['topic']}] {title}",
        f"[{meta['words']}]",
        *lexicon_lines(loc),
        f"[{meta['turns']}]",
        *(
            turn_line(loc, case.topic, index, levels[index], word_slot(case.arc, index))
            for index in range(len(levels))
        ),
        f"[{meta['rules']}]",
        *rules_lines(loc),
    ]
    return "\n".join(lines) + "\n"
