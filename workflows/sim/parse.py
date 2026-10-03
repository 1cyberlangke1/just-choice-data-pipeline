"""游戏与模拟控制 workflow 的反解：把一屏局面读回结构化字段。

输入：`state` 文本 + 标签表。输出：`ParsedBoard`。
预期行为：**不复用渲染时的任何计算**——只按文本本身解析，所以能抓到抬头漏写、格数不对、
某一行缺格这类硬伤。认不出的行（标题行）直接忽略。

标签表里每一格存的都是**候选原文元组**：单语言表只有一项，`merged()` 给的合并表把各语言
的写法都收进来（`Workflow.reparse` 没有语言参数，只能靠合并表做语言无关的反解）。
四张词表（动作 / 耗 / 收益 / 冷却）也是「原文 → 键」的并集，判别器与审计拿**键**去查规则。

抬头三格（回合 / 资源 / 战利品）在文本里是整数，反解按 spec 的三张取值域校验范围——
写出界说明渲染错了，不能当成「认不出来就跳过」。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue
from jc.locale_files import merge_labels
from jc.state_parse import read_field_value

from dataclasses import dataclass
from typing import Final

from . import locales
from .spec import ACTIONS, HAULS, RESOURCES, TURNS

#: 局面行里四格的顺序（与渲染一致，反解按它取格）。
CELL_FIELDS: Final[tuple[str, ...]] = ("action", "cost", "gain", "cooldown")
#: 抬头字段名（跨语言必须互不相同，否则合并表有歧义）。
LABEL_FIELDS: Final[tuple[str, ...]] = ("request", "turn", "resources", "haul")
#: 抬头三格里**是整数**的那几格（单号是字符串）。
NUMERIC_LABELS: Final[tuple[str, ...]] = ("turn", "resources", "haul")
#: 词表名（合并表要逐张做「跨语言不重叠」检查）。
VOCAB_FIELDS: Final[tuple[str, ...]] = ("action", "cost", "gain", "cooldown")
#: 三格整数的取值域。
LABEL_RANGES: Final[dict[str, tuple[int, ...]]] = {
    "turn": TURNS,
    "resources": RESOURCES,
    "haul": HAULS,
}


@dataclass(frozen=True)
class Labels:
    """反解要用的全部原文：四个抬头字段名的候选、四格分隔符、四张「原文 → 键」词表。"""

    label_texts: dict[str, tuple[str, ...]]
    cell_sep: str
    action: dict[str, str]
    cost: dict[str, str]
    gain: dict[str, str]
    cooldown: dict[str, str]

    def table(self, field: str) -> dict[str, str]:
        """输入：词表名。输出：该词表的「原文 → 键」映射。预期行为：认不出直接报错。"""
        if field not in VOCAB_FIELDS:
            raise KeyError(f"没有词表 {field!r}（可用：{', '.join(VOCAB_FIELDS)}）")
        return getattr(self, field)

    def labels_of(self, field: str) -> tuple[str, ...]:
        """输入：抬头字段名。输出：候选原文。预期行为：认不出直接报错。"""
        if field not in LABEL_FIELDS:
            raise KeyError(f"没有标签 {field!r}（可用：{', '.join(LABEL_FIELDS)}）")
        return self.label_texts[field]


@dataclass(frozen=True)
class ParsedRow:
    """反解出来的一行动作：动作 / 耗 / 收益 / 冷却（都是键）。"""

    action: str
    cost: str
    gain: str
    cooldown: str


@dataclass(frozen=True)
class ParsedBoard:
    """反解出来的一屏局面：单号 + 抬头三格 + 六行动作。"""

    request_id: str
    turn: int
    resources: int
    haul: int
    rows: tuple[ParsedRow, ...]


def _pairs(loc: dict[str, JsonValue], field: str) -> dict[str, str]:
    """输入：语言字典、词表名。输出：{原文: 键}。预期行为：原文重复直接报错。"""
    table = loc[field]
    out: dict[str, str] = {}
    for key, text in table.items():
        if text in out:
            raise ValueError(f"{field} 里有两个键共用同一个原文 {text!r}")
        out[str(text)] = str(key)
    return out


def labels_of(loc: dict[str, JsonValue]) -> Labels:
    """输入：语言字典。输出：这一门语言的反解表（每一格只有一项候选）。"""
    return Labels(
        label_texts={field: (str(loc["label"][field]),) for field in LABEL_FIELDS},
        cell_sep=str(loc["meta"]["cell_sep"]),
        action=_pairs(loc, "action"),
        cost=_pairs(loc, "cost"),
        gain=_pairs(loc, "gain"),
        cooldown=_pairs(loc, "cooldown"),
    )


def _union(entries: list[tuple[str, str]], field: str) -> dict[str, str]:
    """输入：[(原文, 键)]、词表名。输出：{原文: 键}。

    预期行为：同一个原文出现在多种语言里、或者同一门语言里两个键撞车，都直接报人话。
    """
    out: dict[str, str] = {}
    for text, key in entries:
        if text in out:
            raise ValueError(f"{field} 的原文 {text!r} 在多种语言里都出现（合并表有歧义）")
        out[text] = key
    return out


def merged() -> Labels:
    """输入：无。输出：所有语言的合并反解表（`reparse` 用它做语言无关的结构反解）。

    预期行为：四格分隔符各语言必须一致（合并表只按一个分隔符切行），字段名与四张词表
    必须跨语言不重叠。
    """
    langs = locales.available()
    if not langs:
        raise ValueError("一个语言文件都没有")
    per_lang = [(lang, labels_of(locales.load(lang))) for lang in langs]
    seps = {labels.cell_sep for _, labels in per_lang}
    if len(seps) != 1:
        raise ValueError(f"各语言的四格分隔符必须一致，现在是 {sorted(seps)}")
    first = per_lang[0][1]
    return Labels(
        label_texts={
            field: merge_labels(
                [text for _, labels in per_lang for text in labels.labels_of(field)], field
            )
            for field in LABEL_FIELDS
        },
        cell_sep=first.cell_sep,
        action=_union(
            [(text, key) for _, labels in per_lang for text, key in labels.action.items()],
            "action",
        ),
        cost=_union(
            [(text, key) for _, labels in per_lang for text, key in labels.cost.items()],
            "cost",
        ),
        gain=_union(
            [(text, key) for _, labels in per_lang for text, key in labels.gain.items()],
            "gain",
        ),
        cooldown=_union(
            [(text, key) for _, labels in per_lang for text, key in labels.cooldown.items()],
            "cooldown",
        ),
    )


def _decode(field: str, text: str, labels: Labels) -> str:
    """输入：词表名、原文、标签表。输出：键。预期行为：认不出的原文直接报人话。"""
    table = labels.table(field)
    if text not in table:
        raise ValueError(f"{field} 认不出来：{text!r}")
    return table[text]


def _parse_cells(line: str, labels: Labels) -> tuple[str, ...]:
    """输入：局面行、标签表。输出：四格词键（顺序固定）。

    预期行为：`[序号] 格 · 格 · 格 · 格`；格数不对直接报错（少一格说明渲染漏写了属性，
    那是硬伤，不能当成「空值」蒙过去）。
    """
    end = line.find("]")
    if end < 0:
        raise ValueError(f"局面行缺右方括号：{line!r}")
    cells = [cell.strip() for cell in line[end + 1 :].strip().split(labels.cell_sep)]
    if len(cells) != len(CELL_FIELDS):
        raise ValueError(f"局面行应该是 {len(CELL_FIELDS)} 格，收到 {len(cells)} 格：{line!r}")
    return tuple(
        _decode(field, cell, labels) for field, cell in zip(CELL_FIELDS, cells, strict=True)
    )


def _parse_number(field: str, text: str) -> int:
    """输入：抬头字段名、原文。输出：整数。预期行为：不是整数或出了取值域都报人话。"""
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"{field} 不是整数：{text!r}") from exc
    if value not in LABEL_RANGES[field]:
        raise ValueError(f"{field} 出了取值域：{value}（可用：{LABEL_RANGES[field]}）")
    return value


def parse_board(state: str, labels: Labels) -> ParsedBoard:
    """输入：state 文本、标签表。输出：反解结果。

    预期行为：逐行认——`[` 开头的是局面行；其余按字段名取，取不到就忽略（标题行）。
    单号、抬头三格与六行动作缺任何一样都报人话。
    """
    if not isinstance(state, str):
        raise TypeError(f"state 必须是字符串，收到 {type(state).__name__}")
    found: dict[str, str] = {}
    rows: list[ParsedRow] = []
    for raw in state.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("["):
            rows.append(ParsedRow(*_parse_cells(line, labels)))
            continue
        for field in LABEL_FIELDS:
            value = read_field_value(line, labels.labels_of(field))
            if value is not None:
                found[field] = value
    absent = [field for field in LABEL_FIELDS if field not in found]
    if absent:
        names = [labels.labels_of(field)[0] for field in absent]
        raise ValueError(f"局面缺这几格：{' / '.join(names)}")
    if len(rows) != len(ACTIONS):
        raise ValueError(f"局面应该是 {len(ACTIONS)} 行，收到 {len(rows)} 行")
    return ParsedBoard(
        request_id=found["request"],
        turn=_parse_number("turn", found["turn"]),
        resources=_parse_number("resources", found["resources"]),
        haul=_parse_number("haul", found["haul"]),
        rows=tuple(rows),
    )
