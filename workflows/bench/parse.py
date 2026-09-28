"""评测与基准 workflow 的反解：把一屏评测记录读回结构化字段。

输入：`state` 文本 + 标签表。输出：`ParsedRecord`。
预期行为：**不复用渲染时的任何计算**——只按文本本身解析，所以能抓到抬头漏写、行数不对、
某一格缺字母、同一行候选重复、各行列序不一致这类硬伤。认不出的行（标题行）直接忽略。

标签表里每一格存的都是**候选原文元组**：单语言表只有一项，`merged()` 给的合并表把各语言的
写法都收进来（`Workflow.reparse` 没有语言参数，只能靠合并表做语言无关的反解）。候选词表是
「原文 → 键」的并集，判别器与审计拿**键**去查规则。

**字母不在语言文件里**：A / B / C / D 是符号不是文案，中英共用同一套，所以它直接从
`spec.LETTERS` 取，也不参与「跨语言不重叠」那道检查（同一段原文在两种语言里都出现是它的
常态，不是冲突）。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from dataclasses import dataclass
from typing import Final

from . import locales
from .spec import BARS, CANDIDATES, LETTERS, TOTAL

#: 一行里「参考 + 三个候选」的格数。
CELL_COUNT: Final[int] = 1 + len(CANDIDATES)
#: 抬头字段名（跨语言必须互不相同，否则合并表有歧义）。
LABEL_FIELDS: Final[tuple[str, ...]] = ("request", "bar")


@dataclass(frozen=True)
class Labels:
    """反解要用的全部原文：抬头字段名的候选、参考格标签的候选、冒号的候选、两个分隔符、候选词表。"""

    label_texts: dict[str, tuple[str, ...]]
    ref_texts: tuple[str, ...]
    colons: tuple[str, ...]
    cell_sep: str
    pair_sep: str
    cand: dict[str, str]

    def labels_of(self, field: str) -> tuple[str, ...]:
        """输入：抬头字段名。输出：它的候选原文。预期行为：认不出直接报错。"""
        if field not in self.label_texts:
            raise KeyError(f"没有这个抬头字段的标签：{field!r}（可用：{list(self.label_texts)}）")
        return self.label_texts[field]

    def cand_key(self, text: str) -> str:
        """输入：候选原文。输出：候选键。预期行为：认不出直接报人话。"""
        if text not in self.cand:
            raise KeyError(f"候选词表里没有 {text!r}（可用：{sorted(self.cand)}）")
        return self.cand[text]


@dataclass(frozen=True)
class ParsedCell:
    """反解出来的一格：候选键 + 答案字母。"""

    cand: str
    letter: str


@dataclass(frozen=True)
class ParsedRow:
    """反解出来的一行：参考字母 + 按列序排的三格。"""

    ref: str
    cells: tuple[ParsedCell, ...]


@dataclass(frozen=True)
class ParsedRecord:
    """反解出来的一屏记录：集号 + 门槛 + 五行。"""

    request_id: str
    bar: int
    rows: tuple[ParsedRow, ...]

    @property
    def layout(self) -> tuple[str, ...]:
        """输入：无。输出：这屏记录的列序（候选键，按列）。预期行为：各行列序一致（解析时已验）。"""
        return tuple(cell.cand for cell in self.rows[0].cells)


def labels_of(loc: dict[str, JsonValue]) -> Labels:
    """输入：语言字典。输出：反解要用的标签表。

    预期行为：候选词表的键必须**恰好**是 `spec.CANDIDATES`，两个键共用同一个原文直接报错——
    少一个键会让某一行反解不出来，共用原文会让反解认错键，两种都是硬伤。
    """
    cand = {str(key): str(text) for key, text in loc["cand"].items()}
    if set(cand) != set(CANDIDATES):
        raise KeyError(f"候选词表的键应该是 {list(CANDIDATES)}，收到 {sorted(cand)}")
    if len(set(cand.values())) != len(cand):
        raise ValueError(f"候选词表里有两条共用同一个原文：{cand}")
    return Labels(
        label_texts={field: (str(loc["label"][field]),) for field in LABEL_FIELDS},
        ref_texts=(str(loc["cell"]["ref"]),),
        colons=(str(loc["meta"]["colon"]),),
        cell_sep=str(loc["meta"]["cell_sep"]),
        pair_sep=str(loc["meta"]["pair_sep"]),
        cand={text: key for key, text in cand.items()},
    )


def _label_union(texts: list[str], what: str) -> tuple[str, ...]:
    """输入：各语言给的原文、名字。输出：去重后的元组。

    预期行为：同一段原文在两种语言里都出现就报错——合并表按原文反查，歧义会让反解认错。
    """
    out: list[str] = []
    for text in texts:
        if text in out:
            raise ValueError(f"{what} 的原文 {text!r} 在多种语言里都出现（合并表会有歧义）")
        out.append(text)
    return tuple(out)


def _union(pairs: list[tuple[str, str]], what: str) -> dict[str, str]:
    """输入：[(原文, 键)]、名字。输出：合并后的「原文 → 键」。预期行为：同原文不同键直接报错。"""
    out: dict[str, str] = {}
    for text, key in pairs:
        if out.get(text, key) != key:
            raise ValueError(f"{what} 的原文 {text!r} 在多种语言里指向不同键（{out[text]} / {key}）")
        out[text] = key
    return out


def merged() -> Labels:
    """输入：无。输出：所有语言合起来的标签表（语言无关的反解用）。

    预期行为：各语言的格分隔符与冒号必须一致（只按一个分隔符切行）；字段名与候选词表跨语言
    必须互不重叠（同一段原文反查不出两个键）。字母是共用的符号，不进这张表。

    冒号**不要求一致**（中文写 `：`、英文写 `: `），所以合并表把各语言的冒号都收进来挨个试；
    格内的名字 / 字母分隔符必须一致——那一格是两种语言都要反解的。
    """
    langs = locales.available()
    if not langs:
        raise ValueError("一个语言文件都没有，没法做语言无关的反解")
    per_lang = [(lang, labels_of(locales.load(lang))) for lang in langs]
    seps = {labels.cell_sep for _, labels in per_lang}
    if len(seps) != 1:
        raise ValueError(f"各语言的格分隔符必须一致，现在是 {sorted(seps)}")
    pairs = {labels.pair_sep for _, labels in per_lang}
    if len(pairs) != 1:
        raise ValueError(f"各语言格内的名字 / 字母分隔符必须一致，现在是 {sorted(pairs)}")
    first = per_lang[0][1]
    return Labels(
        label_texts={
            field: _label_union(
                [text for _, labels in per_lang for text in labels.labels_of(field)],
                f"抬头字段 {field}",
            )
            for field in LABEL_FIELDS
        },
        ref_texts=_label_union(
            [text for _, labels in per_lang for text in labels.ref_texts], "参考格标签"
        ),
        colons=tuple(
            sorted({colon for _, labels in per_lang for colon in labels.colons}, key=len, reverse=True)
        ),
        cell_sep=first.cell_sep,
        pair_sep=first.pair_sep,
        cand=_union(
            [(text, key) for _, labels in per_lang for text, key in labels.cand.items()],
            "候选词表",
        ),
    )


def _field_value(line: str, labels: tuple[str, ...], colons: tuple[str, ...]) -> str | None:
    """输入：一行文本、候选字段名、冒号的候选。输出：冒号后的值；都不是这一格时给 None。

    预期行为：取**最长**的匹配字段名（字段名之间有前缀关系时才不会认错），并且要求字段名
    后面（允许空格）紧跟**某一个**冒号——否则标题行会被认成某一格。故意收紧。
    冒号要挨个试而不是只试第一个：合并表里中文冒号与英文冒号并存。
    """
    best: tuple[int, int] | None = None  # (字段名长度, 值在行里的起点)
    for label in labels:
        if not line.startswith(label):
            continue
        rest = line[len(label) :]
        stripped = rest.lstrip()
        pad = len(rest) - len(stripped)
        for colon in colons:
            if stripped.startswith(colon):
                if best is None or len(label) > best[0]:
                    best = (len(label), len(label) + pad + len(colon))
                break
    if best is None:
        return None
    return line[best[1] :].strip()


def _split_pair(text: str, sep: str, what: str) -> tuple[str, str]:
    """输入：一格文本、名字 / 字母分隔符、名字。输出：(分隔符前的名字, 分隔符后的值)。

    预期行为：必须恰好切成两段——少一段说明渲染漏写了名字或字母，多一段说明名字里混进了
    分隔符，两种都是硬伤，不能靠 `maxsplit=1` 蒙过去。
    """
    parts = [part.strip() for part in text.split(sep)]
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"{what}应该是「名字{sep}值」两段，收到 {text!r}")
    return parts[0], parts[1]


def _check_letter(what: str, letter: str) -> None:
    """输入：名字、字母。输出：无。预期行为：不是四个选项字母之一直接报人话。"""
    if letter not in LETTERS:
        raise ValueError(f"{what}的答案字母认不出来：{letter!r}（可用：{list(LETTERS)}）")


def _parse_row(line: str, labels: Labels) -> ParsedRow:
    """输入：一行记录、标签表。输出：反解出来的一行。

    预期行为：`[序号] 参考格 · 候选格 · 候选格 · 候选格`；格数不对、第一格不是参考格、
    候选名认不出、字母认不出、同一行候选重复，都直接报人话。
    """
    end = line.find("]")
    if end < 0:
        raise ValueError(f"记录行缺右方括号：{line!r}")
    cells = [cell.strip() for cell in line[end + 1 :].strip().split(labels.cell_sep)]
    if len(cells) != CELL_COUNT:
        raise ValueError(f"记录行应该是 {CELL_COUNT} 格，收到 {len(cells)} 格：{line!r}")
    name, letter = _split_pair(cells[0], labels.pair_sep, "参考格")
    if name not in labels.ref_texts:
        raise ValueError(f"第一格应该是参考答案（{labels.ref_texts[0]!r}），收到 {name!r}")
    _check_letter("参考答案", letter)
    out: list[ParsedCell] = []
    for cell in cells[1:]:
        cand_name, cand_letter = _split_pair(cell, labels.pair_sep, "候选格")
        _check_letter(f"候选 {cand_name!r}", cand_letter)
        out.append(ParsedCell(labels.cand_key(cand_name), cand_letter))
    keys = [cell.cand for cell in out]
    if len(set(keys)) != len(keys):
        raise ValueError(f"同一行里候选重复了：{keys}")
    return ParsedRow(ref=letter, cells=tuple(out))


def _parse_bar(text: str) -> int:
    """输入：门槛那一格的原文。输出：整数。预期行为：不是整数或出了取值域都报人话。"""
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"门槛不是整数：{text!r}") from exc
    if value not in BARS:
        raise ValueError(f"门槛出了取值域：{value}（可用：{BARS}）")
    return value


def parse_record(state: str, labels: Labels) -> ParsedRecord:
    """输入：state 文本、标签表。输出：反解结果。

    预期行为：逐行认——`[` 开头的是记录行；其余按字段名取，取不到就忽略（标题行）。
    集号与门槛缺任何一样、行数不对、各行列序不一致都报人话。
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
            rows.append(_parse_row(line, labels))
            continue
        for field in LABEL_FIELDS:
            value = _field_value(line, labels.labels_of(field), labels.colons)
            if value is not None:
                found[field] = value
    absent = [field for field in LABEL_FIELDS if field not in found]
    if absent:
        names = [labels.labels_of(field)[0] for field in absent]
        raise ValueError(f"记录缺这几格：{' / '.join(names)}")
    if len(rows) != TOTAL:
        raise ValueError(f"记录应该是 {TOTAL} 行，收到 {len(rows)} 行")
    orders = {tuple(cell.cand for cell in row.cells) for row in rows}
    if len(orders) != 1:
        raise ValueError(f"各行的列序必须一致，现在是 {sorted(orders)}")
    return ParsedRecord(request_id=found["request"], bar=_parse_bar(found["bar"]), rows=tuple(rows))
