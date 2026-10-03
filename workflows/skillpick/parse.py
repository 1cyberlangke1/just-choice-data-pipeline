"""技能选择 workflow 的反解：把一屏选择单读回结构化字段。

输入：`state` 文本 + 标签表。输出：`ParsedRequest`。
预期行为：**不复用渲染时的任何计算**——只按文本本身解析，所以能抓到抬头漏写、格数不对、
某一行缺格这类硬伤。认不出的行（标题行）直接忽略。

标签表里每一格存的都是**候选原文元组**：单语言表只有一项，`merged()` 给的合并表把各语言
的写法都收进来（`Workflow.reparse` 没有语言参数，只能靠合并表做语言无关的反解）。
四张词表（标签 / 权限范围 / 资源 / 授权）也是「原文 → 键」的并集，判别器与审计拿**键**去查规则。

注意「标签」与「词表」是两个命名空间：前者是**行首的字段名**（`标签：…`），后者是**某一格的
取值集合**（标签列里有哪几个标签）。两边的名字有重名（`tag` / `grant`），但一个是
`labels_of()` 的事、一个是 `table()` 的事，互不干扰。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue
from jc.locale_files import merge_labels
from jc.state_parse import read_field_value

from dataclasses import dataclass
from typing import Final

from . import locales
from .spec import CATALOG

#: 技能表行里三格的顺序（与渲染一致，反解按它取格）。
CELL_FIELDS: Final[tuple[str, ...]] = ("tag", "scope", "resource")
#: 行首字段名（跨语言必须互不相同，否则合并表有歧义）。
LABEL_FIELDS: Final[tuple[str, ...]] = ("request", "tag", "grant", "inuse")
#: 词表名（合并表要逐张做「跨语言不重叠」检查）。
VOCAB_FIELDS: Final[tuple[str, ...]] = ("tag", "scope", "resource", "grant")


@dataclass(frozen=True)
class Labels:
    """反解要用的全部原文：四个行首字段名的候选、三格分隔符、四张「原文 → 键」词表。"""

    request_labels: tuple[str, ...]
    tag_labels: tuple[str, ...]
    grant_labels: tuple[str, ...]
    inuse_labels: tuple[str, ...]
    cell_sep: str
    tag: dict[str, str]
    scope: dict[str, str]
    resource: dict[str, str]
    grant: dict[str, str]

    def table(self, field: str) -> dict[str, str]:
        """输入：词表名。输出：该词表的「原文 → 键」映射。预期行为：认不出直接报错。"""
        if field not in VOCAB_FIELDS:
            raise KeyError(f"没有词表 {field!r}（可用：{', '.join(VOCAB_FIELDS)}）")
        return getattr(self, field)

    def labels_of(self, field: str) -> tuple[str, ...]:
        """输入：行首字段名。输出：候选原文。预期行为：认不出直接报错。"""
        if field not in LABEL_FIELDS:
            raise KeyError(f"没有标签 {field!r}（可用：{', '.join(LABEL_FIELDS)}）")
        return getattr(self, f"{field}_labels")


@dataclass(frozen=True)
class ParsedRow:
    """反解出来的一行技能：适用场景 / 权限范围 / 资源（都是键）。"""

    tag: str
    scope: str
    resource: str


@dataclass(frozen=True)
class ParsedRequest:
    """反解出来的一条请求：请求号 + 标签 / 授权 / 已占用 + 六行技能表。"""

    request_id: str
    tag: str
    grant: str
    inuse: str
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
    meta = loc["meta"]
    return Labels(
        request_labels=(str(meta["request_label"]),),
        tag_labels=(str(meta["tag_label"]),),
        grant_labels=(str(meta["grant_label"]),),
        inuse_labels=(str(meta["inuse_label"]),),
        cell_sep=str(meta["cell_sep"]),
        tag=_pairs(loc, "tag"),
        scope=_pairs(loc, "scope"),
        resource=_pairs(loc, "resource"),
        grant=_pairs(loc, "grant"),
    )


def _union(entries: list[tuple[str, str]], field: str) -> dict[str, str]:
    """输入：[(原文, 键)]、词表名。输出：{原文: 键}。

    预期行为：同一个原文出现在多种语言里、或者同一门语言里两个键撞车，都直接报人话——
    合并表有歧义时反解会认错键，那比直接报错危险得多。
    """
    out: dict[str, str] = {}
    for text, key in entries:
        if text in out:
            raise ValueError(f"{field} 的原文 {text!r} 在多种语言里都出现（合并表有歧义）")
        out[text] = key
    return out


def merged() -> Labels:
    """输入：无。输出：所有语言的合并反解表（`reparse` 用它做语言无关的结构反解）。

    预期行为：三格分隔符各语言必须一致（合并表只按一个分隔符切行），字段名与四张词表
    必须跨语言不重叠。
    """
    langs = locales.available()
    if not langs:
        raise ValueError("一个语言文件都没有")
    per_lang = [(lang, labels_of(locales.load(lang))) for lang in langs]
    seps = {labels.cell_sep for _, labels in per_lang}
    if len(seps) != 1:
        raise ValueError(f"各语言的三格分隔符必须一致，现在是 {sorted(seps)}")
    first = per_lang[0][1]
    return Labels(
        request_labels=merge_labels(
            [text for _, labels in per_lang for text in labels.request_labels], "request"
        ),
        tag_labels=merge_labels(
            [text for _, labels in per_lang for text in labels.tag_labels], "tag"
        ),
        grant_labels=merge_labels(
            [text for _, labels in per_lang for text in labels.grant_labels], "grant"
        ),
        inuse_labels=merge_labels(
            [text for _, labels in per_lang for text in labels.inuse_labels], "inuse"
        ),
        cell_sep=first.cell_sep,
        tag=_union(
            [(text, key) for _, labels in per_lang for text, key in labels.tag.items()], "tag"
        ),
        scope=_union(
            [(text, key) for _, labels in per_lang for text, key in labels.scope.items()],
            "scope",
        ),
        resource=_union(
            [(text, key) for _, labels in per_lang for text, key in labels.resource.items()],
            "resource",
        ),
        grant=_union(
            [(text, key) for _, labels in per_lang for text, key in labels.grant.items()],
            "grant",
        ),
    )


def _decode(field: str, text: str, labels: Labels) -> str:
    """输入：词表名、原文、标签表。输出：键。预期行为：认不出的原文直接报人话。"""
    table = labels.table(field)
    if text not in table:
        raise ValueError(f"{field} 认不出来：{text!r}")
    return table[text]


def _parse_cells(line: str, labels: Labels) -> tuple[str, ...]:
    """输入：技能表行、标签表。输出：三格词键（顺序固定）。

    预期行为：`[序号] 格 · 格 · 格`；格数不对直接报错（少一格说明渲染漏写了属性，
    那是硬伤，不能当成「空值」蒙过去）。
    """
    end = line.find("]")
    if end < 0:
        raise ValueError(f"技能行缺右方括号：{line!r}")
    cells = [cell.strip() for cell in line[end + 1 :].strip().split(labels.cell_sep)]
    if len(cells) != len(CELL_FIELDS):
        raise ValueError(f"技能行应该是 {len(CELL_FIELDS)} 格，收到 {len(cells)} 格：{line!r}")
    return tuple(
        _decode(field, cell, labels) for field, cell in zip(CELL_FIELDS, cells, strict=True)
    )


def parse_request(state: str, labels: Labels) -> ParsedRequest:
    """输入：state 文本、标签表。输出：反解结果。

    预期行为：逐行认——`[` 开头的是技能行；其余按四个字段名取，取不到就忽略（标题行）。
    请求号 / 标签 / 授权 / 已占用 / 技能行缺任何一样都报人话。
    """
    if not isinstance(state, str):
        raise TypeError(f"state 必须是字符串，收到 {type(state).__name__}")
    request_id: str | None = None
    fields: dict[str, str] = {}
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
            if value is None:
                continue
            if field == "request":
                request_id = value
            else:
                fields[field] = _decode("tag" if field == "inuse" else field, value, labels)
            break
    # 请求号存在 `request_id` 里、不进 `fields`，所以点名缺格时要把它单独算，不然
    # 每一条选择单都会被判成「缺请求号」。报的是**原文**（合并表下可能有多门语言的写法），
    # 内部字段名（`tag` / `grant`）对看产物的人没有意义。
    missing = [
        " / ".join(labels.labels_of(name))
        for name in LABEL_FIELDS
        if name != "request" and name not in fields
    ]
    if request_id is None:
        missing.insert(0, " / ".join(labels.request_labels))
    if missing:
        raise ValueError(f"选择单缺这几格：{missing}")
    if len(rows) != len(CATALOG):
        raise ValueError(f"技能表应该是 {len(CATALOG)} 行，收到 {len(rows)} 行")
    return ParsedRequest(
        request_id=request_id,
        tag=fields["tag"],
        grant=fields["grant"],
        inuse=fields["inuse"],
        rows=tuple(rows),
    )
