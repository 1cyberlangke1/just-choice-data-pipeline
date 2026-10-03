"""界面元素定位 workflow 的反解：把一屏元素清单读回结构化字段。

输入：`state` 文本 + 标签表。输出：`ParsedScreen`。
预期行为：**不复用渲染时的任何计算**——只按文本本身解析，所以能抓到抬头漏写、格数不对、
某一行缺格这类硬伤。认不出的行（标题行）直接忽略。

标签表里每一格存的都是**候选原文元组**：单语言表只有一项，`merged()` 给的合并表把各语言
的写法都收进来（`Workflow.reparse` 没有语言参数，只能靠合并表做语言无关的反解）。
四张词表（角色 / 可见性 / 可点性 / 标签）也是「原文 → 键」的并集，判别器与审计拿**键**去查规则。

文本那一格多一层：先按词表原文查**完全一致**；查不到就按各语言的**近似后缀**剥一层再查，
剥得动就还原成 `shadow:<标签>`。渲染侧（`render.text_of`）是这条的镜像，所以「文本写歪」
会在反解时当场现形，而不是静默认成别的键。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue
from jc.locale_files import merge_labels
from jc.state_parse import read_field_value

from dataclasses import dataclass
from typing import Final

from . import locales
from .spec import SHADOW_PREFIX, SIZE

#: 元素表行里四格的顺序（与渲染一致，反解按它取格）。
CELL_FIELDS: Final[tuple[str, ...]] = ("role", "visibility", "clickable", "text")
#: 行首字段名（跨语言必须互不相同，否则合并表有歧义）。
LABEL_FIELDS: Final[tuple[str, ...]] = ("screen", "action")
#: 词表名（合并表要逐张做「跨语言不重叠」检查）。
VOCAB_FIELDS: Final[tuple[str, ...]] = ("role", "visibility", "clickable", "label")


@dataclass(frozen=True)
class Labels:
    """反解要用的全部原文：两个行首字段名的候选、四格分隔符、近似后缀、四张词表。"""

    screen_labels: tuple[str, ...]
    action_labels: tuple[str, ...]
    cell_sep: str
    shadow_suffixes: tuple[str, ...]
    role: dict[str, str]
    visibility: dict[str, str]
    clickable: dict[str, str]
    label: dict[str, str]

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
    """反解出来的一行元素：角色 / 可见性 / 可点性 / 文本（都是键）。"""

    role: str
    visibility: str
    clickable: str
    text: str

    def usable(self) -> bool:
        """输入：无。输出：这一行可见且可点吗。预期行为：与 `spec.ElementRow` 同一条规则。"""
        return self.visibility == "visible" and self.clickable == "yes"


@dataclass(frozen=True)
class ParsedScreen:
    """反解出来的一屏清单：屏号 + 动作标签 + 六行元素。"""

    screen_id: str
    action: str
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
        screen_labels=(str(meta["screen_label"]),),
        action_labels=(str(meta["action_label"]),),
        cell_sep=str(meta["cell_sep"]),
        shadow_suffixes=(str(meta["shadow_suffix"]),),
        role=_pairs(loc, "role"),
        visibility=_pairs(loc, "visibility"),
        clickable=_pairs(loc, "clickable"),
        label=_pairs(loc, "label"),
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


def _suffix_union(per_lang: list[tuple[str, Labels]]) -> tuple[str, ...]:
    """输入：[(语言, 反解表)]。输出：各语言的近似后缀（去重，顺序稳定）。

    预期行为：后缀允许跨语言重名（重名只是多试一次，不会认错键）——所以这里**只去重、
    不报错**，与四张词表的口径不同。
    """
    out: list[str] = []
    for _, labels in per_lang:
        for suffix in labels.shadow_suffixes:
            if suffix and suffix not in out:
                out.append(suffix)
    if not out:
        raise ValueError("所有语言都没有近似后缀（近似那一格没法反解）")
    return tuple(out)


def merged() -> Labels:
    """输入：无。输出：所有语言的合并反解表（`reparse` 用它做语言无关的结构反解）。

    预期行为：四格分隔符各语言必须一致（合并表只按一个分隔符切行），字段名与四张词表
    必须跨语言不重叠，近似后缀只需非空。
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
        screen_labels=merge_labels(
            [text for _, labels in per_lang for text in labels.screen_labels], "screen"
        ),
        action_labels=merge_labels(
            [text for _, labels in per_lang for text in labels.action_labels], "action"
        ),
        cell_sep=first.cell_sep,
        shadow_suffixes=_suffix_union(per_lang),
        role=_union(
            [(text, key) for _, labels in per_lang for text, key in labels.role.items()], "role"
        ),
        visibility=_union(
            [(text, key) for _, labels in per_lang for text, key in labels.visibility.items()],
            "visibility",
        ),
        clickable=_union(
            [(text, key) for _, labels in per_lang for text, key in labels.clickable.items()],
            "clickable",
        ),
        label=_union(
            [(text, key) for _, labels in per_lang for text, key in labels.label.items()], "label"
        ),
    )


def decode_text(raw: str, labels: Labels) -> str:
    """输入：文本原文、标签表。输出：文本键（`<标签>` 或 `shadow:<标签>`）。

    预期行为：先查**完全一致**（原文就是某个标签的写法）；查不到再按各语言的近似后缀剥
    一层，剥完还在词表里就还原成近似键。两步都认不出来直接报人话——静默认成某个标签
    会把「渲染写歪」变成一条答案对不上的样本，而不是一条能定位的错误。
    """
    if raw in labels.label:
        return labels.label[raw]
    for suffix in labels.shadow_suffixes:
        if raw.endswith(suffix) and raw[: -len(suffix)] in labels.label:
            return f"{SHADOW_PREFIX}{labels.label[raw[: -len(suffix)]]}"
    raise ValueError(f"文本认不出来：{raw!r}")


def _decode(field: str, text: str, labels: Labels) -> str:
    """输入：词表名、原文、标签表。输出：键。预期行为：认不出的原文直接报人话。"""
    table = labels.table(field)
    if text not in table:
        raise ValueError(f"{field} 认不出来：{text!r}")
    return table[text]


def _parse_cells(line: str, labels: Labels) -> tuple[str, ...]:
    """输入：元素表行、标签表。输出：四格词键（顺序固定）。

    预期行为：`[序号] 格 · 格 · 格 · 格`；格数不对直接报错（少一格说明渲染漏写了属性，
    那是硬伤，不能当成「空值」蒙过去）。
    """
    end = line.find("]")
    if end < 0:
        raise ValueError(f"元素行缺右方括号：{line!r}")
    cells = [cell.strip() for cell in line[end + 1 :].strip().split(labels.cell_sep)]
    if len(cells) != len(CELL_FIELDS):
        raise ValueError(f"元素行应该是 {len(CELL_FIELDS)} 格，收到 {len(cells)} 格：{line!r}")
    out: list[str] = []
    for field, cell in zip(CELL_FIELDS, cells, strict=True):
        out.append(decode_text(cell, labels) if field == "text" else _decode(field, cell, labels))
    return tuple(out)


def parse_screen(state: str, labels: Labels) -> ParsedScreen:
    """输入：state 文本、标签表。输出：反解结果。

    预期行为：逐行认——`[` 开头的是元素行；其余按两个字段名取，取不到就忽略（标题行）。
    屏号 / 动作 / 元素行缺任何一样都报人话。
    """
    if not isinstance(state, str):
        raise TypeError(f"state 必须是字符串，收到 {type(state).__name__}")
    screen_id: str | None = None
    action: str | None = None
    rows: list[ParsedRow] = []
    for raw in state.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("["):
            rows.append(ParsedRow(*_parse_cells(line, labels)))
            continue
        value = read_field_value(line, labels.screen_labels)
        if value is not None:
            screen_id = value
            continue
        value = read_field_value(line, labels.action_labels)
        if value is not None:
            action = decode_text(value, labels)
    missing: list[str] = []
    if screen_id is None:
        missing.append(" / ".join(labels.screen_labels))
    if action is None:
        missing.append(" / ".join(labels.action_labels))
    if missing:
        raise ValueError(f"清单缺这几格：{missing}")
    if len(rows) != SIZE:
        raise ValueError(f"元素表应该是 {SIZE} 行，收到 {len(rows)} 行")
    return ParsedScreen(screen_id=screen_id, action=action, rows=tuple(rows))
