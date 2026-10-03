"""workflow 语言包（`workflows/<name>/locales/*.toml`）的通用读写，外加按语言字典做的文本件。

输入：一个 workflow 的 `locales` 目录。
输出：`LocaleApi`——`available()` / `load(lang)` / `pick(seq, index)` 三个已经绑好该目录的函数。
预期行为：读取、缓存、缺文件报错、编号回绕这些机制**只在这里有一份**；各 workflow 的
`locales.py` 只声明「目录在哪」，其余由 `make()` 现绑——48 个包各抄一份是纯重复。

下面那几件（`feature_line` / `tagged_prompt` / `header` / `label_leak` / `banned_echo`）是
**按语言字典渲染或检查文本**的通用件：它们只看 `loc` 与调用方给的键，机制同样只留一份。

`make()` 返回的是三个闭包（NamedTuple 装着），这是本模块唯一的「手法」：把目录参数提前绑死，
调用方才能继续写 `locales.load("zh")` 这种不带目录的写法。
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from collections.abc import Sequence
from typing import Callable, Final, NamedTuple

from .jsontypes import JsonValue

#: 语言文件缓存条数：48 个 workflow × 2 种语言还有富余。
CACHE_SIZE: Final[int] = 256

#: 「把提示词那句『用虚构的名字』读成了名字本身」的写法：`虚构项目 Alpha` / `fictional company` 这种。
#: 这不是措辞问题——真机上 12 家写了这句规则的 workflow 里，模型会把它当成一个可以照抄的名字
#: （`batch-sourceselect-zh` 10 条里 4 条、`cp-36b/c/d-en` 各 1 条、`batch-compact-en` 1 条）。
#: 所以它**不归各家的清单管**：与 `banned_echo` 拼在一起，任何一家都不许出现。
#: `虚拟` 只收组合词（`虚拟化` / `虚拟现实` 都是正常说法，不收单字）；`placeholder` 也不收——
#: `subqa` 那一族讲的就是占位符，收了会把整族堵死。
PLACEHOLDER_NAME_MARKS: Final[tuple[str, ...]] = (
    "虚构",
    "虚拟项目",
    "虚拟公司",
    "虚拟产品",
    "虚拟指标",
    "虚拟名称",
    "虚拟平台",
    "虚拟系统",
    "虚拟商城",
    "虚拟技术",
    "fictional",
    "fictitious",
    "hypothetical",
    "imaginary",
    "made-up",
)


class LocaleApi(NamedTuple):
    """一个 workflow 的语言包接口。

    字段：三个函数——`available()` 列语言、`load(lang)` 取字典、`pick(seq, index)` 回绕取项。
    """

    available: Callable[[], list[str]]
    load: Callable[[str], dict[str, JsonValue]]
    pick: Callable[[list[str], int], str]


def available(locale_dir: Path) -> list[str]:
    """输入：语言目录。输出：可用语言列表（按文件名）。"""
    return sorted(path.stem for path in locale_dir.glob("*.toml"))


@lru_cache(maxsize=CACHE_SIZE)
def _read(locale_dir: Path, lang: str) -> dict[str, JsonValue]:
    """输入：语言目录、语言代码。输出：解析后的字典。

    预期行为：结果带缓存（键是「目录 + 语言」，所以不同 workflow 的同名语言互不串味）；
    文件不存在时抛 FileNotFoundError 并列出该目录下可用的语言。
    """
    path = locale_dir / f"{lang}.toml"
    if not path.exists():
        raise FileNotFoundError(f"没有语言文件 {path.name}；可用：{', '.join(available(locale_dir))}")
    with path.open("rb") as handle:
        return tomllib.load(handle)


def pick(seq: list[str], index: int) -> str:
    """输入：候选列表、编号。输出：第 index 个（越界回绕）。

    预期行为：问法池比题族数量少时也照常工作——回绕而不是报错（题面措辞不是判据）。
    """
    return seq[index % len(seq)]


def merge_labels(entries: list[str], what: str) -> tuple[str, ...]:
    """输入：各语言的字段名候选、字段名。输出：合并后的候选元组。

    预期行为：同一个字段名出现在多种语言里直接报人话——合并表有歧义时反解会把某一格
    认成另一格，那比直接报错危险得多。3 家同构（`elementpick` / `sim` / `skillpick`），
    只留这一份。
    """
    out: list[str] = []
    seen: set[str] = set()
    for text in entries:
        if text in seen:
            raise ValueError(f"{what} 的标签 {text!r} 在多种语言里都出现（合并表有歧义）")
        seen.add(text)
        out.append(text)
    return tuple(out)


def make(locale_dir: Path) -> LocaleApi:
    """输入：一个 workflow 的 locales 目录。输出：绑好该目录的三个函数。

    预期行为：返回的 `load` 是闭包，调用方不必知道目录在哪；`pick` 与目录无关，直接共用。
    """

    def list_langs() -> list[str]:
        """输入：无。输出：该目录下的可用语言。"""
        return available(locale_dir)

    def load(lang: str) -> dict[str, JsonValue]:
        """输入：语言代码。输出：解析后的字典。预期行为：缺文件直接报错并列出可用语言。"""
        return _read(locale_dir, lang)

    return LocaleApi(available=list_langs, load=load, pick=pick)


def feature_line(loc: dict[str, JsonValue], dim: str, value: str) -> str:
    """把一条注入特征渲染成提示词里的一行（11 家逐字同构，只留这一份）。

    输入：语言字典、维度名、取值。
    输出：形如「- 用户这一轮想干什么：往一个输入框里填东西（写成像……）」的一行。
    预期行为：短语缺失直接抛 `KeyError`——静默给空字符串会让模型自由发挥，那就等于标签不再由
    特征决定。语言字典的约定：`feature[维度][取值]` 给出 `brief` / `hint`，`dims[维度]` 给出
    维度名，`prompt.bullet` 是那一行的模板。
    """
    spec = loc["feature"][dim][value]
    return str(loc["prompt"]["bullet"]).format(
        dim=loc["dims"][dim], brief=spec["brief"], hint=spec["hint"]
    )


def tagged_prompt(
    loc: dict[str, JsonValue],
    values: dict[str, str],
    dim_order: tuple[str, ...],
    phrasing: int,
) -> str:
    """组装「先列注入特征、再附规则」的提示词（8 家逐字同构，只留这一份）。

    输入：语言字典、{维度: 取值}、维度顺序、问法编号。
    输出：完整提示词（只描述要写的那几维，不含答案）。
    预期行为：特征按 dim_order 固定顺序逐行列出；规则整段附在后面、每条前面加短横线，并按
    phrasing 轮转起点（问法池比规则条数少时也照常工作）；{language} 填成语言名——留着
    占位符会把 {language} 原样发给模型。
    """
    lines = [str(loc["prompt"]["intro"]), ""]
    lines.extend(feature_line(loc, dim, values[dim]) for dim in dim_order)
    lines.append("")
    rules = loc["prompt"]["rules"]
    offset = phrasing % len(rules)
    rotated = list(rules[offset:]) + list(rules[:offset])
    lines.extend(f"- {rule.format(language=loc['language_name'])}" for rule in rotated)
    return "\n".join(lines)


def copy_entry(
    loc: dict[str, JsonValue],
    table: str,
    key: str,
    fields: tuple[str, ...],
    label: str,
) -> dict[str, JsonValue]:
    """取语言文件里某张表的一条文案（10 家同构，只留这一份）。

    输入：语言字典、表名、编号、该表要求的字段、给报错看的名词（如「判据」「信号」）。
    输出：那一条文案（原样返回，不改写）。
    预期行为：缺表、缺这一条、或缺字段都当场报错——判据表与提示词都靠它，缺一个字段模型就会
    拿到半句话。
    """
    block = loc.get(table)
    entry = block.get(key) if isinstance(block, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"语言文件里缺{label} {key} 的文案")
    for field in fields:
        if field not in entry:
            raise ValueError(f"{label} {key} 的文案缺字段 {field}")
    return entry


def entry_line(loc: dict[str, JsonValue], key: str, entry: dict[str, JsonValue]) -> str:
    """把一条判据渲染成 state 里的一行（10 家同构，只留这一份）。

    输入：语言字典、编号、该编号的文案。
    输出：编号 + tag_sep + 判据名 + rule_sep + 判据原文。
    预期行为：分隔符按语言取（state.tag_sep / state.rule_sep），解析时按同一份分隔符切回
    编号——**编号是结构键**，所以它排在分隔符前面，与语言的书写顺序无关。
    """
    sep = str(loc["state"]["tag_sep"])
    rule_sep = str(loc["state"]["rule_sep"])
    return f"{key}{sep}{entry['name']}{rule_sep}{entry['rule']}"


def header(loc: dict[str, JsonValue], title: str) -> str:
    """把段落标题渲染成段落头（11 家同构，只留这一份）。

    输入：语言字典、段落标题。输出：形如「【本块内容】」的段落头。
    预期行为：模板取自 `meta.header`——渲染与反解共用同一份模板，两边才不会各认一种写法。
    """
    return str(loc["meta"]["header"]).format(title=title)


def label_leak(text: str, loc: dict[str, JsonValue], keys: Sequence[str]) -> str:
    """正文开头误抄的标记名（7 家同构，只留这一份）。

    输入：模型写的一段文本、语言字典、要查的键。
    输出：文本开头抄了的那个名字（没有就空串）。
    预期行为：只认「名字 + 冒号」开头这一种写法——正文里正常提到这些词不算；冒号认
    `meta.colon` 与两种全半角写法。`meta[key]` 可以是**一个名字**，也可以是**一列名字**：
    后者用于「一维有好几档、每一档的名字都不许写进正文」的情形——写了就等于把答案印在
    题面里（`compact` 的 `kind_labels` 就是，见范式 §9.101）。
    **大小写不敏感**：en 的档名是小写单词（`side note` / `note` / `assistant`），模型写成
    `Side note: ` 就躲过了按字面量比的版本（`batch-compact-en` 抓到过），而这一行正好就是
    `fam.kind` 的答案——那是真的把答案印进题面。zh 侧没有大小写，行为不变。
    """
    stripped = text.strip().lower()
    meta = loc["meta"]
    separators = {str(meta["colon"]).strip(), "：", ":"}
    for key in keys:
        raw = meta[key]
        names = raw if isinstance(raw, list) else [raw]
        for name in (str(item) for item in names):
            if any(
                stripped.startswith(f"{name}{separator}".lower()) for separator in separators
            ):
                return name
    return ""


def banned_echo(text: str, loc: dict[str, JsonValue]) -> list[str]:
    """正文里回显了那几条禁用话术（7 家同构，只留这一份）。

    输入：模型写的一段文本、语言字典。输出：文本里出现的禁用话术（没有就空列表）。
    预期行为：只认 `banned_intent_phrases` 里点名的那几条**高精度**短语——宁可漏检也不误判，
    误判会在渲染期白拒一条本来能用的数据。大小写不敏感（英文里 `Pinned` 与 `pinned` 一样泄底）。
    另外**一律**认 `PLACEHOLDER_NAME_MARKS`：那是「把提示词那句『用虚构的名字』读成了名字本身」，
    任何一家都不该出现，与各家的清单无关（见那个常量的注释）。
    """
    lowered = text.lower()
    hits = [
        str(phrase) for phrase in loc["banned_intent_phrases"] if str(phrase).lower() in lowered
    ]
    return hits + [mark for mark in PLACEHOLDER_NAME_MARKS if mark.lower() in lowered]
