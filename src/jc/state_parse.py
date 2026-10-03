"""渲染好的 state 的**结构化反解**（通用层）：抬头认语言、按语言拆表、认不出来给空。

本仓库「特征注入式」的那一族 workflow（`adblock` / `logtriage` / `memorygate` /
`promptinject` / `sqlreview` / `subqa` / `tracescore` / `turntake` / `debugstep`）的 state
是同一套排版：**判据表 / 档位表 / 素材抬头**三段，标记词写在各自的 `locales/*.toml` 里
（`state.tag_mark` / `rule_mark` / `content_mark` / `tag_sep`）。反解机制只在这里有一份，
9 个包各抄一份纯属重复——抄歪了还会让「审计按 state 重算」悄悄对不上。

输入：state（字符串或任意对象）、语言字典、语言包接口、按语言反解的解析函数。
输出：`ParsedState`（判据编号 + 素材原文，认不出来时 `text` 为 None）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from .jsontypes import JsonValue
from .locale_files import LocaleApi, header


class Parsed(Protocol):
    """反解结果的最小形状：只要求能回答「素材那一块认出来了没有」。"""

    @property
    def found(self) -> bool:
        """输入：无。输出：素材抬头认出来了没有。"""


@dataclass(frozen=True)
class ParsedState:
    """从 state 反解出来的、纯代码看得见的部分。

    字段：`tags` 判据表里列出的判据编号；`text` 素材原文（认不出来给 None）。
    """

    tags: tuple[str, ...]
    text: str | None

    @property
    def found(self) -> bool:
        """输入：无。输出：素材那一块认出来了没有。"""
        return self.text is not None


def parse_sections(state: str, loc: dict[str, JsonValue], *, line_prefix: str) -> ParsedState:
    """输入：state、语言字典、判据行的行前缀。输出：按这一份语言反解的结果。

    预期行为：认判据表抬头、档位表抬头、素材抬头；判据表里带行前缀的那些行取编号（按
    `state.tag_sep` 切），**素材抬头之后的所有行原样拼回正文**——抬头之外的内容一律不当结构，
    否则素材里出现一行以行前缀开头的文本就会把判据表数歪。
    """
    marks = loc["state"]
    tag_mark = str(marks["tag_mark"])
    rule_mark = str(marks["rule_mark"])
    content_mark = str(marks["content_mark"])
    sep = str(marks["tag_sep"])
    tags: list[str] = []
    body: list[str] = []
    section = ""
    for raw in state.splitlines():
        text = raw.strip()
        if section == "content":
            body.append(raw)
            continue
        if text.startswith(content_mark):
            section = "content"
            body.append(text[len(content_mark) :].strip())
            continue
        if text.startswith(tag_mark):
            section = "tag"
            continue
        if text.startswith(rule_mark):
            section = "rule"
            continue
        if section == "tag" and text.startswith(line_prefix):
            tags.append(text[len(line_prefix) :].partition(sep)[0].strip())
    content = "\n".join(body).strip() if body else None
    return ParsedState(tags=tuple(tags), text=content)


def split_sections(
    state: str,
    loc: dict[str, JsonValue],
    keys: Sequence[str],
    *,
    keep_indent: bool = False,
) -> dict[str, list[str]]:
    """把渲染好的 state 按段落头切成 `{段落名: 该段的行列表}`（11 家同构，只留这一份）。

    输入：state、语言字典、要认的段落名键（`meta` 里那些 `*_label`）、是否保留行首空白。
    输出：段落名 → 该段的行列表；段落头那一行剩下的内容算该段的第一行。
    预期行为：段落头是**前缀匹配**（后面可能跟着括号里的列名 / 说明），只认「行首」这一种写法
    ——正文里正常提到这些词不算。`keep_indent=True` 时数据行只去尾部空白：正文是代码或分条
    材料时，行首那个 `+` / `-` / 空格就是它的凭据，去掉就解不出来了。
    """
    heads = [(str(loc["meta"][key]), header(loc, str(loc["meta"][key]))) for key in keys]
    out: dict[str, list[str]] = {}
    current: str | None = None
    for raw in state.splitlines():
        line = raw.rstrip() if keep_indent else raw.strip()
        stripped = line.strip()
        hit = next(((label, head) for label, head in heads if stripped.startswith(head)), None)
        if hit is not None:
            label, head = hit
            current = label
            out.setdefault(current, [])
            rest = stripped[len(head) :].strip()
            if rest:
                out[current].append(rest)
            continue
        if current is not None and stripped:
            out[current].append(line)
    return out


def read_after_label(line: str, labels: Sequence[str], what: str) -> str:
    """输入：一行文本、标签的几种写法、报错用的名字。输出：标签后面的原文。

    预期行为：**最长前缀优先**（标签之间有前缀关系时才不会认错）；标签后面必须跟东西，
    否则更长的标签会被认成短标签本身；一个都认不出来就报人话（渲染版式写歪了）。
    6 家同构（`cachereuse` / `clipcut` / `planstep` / `quoteread` / `sentiment` / `siftrank`），
    只留这一份——抄歪了会让「按 state 反解」悄悄对不上。
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


def read_field_value(line: str, labels: Sequence[str]) -> str | None:
    """输入：一行文本、候选字段名。输出：分隔符后的值；都不是这一格时给 None。

    预期行为：取**最长**的匹配字段名（字段名之间有前缀关系时才不会认错），并且要求字段名
    后面紧跟分隔符——否则标题行会被认成某一格。故意收紧。3 家同构
    （`elementpick` / `sim` / `skillpick`），只留这一份。
    """
    best: str | None = None
    for label in labels:
        if not line.startswith(label):
            continue
        if line[len(label) :].lstrip()[:1] not in ("：", ":"):
            continue
        if best is None or len(label) > len(best):
            best = label
    if best is None:
        return None
    return line[len(best) :].strip().lstrip("：:").strip()


def recognize_locale(
    state: object, api: LocaleApi, parse: Callable[[str, dict[str, JsonValue]], Parsed]
) -> str | None:
    """输入：state、语言包接口、按语言反解的解析函数。输出：能认出素材抬头的那一种语言。

    预期行为：认不出（不是字符串、或哪一份语言都对不上）给 None。审计拿它按 state **自己**的
    语言取规范文案——一次 run 里混了 zh / en 时，用调用方给的那一种去比会平白报一堆「文案对不上」。

    为什么认的是**素材抬头**而不是判据编号：编号（A1…、R1…）是结构键、与语言无关，拿它当判据会把
    别的语言的 state 也抢过来；素材抬头是渲染时按语言写死的、整段最后那一块的开头，才分得开。
    """
    if not isinstance(state, str):
        return None
    for lang in api.available():
        if parse(state, api.load(lang)).found:
            return lang
    return None


def parse_any(
    state: object, api: LocaleApi, parse: Callable[[str, dict[str, JsonValue]], ParsedState]
) -> ParsedState:
    """输入：state、语言包接口、按语言反解的解析函数。输出：纯代码能反解的部分。

    预期行为：与调用方声明的语言无关——把各语言的标记都试一遍，认出素材抬头的那一份就用它；
    一份都认不出来时返回空结果（`text` 为 None），不报错、不猜。
    """
    lang = recognize_locale(state, api, parse)
    if lang is None:
        return ParsedState(tags=(), text=None)
    return parse(str(state), api.load(lang))
