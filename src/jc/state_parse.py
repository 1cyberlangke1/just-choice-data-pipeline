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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .jsontypes import JsonValue
from .locale_files import LocaleApi


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
