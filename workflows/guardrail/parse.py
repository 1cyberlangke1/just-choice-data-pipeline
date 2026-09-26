"""内容审核（护栏）workflow 的反解：把一屏待审内容读回结构化字段。

输入：`state` 文本 + 标签表。输出：`ParsedMessage`。
预期行为：**不复用渲染时的任何计算**——只按文本本身解析，所以能抓到字段漏写、
标签写错、正文被截断这类硬伤。认不出的行（抬头）直接忽略；`正文` 那一行之后的
行**全部**按正文收（正文是多句、每句一行，与工单那种单行正文不同）。
这里只做**结构**反解（各格都是原文），把「原文 → 键」的翻译留给判别器——
判别器手上有语言文件，而 `Workflow.reparse` 没有语言参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from . import locales

#: 语言文件 `[meta]` 里那几个标签（反解要认全，缺一格就报错）。
META_KEYS: Final[tuple[str, ...]] = (
    "content_label",
    "reach_label",
    "direction_label",
    "audience_label",
    "history_label",
    "body_label",
)

#: 分隔符：中英两版各用一种，反解两种都认。
COLONS: Final[tuple[str, ...]] = ("：", ":")


@dataclass(frozen=True)
class ParsedMessage:
    """反解出来的一条待审内容：字段是**原始文本**，正文是逐句的元组。"""

    content_id: str
    reach: str
    direction: str
    audience: str
    history: str
    body: tuple[str, ...]


def label_index(loc: dict[str, Any]) -> dict[str, str]:
    """输入：语言字典。输出：{标签原文: 槽位名}（只收 `META_KEYS` 里那几个）。

    预期行为：两个标签文案相同时报错——否则反解会把两格认成一格。
    """
    out: dict[str, str] = {}
    for key in META_KEYS:
        text = loc["meta"][key]
        if text in out:
            raise ValueError(f"标签文案重复：{text!r}（{out[text]} 与 {key}）")
        out[text] = key
    return out


def merged_labels() -> dict[str, str]:
    """输入：无。输出：全部语言的标签合并表。

    预期行为：供语言无关的 `reparse` 用；各语言的标签互不重叠（由 test_locales 锁住），
    重叠了这里会报错而不是静默取其一。
    """
    out: dict[str, str] = {}
    for lang in locales.available():
        for text, slot in label_index(locales.load(lang)).items():
            if text in out:
                raise ValueError(f"标签 {text!r} 在多种语言里都出现（{out[text]} / {slot}）")
            out[text] = slot
    return out


def _match_label(line: str, labels: dict[str, str]) -> tuple[str, str] | None:
    """输入：一行文本、标签表。输出：(槽位, 值) 或 None。

    预期行为：取**最长**的匹配标签（标签之间有前缀关系时才不会认错），
    并且要求标签后面紧跟分隔符——否则抬头「内容审核请求」会被认成「内容号」那一格
    （它确实以「内容」开头）。
    """
    best: tuple[str, str] | None = None
    for text, slot in labels.items():
        if not line.startswith(text):
            continue
        rest = line[len(text) :].lstrip()
        if rest[:1] not in COLONS:
            continue
        if best is None or len(text) > len(best[0]):
            best = (text, slot)
    if best is None:
        return None
    value = line[len(best[0]) :].strip().lstrip("：:").strip()
    return best[1], value


def parse_message(state: str, labels: dict[str, str]) -> ParsedMessage:
    """输入：state 文本、标签表。输出：反解结果。

    预期行为：逐行认标签；认不出的行忽略；`body_label` 之后的行全是正文；
    `META_KEYS` 缺任何一格都报人话——缺格说明渲染漏写了字段，是硬伤，不能当空值蒙过去。
    """
    fields: dict[str, str] = {}
    body: list[str] = []
    in_body = False
    for raw in state.splitlines():
        line = raw.strip()
        if not line:
            continue
        if in_body:
            body.append(line)
            continue
        found = _match_label(line, labels)
        if found is None:
            continue
        fields[found[0]] = found[1]
        if found[0] == "body_label":
            in_body = True
    missing = [key for key in META_KEYS if key not in fields]
    if missing:
        raise ValueError(f"待审内容缺这几格：{missing}")
    return ParsedMessage(
        content_id=fields["content_label"],
        reach=fields["reach_label"],
        direction=fields["direction_label"],
        audience=fields["audience_label"],
        history=fields["history_label"],
        body=tuple(body),
    )
