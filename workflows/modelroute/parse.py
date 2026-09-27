"""模型路由与降级 workflow 的反解：把画像文本读回结构化字段。

输入：`state` 文本 + 标签表。输出：`ParsedRequest`。
预期行为：**不复用渲染时的任何计算**——只按文本本身解析，所以能抓到字段漏写、标签写错、
正文被截断这类硬伤。认不出的行（标题行）直接忽略。

这里只做**结构**反解（各格都是原文），把「原文 → 键」的翻译与「正文 → 任务类型」留给
判别器——判别器手上有语言文件，而 `Workflow.reparse` 没有语言参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from . import locales
from .render import FIELD_ORDER

#: 语言文件 `[meta]` 里要认的标签（反解要认全，缺一格就报错）。
META_KEYS: Final[tuple[str, ...]] = (*FIELD_ORDER, "body_label")

#: 反解结果里「标签键 → 字段名」的对应（两处名字不同：标签键是渲染用的，字段名是读法用的）。
SLOT_FIELDS: Final[dict[str, str]] = {
    "request_label": "request_id",
    "complexity_label": "complexity",
    "latency_label": "latency",
    "cost_label": "cost",
    "retry_label": "retry",
    "data_label": "data",
    "body_label": "body",
}


@dataclass(frozen=True)
class ParsedRequest:
    """反解出来的一条请求画像：全是**原始文本**。"""

    request_id: str
    complexity: str
    latency: str
    cost: str
    retry: str
    data: str
    body: str


def label_index(loc: dict[str, Any]) -> dict[str, str]:
    """输入：语言字典。输出：{标签原文: 标签键}（只收 `META_KEYS` 里那几个）。

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
    """输入：一行文本、标签表。输出：(标签键, 值) 或 None。

    预期行为：取**最长**的匹配标签（标签之间有前缀关系时才不会认错），并且要求标签后面
    紧跟分隔符——否则标题行会被认成某一格（它确实可能以某个标签开头）。故意收紧。
    """
    best: tuple[str, str] | None = None
    for text, slot in labels.items():
        if not line.startswith(text):
            continue
        rest = line[len(text) :].lstrip()
        if not rest[:1] in ("：", ":"):
            continue
        if best is None or len(text) > len(best[0]):
            best = (text, slot)
    if best is None:
        return None
    value = line[len(best[0]) :].strip().lstrip("：:").strip()
    return best[1], value


def parse_request(state: str, labels: dict[str, str]) -> ParsedRequest:
    """输入：state 文本、标签表。输出：反解结果。

    预期行为：逐行认标签；认不出的行忽略；`META_KEYS` 缺任何一格都报人话——
    缺格说明渲染漏写了字段，是硬伤，不能当成「空值」蒙过去。非字符串直接报类型错
    （不然会炸出一句 `int` 没有 `splitlines` 的 AttributeError，不是人话）。
    """
    if not isinstance(state, str):
        raise TypeError(f"state 必须是字符串，收到 {type(state).__name__}")
    fields: dict[str, str] = {}
    for raw in state.splitlines():
        line = raw.strip()
        if not line:
            continue
        found = _match_label(line, labels)
        if found is None:
            continue
        fields[found[0]] = found[1]
    missing = [key for key in META_KEYS if key not in fields]
    if missing:
        raise ValueError(f"画像缺这几格：{missing}")
    return ParsedRequest(**{SLOT_FIELDS[key]: fields[key] for key in META_KEYS})
