"""产物文本的**书写系统**必须与它声明的语言一致：判据 + 实测阈值。

为什么要有：`rerank` 的 en 侧真机小批里，模型把整段 state 写成了中文（实测汉字占比 0.577），
`citation` 的 en 侧有一条写成了中文文档（0.770）——**此前没有任何机制保证产物文本与它声明的
语言一致**，模型答非所问也照样入库：判别器只查「答案对不对」，审计只查「题族齐不齐」，两边
都不看语言。这个模块补上那条判据。

输入：一段文本 + 语言标签（`zh` / `en`）。
输出：`None`（一致 / 判不了）或人话理由（不一致）。
预期行为：只数**汉字**与**拉丁字母**两类字符，算「汉字占两者的比例」；数字、标点、空白、
代码符号（`#` `{` `|`）两种语言里都常见，算进去只会稀释信号。这是**书写系统层面的粗判据**，
不是语言识别：它只回答「这段文本的汉字占比配不配得上 zh / en 这个标签」。

阈值是量出来的，不是拍的（2026-09-30，只统计 manifest 里 `backend.name == "llm"` 的真机
产物，量法见 `tmp/probe_script_by_run.py`）：

| 侧 | 正常范围（真机） | 坏样本 | 取的阈值 |
|---|---|---|---|
| en（上限） | 0.000（绝大多数）、`browser` 0.068~0.084（它的元素名是中文固定词表） | `rerank` 0.577、`citation` 0.770 | **0.30** |
| zh（下限） | `review` 0.241、`shell` 0.246、`compact` 0.344、`search` 0.384、`toolgate` 0.401、`routing` 0.469、`browser` 0.505、`rubric` 0.594、`guardrail` 0.697、`autoformat` 0.823、`citation` 0.912 | `rerank` 0.081 / 0.089 / 0.092 / 0.117 | **0.15** |

两条阈值都留了余量：en 侧离正常最大值（0.084）有 3.5 倍，zh 侧离正常最小值（0.241）还有 0.09。
**前提是只把自然语言正文交给它**：`lint` 的 state 是「一行中文抬头 + 一段源码」（真机 zh 侧
占比低到 0.099）、`shell` 的 state 是命令行、`maze` 的 state 是 ASCII 迷宫图——那些不该按语言
判。所以判据挂在各 workflow 的 `check_draft` 上、只喂它认得出语言的那几个字段，**不在通用层
对整份 state 一刀切**。

短文本不判：`MIN_SCRIPTED` 个「有书写系统的字符」都不到就返回 `None`——`m1`、`yes`、`3` 这种
短标签本来就推不出语言，判它只会误伤。这条也是「选项名 / 编号 / 布尔值」能安全混进样本的原因。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Final

from .workflow import DraftError

#: 汉字区间：CJK 统一表意文字基本区（U+4E00–U+9FFF）。只收基本区——扩展区的生僻字与古字在
#: 本仓库的产物里没出现过，收进来只会让「哪些字符算汉字」这件事变得难解释。
_HAN_START: Final[int] = 0x4E00
_HAN_END: Final[int] = 0x9FFF

#: 语言标签 → 它期望的书写系统。只收本仓库真在用的两种；**表外的语言一律不发表意见**（返回
#: None）：「不知道」与「不一致」是两件事，把不知道报成不一致，新加一种语言就会全线误伤。
LOCALE_SCRIPT: Final[dict[str, str]] = {"zh": "han", "en": "latin"}

#: zh 侧下限：汉字占比低于它就判不一致。
ZH_FLOOR: Final[float] = 0.15

#: en 侧上限：汉字占比高于它就判不一致（等价于「拉丁字母占比低于 0.70」）。
EN_HAN_CEILING: Final[float] = 0.30

#: 参与判据的字符数不到这个数就不判（理由见模块文档「短文本不判」）。
MIN_SCRIPTED: Final[int] = 12


def scripted_chars(text: str) -> int:
    """输入：一段文本。输出：其中「有书写系统的字符」个数（汉字 + 拉丁字母）。

    预期行为：只数这两类——其余字符（数字、标点、空白、emoji、代码符号）既不支持 zh 也不支持
    en，数进来只会让比例失真。
    """
    return sum(1 for ch in text if _is_han(ch) or _is_latin(ch))


def script_ratio(text: str) -> float:
    """输入：一段文本。输出：汉字占「汉字 + 拉丁字母」的比例（两类都没有时给 0.0）。

    预期行为：分母为 0（纯数字 / 纯标点 / 空串）时返回 0.0——调用方要先看 `scripted_chars`
    够不够 `MIN_SCRIPTED`，`language_mismatch` 已经把这一步包在里面了。
    """
    han = sum(1 for ch in text if _is_han(ch))
    total = scripted_chars(text)
    return han / total if total else 0.0


def language_mismatch(text: str, locale: str) -> str | None:
    """输入：一段文本、语言标签。输出：不一致时的人话理由；一致 / 判不了时 None。

    预期行为：三道闸依次过——表外的语言不发表意见（None）、有书写系统的字符不够 `MIN_SCRIPTED`
    不判（None）、比例越界才报。报的理由带上实测比例与阈值，省得再回去数一遍。
    """
    script = LOCALE_SCRIPT.get(locale)
    if script is None:
        return None
    total = scripted_chars(text)
    if total < MIN_SCRIPTED:
        return None
    ratio = script_ratio(text)
    if script == "han" and ratio < ZH_FLOOR:
        return (
            f"这段文本声明的语言是 {locale}（该以汉字为主），但汉字只占 "
            f"{ratio:.3f}（下限 {ZH_FLOOR}）——它写成了别的语言：{_preview(text)}"
        )
    if script == "latin" and ratio > EN_HAN_CEILING:
        return (
            f"这段文本声明的语言是 {locale}（该以拉丁字母为主），但汉字占到 "
            f"{ratio:.3f}（上限 {EN_HAN_CEILING}）——它写成了别的语言：{_preview(text)}"
        )
    return None


def require_language(texts: Mapping[str, str], locale: str) -> None:
    """输入：{字段名: 文本} 与语言标签。输出：无（不一致抛 `DraftError`，管线据此重渲染）。

    预期行为：逐个字段判，**报出第一个不一致的字段名**——`DraftError` 的
    `retryable_for_case` 会让管线重渲染一次（模型采样抖动，换个说法往往就对了），所以理由要
    说得让人（和下一次的提示词）知道是哪一段写错了。字段名由调用方给中文（「查询」「候选」
    这种），报错才读得懂。
    """
    for field, text in texts.items():
        problem = language_mismatch(text, locale)
        if problem is not None:
            raise DraftError(f"{field}：{problem}")


def sample_problem(texts: Mapping[str, str], locales: Iterable[str]) -> str | None:
    """输入：{字段名: 文本} 与这一份产物**声明的**语言（可以有好几种）。输出：人话理由 / None。

    预期行为：只要**有一种**声明语言能全过就算过——一次 run 可以把两种语言轮转着写进同一份
    产物（`jc.pipeline` 按 `locales[seq % len]` 轮转），逐行判「配不配得上某种声明语言」才是
    正确的问法。一种都过不了才报，报的是每一种声明语言各自的第一条理由。

    声明语言为空（或全是表外语言）时**不发表意见**：没有声明就无从谈「与声明一致」，
    把「不知道」报成「不一致」会误伤老产物与新加的语言。
    """
    declared = [locale for locale in locales if locale in LOCALE_SCRIPT]
    if not declared:
        return None
    reasons: list[str] = []
    for locale in declared:
        first: str | None = None
        for field, text in texts.items():
            problem = language_mismatch(text, locale)
            if problem is not None:
                first = f"{field}：{problem}"
                break
        if first is None:
            return None
        reasons.append(first)
    return f"声明的语言 {'、'.join(declared)} 都对不上——{'；'.join(reasons)}"


def _is_han(ch: str) -> bool:
    """输入：一个字符。输出：它是不是汉字（基本区）。"""
    return _HAN_START <= ord(ch) <= _HAN_END


def _is_latin(ch: str) -> bool:
    """输入：一个字符。输出：它是不是拉丁字母。

    预期行为：ASCII 字母才算——`str.isalpha()` 会把汉字、假名、西里尔字母一起算成「字母」，
    那样 en 侧的分母会被别种文字污染。
    """
    return ch.isascii() and ch.isalpha()


def _preview(text: str, limit: int = 60) -> str:
    """输入：一段文本、可选长度上限。输出：折成一行、截断后的预览（报错时给人看）。"""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"
