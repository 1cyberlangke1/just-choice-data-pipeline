"""社交帖子分诊的文本外壳：信号判据表、处置阈值表与帖子抬头由代码写，帖子正文由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **判据表、阈值表、正文抬头由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**一条帖子正文**（社交平台上的一条短文本是长尾的，模板拼出来的脱离现实分布）；
- **命中哪几条判据由代码按字面算**（`features.PATTERNS` 那四条正则），模型碰不到：提示词里逐条
  写明「必须出现哪些片段、不许出现哪些」，`check_draft` 在渲染期按同一份判据复核一遍（不合格就
  重渲染），`judges.HygieneJudge` / `FrameJudge` 再各查一遍，兜住不是本 workflow 渲染出来的
  state。

**为什么判据写成机械判据**：这一族的官方形态就是「有没有给股票代码 / 有没有给价位 / 有没有给
涨跌幅 / 有没有给外链」这种按字面能判的规则。写成正则之后，「命中哪几条」是**算**出来的，模型
只负责写一条读得通、长尾的帖子——两边都不越界（范式 §9.22 / §9.23 / §9.24 同一条路子）。

**诱饵（`decoy=lower`）为什么要逐字给出写法**：模型不会自发写出 `$acme` 这种「小写 cashtag」
形状——含糊地说「写个像股票代码的东西」，它会直接写 `$ACME`，那就真踩中 S1 了，计划当场落空。
所以提示词把 `features.DECOY_EXAMPLE` 原样摆出来，并明说「不许改写成大写」。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.workflow import DraftError

from . import features, locales
from .features import PostTriageCase
from .spec import PostTriageSpec

#: 判据表与阈值表每行的前缀。**这是 state 格式的一部分，不是文案**，所以不进语言文件：
#: 渲染与解析共用这一个常量，两边不可能漂移。
LINE_PREFIX: Final[str] = "- "

#: 正文长度上下限。太短读不出「像不像一条正经帖子」，太长就只是灌水（题面也没必要）。
MIN_CONTENT_CHARS: Final[int] = features.MIN_CONTENT_CHARS
MAX_CONTENT_CHARS: Final[int] = features.MAX_CONTENT_CHARS


class ContentDraft(BaseModel):
    """后端要填的草稿：帖子正文那一段。

    `text` 是**自然语言**（帖子正文是长尾的、由模型写）；结构化事实（判据表、阈值表、抬头）全由
    组装器渲染，模型碰不到。
    """

    text: str = Field(
        description="帖子正文：一段读得通的自然语言，不要标题、不要分点、不要围栏"
    )


@dataclass(frozen=True)
class ParsedContent:
    """从 state 反解出来的、纯代码看得见的部分。

    字段：`signals` 判据表里列出的判据编号；`text` 帖子正文（认不出来给 None）。
    """

    signals: tuple[str, ...]
    text: str | None

    @property
    def found(self) -> bool:
        """输入：无。输出：正文那一块认出来了没有。"""
        return self.text is not None


def signal_copy(loc: dict[str, JsonValue], signal: str) -> dict[str, JsonValue]:
    """输入：语言字典、判据编号。输出：该判据的文案（name / rule / example / avoid）。

    预期行为：缺字段就当场报错——判据表与提示词都靠它，缺一个字段模型就会拿到半句话。
    """
    table = loc.get("signal")
    entry = table.get(signal) if isinstance(table, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"语言文件里缺判据 {signal} 的文案")
    for key in ("name", "rule", "example", "avoid"):
        if key not in entry:
            raise ValueError(f"判据 {signal} 的文案缺字段 {key}")
    return entry


def signal_line(loc: dict[str, JsonValue], signal: str) -> str:
    """输入：语言字典、判据编号。输出：判据表里那一行（不含行前缀）。

    预期行为：`编号 + 分隔符 + 判据名`。分隔符按语言取（`state.signal_sep`），解析时按同一份
    分隔符切回编号——**编号是结构键**，所以它排在分隔符前面，与语言的书写顺序无关。
    """
    sep = str(loc["state"]["signal_sep"])
    return f"{signal}{sep}{signal_copy(loc, signal)['name']}"


def threshold_line(loc: dict[str, JsonValue], level: str) -> str:
    """输入：语言字典、处置档位名。输出：阈值表里那一行（不含行前缀）。

    预期行为：`阈值描述 + 分隔符 + 处置动作`；处置动作与 `route` 那一族的选项取的是**同一份
    文案**（`fam.route.opt`），所以 state 里那张表和题面不可能各说各的。
    """
    sep = str(loc["state"]["signal_sep"])
    action = str(loc["fam"]["route"]["opt"][level])
    return f"{loc['state']['threshold'][level]}{sep}{action}"


def build_prompt(loc: dict[str, JsonValue], case: PostTriageCase) -> str:
    """组装提示词。

    输入：语言字典、注入特征。输出：完整提示词（不含答案）。
    预期行为：**每一条硬性要求都写清楚**——必须出现哪些片段（逐条给例子，并**明说那只是例子、
    要自己另编**）、不许出现哪些（逐条给反面写法）、`lower` 时把那个小写 cashtag 原样给出、
    长度区间、输出前自查。这些就是「受控合成」的规格，判据全在 `check_draft` 里机械复核，模型
    写歪了当场重渲染。

    为什么必须明说「例子只是例子」：只给一个具体片段当例子，模型会把它**逐条照抄**进每一条
    正文——那等于把「背下这个片段 = 命中这条判据」的捷径长出来（范式 §9.25）。样本本身没错
    （标签由正则算），但这样的数据在训练里是废的。

    为什么把「不许出现」也逐条列出来：只写「必须出现什么」时，模型会顺手带上别的形状（写个带
    涨跌幅的句子、句末来一个链接），命中组合当场多出几条——那一条的标签就废了。
    """
    prompt = loc["prompt"]
    hits = case.hits()
    missing = [signal for signal in features.SIGNAL_IDS if signal not in hits]
    lines = [
        str(prompt["intro"]).format(min=MIN_CONTENT_CHARS, max=MAX_CONTENT_CHARS),
        str(prompt["lang_rule"]).format(language=loc["language_name"]),
    ]
    if hits:
        lines.append(str(prompt["signals_header"]))
        for signal in hits:
            entry = signal_copy(loc, signal)
            lines.append(
                str(prompt["signals_line"]).format(
                    name=entry["name"], example=entry["example"], rule=entry["rule"]
                )
            )
        lines.append(str(prompt["vary_rule"]))
    if missing:
        lines.append(str(prompt["miss_header"]) if hits else str(prompt["no_signals"]))
        for signal in missing:
            entry = signal_copy(loc, signal)
            lines.append(
                str(prompt["miss_line"]).format(
                    name=entry["name"], avoid=entry["avoid"], rule=entry["rule"]
                )
            )
    if case.decoy == features.DECOY_LOWER:
        lines.append(str(prompt["decoy_line"]).format(decoy=features.DECOY_EXAMPLE))
    lines.append(str(prompt["extra_rule"]))
    lines.append(str(prompt["length_rule"]).format(min=MIN_CONTENT_CHARS, max=MAX_CONTENT_CHARS))
    lines.append(str(prompt["self_check"]))
    lines.append(str(prompt["output"]))
    return "\n".join(lines)


def build_task(spec: PostTriageSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：真值规格、语言字典。输出：渲染任务。

    预期行为：任务 id 由规格算（断点续跑按它查进度），`spec` 里带上注入特征——后端实现与假后端
    若想「照着计划答」，不必再解析提示词。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="posttriage",
        prompt=build_prompt(loc, spec.case),
        response_model=ContentDraft,
        seeds=(),
        spec={
            "case": spec.case.as_dict(),
            "signals": list(spec.case.hits()),
            "decoy": spec.case.decoy,
        },
    )


def check_draft(draft: ContentDraft, case: PostTriageCase) -> None:
    """校验「帖子正文」是否兑现了注入特征。

    输入：后端草稿、注入特征。输出：无（不合格抛 `DraftError`，管线据此重渲染）。
    预期行为：逐条查——

    1. 正文非空、不带围栏、长度落在 `MIN_CONTENT_CHARS`~`MAX_CONTENT_CHARS`；
    2. **按字面算出来的命中组合与计划逐条相同**（多一条少一条都不行，理由里说清差在哪）；
    3. 诱饵：`lower` 时正文里必须真有那个小写 cashtag 形状，`none` 时一处都不许有。

    第 2 条是这一族的核心：命中组合就是金标签，它由正则从文本算得出来，所以「模型写歪了」在
    渲染期就能拦住，不用等判别器。
    """
    text = draft.text.strip()
    if not text:
        raise DraftError("正文是空的")
    if "```" in text:
        raise DraftError(f"正文里带围栏：{text[:60]!r}")
    if len(text) < MIN_CONTENT_CHARS:
        raise DraftError(f"正文太短（至少 {MIN_CONTENT_CHARS} 个字符）：{text!r}")
    if len(text) > MAX_CONTENT_CHARS:
        raise DraftError(f"正文太长（最多 {MAX_CONTENT_CHARS} 个字符，实际 {len(text)}）")
    found = features.signals_of(text)
    if found != case.signals:
        raise DraftError(
            f"正文按字面算出来的命中组合是 {found or '无'}，计划要的是 {case.signals or '无'}"
        )
    decoy = features.has_decoy(text)
    if case.decoy == features.DECOY_LOWER and not decoy:
        raise DraftError(f"计划要埋一个小写 cashtag 的诱饵（像 {features.DECOY_EXAMPLE}），正文里没有")
    if case.decoy == features.DECOY_NONE and decoy:
        raise DraftError("正文里出现了小写 cashtag 形状的诱饵，但计划没要")


def to_state(draft: ContentDraft, loc: dict[str, JsonValue]) -> str:
    """输入：后端草稿、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：判据表 + 阈值表 + 正文抬头 + 正文。**判据表与阈值表印在 state 里**，所以「这条帖子
    命中了哪几条、该给什么处置」是自明的——考生只读 state 与题面就能答，不必知道外部政策
    （范式 §9.8）。
    """
    marks = loc["state"]
    parts = [str(marks["signal_mark"])]
    parts.extend(f"{LINE_PREFIX}{signal_line(loc, signal)}" for signal in features.SIGNAL_IDS)
    parts.append(str(marks["route_mark"]))
    parts.extend(f"{LINE_PREFIX}{threshold_line(loc, level)}" for level in features.ROUTE_LEVELS)
    parts.append(str(marks["content_mark"]))
    parts.append(draft.text.strip())
    return "\n".join(parts)


def parse_with(state: str, loc: dict[str, JsonValue]) -> ParsedContent:
    """输入：state、语言字典。输出：按这一份语言反解的结果。

    预期行为：认判据表抬头、阈值表抬头、正文抬头；判据表里带行前缀的那些行取编号（按
    `state.signal_sep` 切），**正文抬头之后的所有行原样拼回正文**——正文是多行的，抬头之外的
    内容一律不当结构，否则正文里提到「判据」两个字就会被数成判据表。
    """
    marks = loc["state"]
    signal_mark = str(marks["signal_mark"])
    route_mark = str(marks["route_mark"])
    content_mark = str(marks["content_mark"])
    sep = str(marks["signal_sep"])
    signals: list[str] = []
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
        if text.startswith(signal_mark):
            section = "signal"
            continue
        if text.startswith(route_mark):
            section = "route"
            continue
        if section == "signal" and text.startswith(LINE_PREFIX):
            signals.append(text[len(LINE_PREFIX) :].partition(sep)[0].strip())
    content = "\n".join(body).strip() if body else None
    return ParsedContent(signals=tuple(signals), text=content)


def detect_locale(state: object) -> str | None:
    """输入：state。输出：能认出正文抬头的那一种语言（认不出给 None）。

    预期行为：审计拿它按 state **自己**的语言取规范文案——一次 run 里混了 zh / en 时，用调用方
    给的那一种去比会平白报一堆「文案对不上」。

    为什么认的是**正文抬头**而不是判据编号：编号（S1…S4）是结构键，与语言无关，拿它当判据会把
    别的语言的 state 也抢过来（这条坑在 lint 上踩过一次，见范式 §9.22）；正文抬头是渲染时按语言
    写死的、整段最后那一块的开头，才分得开。
    """
    if not isinstance(state, str):
        return None
    for lang in locales.available():
        if parse_with(state, locales.load(lang)).found:
            return lang
    return None


def parse_state(state: object) -> ParsedContent:
    """输入：state。输出：纯代码能反解的部分（判据表 / 正文）。

    预期行为：与语言无关——把各语言的标记都试一遍，认出正文抬头的那一份就用它；一份都认不出来
    时返回空结果（`text` 为 None）。
    """
    if not isinstance(state, str):
        return ParsedContent(signals=(), text=None)
    lang = detect_locale(state)
    if lang is None:
        return ParsedContent(signals=(), text=None)
    return parse_with(state, locales.load(lang))
