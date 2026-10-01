"""引文核查的文本外壳：文档骨架、引文标记与「引文」都由代码写，短文的正文由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **文档抬头、节抬头、句子前缀、引文抬头、声称出处抬头由代码渲染**——它们是结构化事实，
  让模型去写只会写歪；
- 后端写**标题与 6 句话**（一篇短文的正文是长尾的，模板拼的脱离现实）；
- **引文由代码从这篇短文里剪**：`exact` 整句照抄、`partial` 截前半句、`altered` 把句子里
  的一个数字换掉。所以「引文到底在不在文档里」是**算**出来的，不靠模型自己说。
  `check_draft` 在渲染期查一遍是为了**重试**，`judges.HygieneJudge` 再查一遍是为了兜住
  不是本 workflow 渲染出来的 state。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.state_parse import recognize_locale
from jc.workflow import DraftError

from . import features, locales
from .features import CitationPlan
from .spec import CitationSpec

#: 每句话的行前缀。**这是 state 格式的一部分，不是文案**，所以不进语言文件：渲染与解析
#: 共用这一个常量，两边不可能漂移。
LINE_PREFIX: Final[str] = "- "

#: 句子太短时 `partial` 会退化成整句，所以文档被拒的门槛也印在 state 格式里（不是文案）。
MIN_SENTENCE_CHARS: Final[int] = features.MIN_SENTENCE_CHARS


class CitationDraft(BaseModel):
    """后端要填的草稿：短文标题 + 6 句话。

    两个字段都是**自然语言**（正文是长尾的、由模型写）；结构化事实（抬头、节号、引文、声称
    出处）全由组装器渲染，模型碰不到。
    """

    title: str = Field(description="短文的标题，单独一行，不超过 30 个字符")
    sentences: list[str] = Field(
        min_length=features.SENTENCE_COUNT,
        max_length=features.SENTENCE_COUNT,
        description="6 句话，按「第 1 节两句、第 2 节两句、第 3 节两句」的顺序",
    )


@dataclass(frozen=True)
class ParsedCitation:
    """从 state 反解出来的、纯代码看得见的部分。"""

    title: str | None
    sections: tuple[tuple[str, ...], ...]
    quote: str | None
    claimed: int | None

    @property
    def found(self) -> bool:
        """输入：无。输出：文档抬头认出来了没有。预期行为：就是 `title` 在不在。"""
        return self.title is not None

    @property
    def sentences(self) -> tuple[str, ...]:
        """输入：无。输出：6 句话（按节顺序摊平）。预期行为：节数不对时如实少给。"""
        return tuple(item for group in self.sections for item in group)


def sec_ref(loc: dict[str, JsonValue], index: int) -> str:
    """输入：语言字典、节下标（0 起）。输出：节的可读指代（`第 3 节`）。"""
    return str(loc["state"]["sec_ref"]).format(n=index + 1)


def sections_of(sentences: list[str] | tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """输入：摊平的句子。输出：按节切好的句子。

    预期行为：每 `SENTENCES_PER_SECTION` 句一节，顺序不变；句子数不是整数倍时最后一节短一截
    （由 `check_draft` 负责把它判成不合格）。
    """
    step = features.SENTENCES_PER_SECTION
    return tuple(
        tuple(sentences[start : start + step]) for start in range(0, len(sentences), step)
    )


def make_quote(sentence: str, form: str, sentences: tuple[str, ...]) -> str:
    """输入：被引用的那句话、造法、整篇文档的句子。输出：引文原文。

    预期行为：三种造法各走一条路——
    - `exact`：整句照抄（含句末标点）；
    - `partial`：截前半句（至少 `MIN_PARTIAL` 个字符），**是原句的真子串**；以空格分词的文本
      会退到最近的词边界（见 `_partial_cut`）；
    - `altered`：把句子里**第一个数字**换掉，换到「整句一处都不出现」为止——这才是「查无此
      文」。取整篇文档而不只是原句，是因为换完的句子有可能**正好等于文档里的另一句**（两句
      只在数字上不同时就会这样），那时换一个数字就行。
    """
    text = sentence.strip()
    if form == features.FORM_EXACT:
        return text
    if form == features.FORM_PARTIAL:
        return text[: _partial_cut(text)].strip()
    if form == features.FORM_ALTERED:
        found = features.DIGITS.search(text)
        if found is None:
            raise DraftError(f"句子里没有具体数字，改不了：{text!r}")
        original = int(found.group())
        for delta in range(1, 10):
            replacement = str(original + delta)
            quote = text[: found.start()] + replacement + text[found.end() :]
            if all(quote not in item for item in sentences):
                return quote
        raise DraftError(f"换不出一个不在文档里的数字：{text!r}")
    raise ValueError(f"未知造法：{form!r}")


def _partial_cut(text: str) -> int:
    """输入：一句话。输出：截前半句的切点。

    预期行为：从中点切开，但**以空格分词的文本要退到最近的词边界**——真机第二批里有一条被读者
    模型判成「文档里没有这句」，引文是 `The final release includes 8 downloadable CSV fi`：
    字符确实在，可它砍在 `files` 中间，读起来像一个不完整的片段，读者（人和模型都一样）会
    当成「不是原样引用」。退到词边界之后引文长这样：`... 8 downloadable CSV`，一眼就能对上。

    中文没有空格，`rfind` 找不到边界，于是保持字符切分——中文按字数截断本来就是常态。退到
    词边界会让引文变短，短于 `MIN_PARTIAL` 时就不退（宁可砍在词中间，也不要一个没有信息量的
    片段）。
    """
    cut = max(features.MIN_PARTIAL, len(text) // 2)
    if cut >= len(text):
        raise DraftError(f"句子太短，截不出真子串：{text!r}")
    if text[cut - 1] != " " and text[cut] != " ":
        back = text.rfind(" ", 0, cut)
        if back >= features.MIN_PARTIAL:
            cut = back
    return cut


def build_prompt(loc: dict[str, JsonValue], topic: int) -> str:
    """组装提示词。

    输入：语言字典、题材方向下标。输出：完整提示词（不含答案）。
    预期行为：**每条硬性要求都写清楚**——节数句数、每句一个具体数字、最短长度、两两不重复。
    这些是「受控合成」的规格：判据全在 `check_draft` 里机械复核，模型写歪了当场重渲染。
    **语言约束紧跟开场白**：真机 en 侧出过整篇写成中文的样本（汉字占比 0.770）——提示词
    通篇是目标语言并不足以让模型不换语言，得有一句明说的。

    为什么还要给一个**题材方向**（范式 §9.57）：首版**一个样本级变量都没有**，同切分的提示词
    逐字相同，标题当场塌成同一个字符串（真机 9 条 en 产物全是 `System Log Report 2026`）。
    方向由 `spec.topic` 抽出来、只影响短文与标题怎么写，与判据无关。
    """
    prompt = loc["prompt"]
    return "\n".join(
        [
            str(prompt["intro"]).format(
                sections=features.SECTIONS,
                per=features.SENTENCES_PER_SECTION,
                n=features.SENTENCE_COUNT,
            ),
            str(prompt["lang_rule"]).format(language=loc["language_name"]),
            str(prompt["topic_line"]).format(
                topic=locales.pick(loc["prompt"]["topics"], topic)
            ),
            str(prompt["digits_rule"]),
            str(prompt["length_rule"]).format(min=MIN_SENTENCE_CHARS),
            str(prompt["title_rule"]),
            str(prompt["self_check"]),
            str(prompt["output"]).format(n=features.SENTENCE_COUNT),
        ]
    )


def build_task(spec: CitationSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：真值规格、语言字典。输出：渲染任务。

    预期行为：任务 id 由规格算（断点续跑按它查进度），`spec` 里带上造法与出处——后端实现与
    假后端若想「照着计划答」，不必再解析提示词。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="citation",
        prompt=build_prompt(loc, spec.topic),
        response_model=CitationDraft,
        spec={
            "case": spec.case.as_dict(),
            "form": spec.plan.form,
            "true_section": spec.plan.true_section,
            "claimed": spec.plan.claimed,
        },
    )


def check_draft(draft: CitationDraft, plan: CitationPlan, loc: dict[str, JsonValue]) -> None:
    """校验「短文 + 引文」是否兑现了引文计划。

    输入：后端草稿、引文计划、语言字典。输出：无（不合格抛 `DraftError`，管线据此重渲染）。
    预期行为：逐条查——

    1. 标题与 6 句话都是单行、非空、不带围栏；句子够长、每句都有具体数字、两两不重复；
    2. 按计划剪出来的引文**与计划一致**：`exact` 就是原句、`partial` 是原句的真子串、
       `altered` 是原句换一个数字且**整篇文档一处都不出现**；
    3. 引文**只出在那一节里**（`exact` / `partial`）——否则「引文出自第几节」就没有唯一解。

    为什么第 3 条要机械查：`partial` 截的是半句，半句完全可能在别的小节里也出现；一旦撞上，
    那一条的 `section` 答案就是错的，必须在渲染期拦下重写。
    """
    text = draft.title.strip()
    if not text:
        raise DraftError("标题是空的")
    if "\n" in text:
        raise DraftError("标题必须是单行")
    if "```" in text:
        raise DraftError(f"标题里带围栏：{text!r}")
    sentences = [item.strip() for item in draft.sentences]
    if len(sentences) != features.SENTENCE_COUNT:
        raise DraftError(
            f"句子该有 {features.SENTENCE_COUNT} 句，实际 {len(sentences)} 句"
        )
    if any(not item for item in sentences):
        raise DraftError(f"句子里有空串：{sentences!r}")
    if any("\n" in item for item in sentences):
        raise DraftError("每句话必须是单行（出现换行说明模型把整段塞进了一句）")
    if any("```" in item for item in sentences):
        raise DraftError(f"句子里带围栏：{sentences!r}")
    short = [item for item in sentences if len(item) < MIN_SENTENCE_CHARS]
    if short:
        raise DraftError(f"句子太短（至少 {MIN_SENTENCE_CHARS} 个字符）：{short!r}")
    missing = [item for item in sentences if not features.DIGITS.search(item)]
    if missing:
        raise DraftError(f"句子里没有具体数字（引文要拿它换）：{missing!r}")
    if len(set(sentences)) != len(sentences):
        raise DraftError(f"句子重复：{sentences!r}")
    groups = sections_of(sentences)
    if len(groups) != features.SECTIONS:
        raise DraftError(f"小节该有 {features.SECTIONS} 节，实际 {len(groups)} 节")
    index = plan.sentence_index
    target = sentences[index]
    quote = make_quote(target, plan.form, tuple(sentences))
    if plan.form == features.FORM_EXACT:
        if quote != target:
            raise DraftError(f"整句照抄的引文与原句不一致：{quote!r} != {target!r}")
    elif plan.form == features.FORM_PARTIAL:
        if quote == target or quote not in target:
            raise DraftError(f"截半句的引文必须是原句的真子串：{quote!r} 不在 {target!r} 里")
    else:
        if quote == target:
            raise DraftError(f"改数字的引文必须与原句不同：{quote!r}")
        if any(quote in item for item in sentences):
            raise DraftError(f"改数字的引文居然还在文档里：{quote!r}")
    if plan.form != features.FORM_ALTERED:
        hits = features.locate_sections(groups, quote)
        if hits != (plan.true_section,):
            shown = "、".join(f"第 {index + 1} 节" for index in hits) or "哪一节都没有"
            raise DraftError(
                f"引文该只出在第 {plan.true_section + 1} 节，实际出现在{shown}：{quote!r}"
            )


def to_state(draft: CitationDraft, plan: CitationPlan, loc: dict[str, JsonValue]) -> str:
    """输入：后端草稿、引文计划、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：标题 + 3 节（每节两句）+ 引文 + 声称出处；**引文与声称出处印在 state 里**，
    所以「这条引文出自哪一节」是自明的（范式 §9.8）。
    """
    state = loc["state"]
    sentences = [item.strip() for item in draft.sentences]
    groups = sections_of(sentences)
    quote = make_quote(sentences[plan.sentence_index], plan.form, tuple(sentences))
    parts = [f"{state['doc_mark']}{draft.title.strip()}"]
    for index, group in enumerate(groups):
        parts.append(str(state["sec_mark"]).format(n=index + 1))
        parts.extend(f"{LINE_PREFIX}{item}" for item in group)
    parts.append(f"{state['cite_mark']}{quote}")
    parts.append(f"{state['claim_mark']}{str(state['claim_value']).format(n=plan.claimed + 1)}")
    return "\n".join(parts)


def parse_with(state: str, loc: dict[str, JsonValue]) -> ParsedCitation:
    """输入：state、语言字典。输出：按这一份语言反解的结果。

    预期行为：认文档抬头、节抬头（三节各一个）、引文抬头、声称出处抬头；句子是**当前节**里的
    行前缀行。抬头之外的行一律不当句子——否则引文那几行会被数成正文。
    """
    marks = loc["state"]
    doc_mark = str(marks["doc_mark"])
    cite_mark = str(marks["cite_mark"])
    claim_mark = str(marks["claim_mark"])
    sec_marks = [
        str(marks["sec_mark"]).format(n=index + 1) for index in range(features.SECTIONS)
    ]
    title: str | None = None
    quote: str | None = None
    claimed: int | None = None
    groups: list[list[str]] = []
    current: list[str] | None = None
    for raw in state.splitlines():
        text = raw.strip()
        if text.startswith(doc_mark):
            title = text[len(doc_mark) :].strip()
            current = None
            continue
        hit = next(
            (index for index, mark in enumerate(sec_marks) if text.startswith(mark)), None
        )
        if hit is not None:
            groups.append([])
            current = groups[-1]
            continue
        if text.startswith(cite_mark):
            quote = text[len(cite_mark) :].strip()
            current = None
            continue
        if text.startswith(claim_mark):
            body = text[len(claim_mark) :].strip()
            found = features.DIGITS.search(body)
            claimed = int(found.group()) - 1 if found else None
            current = None
            continue
        if current is not None and text.startswith(LINE_PREFIX):
            current.append(text[len(LINE_PREFIX) :].strip())
    return ParsedCitation(
        title=title,
        sections=tuple(tuple(group) for group in groups),
        quote=quote,
        claimed=claimed,
    )


def detect_locale(state: object) -> str | None:
    """输入：state。输出：能认出文档抬头的那一种语言（认不出给 None）。

    预期行为：审计拿它按 state **自己**的语言取规范文案——一次 run 里混了 zh / en 时，用调用方
    给的那一种去比会平白报一堆「文案对不上」。

    为什么认的是**文档抬头**而不是节抬头：节抬头只有「第 1 节」这几个字，它与句子行的关系太
    弱；文档抬头是渲染时按语言写死的、整段开头那一行，拿它当判据才分得开（这条坑在 lint 上
    踩过一次，见范式 §9.22）。
    """
    return recognize_locale(state, locales, parse_with)


def parse_state(state: object) -> ParsedCitation:
    """输入：state。输出：纯代码能反解的部分（标题 / 分节句子 / 引文 / 声称出处）。

    预期行为：与语言无关——把各语言的标记都试一遍，认出文档抬头的那一份就用它；一份都认不
    出来时返回空结果（`title` 为 None）。
    """
    lang = detect_locale(state)
    if lang is None:
        return ParsedCitation(title=None, sections=(), quote=None, claimed=None)
    return parse_with(state, locales.load(lang))
