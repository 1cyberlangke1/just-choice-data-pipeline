"""广告元素判定的文本外壳：判据表、处置档位表与片段抬头由代码写，HTML 片段由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **判据表、档位表、片段抬头由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**一段 HTML 元素片段**（真实页面里的广告位长尾得很，模板拼出来的脱离现实分布）；
- **命中哪几条判据由代码按字面算**（`features.PATTERNS` 那四条正则），模型碰不到：提示词里逐条
  写明「必须出现哪些形状、出现在哪个位置、不许出现哪些」，`check_draft` 在渲染期按同一份判据复核
  一遍（不合格就重渲染），`judges.HygieneJudge` / `FrameJudge` 再各查一遍。

**判据原文也印进 state**（范式 §9.44 的教训）：反解模型要判「这段片段踩了哪几条」，而判据名是
**压缩过的标签**——「广告位命名」推不出「class / id 的取值里要有 ad / banner / sponsor 这类词」。

**提示词里一个示例片段都不给**（范式 §9.45）：`logtriage` 上量过一次——给了成品句，20 条里 16 条
整段照抄，写一句「那只是例子」也拦不住。所以这里只印**判据原文**（它本来就是完整规格），
`pasted_rules` 再兜住「把判据原文整段贴进去」这条残余捷径。

**位置也要在提示词里说清**（范式 §9.46）：四条判据各占一处（class / id 取值、标签之间的文字、
资源与投放属性、style 与尺寸属性），不说清的话模型会把它们写进同一处、互相遮蔽。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
"""

from __future__ import annotations

import re
from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.locale_files import copy_entry, entry_line
from jc.state_parse import ParsedState as ParsedContent, parse_any, parse_sections, recognize_locale
from jc.workflow import DraftError

from . import features, locales
from .features import AdBlockCase
from .spec import AdBlockSpec

#: 判据表与档位表每行的前缀。**这是 state 格式的一部分，不是文案**，所以不进语言文件：
#: 渲染与解析共用这一个常量，两边不可能漂移。
LINE_PREFIX: Final[str] = "- "

#: 标签本身（`<...>`）。`visible_text` 拿它把片段切成「标签」与「标签之间的文字」两类。
_TAG_RE: Final[re.Pattern[str]] = re.compile(r"<[^>]*>")


def visible_text(snippet: str) -> str:
    """输入：一段 HTML 片段。输出：元素里**标签之间的可见文字**（空白折叠成单个空格）。

    预期行为：按标签切开，只留下标签之外的那些块——属性值里的字不算可见文字（A2 那条判据也是
    这么算的），标签名与属性名更不算。

    为什么需要它：`Workflow.language_texts` 要把「模型写的自然语言正文」交给 `jc.language` 那把
    尺子，而这一族的正文就是**元素里的可见文字**。整段片段拿过去比会把标签名 / 属性名（全是
    ASCII）也算进语言判定——那些不是自然语言正文，不该交给那把尺子（真机 `ab-10b-zh` 10 条里
    有 5 条的整段按**字母**比掉到 0.15 以下，范式 §9.49）。
    """
    return " ".join(part.strip() for part in _TAG_RE.split(snippet) if part.strip())

#: 片段的字符数上下限。太短装不下一个像样的元素（标签 + 属性 + 文字），太长就只是一整段页面。
MIN_CONTENT_CHARS: Final[int] = features.MIN_CONTENT_CHARS
MAX_CONTENT_CHARS: Final[int] = features.MAX_CONTENT_CHARS


class ContentDraft(BaseModel):
    """后端要填的草稿：那一段 HTML 元素片段。

    `snippet` 是**一段 HTML**（真实页面的广告位长尾得很、由模型写）；结构化事实（判据表、档位表、
    抬头）全由组装器渲染，模型碰不到。
    """

    snippet: str = Field(
        description="那个 DOM 元素（含它最直接的上下文）的 HTML 片段；不要围栏、不要解释"
    )


def tag_copy(loc: dict[str, JsonValue], tag: str) -> dict[str, JsonValue]:
    """输入：语言字典、判据编号。输出：该判据的文案（name / rule / avoid）。

    预期行为：缺字段就当场报错——判据表与提示词都靠它，缺一个字段模型就会拿到半句话。
    """
    return copy_entry(loc, "tag", tag, ("name", "rule", "avoid"), "判据")


def tag_line(loc: dict[str, JsonValue], tag: str) -> str:
    """输入：语言字典、判据编号。输出：判据表里那一行（不含行前缀）。

    预期行为：`编号 + 分隔符 + 判据名 + 规则分隔符 + 判据原文`。分隔符按语言取
    （`state.tag_sep`），解析时按同一份分隔符切回编号——**编号是结构键**，所以它排在分隔符前面，
    与语言的书写顺序无关。

    **为什么判据原文也要印进 state**：判据名是压缩过的标签，不是判据本身。反解模型要照着同一份
    机械口径判，材料里就必须有那份口径（范式 §9.44）。
    """
    return entry_line(loc, tag, tag_copy(loc, tag))


def surface_copy(loc: dict[str, JsonValue], surface: str) -> str:
    """输入：语言字典、页面类型。输出：该页面类型的形态描述。

    预期行为：缺这一档就当场报错（新加一档页面却忘了写文案时，渲染会拿不到句子）。
    """
    table = loc.get("surface")
    entry = table.get(surface) if isinstance(table, dict) else None
    if not isinstance(entry, dict) or "desc" not in entry:
        raise ValueError(f"语言文件里缺页面类型 {surface} 的描述")
    return str(entry["desc"])


def threshold_line(loc: dict[str, JsonValue], level: str) -> str:
    """输入：语言字典、处置档位。输出：档位表里那一行（不含行前缀）。

    预期行为：`阈值描述 + 分隔符 + 处置动作`；处置动作与 `act` 那一族的选项取的是**同一份文案**
    （`fam.act.opt`），所以 state 里那张表和题面不可能各说各的。
    """
    sep = str(loc["state"]["tag_sep"])
    action = str(loc["fam"]["act"]["opt"][level])
    return f"{loc['state']['threshold'][level]}{sep}{action}"


def build_prompt(loc: dict[str, JsonValue], case: AdBlockCase) -> str:
    """组装提示词。

    输入：语言字典、注入特征。输出：完整提示词（不含答案）。
    预期行为：**每一条硬性要求都写清楚**——必须出现哪些形状（逐条印上判据原文）、出现在哪个位置、
    不许出现哪些（逐条给反面写法）、这个元素在哪种页面上、长度区间、必须是能读的 HTML、输出前自查。
    这些就是「受控合成」的规格，判据全在 `check_draft` 里机械复核，模型写歪了当场重渲染。

    为什么**一个示例片段都不给**（范式 §9.45）：只给一个具体片段当例子，模型会把它**整段照抄**进
    样本，而且「那只是例子、请自己另编」这句约束写进提示词也拦不住。判据原文本身就是完整规格，
    所以提示词只印判据原文、不给可抄的成品片段。

    为什么把「不许出现」也逐条列出来：只写「必须出现什么」时，模型会顺手带上别的形状（写了
    `class="ad-slot"` 又顺手给个 `width="1"`）——那一条的标签就废了。
    """
    prompt = loc["prompt"]
    hits = case.tags()
    missing = [tag for tag in features.TAG_IDS if tag not in hits]
    lines = [
        str(prompt["intro"]).format(min=MIN_CONTENT_CHARS, max=MAX_CONTENT_CHARS),
        str(prompt["lang_rule"]).format(language=loc["language_name"]),
        str(prompt["surface_line"]).format(surface=surface_copy(loc, case.surface)),
    ]
    if hits:
        lines.append(str(prompt["hits_header"]))
        for tag in hits:
            entry = tag_copy(loc, tag)
            lines.append(str(prompt["hits_line"]).format(name=entry["name"], rule=entry["rule"]))
        lines.append(str(prompt["vary_rule"]))
    if missing:
        lines.append(str(prompt["miss_header"]) if hits else str(prompt["no_hits"]))
        for tag in missing:
            entry = tag_copy(loc, tag)
            lines.append(
                str(prompt["miss_line"]).format(
                    name=entry["name"], avoid=entry["avoid"], rule=entry["rule"]
                )
            )
    lines.append(str(prompt["length_rule"]).format(min=MIN_CONTENT_CHARS, max=MAX_CONTENT_CHARS))
    lines.append(str(prompt["self_check"]))
    lines.append(str(prompt["output"]))
    return "\n".join(lines)


def build_task(spec: AdBlockSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：真值规格、语言字典。输出：渲染任务。

    预期行为：任务 id 由规格算（断点续跑按它查进度），`spec` 里带上注入特征——后端实现与假后端
    若想「照着计划答」，不必再解析提示词。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="adblock",
        prompt=build_prompt(loc, spec.case),
        response_model=ContentDraft,
        seeds=(),
        spec={
            "case": spec.case.as_dict(),
            "hits": list(spec.case.tags()),
            "surface": spec.case.surface,
        },
    )


def pasted_rules(text: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：HTML 片段、语言字典。输出：被**逐字贴进**样本的判据原文（按 `TAG_IDS` 顺序）。

    预期行为：只认**完全相同的整段**判据原文（`tag_copy(...)["rule"]`）——改一个词就不再算。

    为什么要有这条：提示词里给模型看的**唯一**一段具体文字就是判据原文（示例片段一律不给，见
    `build_prompt`），所以模型能「抄」的只剩它——把判据的词表整段贴进片段，标签照样算命中（正则
    只看子串），但那段片段读起来不是 HTML，而是判据本身。这条把这种残余捷径变成机械可查：贴了就
    重渲染。真实页面元素不会包含整段判据原文，所以这条**不会误伤**正常产物。
    """
    return [
        entry
        for entry in (str(tag_copy(loc, tag)["rule"]) for tag in features.TAG_IDS)
        if entry in text
    ]


def check_draft(draft: ContentDraft, case: AdBlockCase, loc: dict[str, JsonValue]) -> None:
    """渲染期校验：草稿必须**兑现注入特征**。

    输入：草稿、注入特征、语言字典。输出：无（不合格抛 `DraftError`，管线据此重渲染）。
    预期行为：逐条查——

    1. **非空、不带围栏、长度落在区间内、真的是一段 HTML**（至少有一个标签）；
    2. **按字面算出来的命中组合与计划逐条相同**（多一条少一条都报错，并把两边都写进理由）；
    3. **不许把判据原文原样贴进片段**（`pasted_rules` 逐字比对）。
    第 2 条是这一族的核心：命中组合就是金标签，它由正则从文本算得出来，所以「模型写歪了」在
    渲染期就能拦住，不用等判别器。
    """
    text = draft.snippet.strip()
    if not text:
        raise DraftError("片段是空的")
    if "```" in text:
        raise DraftError(f"片段里带围栏：{text[:60]!r}")
    if "<" not in text:
        raise DraftError(f"这不是一段 HTML（一个标签都没有）：{text[:60]!r}")
    if len(text) < MIN_CONTENT_CHARS:
        raise DraftError(f"片段太短（至少 {MIN_CONTENT_CHARS} 个字符）：{text!r}")
    if len(text) > MAX_CONTENT_CHARS:
        raise DraftError(f"片段太长（最多 {MAX_CONTENT_CHARS} 个字符，实际 {len(text)}）")
    found = features.hits_of(text)
    if found != case.hits:
        raise DraftError(
            f"片段按字面算出来的命中组合是 {found or '无'}，计划要的是 {case.hits or '无'}"
        )
    pasted = pasted_rules(text, loc)
    if pasted:
        raise DraftError(f"片段把判据原文原样贴了进去：{pasted}")


def to_state(draft: ContentDraft, loc: dict[str, JsonValue]) -> str:
    """输入：后端草稿、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：判据表 + 档位表 + 片段抬头 + HTML 片段。**判据表与档位表印在 state 里**，所以
    「这段片段踩了哪几条、该给什么处置」是自明的——考生只读 state 与题面就能答，不必知道外部
    规则（范式 §9.8）。
    """
    marks = loc["state"]
    parts = [str(marks["tag_mark"])]
    parts.extend(f"{LINE_PREFIX}{tag_line(loc, tag)}" for tag in features.TAG_IDS)
    parts.append(str(marks["rule_mark"]))
    parts.extend(f"{LINE_PREFIX}{threshold_line(loc, level)}" for level in features.ACTS)
    parts.append(str(marks["content_mark"]))
    parts.append(draft.snippet.strip())
    return "\n".join(parts)


def parse_with(state: str, loc: dict[str, JsonValue]) -> ParsedContent:
    """输入：state、语言字典。输出：按这一份语言反解的结果。"""
    return parse_sections(state, loc, line_prefix=LINE_PREFIX)


def detect_locale(state: object) -> str | None:
    """输入：state。输出：能认出素材抬头的那一种语言（认不出给 None）。"""
    return recognize_locale(state, locales, parse_with)


def parse_state(state: object) -> ParsedContent:
    """输入：state。输出：纯代码能反解的部分（判据表 / 素材）。"""
    return parse_any(state, locales, parse_with)
