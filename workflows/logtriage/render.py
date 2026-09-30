"""日志行分诊的文本外壳：噪声判据表、处置档位表与日志抬头由代码写，日志行由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **判据表、档位表、日志抬头由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**一行日志**（应用日志、编排日志、访问日志都是长尾的，模板拼出来的脱离现实分布）；
- **命中哪几条判据由代码按字面算**（`features.PATTERNS` 那四条正则），模型碰不到：提示词里逐条
  写明「必须出现哪些形状、不许出现哪些」，`check_draft` 在渲染期按同一份判据复核一遍（不合格就
  重渲染），`judges.HygieneJudge` / `FrameJudge` 再各查一遍，兜住不是本 workflow 渲染出来的
  state。

**判据原文也印进 state**（范式 §9.44 的教训，这一族一开始就照做）：反解模型要判「这行日志踩了
哪几条」，而判据名是**压缩过的标签**——「已自愈的重试与降级」推不出「出现『重试中 / 退避 /
falling back』这类词」。`memorygate` 上量过一次：只印判据名时反解 1/8，补上原文 8/8。

**通道（`source`）为什么要写进提示词**：同一份命中组合可以长在应用日志、编排日志或访问日志上，
而这三种形态的**排版差别很大**（访问日志是 `时间 IP 方法 路径 状态码` 的定宽行，编排日志带
容器名与重启计数，应用日志带 logger 名与线程）。不点明形态，模型写出来的东西会四不像；点明之后，
这一维就成了一条**只影响渲染、不影响标签**的留出维（`test_ood` 侧是训练侧一次都没见过的
`access`）。

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
from .features import LogTriageCase
from .spec import LogTriageSpec

#: 判据表与档位表每行的前缀。**这是 state 格式的一部分，不是文案**，所以不进语言文件：
#: 渲染与解析共用这一个常量，两边不可能漂移。
LINE_PREFIX: Final[str] = "- "

#: 日志行的长度上下限。太短读不出「像不像一行真日志」，太长就只是灌水。
MIN_CONTENT_CHARS: Final[int] = features.MIN_CONTENT_CHARS
MAX_CONTENT_CHARS: Final[int] = features.MAX_CONTENT_CHARS


class ContentDraft(BaseModel):
    """后端要填的草稿：那一行日志。

    `text` 是**一行文本**（日志本身是长尾的、由模型写）；结构化事实（判据表、档位表、抬头）
    全由组装器渲染，模型碰不到。
    """

    text: str = Field(
        description="日志行：一行读得通的日志（应用日志 / 编排日志 / 访问日志），"
        "不要标题、不要围栏"
    )


@dataclass(frozen=True)
class ParsedContent:
    """从 state 反解出来的、纯代码看得见的部分。

    字段：`tags` 判据表里列出的判据编号；`text` 日志行（认不出来给 None）。
    """

    tags: tuple[str, ...]
    text: str | None

    @property
    def found(self) -> bool:
        """输入：无。输出：日志那一块认出来了没有。"""
        return self.text is not None


def tag_copy(loc: dict[str, JsonValue], tag: str) -> dict[str, JsonValue]:
    """输入：语言字典、判据编号。输出：该判据的文案（name / rule / avoid）。

    预期行为：缺字段就当场报错——判据表与提示词都靠它，缺一个字段模型就会拿到半句话。
    """
    table = loc.get("tag")
    entry = table.get(tag) if isinstance(table, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"语言文件里缺判据 {tag} 的文案")
    for key in ("name", "rule", "avoid"):
        if key not in entry:
            raise ValueError(f"判据 {tag} 的文案缺字段 {key}")
    return entry


def tag_line(loc: dict[str, JsonValue], tag: str) -> str:
    """输入：语言字典、判据编号。输出：判据表里那一行（不含行前缀）。

    预期行为：`编号 + 分隔符 + 判据名 + 规则分隔符 + 判据原文`。分隔符按语言取
    （`state.tag_sep`），解析时按同一份分隔符切回编号——**编号是结构键**，所以它排在分隔符
    前面，与语言的书写顺序无关。

    **为什么判据原文也要印进 state**：判据名是压缩过的标签，不是判据本身。反解模型要照着
    同一份机械口径判，材料里就必须有那份口径（范式 §9.44，在 `memorygate` 上量过 1/8 → 8/8）。
    """
    sep = str(loc["state"]["tag_sep"])
    rule_sep = str(loc["state"]["rule_sep"])
    entry = tag_copy(loc, tag)
    return f"{tag}{sep}{entry['name']}{rule_sep}{entry['rule']}"


def source_copy(loc: dict[str, JsonValue], source: str) -> str:
    """输入：语言字典、通道。输出：该通道的形态描述。

    预期行为：缺这一档就当场报错（新加一档通道却忘了写文案时，渲染会拿不到句子）。
    """
    table = loc.get("source")
    entry = table.get(source) if isinstance(table, dict) else None
    if not isinstance(entry, dict) or "desc" not in entry:
        raise ValueError(f"语言文件里缺通道 {source} 的描述")
    return str(entry["desc"])


def threshold_line(loc: dict[str, JsonValue], level: str) -> str:
    """输入：语言字典、处置档位。输出：档位表里那一行（不含行前缀）。

    预期行为：`阈值描述 + 分隔符 + 处置动作`；处置动作与 `route` 那一族的选项取的是**同一份
    文案**（`fam.route.opt`），所以 state 里那张表和题面不可能各说各的。
    """
    sep = str(loc["state"]["tag_sep"])
    action = str(loc["fam"]["route"]["opt"][level])
    return f"{loc['state']['threshold'][level]}{sep}{action}"


def build_prompt(loc: dict[str, JsonValue], case: LogTriageCase) -> str:
    """组装提示词。

    输入：语言字典、注入特征。输出：完整提示词（不含答案）。
    预期行为：**每一条硬性要求都写清楚**——必须出现哪些形状（逐条印上判据原文）、不许出现哪些
    （逐条给反面写法）、这行日志从哪种通道来、长度区间、输出前自查。这些就是「受控合成」的规格，
    判据全在 `check_draft` 里机械复核，模型写歪了当场重渲染。

    为什么**一个示例片段都不给**（范式 §9.45）：只给一个具体片段当例子，模型会把它**整段照抄**
    进日志——首轮真机小批量（`lt-10a`）里 8/10 条照抄，而且「那只是例子、请自己另编」这句约束
    写进提示词也拦不住。判据原文本身就是完整规格（`rule` 把要匹配的词逐个列了出来），所以提示词
    只印判据原文、不给可抄的成品句；`pasted_rules` 再兜住「把判据原文整段贴进日志」这条残余捷径。

    为什么把「不许出现」也逐条列出来：只写「必须出现什么」时，模型会顺手带上别的形状
    （写日志时顺手带上时间戳与级别词不算，但顺手写一句「正在重试」就会多出一条命中）——
    那一条的标签就废了。
    """
    prompt = loc["prompt"]
    hits = case.tags()
    missing = [tag for tag in features.TAG_IDS if tag not in hits]
    lines = [
        str(prompt["intro"]).format(min=MIN_CONTENT_CHARS, max=MAX_CONTENT_CHARS),
        str(prompt["lang_rule"]).format(language=loc["language_name"]),
        str(prompt["source_line"]).format(source=source_copy(loc, case.source)),
    ]
    if hits:
        lines.append(str(prompt["hits_header"]))
        for tag in hits:
            entry = tag_copy(loc, tag)
            lines.append(
                str(prompt["hits_line"]).format(
                    name=entry["name"], rule=entry["rule"]
                )
            )
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


def build_task(spec: LogTriageSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：真值规格、语言字典。输出：渲染任务。

    预期行为：任务 id 由规格算（断点续跑按它查进度），`spec` 里带上注入特征——后端实现与假后端
    若想「照着计划答」，不必再解析提示词。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="logtriage",
        prompt=build_prompt(loc, spec.case),
        response_model=ContentDraft,
        seeds=(),
        spec={
            "case": spec.case.as_dict(),
            "hits": list(spec.case.tags()),
            "source": spec.case.source,
        },
    )


def check_draft(
    draft: ContentDraft, case: LogTriageCase, loc: dict[str, JsonValue]
) -> None:
    """校验「日志行」是否兑现了注入特征。

    输入：后端草稿、注入特征、语言字典。输出：无（不合格抛 `DraftError`，管线据此重渲染）。
    预期行为：逐条查——

    1. 日志非空、不带围栏、长度落在 `MIN_CONTENT_CHARS`~`MAX_CONTENT_CHARS`；
    2. **按字面算出来的命中组合与计划逐条相同**（多一条少一条都不行，理由里说清差在哪）；
    3. **不许把判据原文原样贴进日志**（`pasted_rules` 逐字比对）。
    第 2 条是这一族的核心：命中组合就是金标签，它由正则从文本算得出来，所以「模型写歪了」在
    渲染期就能拦住，不用等判别器。
    """
    text = draft.text.strip()
    if not text:
        raise DraftError("日志是空的")
    if "```" in text:
        raise DraftError(f"日志里带围栏：{text[:60]!r}")
    if len(text) < MIN_CONTENT_CHARS:
        raise DraftError(f"日志太短（至少 {MIN_CONTENT_CHARS} 个字符）：{text!r}")
    if len(text) > MAX_CONTENT_CHARS:
        raise DraftError(f"日志太长（最多 {MAX_CONTENT_CHARS} 个字符，实际 {len(text)}）")
    found = features.hits_of(text)
    if found != case.hits:
        raise DraftError(
            f"日志按字面算出来的命中组合是 {found or '无'}，计划要的是 {case.hits or '无'}"
        )
    pasted = pasted_rules(text, loc)
    if pasted:
        raise DraftError(f"日志把判据原文原样贴了进去：{pasted}")


def pasted_rules(text: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：日志行、语言字典。输出：被**逐字贴进**日志的判据原文（按 `TAG_IDS` 顺序）。

    预期行为：只认**完全相同的整段**判据原文（`tag_copy(...)["rule"]`）——改一个词就不再算。

    为什么要有这条：提示词里给模型看的**唯一**一段具体文字就是判据原文（示例片段已经不再给，
    见 `build_prompt`），所以模型能「抄」的只剩它——把判据的词表整段贴进日志，标签照样算命中
    （正则只看子串），但那条日志读起来不是日志，而是判据本身。这条把这种残余捷径变成机械可查：
    贴了就重渲染。真实日志不会包含整段判据原文，所以这条**不会误伤**正常产物。
    """
    return [
        entry
        for entry in (str(tag_copy(loc, tag)["rule"]) for tag in features.TAG_IDS)
        if entry in text
    ]


def to_state(draft: ContentDraft, loc: dict[str, JsonValue]) -> str:
    """输入：后端草稿、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：判据表 + 档位表 + 日志抬头 + 日志。**判据表与档位表印在 state 里**，所以「这行
    日志踩了哪几条、该怎么处置」是自明的——考生只读 state 与题面就能答，不必知道外部规则
    （范式 §9.8）。
    """
    marks = loc["state"]
    parts = [str(marks["tag_mark"])]
    parts.extend(f"{LINE_PREFIX}{tag_line(loc, tag)}" for tag in features.TAG_IDS)
    parts.append(str(marks["rule_mark"]))
    parts.extend(
        f"{LINE_PREFIX}{threshold_line(loc, level)}" for level in features.HANDLING_LEVELS
    )
    parts.append(str(marks["content_mark"]))
    parts.append(draft.text.strip())
    return "\n".join(parts)


def parse_with(state: str, loc: dict[str, JsonValue]) -> ParsedContent:
    """输入：state、语言字典。输出：按这一份语言反解的结果。

    预期行为：认判据表抬头、档位表抬头、日志抬头；判据表里带行前缀的那些行取编号（按
    `state.tag_sep` 切），**日志抬头之后的所有行原样拼回日志**——抬头之外的内容一律不当结构，
    否则日志正文里出现一行以 `- ` 开头的堆栈就会把判据表数歪。
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
        if section == "tag" and text.startswith(LINE_PREFIX):
            tags.append(text[len(LINE_PREFIX) :].partition(sep)[0].strip())
    content = "\n".join(body).strip() if body else None
    return ParsedContent(tags=tuple(tags), text=content)


def detect_locale(state: object) -> str | None:
    """输入：state。输出：能认出日志抬头的那一种语言（认不出给 None）。

    预期行为：审计拿它按 state **自己**的语言取规范文案——一次 run 里混了 zh / en 时，用调用方
    给的那一种去比会平白报一堆「文案对不上」。

    为什么认的是**日志抬头**而不是判据编号：编号（N1…N4）是结构键，与语言无关，拿它当判据会把
    别的语言的 state 也抢过来（这条坑在 lint 上踩过一次，见范式 §9.22）；日志抬头是渲染时按
    语言写死的、整段最后那一块的开头，才分得开。
    """
    if not isinstance(state, str):
        return None
    for lang in locales.available():
        if parse_with(state, locales.load(lang)).found:
            return lang
    return None


def parse_state(state: object) -> ParsedContent:
    """输入：state。输出：纯代码能反解的部分（判据表 / 日志）。

    预期行为：与语言无关——把各语言的标记都试一遍，认出日志抬头的那一份就用它；一份都认不出来
    时返回空结果（`text` 为 None）。
    """
    if not isinstance(state, str):
        return ParsedContent(tags=(), text=None)
    lang = detect_locale(state)
    if lang is None:
        return ParsedContent(tags=(), text=None)
    return parse_with(state, locales.load(lang))
