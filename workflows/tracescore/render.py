"""代理轨迹自评的文本外壳：行为信号表、处置档位表与轨迹抬头由代码写，执行轨迹由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **信号表、档位表、轨迹抬头由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**一段执行轨迹**（命令行代理、编辑器内代理、对话式代理的会话片段读起来都是长尾的，
  模板拼出来的脱离现实分布）；
- **命中哪几条信号由代码按字面算**（`features.PATTERNS` 那四条正则），模型碰不到：提示词里逐条
  写明「必须出现哪些形状、不许出现哪些」，`check_draft` 在渲染期按同一份判据复核一遍（不合格就
  重渲染），`judges.HygieneJudge` / `FrameJudge` 再各查一遍，兜住不是本 workflow 渲染出来的
  state。

**信号原文也印进 state**（范式 §9.44 的教训，这一族一开始就照做）：反解模型要判「这段轨迹命中了
哪几条信号」，而信号名是**压缩过的标签**——「跑过验证」推不出「出现『跑了测试 / pytest / npm
test』这类验证动作」。`memorygate` 上量过一次：只印判据名时反解 1/8，补上原文 8/8。

**入口（`source`）为什么要写进提示词**：同一份命中组合可以长在命令行代理、编辑器内代理或对话式
代理的轨迹上，而这三者的**形态差别很大**（命令行代理的轨迹是一串「跑了什么命令 + 看到了什么
输出」，编辑器内代理带文件路径与行号，对话式代理是「用户说 / 助手说」的轮次）。不点明形态，
模型写出来的东西会四不像；点明之后，这一维就成了一条**只影响渲染、不影响标签**的留出维
（`test_ood` 侧是训练侧一次都没见过的 `chat`）。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.state_parse import ParsedState as ParsedContent, parse_any, parse_sections, recognize_locale
from jc.workflow import DraftError

from . import features, locales
from .features import TraceScoreCase
from .spec import TraceScoreSpec

#: 信号表与档位表每行的前缀。**这是 state 格式的一部分，不是文案**，所以不进语言文件：
#: 渲染与解析共用这一个常量，两边不可能漂移。
LINE_PREFIX: Final[str] = "- "

#: 执行轨迹的长度上下限。太短读不出「像不像一段真轨迹」，太长就只是灌水。
MIN_CONTENT_CHARS: Final[int] = features.MIN_CONTENT_CHARS
MAX_CONTENT_CHARS: Final[int] = features.MAX_CONTENT_CHARS


class ContentDraft(BaseModel):
    """后端要填的草稿：那段执行轨迹。

    `text` 是**一段多行文本**（轨迹本身是长尾的、由模型写）；结构化事实（信号表、档位表、抬头）
    全由组装器渲染，模型碰不到。
    """

    text: str = Field(
        description="执行轨迹：一段读得通的代理会话片段（命令行代理 / 编辑器内代理 / 对话式代理），"
        "不要标题、不要围栏"
    )


def tag_copy(loc: dict[str, JsonValue], tag: str) -> dict[str, JsonValue]:
    """输入：语言字典、信号编号。输出：该信号的文案（name / rule / avoid）。

    预期行为：缺字段就当场报错——信号表与提示词都靠它，缺一个字段模型就会拿到半句话。
    """
    table = loc.get("tag")
    entry = table.get(tag) if isinstance(table, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"语言文件里缺信号 {tag} 的文案")
    for key in ("name", "rule", "avoid"):
        if key not in entry:
            raise ValueError(f"信号 {tag} 的文案缺字段 {key}")
    return entry


def tag_line(loc: dict[str, JsonValue], tag: str) -> str:
    """输入：语言字典、信号编号。输出：信号表里那一行（不含行前缀）。

    预期行为：`编号 + 分隔符 + 信号名 + 规则分隔符 + 判据原文`。分隔符按语言取
    （`state.tag_sep`），解析时按同一份分隔符切回编号——**编号是结构键**，所以它排在分隔符
    前面，与语言的书写顺序无关。

    **为什么判据原文也要印进 state**：信号名是压缩过的标签，不是判据本身。反解模型要照着同一份
    机械口径判，材料里就必须有那份口径（范式 §9.44，在 `memorygate` 上量过 1/8 → 8/8）。
    """
    sep = str(loc["state"]["tag_sep"])
    rule_sep = str(loc["state"]["rule_sep"])
    entry = tag_copy(loc, tag)
    return f"{tag}{sep}{entry['name']}{rule_sep}{entry['rule']}"


def source_copy(loc: dict[str, JsonValue], source: str) -> str:
    """输入：语言字典、入口。输出：该入口的形态描述。

    预期行为：缺这一档就当场报错（新加一档入口却忘了写文案时，渲染会拿不到句子）。
    """
    table = loc.get("source")
    entry = table.get(source) if isinstance(table, dict) else None
    if not isinstance(entry, dict) or "desc" not in entry:
        raise ValueError(f"语言文件里缺入口 {source} 的描述")
    return str(entry["desc"])


def threshold_line(loc: dict[str, JsonValue], level: str) -> str:
    """输入：语言字典、处置档位。输出：档位表里那一行（不含行前缀）。

    预期行为：`分数区间 + 分隔符 + 处置动作`；处置动作与 `route` 那一族的选项取的是**同一份
    文案**（`fam.route.opt`），所以 state 里那张表和题面不可能各说各的。
    """
    sep = str(loc["state"]["tag_sep"])
    action = str(loc["fam"]["route"]["opt"][level])
    return f"{loc['state']['threshold'][level]}{sep}{action}"


def build_prompt(loc: dict[str, JsonValue], case: TraceScoreCase) -> str:
    """组装提示词。

    输入：语言字典、注入特征。输出：完整提示词（不含答案）。
    预期行为：**每一条硬性要求都写清楚**——必须出现哪些形状（逐条印上判据原文）、不许出现哪些
    （逐条给反面写法）、这段轨迹从哪种入口来、长度区间、输出前自查。这些就是「受控合成」的规格，
    判据全在 `check_draft` 里机械复核，模型写歪了当场重渲染。

    为什么**一个示例片段都不给**（范式 §9.45）：只给一个具体片段当例子，模型会把它**整段照抄**
    进正文——首轮真机小批量（`lt-10a`）里 8/10 条照抄，而且「那只是例子、请自己另编」这句约束
    写进提示词也拦不住。判据原文本身就是完整规格（`rule` 把要匹配的词逐个列了出来），所以提示词
    只印判据原文、不给可抄的成品句；`pasted_rules` 再兜住「把判据原文整段贴进轨迹」这条残余捷径。

    为什么把「不许出现」也逐条列出来：只写「必须出现什么」时，模型会顺手带上别的形状（写轨迹时
    顺手写一句「重试了一次」就会多出一条命中）——那一条的标签就废了。
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


def build_task(spec: TraceScoreSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：真值规格、语言字典。输出：渲染任务。

    预期行为：任务 id 由规格算（断点续跑按它查进度），`spec` 里带上注入特征——后端实现与假后端
    若想「照着计划答」，不必再解析提示词。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="tracescore",
        prompt=build_prompt(loc, spec.case),
        response_model=ContentDraft,
        seeds=(),
        spec={
            "case": spec.case.as_dict(),
            "hits": list(spec.case.tags()),
            "source": spec.case.source,
        },
    )


def check_draft(draft: ContentDraft, case: TraceScoreCase, loc: dict[str, JsonValue]) -> None:
    """校验「执行轨迹」是否兑现了注入特征。

    输入：后端草稿、注入特征、语言字典。输出：无（不合格抛 `DraftError`，管线据此重渲染）。
    预期行为：逐条查——

    1. 轨迹非空、不带围栏、长度落在 `MIN_CONTENT_CHARS`~`MAX_CONTENT_CHARS`；
    2. **按字面算出来的命中组合与计划逐条相同**（多一条少一条都不行，理由里说清差在哪）；
    3. **不许把判据原文原样贴进轨迹**（`pasted_rules` 逐字比对）。
    第 2 条是这一族的核心：命中组合就是金标签，它由正则从文本算得出来，所以「模型写歪了」在
    渲染期就能拦住，不用等判别器。
    """
    text = draft.text.strip()
    if not text:
        raise DraftError("轨迹是空的")
    if "```" in text:
        raise DraftError(f"轨迹里带围栏：{text[:60]!r}")
    if len(text) < MIN_CONTENT_CHARS:
        raise DraftError(f"轨迹太短（至少 {MIN_CONTENT_CHARS} 个字符）：{text!r}")
    if len(text) > MAX_CONTENT_CHARS:
        raise DraftError(f"轨迹太长（最多 {MAX_CONTENT_CHARS} 个字符，实际 {len(text)}）")
    found = features.hits_of(text)
    if found != case.hits:
        raise DraftError(
            f"轨迹按字面算出来的命中组合是 {found or '无'}，计划要的是 {case.hits or '无'}"
        )
    pasted = pasted_rules(text, loc)
    if pasted:
        raise DraftError(f"轨迹把判据原文原样贴了进去：{pasted}")


def pasted_rules(text: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：轨迹正文、语言字典。输出：被**逐字贴进**轨迹的判据原文（按 `TAG_IDS` 顺序）。

    预期行为：只认**完全相同的整段**判据原文（`tag_copy(...)["rule"]`）——改一个词就不再算。

    为什么要有这条：提示词里给模型看的**唯一**一段具体文字就是判据原文（示例片段已经不再给，
    见 `build_prompt`），所以模型能「抄」的只剩它——把判据的词表整段贴进轨迹，标签照样算命中
    （正则只看子串），但那段文本读起来不是轨迹，而是判据本身。这条把这种残余捷径变成机械可查：
    贴了就重渲染。真实轨迹不会包含整段判据原文，所以这条**不会误伤**正常产物。
    """
    return [
        entry
        for entry in (str(tag_copy(loc, tag)["rule"]) for tag in features.TAG_IDS)
        if entry in text
    ]


def to_state(draft: ContentDraft, loc: dict[str, JsonValue]) -> str:
    """输入：后端草稿、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：信号表 + 档位表 + 轨迹抬头 + 轨迹。**信号表与档位表印在 state 里**，所以「这段轨迹
    命中了哪几条、该怎么处置」是自明的——考生只读 state 与题面就能答，不必知道外部规则
    （范式 §9.8）。
    """
    marks = loc["state"]
    parts = [str(marks["tag_mark"])]
    parts.extend(f"{LINE_PREFIX}{tag_line(loc, tag)}" for tag in features.TAG_IDS)
    parts.append(str(marks["rule_mark"]))
    parts.extend(f"{LINE_PREFIX}{threshold_line(loc, level)}" for level in features.ROUTE_LEVELS)
    parts.append(str(marks["content_mark"]))
    parts.append(draft.text.strip())
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
