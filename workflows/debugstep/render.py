"""调试器下一步动作的文本外壳：判据表、处置档位表与格式规格由代码写，一屏会话由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **判据表、档位表、会话抬头由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**一屏调试会话**（真实调试里每一屏长尾得很，模板拼出来的脱离现实分布）；
- **命中哪几条判据由代码按字面算**（`features.PATTERNS` 那四条正则），模型碰不到：提示词里逐条
  写明「哪一行必须出现什么形状、别的地方不许出现」，`check_draft` 在渲染期按同一份判据复核一遍
  （不合格就重渲染），`judges.HygieneJudge` / `FrameJudge` 再各查一遍。

**判据原文也印进 state**（范式 §9.44 的教训）：反解模型要判「这屏会话踩了哪几条」，而判据名是
**压缩过的标签**——「空值变量」推不出「Locals 那一行里要有 `= None` 这种写法」。

**提示词里一个示例会话都不给**（范式 §9.45）：`logtriage` 上量过一次——给了成品句，20 条里 16 条
整段照抄，写一句「那只是例子」也拦不住。所以这里只印**判据原文**（它本来就是完整规格）与一份
**格式规格**（键名 + 占位符，不是可抄的成品），`pasted_rules` 再兜住「把判据原文整段贴进去」。

**格式规格为什么是「键名 + 占位符」而不是示例**：键名是 state 格式的一部分（与 `LINE_PREFIX`
同性质），模型必须知道五行的名字与冒号才写得出来；但每一行的**取值**都得它自己编，所以规格里
只有 `<...>` 占位符，没有一个能整段照抄的成品。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.locale_files import copy_entry, entry_line
from jc.state_parse import ParsedState as ParsedContent, parse_any, parse_sections, recognize_locale
from jc.workflow import DraftError

from . import features, locales
from .features import DebugStepCase
from .spec import DebugStepSpec

#: 判据表与档位表每行的前缀。**这是 state 格式的一部分，不是文案**，所以不进语言文件：
#: 渲染与解析共用这一个常量，两边不可能漂移。
LINE_PREFIX: Final[str] = "- "

#: 片段的字符数上下限。太短装不下五行（四个键 + 说明），太长就成了一整份调试日志。
MIN_CONTENT_CHARS: Final[int] = features.MIN_CONTENT_CHARS
MAX_CONTENT_CHARS: Final[int] = features.MAX_CONTENT_CHARS


class ContentDraft(BaseModel):
    """后端要填的草稿：那一屏调试会话。

    `session` 是**一屏调试会话**（真实会话长尾得很、由模型写）；结构化事实（判据表、档位表、
    抬头、格式规格）全由组装器渲染，模型碰不到。
    """

    session: str = Field(
        description="五行调试会话（Stopped / Stack / Locals / Breakpoints / Note）；不要围栏、不要解释"
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


def runtime_copy(loc: dict[str, JsonValue], runtime: str) -> str:
    """输入：语言字典、运行时。输出：该运行时的形态描述。

    预期行为：缺这一档就当场报错（新加一档运行时却忘了写文案时，渲染会拿不到句子）。
    """
    table = loc.get("runtime")
    entry = table.get(runtime) if isinstance(table, dict) else None
    if not isinstance(entry, dict) or "desc" not in entry:
        raise ValueError(f"语言文件里缺运行时 {runtime} 的描述")
    return str(entry["desc"])


def threshold_line(loc: dict[str, JsonValue], level: str) -> str:
    """输入：语言字典、处置档位。输出：档位表里那一行（不含行前缀）。

    预期行为：`阈值描述 + 分隔符 + 处置动作`；处置动作与 `act` 那一族的选项取的是**同一份文案**
    （`fam.act.opt`），所以 state 里那张表和题面不可能各说各的。
    """
    sep = str(loc["state"]["tag_sep"])
    action = str(loc["fam"]["act"]["opt"][level])
    return f"{loc['state']['threshold'][level]}{sep}{action}"


def format_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：格式规格那几行（键名 + 占位符）。

    预期行为：每个键一行，行首带该键自己的说明（`prompt.format` 里按 `features.KEY_LINES` 取）。
    键名与冒号是 state 格式的一部分，所以它们**不进语言文件**，只有说明文字按语言取。
    """
    spec = loc["prompt"]["format"]
    return [f"{LINE_PREFIX}{spec[key]}" for key in features.KEY_LINES]


def build_prompt(loc: dict[str, JsonValue], case: DebugStepCase) -> str:
    """组装提示词。

    输入：语言字典、注入特征。输出：完整提示词（不含答案）。
    预期行为：**每一条硬性要求都写清楚**——必须出现哪些形状（逐条印上判据原文）、出现在哪一行、
    不许出现哪些（逐条给反面写法）、这个会话来自哪种运行时、长度区间、五行格式规格、输出前自查。
    这些就是「受控合成」的规格，判据全在 `check_draft` 里机械复核，模型写歪了当场重渲染。

    为什么**一个示例会话都不给**（范式 §9.45）：只给一份具体会话当例子，模型会把它**整段照抄**进
    样本，而且「那只是例子、请自己另编」这句约束写进提示词也拦不住。判据原文本身就是完整规格，
    所以提示词只印判据原文 + 一份带占位符的格式规格。

    为什么把「不许出现」也逐条列出来：只写「必须出现什么」时，模型会顺手带上别的形状（写了
    `Stopped: exception ...` 又顺手在 `Locals:` 行里给个 `= None`）——那一条的标签就废了。
    """
    prompt = loc["prompt"]
    hits = case.tags()
    missing = [tag for tag in features.TAG_IDS if tag not in hits]
    lines = [
        str(prompt["intro"]).format(min=MIN_CONTENT_CHARS, max=MAX_CONTENT_CHARS),
        str(prompt["lang_rule"]).format(language=loc["language_name"]),
        str(prompt["runtime_line"]).format(runtime=runtime_copy(loc, case.runtime)),
        str(prompt["format_header"]),
        *format_lines(loc),
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


def build_task(spec: DebugStepSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：真值规格、语言字典。输出：渲染任务。

    预期行为：任务 id 由规格算（断点续跑按它查进度），`spec` 里带上注入特征——后端实现与假后端
    若想「照着计划答」，不必再解析提示词。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="debugstep",
        prompt=build_prompt(loc, spec.case),
        response_model=ContentDraft,
        seeds=(),
        spec={
            "case": spec.case.as_dict(),
            "hits": list(spec.case.tags()),
            "runtime": spec.case.runtime,
        },
    )


def pasted_rules(text: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：一屏会话、语言字典。输出：被**逐字贴进**样本的判据原文（按 `TAG_IDS` 顺序）。

    预期行为：只认**完全相同的整段**判据原文（`tag_copy(...)["rule"]`）——改一个词就不再算。

    为什么要有这条：提示词里给模型看的**唯一**一段具体文字就是判据原文（示例会话一律不给，见
    `build_prompt`），所以模型能「抄」的只剩它——把判据的词表整段贴进会话，标签照样算命中（正则
    只看子串），但那段会话读起来不是调试器输出，而是判据本身。这条把这种残余捷径变成机械可查：
    贴了就重渲染。真实会话不会包含整段判据原文，所以这条**不会误伤**正常产物。
    """
    return [
        entry
        for entry in (str(tag_copy(loc, tag)["rule"]) for tag in features.TAG_IDS)
        if entry in text
    ]


def check_draft(draft: ContentDraft, case: DebugStepCase, loc: dict[str, JsonValue]) -> None:
    """渲染期校验：草稿必须**兑现注入特征**。

    输入：草稿、注入特征、语言字典。输出：无（不合格抛 `DraftError`，管线据此重渲染）。
    预期行为：逐条查——

    1. **非空、不带围栏、长度落在区间内、五个结构键都在**（少一行就定位不到判据）；
    2. **按字面算出来的命中组合与计划逐条相同**（多一条少一条都报错，并把两边都写进理由）；
    3. **不许把判据原文原样贴进会话**（`pasted_rules` 逐字比对）。
    第 2 条是这一族的核心：命中组合就是金标签，它由正则从文本算得出来，所以「模型写歪了」在
    渲染期就能拦住，不用等判别器。
    """
    text = draft.session.strip()
    if not text:
        raise DraftError("会话是空的")
    if "```" in text:
        raise DraftError(f"会话里带围栏：{text[:60]!r}")
    if len(text) < MIN_CONTENT_CHARS:
        raise DraftError(f"会话太短（至少 {MIN_CONTENT_CHARS} 个字符）：{text!r}")
    if len(text) > MAX_CONTENT_CHARS:
        raise DraftError(f"会话太长（最多 {MAX_CONTENT_CHARS} 个字符，实际 {len(text)}）")
    missing = features.missing_keys(text)
    if missing:
        raise DraftError(f"会话缺这几行（每个键要单独占一行、后面跟冒号）：{list(missing)}")
    found = features.hits_of(text)
    if found != case.hits:
        raise DraftError(
            f"会话按字面算出来的命中组合是 {found or '无'}，计划要的是 {case.hits or '无'}"
        )
    pasted = pasted_rules(text, loc)
    if pasted:
        raise DraftError(f"会话把判据原文原样贴了进去：{pasted}")


def to_state(draft: ContentDraft, loc: dict[str, JsonValue]) -> str:
    """输入：后端草稿、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：判据表 + 档位表 + 会话抬头 + 一屏会话。**判据表与档位表印在 state 里**，所以
    「这屏会话踩了哪几条、该走哪一步」是自明的——考生只读 state 与题面就能答，不必知道外部规则
    （范式 §9.8）。
    """
    marks = loc["state"]
    parts = [str(marks["tag_mark"])]
    parts.extend(f"{LINE_PREFIX}{tag_line(loc, tag)}" for tag in features.TAG_IDS)
    parts.append(str(marks["rule_mark"]))
    parts.extend(f"{LINE_PREFIX}{threshold_line(loc, level)}" for level in features.ACTS)
    parts.append(str(marks["content_mark"]))
    parts.append(draft.session.strip())
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


def note_text(session: str) -> str:
    """输入：一屏会话。输出：`Note:` 那一行的说明文字（没有就给空串）。

    预期行为：这是这一族里**唯一的自然语言正文**——`Workflow.language_texts` 报的就是它（范式
    §9.50：语言尺子只喂认得出语言的字段，键名、文件路径、变量名那些 ASCII 一律不算）。
    """
    return features.line_value(session, "note")

