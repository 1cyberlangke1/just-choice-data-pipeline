"""字幕对质检的文本外壳：判据表、处置档位表与格式规格由代码写，一条字幕对由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **判据表、档位表、字幕对抬头由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**一条字幕对**（真实字幕长尾得很，模板拼出来的脱离现实分布）；
- **命中哪几条判据由代码按字面算**（`features.PATTERNS` 那四条正则），模型碰不到：提示词里逐条
  写明「哪一行必须出现什么形状、别的地方不许出现」，`check_draft` 在渲染期按同一份判据复核一遍
  （不合格就重渲染），`judges.HygieneJudge` / `FrameJudge` 再各查一遍。

**判据原文也印进 state**（范式 §9.44 的教训）：反解模型要判「这条字幕对踩了哪几条」，而判据名是
**压缩过的标签**——「时间码零时长」推不出「`Timing:` 那一行里 `-->` 两侧要一模一样」。

**提示词里一个示例字幕对都不给**（范式 §9.45）：`logtriage` 上量过一次——给了成品句，20 条里
16 条整段照抄，写一句「那只是例子」也拦不住。所以这里只印**判据原文**（它本来就是完整规格）与一份
**格式规格**（键名 + 占位符，不是可抄的成品），`pasted_rules` 再兜住「把判据原文整段贴进去」。

**格式规格为什么是「键名 + 占位符」而不是示例**（范式 §9.52）：键名是 state 格式的一部分（与
`LINE_PREFIX` 同性质），模型必须知道四行的名字与冒号才写得出来；但每一行的**取值**都得它自己编
——时间码那一行更是不能给成品，给一对填好的时间码，模型照抄下去正好是「零时长」。

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
from .features import SubQaCase
from .spec import SubQaSpec

#: 判据表与档位表每行的前缀。**这是 state 格式的一部分，不是文案**，所以不进语言文件：
#: 渲染与解析共用这一个常量，两边不可能漂移。
LINE_PREFIX: Final[str] = "- "

#: 字幕对的字符数上下限。太短装不下四行，太长就不是「一屏」了。
MIN_CONTENT_CHARS: Final[int] = features.MIN_CONTENT_CHARS
MAX_CONTENT_CHARS: Final[int] = features.MAX_CONTENT_CHARS


class ContentDraft(BaseModel):
    """后端要填的草稿：那一条字幕对。

    `cue` 是**一条双语字幕对**（真实字幕长尾得很、由模型写）；结构化事实（判据表、档位表、抬头、
    格式规格）全由组装器渲染，模型碰不到。
    """

    cue: str = Field(
        description="四行字幕对（Source / Target / Timing / Note）；不要围栏、不要解释"
    )


@dataclass(frozen=True)
class ParsedContent:
    """从 state 反解出来的、纯代码看得见的部分。

    字段：`tags` 判据表里列出的判据编号；`text` 一条字幕对（认不出来给 None）。
    """

    tags: tuple[str, ...]
    text: str | None

    @property
    def found(self) -> bool:
        """输入：无。输出：字幕对那一块认出来了没有。"""
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
    （`state.tag_sep`），解析时按同一份分隔符切回编号——**编号是结构键**，所以它排在分隔符前面，
    与语言的书写顺序无关。

    **为什么判据原文也要印进 state**：判据名是压缩过的标签，不是判据本身。反解模型要照着同一份
    机械口径判，材料里就必须有那份口径（范式 §9.44）。
    """
    sep = str(loc["state"]["tag_sep"])
    rule_sep = str(loc["state"]["rule_sep"])
    entry = tag_copy(loc, tag)
    return f"{tag}{sep}{entry['name']}{rule_sep}{entry['rule']}"


def track_copy(loc: dict[str, JsonValue], track: str) -> str:
    """输入：语言字典、素材类型。输出：该素材类型的形态描述。

    预期行为：缺这一档就当场报错（新加一档素材却忘了写文案时，渲染会拿不到句子）。
    """
    table = loc.get("track")
    entry = table.get(track) if isinstance(table, dict) else None
    if not isinstance(entry, dict) or "desc" not in entry:
        raise ValueError(f"语言文件里缺素材类型 {track} 的描述")
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
    键名与冒号是 state 格式的一部分，所以它们**不进语言文件**，只有说明文字按语言取；每一行的取值
    都写成 `<...>` 占位符——规格里不许有能整段照抄的成品（范式 §9.52）。
    """
    spec = loc["prompt"]["format"]
    return [f"{LINE_PREFIX}{spec[key]}" for key in features.KEY_LINES]


def build_prompt(loc: dict[str, JsonValue], case: SubQaCase, timecode: str) -> str:
    """组装提示词。

    输入：语言字典、注入特征、开始时间码。输出：完整提示词（不含答案）。
    预期行为：**每一条硬性要求都写清楚**——必须出现哪些形状（逐条印上判据原文）、出现在哪一行、
    不许出现哪些（逐条给反面写法）、这条字幕对来自哪种素材、长度区间、四行格式规格、输出前自查。
    这些就是「受控合成」的规格，判据全在 `check_draft` 里机械复核，模型写歪了当场重渲染。

    为什么要给**样本级变量**（范式 §9.61）：只给素材类型时，提示词在同一素材内逐字相同，模型把
    自由字段迅速塌成同一个值——`Timing:` 上量到过（`sq-10c-zh` 的 9 条里 4 条同一串数字）。这里
    钉的是**开始**那一侧：结束那一侧要么按 `S3` 的判据与它一模一样、要么自己编一个更晚的，两种
    写法都留着自由度，所以这个旋钮不与任何判据打架。

    为什么**一个示例字幕对都不给**（范式 §9.45）：只给一份具体字幕当例子，模型会把它**整段照抄**
    进样本，而且「那只是例子、请自己另编」这句约束写进提示词也拦不住。判据原文本身就是完整规格，
    所以提示词只印判据原文 + 一份带占位符的格式规格。

    为什么把「不许出现」也逐条列出来：只写「必须出现什么」时，模型会顺手带上别的形状（写了
    `♪` 又顺手在备注里留个 `待确认`）——那一条的标签就废了。
    """
    prompt = loc["prompt"]
    hits = case.tags()
    missing = [tag for tag in features.TAG_IDS if tag not in hits]
    lines = [
        str(prompt["intro"]).format(min=MIN_CONTENT_CHARS, max=MAX_CONTENT_CHARS),
        str(prompt["lang_rule"]).format(
            language=loc["language_name"], source_language=loc["source_language"]
        ),
        str(prompt["track_line"]).format(track=track_copy(loc, case.track)),
        str(prompt["timing_line"]).format(timecode=timecode),
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
    lines.append(
        str(prompt["self_check"]).format(
            language=loc["language_name"], source_language=loc["source_language"]
        )
    )
    lines.append(str(prompt["output"]))
    return "\n".join(lines)


def build_task(spec: SubQaSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：真值规格、语言字典。输出：渲染任务。

    预期行为：任务 id 由规格算（断点续跑按它查进度），`spec` 里带上注入特征——后端实现与假后端
    若想「照着计划答」，不必再解析提示词。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="subqa",
        prompt=build_prompt(loc, spec.case, spec.timecode),
        response_model=ContentDraft,
        seeds=(),
        spec={
            "case": spec.case.as_dict(),
            "hits": list(spec.case.tags()),
            "track": spec.case.track,
        },
    )


def pasted_rules(text: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：一条字幕对、语言字典。输出：被**逐字贴进**样本的判据原文（按 `TAG_IDS` 顺序）。

    预期行为：只认**完全相同的整段**判据原文（`tag_copy(...)["rule"]`）——改一个词就不再算。

    为什么要有这条：提示词里给模型看的**唯一**一段具体文字就是判据原文（示例字幕一律不给，见
    `build_prompt`），所以模型能「抄」的只剩它——把判据的词表整段贴进字幕，标签照样算命中（正则
    只看子串），但那条字幕读起来不是字幕，而是判据本身。这条把这种残余捷径变成机械可查：贴了就
    重渲染。真实字幕不会包含整段判据原文，所以这条**不会误伤**正常产物。
    """
    return [
        entry
        for entry in (str(tag_copy(loc, tag)["rule"]) for tag in features.TAG_IDS)
        if entry in text
    ]


def check_draft(draft: ContentDraft, case: SubQaCase, loc: dict[str, JsonValue]) -> None:
    """渲染期校验：草稿必须**兑现注入特征**。

    输入：草稿、注入特征、语言字典。输出：无（不合格抛 `DraftError`，管线据此重渲染）。
    预期行为：逐条查——

    1. **非空、不带围栏、长度落在区间内、四个结构键都在**（少一行就定位不到判据）；
    2. **按字面算出来的命中组合与计划逐条相同**（多一条少一条都报错，并把两边都写进理由）；
    3. **不许把判据原文原样贴进字幕**（`pasted_rules` 逐字比对）。
    第 2 条是这一族的核心：命中组合就是金标签，它由正则从文本算得出来，所以「模型写歪了」在
    渲染期就能拦住，不用等判别器。
    """
    text = draft.cue.strip()
    if not text:
        raise DraftError("字幕对是空的")
    if "```" in text:
        raise DraftError(f"字幕对里带围栏：{text[:60]!r}")
    if len(text) < MIN_CONTENT_CHARS:
        raise DraftError(f"字幕对太短（至少 {MIN_CONTENT_CHARS} 个字符）：{text!r}")
    if len(text) > MAX_CONTENT_CHARS:
        raise DraftError(f"字幕对太长（最多 {MAX_CONTENT_CHARS} 个字符，实际 {len(text)}）")
    missing = features.missing_keys(text)
    if missing:
        raise DraftError(f"字幕对缺这几行（每个键要单独占一行、后面跟冒号）：{list(missing)}")
    found = features.hits_of(text)
    if found != case.hits:
        raise DraftError(
            f"字幕对按字面算出来的命中组合是 {found or '无'}，计划要的是 {case.hits or '无'}"
        )
    pasted = pasted_rules(text, loc)
    if pasted:
        raise DraftError(f"字幕对把判据原文原样贴了进去：{pasted}")


def to_state(draft: ContentDraft, loc: dict[str, JsonValue]) -> str:
    """输入：后端草稿、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：判据表 + 档位表 + 字幕对抬头 + 一条字幕对。**判据表与档位表印在 state 里**，所以
    「这条字幕对踩了哪几条、该走哪一步」是自明的——考生只读 state 与题面就能答，不必知道外部规则
    （范式 §9.8）。
    """
    marks = loc["state"]
    parts = [str(marks["tag_mark"])]
    parts.extend(f"{LINE_PREFIX}{tag_line(loc, tag)}" for tag in features.TAG_IDS)
    parts.append(str(marks["rule_mark"]))
    parts.extend(f"{LINE_PREFIX}{threshold_line(loc, level)}" for level in features.ACTS)
    parts.append(str(marks["content_mark"]))
    parts.append(draft.cue.strip())
    return "\n".join(parts)


def parse_with(state: str, loc: dict[str, JsonValue]) -> ParsedContent:
    """输入：state、语言字典。输出：按这一份语言反解的结果。

    预期行为：认判据表抬头、档位表抬头、字幕对抬头；判据表里带行前缀的那些行取编号（按
    `state.tag_sep` 切），**字幕对抬头之后的所有行原样拼回正文**——抬头之外的内容一律不当结构，
    否则字幕里出现一行以 `- ` 开头的文本就会把判据表数歪。
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
    """输入：state。输出：能认出字幕对抬头的那一种语言（认不出给 None）。

    预期行为：审计拿它按 state **自己**的语言取规范文案——一次 run 里混了 zh / en 时，用调用方
    给的那一种去比会平白报一堆「文案对不上」。

    为什么认的是**字幕对抬头**而不是判据编号：编号（S1…S4）是结构键，与语言无关，拿它当判据会把
    别的语言的 state 也抢过来（范式 §9.22）；字幕对抬头是渲染时按语言写死的、整段最后那一块的开头，
    才分得开。
    """
    if not isinstance(state, str):
        return None
    for lang in locales.available():
        if parse_with(state, locales.load(lang)).found:
            return lang
    return None


def parse_state(state: object) -> ParsedContent:
    """输入：state。输出：纯代码能反解的部分（判据表 / 一条字幕对）。

    预期行为：与语言无关——把各语言的标记都试一遍，认出字幕对抬头的那一份就用它；一份都认不出来
    时返回空结果（`text` 为 None）。
    """
    if not isinstance(state, str):
        return ParsedContent(tags=(), text=None)
    lang = detect_locale(state)
    if lang is None:
        return ParsedContent(tags=(), text=None)
    return parse_with(state, locales.load(lang))


def target_text(cue: str) -> str:
    """输入：一条字幕对。输出：`Target:` 那一行的译文（没有就给空串）。

    预期行为：这是这一族里**两段自然语言正文**之一（另一段是备注），`Workflow.language_texts`
    会把它报给语言尺子（范式 §9.50：只喂认得出语言的字段，时间码与键名那些 ASCII 一律不算）。
    """
    return features.line_value(cue, "target")


def note_text(cue: str) -> str:
    """输入：一条字幕对。输出：`Note:` 那一行的备注（没有就给空串）。"""
    return features.line_value(cue, "note")
