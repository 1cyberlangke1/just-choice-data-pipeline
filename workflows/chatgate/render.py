"""聊天回复门控的文本外壳：判据表、毛病表、档位表由代码写，一条消息与三条候选由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **判据表、毛病表、档位表由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**一条消息 + 三条候选回复**（聊天内容长尾得厉害，模板拼出来的脱离现实分布）；
- **踩中哪几条判据、哪条候选干净，都由代码按字面算**（`features.PATTERNS` / `FLAW_PATTERNS`
  那七条正则），模型碰不到：提示词里逐条写明「必须出现哪些形状、不许出现哪些、三条候选的毛病
  怎么分」，`check_draft` 在渲染期按同一份判据复核一遍（不合格就重渲染），
  `judges.HygieneJudge` / `FrameJudge` 再各查一遍，兜住不是本 workflow 渲染出来的 state。

**判据原文与毛病原文也印进 state**（范式 §9.44 的教训）：反解模型要判「这条消息踩了哪几条、
哪条候选干净」，而判据名是**压缩过的标签**——「把决定推给我」推不出「出现『你定 / 行不行』
这类词」。`memorygate` 上量过一次：只印判据名时反解 1/8，补上原文 8/8。

**场景（`scene`）为什么要写进提示词**：同一份命中组合可以长在同事私聊、工作群或陌生人私信上，
而这三处的**口吻与分寸差别很大**（同事之间会直接开口要东西，群里要顾着别人也在看，陌生人私信
天然更可疑）。不点明来路，模型写出来的东西会四不像；点明之后，这一维就成了一条**只影响渲染、
不影响标签**的留出维（`test_ood` 侧是训练侧一次都没见过的 `stranger`）。

**一条示例片段都不给**（范式 §9.45）：只给一个具体片段当例子，模型会把它**整段照抄**进产物
（`lt-10a` 上 8/10 条照抄，`shell-hint-10a` 上路径例子被整批抄走）。判据原文本身就是完整规格
（`rule` 把要匹配的词逐个列了出来），所以提示词只印判据原文、不给可抄的成品句。

提示词里剩下的具体文字有两类，能不能被整段抄进去是**算得出来的**：判据 / 毛病原文（`rule`）
比消息与候选的长度上限都长，整段贴不进去（`test_render` 把这条不变量钉住）；反面清单
（`avoid`）短得多，是唯一真能抄进去的东西，`pasted_lists` 就盯它。

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
from .features import ChatCase, ChatPlan
from .spec import ChatGateSpec

#: 判据表、毛病表、档位表、候选回复每行的前缀。**这是 state 格式的一部分，不是文案**，
#: 所以不进语言文件：渲染与解析共用这一个常量，两边不可能漂移。
LINE_PREFIX: Final[str] = "- "

#: 候选回复的「序号 + 正文」之间的分隔符。同样是结构（不是文案）：解析靠它把序号摘掉，
#: 而 `partition` 只切**第一处**，所以候选正文里再出现一次也不会切歪。
REPLY_SEP: Final[str] = ". "

#: 消息与候选的长度上下限（渲染期与提示词共用同一组数）。
MIN_MESSAGE_CHARS: Final[int] = features.MIN_MESSAGE_CHARS
MAX_MESSAGE_CHARS: Final[int] = features.MAX_MESSAGE_CHARS
MIN_CANDIDATE_CHARS: Final[int] = features.MIN_CANDIDATE_CHARS
MAX_CANDIDATE_CHARS: Final[int] = features.MAX_CANDIDATE_CHARS


class MessageDraft(BaseModel):
    """后端要填的草稿：一条消息 + 三条候选回复。

    `text` 与 `candidates` 是**自然语言正文**（聊天内容长尾得厉害、由模型写）；结构化事实
    （判据表、毛病表、档位表）全由组装器渲染，模型碰不到。
    """

    text: str = Field(description="对方发来的那条消息：一行读得通的聊天消息，不要标题、不要围栏")
    candidates: list[str] = Field(
        description="三条候选回复，按第 1、2、3 条的顺序各给一个字符串；每条都要读得通"
    )


@dataclass(frozen=True)
class ParsedState:
    """从 state 反解出来的、纯代码看得见的部分。

    字段：`tags` 判据表里列出的判据编号；`flaws` 毛病表里列出的毛病编号；`message` 那条消息
    （认不出来给 None）；`replies` 三条候选回复的正文（按原顺序）。
    """

    tags: tuple[str, ...]
    flaws: tuple[str, ...]
    message: str | None
    replies: tuple[str, ...]

    @property
    def found(self) -> bool:
        """输入：无。输出：消息那一块认出来了没有。"""
        return self.message is not None


def _copy(loc: dict[str, JsonValue], table: str, key: str, fields: tuple[str, ...]) -> dict[str, JsonValue]:
    """输入：语言字典、表名、编号、该表要求的字段。输出：那一条文案。

    预期行为：缺表、缺这一条、或缺字段都当场报错——判据表与提示词都靠它，缺一个字段模型就会
    拿到半句话。
    """
    block = loc.get(table)
    entry = block.get(key) if isinstance(block, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"语言文件里缺 {table} 的 {key} 文案")
    for field in fields:
        if field not in entry:
            raise ValueError(f"{table} 的 {key} 缺字段 {field}")
    return entry


def tag_copy(loc: dict[str, JsonValue], tag: str) -> dict[str, JsonValue]:
    """输入：语言字典、判据编号。输出：该判据的文案（name / rule / avoid）。"""
    return _copy(loc, "tag", tag, ("name", "rule", "avoid"))


def flaw_copy(loc: dict[str, JsonValue], flaw: str) -> dict[str, JsonValue]:
    """输入：语言字典、毛病编号。输出：该毛病的文案（name / rule / avoid）。"""
    return _copy(loc, "flaw", flaw, ("name", "rule", "avoid"))


def scene_copy(loc: dict[str, JsonValue], scene: str) -> str:
    """输入：语言字典、场景。输出：该场景的来路描述。

    预期行为：缺这一档就当场报错（新加一档场景却忘了写文案时，渲染会拿不到句子）。
    """
    return str(_copy(loc, "scene", scene, ("desc",))["desc"])


def tag_line(loc: dict[str, JsonValue], tag: str) -> str:
    """输入：语言字典、判据编号。输出：判据表里那一行（不含行前缀）。

    预期行为：`编号 + 分隔符 + 判据名 + 规则分隔符 + 判据原文`。分隔符按语言取
    （`state.tag_sep` / `state.rule_sep`），解析时按同一份分隔符切回编号——**编号是结构键**，
    所以它排在分隔符前面，与语言的书写顺序无关。
    """
    entry = tag_copy(loc, tag)
    sep = str(loc["state"]["tag_sep"])
    rule_sep = str(loc["state"]["rule_sep"])
    return f"{tag}{sep}{entry['name']}{rule_sep}{entry['rule']}"


def flaw_line(loc: dict[str, JsonValue], flaw: str) -> str:
    """输入：语言字典、毛病编号。输出：毛病表里那一行（不含行前缀）。形状与 `tag_line` 相同。"""
    entry = flaw_copy(loc, flaw)
    sep = str(loc["state"]["tag_sep"])
    rule_sep = str(loc["state"]["rule_sep"])
    return f"{flaw}{sep}{entry['name']}{rule_sep}{entry['rule']}"


def threshold_line(loc: dict[str, JsonValue], level: str) -> str:
    """输入：语言字典、处置档位。输出：档位表里那一行（不含行前缀）。

    预期行为：`阈值描述 + 分隔符 + 处置动作`；处置动作与 `route` 那一族的选项取的是**同一份
    文案**（`fam.route.opt`），所以 state 里那张表和题面不可能各说各的。
    """
    sep = str(loc["state"]["tag_sep"])
    action = str(loc["fam"]["route"]["opt"][level])
    return f"{loc['state']['threshold'][level]}{sep}{action}"


def build_prompt(loc: dict[str, JsonValue], case: ChatCase, plan: ChatPlan) -> str:
    """组装提示词。

    输入：语言字典、注入特征、完整计划。输出：完整提示词（不含答案）。
    预期行为：**每一条硬性要求都写清楚**——消息必须出现哪些形状（逐条印上判据原文）、不许出现
    哪些（逐条给反面写法）、这条消息从哪种场景来、三条候选的毛病怎么分、长度区间、输出前自查。
    这些就是「受控合成」的规格，判据全在 `check_draft` 里机械复核，模型写歪了当场重渲染。

    为什么**一个示例片段都不给**（范式 §9.45）：给一个具体片段，模型会把它**整段照抄**；
    判据原文本身就是完整规格，所以提示词只印判据原文、不给可抄的成品句。

    为什么把「不许出现」也逐条列出来：只写「必须出现什么」时，模型会顺手带上别的形状——那一条
    的标签就废了。同理，干净的那条候选要把三条毛病的反面写法一起列出来，否则模型会顺手带上一个。
    """
    prompt = loc["prompt"]
    hits = plan.hits
    missing = [tag for tag in features.TAG_IDS if tag not in hits]
    lines = [
        str(prompt["intro"]).format(
            min_msg=MIN_MESSAGE_CHARS,
            max_msg=MAX_MESSAGE_CHARS,
            min_cand=MIN_CANDIDATE_CHARS,
            max_cand=MAX_CANDIDATE_CHARS,
        ),
        str(prompt["lang_rule"]).format(language=loc["language_name"]),
        str(prompt["scene_line"]).format(scene=scene_copy(loc, case.scene)),
    ]
    if hits:
        lines.append(str(prompt["hits_header"]))
        for tag in hits:
            entry = tag_copy(loc, tag)
            lines.append(str(prompt["hits_line"]).format(name=entry["name"], rule=entry["rule"]))
    # 「不许照抄」这一句**任何一档都要印**：命中为空时，反面清单同样是具体文字，模型照样能抄。
    lines.append(str(prompt["vary_rule"]))
    if missing:
        lines.append(str(prompt["miss_header"]) if hits else str(prompt["no_hits"]))
        for tag in missing:
            entry = tag_copy(loc, tag)
            lines.append(
                str(prompt["miss_line"]).format(
                    name=entry["name"], rule=entry["rule"], avoid=entry["avoid"]
                )
            )
    lines.append(str(prompt["reply_header"]))
    for index in features.CANDIDATE_SLOTS:
        flaws = plan.reply.flaws[index]
        if flaws:
            for flaw in flaws:
                entry = flaw_copy(loc, flaw)
                lines.append(
                    str(prompt["reply_line"]).format(
                        index=index + 1, name=entry["name"], rule=entry["rule"]
                    )
                )
        else:
            # 分隔符按语言取（中文用全角、英文用半角加空格）：写死会在英文提示词里漏出全角分号。
            avoid = str(prompt["avoid_sep"]).join(
                str(flaw_copy(loc, item)["avoid"]) for item in features.FLAW_IDS
            )
            lines.append(str(prompt["clean_line"]).format(index=index + 1, avoid=avoid))
    lines.append(
        str(prompt["length_rule"]).format(
            min_msg=MIN_MESSAGE_CHARS,
            max_msg=MAX_MESSAGE_CHARS,
            min_cand=MIN_CANDIDATE_CHARS,
            max_cand=MAX_CANDIDATE_CHARS,
        )
    )
    lines.append(str(prompt["self_check"]))
    lines.append(str(prompt["output"]))
    return "\n".join(lines)


def build_task(spec: ChatGateSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：真值规格、语言字典。输出：渲染任务。

    预期行为：任务 id 由规格算（断点续跑按它查进度），`spec` 里带上注入特征与候选毛病计划——
    后端实现与假后端若想「照着计划答」，不必再解析提示词。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="chatgate",
        prompt=build_prompt(loc, spec.case, spec.plan),
        response_model=MessageDraft,
        seeds=(),
        spec={
            "case": spec.case.as_dict(),
            "hits": list(spec.plan.hits),
            "clean": spec.plan.reply.clean_answer,
            "flaws": [list(flaws) for flaws in spec.plan.reply.flaws],
        },
    )


def pasted_rules(text: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：一段正文、语言字典。输出：被**逐字贴进**正文的判据 / 毛病原文。

    预期行为：只认**完全相同的整段**原文（`tag_copy(...)["rule"]` / `flaw_copy(...)["rule"]`）
    ——改一个词就不再算。

    为什么要有这条：提示词里给模型看的**唯一**一段具体文字就是判据与毛病的原文（示例片段一律
    不给），所以模型能「抄」的只剩它——整段贴进去，标签照样算命中（正则只看子串），但那条消息
    读起来不是消息，而是判据本身。真实聊天里不会出现整段判据原文，所以这条**不会误伤**正常产物。
    """
    rules = [str(tag_copy(loc, tag)["rule"]) for tag in features.TAG_IDS]
    rules += [str(flaw_copy(loc, flaw)["rule"]) for flaw in features.FLAW_IDS]
    return [rule for rule in rules if rule in text]


def to_state(draft: MessageDraft, loc: dict[str, JsonValue]) -> str:
    """输入：后端草稿、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：判据表 + 毛病表 + 档位表 + 消息 + 三条候选。**三张表都印在 state 里**，所以
    「这条消息踩了哪几条、哪条候选干净、该走哪一档」是自明的——考生只读 state 与题面就能答，
    不必知道外部规则（范式 §9.8）。候选按第 1、2、3 条编号印出来，`reply` 那一族的选项就指这个。
    """
    marks = loc["state"]
    parts = [str(marks["tag_mark"])]
    parts.extend(f"{LINE_PREFIX}{tag_line(loc, tag)}" for tag in features.TAG_IDS)
    parts.append(str(marks["flaw_mark"]))
    parts.extend(f"{LINE_PREFIX}{flaw_line(loc, flaw)}" for flaw in features.FLAW_IDS)
    parts.append(str(marks["rule_mark"]))
    parts.extend(
        f"{LINE_PREFIX}{threshold_line(loc, level)}" for level in features.HANDLING_LEVELS
    )
    parts.append(str(marks["content_mark"]))
    parts.append(draft.text.strip())
    parts.append(str(marks["reply_mark"]))
    parts.extend(
        f"{LINE_PREFIX}{index + 1}{REPLY_SEP}{candidate.strip()}"
        for index, candidate in enumerate(draft.candidates)
    )
    return "\n".join(parts)


def parse_with(state: str, loc: dict[str, JsonValue]) -> ParsedState:
    """输入：state、语言字典。输出：按这一份语言反解的结果。

    预期行为：认五个抬头（判据表 / 毛病表 / 档位表 / 消息 / 候选回复）。三张表里带行前缀的那些
    行按 `state.tag_sep` 切出编号；消息那一段的**所有行原样拼回消息**；候选那一段的每个带前缀的
    行切掉前缀与序号之后就是一条候选。抬头之外的内容一律不当结构。
    """
    marks = loc["state"]
    tag_mark = str(marks["tag_mark"])
    flaw_mark = str(marks["flaw_mark"])
    rule_mark = str(marks["rule_mark"])
    content_mark = str(marks["content_mark"])
    reply_mark = str(marks["reply_mark"])
    sep = str(marks["tag_sep"])
    tags: list[str] = []
    flaws: list[str] = []
    message: list[str] = []
    replies: list[str] = []
    section = ""
    seen_message = False
    for raw in state.splitlines():
        text = raw.strip()
        if text.startswith(tag_mark):
            section = "tag"
            continue
        if text.startswith(flaw_mark):
            section = "flaw"
            continue
        if text.startswith(rule_mark):
            section = "rule"
            continue
        if text.startswith(content_mark):
            section = "content"
            seen_message = True
            message.append(text[len(content_mark) :].strip())
            continue
        if text.startswith(reply_mark):
            section = "reply"
            continue
        if section == "tag" and text.startswith(LINE_PREFIX):
            tags.append(text[len(LINE_PREFIX) :].partition(sep)[0].strip())
        elif section == "flaw" and text.startswith(LINE_PREFIX):
            flaws.append(text[len(LINE_PREFIX) :].partition(sep)[0].strip())
        elif section == "content":
            message.append(raw)
        elif section == "reply" and text.startswith(LINE_PREFIX):
            body = text[len(LINE_PREFIX) :]
            replies.append(body.partition(REPLY_SEP)[2].strip() if REPLY_SEP in body else body.strip())
    return ParsedState(
        tags=tuple(tags),
        flaws=tuple(flaws),
        message="\n".join(message).strip() if seen_message else None,
        replies=tuple(replies),
    )


def detect_locale(state: object) -> str | None:
    """输入：state。输出：能认出消息抬头的那一种语言（认不出给 None）。

    预期行为：审计拿它按 state **自己**的语言取规范文案——一次 run 里混了 zh / en 时，用调用方
    给的那一种去比会平白报一堆「文案对不上」。

    为什么认的是**消息抬头**而不是判据编号：编号（C1…C4）是结构键，与语言无关，拿它当判据会把
    别的语言的 state 也抢过来（这条坑在 lint 上踩过一次，见范式 §9.22）；消息抬头是渲染时按
    语言写死的、整段最后那几块的开头，才分得开。
    """
    return recognize_locale(state, locales, parse_with)


def parse_state(state: object) -> ParsedState:
    """输入：state。输出：纯代码能反解的部分（三张表 / 消息 / 候选）。

    预期行为：与语言无关——把各语言的标记都试一遍，认出消息抬头的那一份就用它；一份都认不出来
    时返回空结果（`message` 为 None）。
    """
    if not isinstance(state, str):
        return ParsedState(tags=(), flaws=(), message=None, replies=())
    lang = detect_locale(state)
    if lang is None:
        return ParsedState(tags=(), flaws=(), message=None, replies=())
    return parse_with(state, locales.load(lang))


def check_draft(draft: MessageDraft, plan: ChatPlan, loc: dict[str, JsonValue]) -> None:
    """校验「一条消息 + 三条候选」是否兑现了计划。

    输入：后端草稿、完整计划、语言字典。输出：无（不合格抛 `DraftError`，管线据此重渲染）。
    预期行为：逐条查——

    1. 消息非空、不带围栏；恰好 `len(CANDIDATE_SLOTS)` 条候选，每条非空、不带围栏；
    2. **不许把判据 / 毛病原文原样贴进消息或候选**（`pasted_rules` 逐字比对）。这一条排在
       长度之前量：判据与毛病的原文都比长度上限长（实测 132~306 个字符，上限 120 / 90），
       先量长度的话理由会变成「太长」，把真正的毛病盖掉——这条闸门也就成了永远撞不到的死代码；
    3. 消息长度落在 `MIN_MESSAGE_CHARS`~`MAX_MESSAGE_CHARS`，候选长度落在候选区间；
    4. **消息按字面算出来的命中组合与计划逐字相同**（多一条少一条都不行，理由里说清差在哪）；
    5. **三条候选各自的毛病与计划逐条相同**（含「该发的那条一个毛病都不踩」——它由计划的
       `ReplyPlan` 保证，这里只是把它落到文本上验一遍）。
    第 4、5 条是这一族的核心：命中组合与候选毛病就是金标签，它们由正则从文本算得出来，所以
    「模型写歪了」在渲染期就能拦住，不用等判别器。
    """
    text = draft.text.strip()
    if not text:
        raise DraftError("消息是空的")
    if "```" in text:
        raise DraftError(f"消息里带围栏：{text[:60]!r}")
    if len(draft.candidates) != len(features.CANDIDATE_SLOTS):
        raise DraftError(
            f"候选回复要有 {len(features.CANDIDATE_SLOTS)} 条，实际 {len(draft.candidates)} 条"
        )
    bodies = [text]
    for index, candidate in enumerate(draft.candidates):
        body = candidate.strip()
        if not body:
            raise DraftError(f"第 {index + 1} 条候选是空的")
        if "```" in body:
            raise DraftError(f"第 {index + 1} 条候选里带围栏：{body[:60]!r}")
        bodies.append(body)
    pasted = sorted({rule for body in bodies for rule in pasted_rules(body, loc)})
    if pasted:
        raise DraftError(f"消息或候选把判据 / 毛病原文原样贴了进去：{pasted}")
    if len(text) < MIN_MESSAGE_CHARS:
        raise DraftError(f"消息太短（至少 {MIN_MESSAGE_CHARS} 个字符）：{text!r}")
    if len(text) > MAX_MESSAGE_CHARS:
        raise DraftError(f"消息太长（最多 {MAX_MESSAGE_CHARS} 个字符，实际 {len(text)}）")
    for index, body in enumerate(bodies[1:]):
        if len(body) < MIN_CANDIDATE_CHARS:
            raise DraftError(
                f"第 {index + 1} 条候选太短（至少 {MIN_CANDIDATE_CHARS} 个字符）：{body!r}"
            )
        if len(body) > MAX_CANDIDATE_CHARS:
            raise DraftError(
                f"第 {index + 1} 条候选太长（最多 {MAX_CANDIDATE_CHARS} 个字符，实际 {len(body)}）"
            )
    found = features.hits_of(text)
    wanted = "+".join(plan.hits)
    if found != wanted:
        raise DraftError(
            f"消息按字面算出来的命中组合是 {found or '无'}，计划要的是 {wanted or '无'}"
        )
    for index, body in enumerate(bodies[1:]):
        got = features.flaw_set(body)
        want = plan.reply.flaws[index]
        if got != want:
            raise DraftError(
                f"第 {index + 1} 条候选的毛病是 {list(got) or '无'}，"
                f"计划要的是 {list(want) or '无'}"
            )
