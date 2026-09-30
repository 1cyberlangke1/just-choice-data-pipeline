"""数据库变更评审的文本外壳：判据表、处置档位表与格式规格由代码写，一屏记录由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **判据表、档位表、记录抬头由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**一屏评审记录**（真实的库名、表名、语句、执行计划长尾得很，模板拼出来的脱离现实
  分布）；
- **命中哪几条判据由代码按字面算**（`features.PATTERNS` 那四条正则），模型碰不到：提示词里逐条
  写明「哪一行必须出现什么形状、别的地方不许出现」，`check_draft` 在渲染期按同一份判据复核一遍
  （不合格就重渲染），`judges.HygieneJudge` / `FrameJudge` 再各查一遍。

**判据原文也印进 state**（范式 §9.44 的教训）：反解模型要判「这屏记录踩了哪几条」，而判据名是
**压缩过的标签**——「大表全表扫描」推不出「`plan:` 那一行要同时有 `Seq Scan` 与七位的 `rows=`」。

**提示词里一个示例记录都不给**（范式 §9.45）：`logtriage` 上量过一次——给了成品句，20 条里
16 条整段照抄，写一句「那只是例子」也拦不住。所以这里只印**判据原文**（它本来就是完整规格）与
一份**格式规格**（键名 + 占位符，不是可抄的成品），`pasted_rules` 再兜住「把判据原文整段贴进去」。

**格式规格为什么是「键名 + 占位符」而不是示例**（范式 §9.52）：键名是 state 格式的一部分，模型
必须知道四行的名字与冒号才写得出来；但每一行的**取值**都得它自己编——计划那一行更是不能给成品，
给一条填好的 `Seq Scan ... rows=1000000`，模型照抄下去正好是「大表全扫」。

**SQL 只写不执行**：这一族要的是**文本**，提示词里明说语句与结构变更只写在记录里、不许真的去跑
（判据看的也只是字面，跑不跑都不改变标签）。

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
from .features import SqlReviewCase
from .spec import SqlReviewSpec

#: 判据表与档位表每行的前缀。**这是 state 格式的一部分，不是文案**，所以不进语言文件：
#: 渲染与解析共用这一个常量，两边不可能漂移。
LINE_PREFIX: Final[str] = "- "

#: 一屏评审记录的字符数上下限。太短装不下四行，太长就不是「一屏」了。
MIN_CONTENT_CHARS: Final[int] = features.MIN_CONTENT_CHARS
MAX_CONTENT_CHARS: Final[int] = features.MAX_CONTENT_CHARS


class ContentDraft(BaseModel):
    """后端要填的草稿：那一屏评审记录。

    `packet` 是**一屏数据库变更评审记录**（真实的库名、表名与语句长尾得很、由模型写）；结构化
    事实（判据表、档位表、抬头、格式规格）全由组装器渲染，模型碰不到。
    """

    packet: str = Field(
        description="四行评审记录（statement / plan / ddl / note）；不要围栏、不要解释"
    )


@dataclass(frozen=True)
class ParsedContent:
    """从 state 反解出来的、纯代码看得见的部分。

    字段：`tags` 判据表里列出的判据编号；`text` 一屏评审记录（认不出来给 None）。
    """

    tags: tuple[str, ...]
    text: str | None

    @property
    def found(self) -> bool:
        """输入：无。输出：评审记录那一块认出来了没有。"""
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


def workload_copy(loc: dict[str, JsonValue], workload: str) -> str:
    """输入：语言字典、变更场景。输出：该场景的形态描述。

    预期行为：缺这一档就当场报错（新加一档场景却忘了写文案时，渲染会拿不到句子）。
    """
    table = loc.get("workload")
    entry = table.get(workload) if isinstance(table, dict) else None
    if not isinstance(entry, dict) or "desc" not in entry:
        raise ValueError(f"语言文件里缺变更场景 {workload} 的描述")
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

    说明文字里的 `{language}` 在这里就地填掉（备注那一行的说明要指名道姓说清用哪种语言写，
    只写「用目标语言」模型会照它自己的先验挑一种）。
    """
    spec = loc["prompt"]["format"]
    return [
        f"{LINE_PREFIX}{str(spec[key]).format(language=loc['language_name'])}"
        for key in features.KEY_LINES
    ]


def build_prompt(loc: dict[str, JsonValue], case: SqlReviewCase) -> str:
    """组装提示词。

    输入：语言字典、注入特征。输出：完整提示词（不含答案）。
    预期行为：**每一条硬性要求都写清楚**——必须出现哪些形状（逐条印上判据原文）、出现在哪一行、
    不许出现哪些（逐条给反面写法）、这次变更发生在哪一类业务上、长度区间、四行格式规格、输出前
    自查。这些就是「受控合成」的规格，判据全在 `check_draft` 里机械复核，模型写歪了当场重渲染。

    为什么**一个示例记录都不给**（范式 §9.45）：只给一份具体记录当例子，模型会把它**整段照抄**
    进样本，而且「那只是例子、请自己另编」这句约束写进提示词也拦不住。判据原文本身就是完整规格，
    所以提示词只印判据原文 + 一份带占位符的格式规格。

    为什么把「不许出现」也逐条列出来：只写「必须出现什么」时，模型会顺手带上别的形状（写了个
    没有 `WHERE` 的 `UPDATE`，又顺手在计划里写个 `Seq Scan`）——那一条的标签就废了。
    """
    prompt = loc["prompt"]
    hits = case.tags()
    missing = [tag for tag in features.TAG_IDS if tag not in hits]
    lines = [
        str(prompt["intro"]).format(min=MIN_CONTENT_CHARS, max=MAX_CONTENT_CHARS),
        str(prompt["lang_rule"]).format(language=loc["language_name"]),
        str(prompt["workload_line"]).format(workload=workload_copy(loc, case.workload)),
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
    lines.append(str(prompt["self_check"]).format(language=loc["language_name"]))
    lines.append(str(prompt["output"]))
    return "\n".join(lines)


def build_task(spec: SqlReviewSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：真值规格、语言字典。输出：渲染任务。

    预期行为：任务 id 由规格算（断点续跑按它查进度），`spec` 里带上注入特征——后端实现与假后端
    若想「照着计划答」，不必再解析提示词。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="sqlreview",
        prompt=build_prompt(loc, spec.case),
        response_model=ContentDraft,
        seeds=(),
        spec={
            "case": spec.case.as_dict(),
            "hits": list(spec.case.tags()),
            "workload": spec.case.workload,
        },
    )


def pasted_rules(text: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：一屏评审记录、语言字典。输出：被**逐字贴进**样本的判据原文（按 `TAG_IDS` 顺序）。

    预期行为：只认**完全相同的整段**判据原文（`tag_copy(...)["rule"]`）——改一个词就不再算。

    为什么要有这条：提示词里给模型看的**唯一**一段具体文字就是判据原文（示例记录一律不给，见
    `build_prompt`），所以模型能「抄」的只剩它——把判据的词表整段贴进记录，标签照样算命中（正则
    只看子串），但那一屏记录读起来是「写手在照着规格交作业」，不是一次真实变更。
    """
    body = " ".join(text.split())
    out: list[str] = []
    for tag in features.TAG_IDS:
        rule = " ".join(str(tag_copy(loc, tag)["rule"]).split())
        if rule and rule in body:
            out.append(str(tag_copy(loc, tag)["rule"]))
    return out


def check_draft(draft: ContentDraft, case: SqlReviewCase, loc: dict[str, JsonValue]) -> None:
    """输入：后端草稿、注入特征、语言字典。输出：无（不合格抛 `DraftError`）。

    预期行为：五道闸依次过——非空、不带围栏、长度落在区间内、四个结构键齐全、按字面算出来的命中
    组合与计划逐条相同、没有把判据原文整段贴进去。抛 `DraftError`（`retryable_for_case = True`）
    让管线重渲染一次。
    """
    text = draft.packet.strip()
    if not text:
        raise DraftError("评审记录是空的")
    if "```" in text:
        raise DraftError("评审记录里带了围栏")
    if len(text) < MIN_CONTENT_CHARS:
        raise DraftError(f"评审记录太短（至少 {MIN_CONTENT_CHARS} 个字符，实际 {len(text)}）")
    if len(text) > MAX_CONTENT_CHARS:
        raise DraftError(f"评审记录太长（最多 {MAX_CONTENT_CHARS} 个字符，实际 {len(text)}）")
    missing = features.missing_keys(text)
    if missing:
        raise DraftError(f"评审记录缺这几行（每个键要单独占一行、后面跟冒号）：{list(missing)}")
    found = features.hits_of(text)
    if found != case.hits:
        raise DraftError(
            f"评审记录按字面算出来的命中组合是 {found or '无'}，计划要的是 {case.hits or '无'}"
        )
    pasted = pasted_rules(text, loc)
    if pasted:
        raise DraftError(f"评审记录把判据原文原样贴了进去：{pasted}")


def to_state(draft: ContentDraft, loc: dict[str, JsonValue]) -> str:
    """输入：后端草稿、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：判据表 + 档位表 + 记录抬头 + 一屏评审记录。**判据表与档位表印在 state 里**，所以
    「这屏记录踩了哪几条、该走哪一步」是自明的——考生只读 state 与题面就能答，不必知道外部规则
    （范式 §9.8）。
    """
    marks = loc["state"]
    parts = [str(marks["tag_mark"])]
    parts.extend(f"{LINE_PREFIX}{tag_line(loc, tag)}" for tag in features.TAG_IDS)
    parts.append(str(marks["rule_mark"]))
    parts.extend(f"{LINE_PREFIX}{threshold_line(loc, level)}" for level in features.ACTS)
    parts.append(str(marks["content_mark"]))
    parts.append(draft.packet.strip())
    return "\n".join(parts)


def parse_with(state: str, loc: dict[str, JsonValue]) -> ParsedContent:
    """输入：state、语言字典。输出：按这一份语言反解的结果。

    预期行为：认判据表抬头、档位表抬头、记录抬头；判据表里带行前缀的那些行取编号（按
    `state.tag_sep` 切），**记录抬头之后的所有行原样拼回正文**——抬头之外的内容一律不当结构，
    否则记录里出现一行以 `- ` 开头的 SQL 注释就会把判据表数歪。
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
    """输入：state。输出：能认出记录抬头的那一种语言（认不出给 None）。

    预期行为：审计拿它按 state **自己**的语言取规范文案——一次 run 里混了 zh / en 时，用调用方
    给的那一种去比会平白报一堆「文案对不上」。

    为什么认的是**记录抬头**而不是判据编号：编号（Q1…Q4）是结构键，与语言无关，拿它当判据会把
    别的语言的 state 也抢过来（范式 §9.22）；记录抬头是渲染时按语言写死的、整段最后那一块的开头，
    才分得开。
    """
    if not isinstance(state, str):
        return None
    for lang in locales.available():
        if parse_with(state, locales.load(lang)).found:
            return lang
    return None


def parse_state(state: object) -> ParsedContent:
    """输入：state。输出：纯代码能反解的部分（判据表 / 一屏评审记录）。

    预期行为：与语言无关——把各语言的标记都试一遍，认出记录抬头的那一份就用它；一份都认不出来
    时返回空结果（`text` 为 None）。
    """
    if not isinstance(state, str):
        return ParsedContent(tags=(), text=None)
    lang = detect_locale(state)
    if lang is None:
        return ParsedContent(tags=(), text=None)
    return parse_with(state, locales.load(lang))


def note_text(packet: str) -> str:
    """输入：一屏评审记录。输出：`note:` 那一行的评审备注（没有就给空串）。

    预期行为：这是这一族里**唯一一段自然语言正文**（语句、计划、结构变更都是 SQL / 英文格式），
    `Workflow.language_texts` 会把它报给语言尺子（范式 §9.50：只喂认得出语言的字段）。
    """
    return features.line_value(packet, "note")
