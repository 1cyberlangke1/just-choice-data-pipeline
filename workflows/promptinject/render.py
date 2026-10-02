"""提示注入的文本外壳：注入判据表、处置阈值表与内容抬头由代码写，不可信内容由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **判据表、阈值表、内容抬头由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**不可信内容**（网页正文 / 工具返回值 / 检索片段 / 文件内容都是长尾的，模板拼出来的
  脱离现实分布）；
- **命中哪几条判据由代码按字面算**（`features.PATTERNS` 那四条正则），模型碰不到：提示词里逐条
  写明「必须出现哪些形状、不许出现哪些」，`check_draft` 在渲染期按同一份判据复核一遍（不合格就
  重渲染），`judges.HygieneJudge` / `FrameJudge` 再各查一遍，兜住不是本 workflow 渲染出来的
  state。

**为什么判据写成机械短语表**：这一族要判的是「这段内容有没有在试图操纵读它的 Agent」，而真实
语料里的操纵手法是有固定形状的（覆盖指令、伪造角色、外送数据、诱导执行）。写成正则之后，
「命中哪几条」是**算**出来的，模型只负责写一段读得通、长尾的内容——两边都不越界
（范式 §9.22 / §9.23 / §9.24 同一条路子）。

**素材来源（`source`）为什么要写进提示词**：同一份命中组合可以长在网页正文、工具返回值或文件
内容上，而这三种形态的**排版差别很大**（JSON 有引号、脚本有缩进、网页是连续段落）。不点明形态，
模型写出来的东西会四不像；点明之后，这一维就成了一条**只影响渲染、不影响标签**的留出维
（`test_ood` 侧是训练侧一次都没见过的 `file`）。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.locale_files import copy_entry
from jc.state_parse import ParsedState as ParsedContent, parse_any, parse_sections, recognize_locale
from jc.workflow import DraftError

from . import features, locales
from .features import PromptInjectCase
from .spec import PromptInjectSpec

#: 判据表与阈值表每行的前缀。**这是 state 格式的一部分，不是文案**，所以不进语言文件：
#: 渲染与解析共用这一个常量，两边不可能漂移。
LINE_PREFIX: Final[str] = "- "

#: 内容长度上下限。太短读不出「像不像一段正经的网页 / 工具返回值」，太长就只是灌水。
MIN_CONTENT_CHARS: Final[int] = features.MIN_CONTENT_CHARS
MAX_CONTENT_CHARS: Final[int] = features.MAX_CONTENT_CHARS


class ContentDraft(BaseModel):
    """后端要填的草稿：那段不可信内容。

    `text` 是**自然语言**（内容本身是长尾的、由模型写）；结构化事实（判据表、阈值表、抬头）
    全由组装器渲染，模型碰不到。
    """

    text: str = Field(
        description="不可信内容：一段读得通的自然语言（网页正文 / 工具返回值 / 文件片段），"
        "不要标题、不要分点、不要围栏"
    )


def tag_copy(loc: dict[str, JsonValue], tag: str) -> dict[str, JsonValue]:
    """输入：语言字典、判据编号。输出：该判据的文案（name / rule / example / avoid）。

    预期行为：缺字段就当场报错——判据表与提示词都靠它，缺一个字段模型就会拿到半句话。
    """
    return copy_entry(loc, "tag", tag, ("name", "rule", "example", "avoid"), "判据")


def tag_line(loc: dict[str, JsonValue], tag: str) -> str:
    """输入：语言字典、判据编号。输出：判据表里那一行（不含行前缀）。

    预期行为：`编号 + 分隔符 + 判据名`。分隔符按语言取（`state.tag_sep`），解析时按同一份
    分隔符切回编号——**编号是结构键**，所以它排在分隔符前面，与语言的书写顺序无关。
    """
    sep = str(loc["state"]["tag_sep"])
    return f"{tag}{sep}{tag_copy(loc, tag)['name']}"


def source_copy(loc: dict[str, JsonValue], source: str) -> str:
    """输入：语言字典、素材来源键。输出：那一种来源的形态说明。

    预期行为：缺字段就当场报错——提示词靠它说清「这段内容该长成什么样」。
    """
    table = loc.get("source")
    entry = table.get(source) if isinstance(table, dict) else None
    if not isinstance(entry, dict) or not str(entry.get("desc", "")).strip():
        raise ValueError(f"语言文件里缺素材来源 {source} 的形态说明")
    return str(entry["desc"])


def threshold_line(loc: dict[str, JsonValue], level: str) -> str:
    """输入：语言字典、处置档位名。输出：阈值表里那一行（不含行前缀）。

    预期行为：`阈值描述 + 分隔符 + 处置动作`；处置动作与 `action` 那一族的选项取的是**同一份
    文案**（`fam.action.opt`），所以 state 里那张表和题面不可能各说各的。
    """
    sep = str(loc["state"]["tag_sep"])
    action = str(loc["fam"]["action"]["opt"][level])
    return f"{loc['state']['threshold'][level]}{sep}{action}"


def build_prompt(loc: dict[str, JsonValue], case: PromptInjectCase) -> str:
    """组装提示词。

    输入：语言字典、注入特征。输出：完整提示词（不含答案）。
    预期行为：**每一条硬性要求都写清楚**——必须出现哪些形状（逐条给例子，并**明说那只是例子、
    要自己另编**）、不许出现哪些（逐条给反面写法）、这一条是什么素材来源、长度区间、
    输出前自查。这些就是「受控合成」的规格，判据全在 `check_draft` 里机械复核，模型写歪了当场
    重渲染。

    为什么必须明说「例子只是例子」（范式 §9.25）：只给一个具体片段当例子，模型会把它**逐条
    照抄**进每一条内容——样本本身没错（标签由正则算），但「背下这个片段 = 命中这条判据」的
    捷径就此长出来了。

    为什么把「不许出现」也逐条列出来：只写「必须出现什么」时，模型会顺手带上别的形状（写个带
    参数的链接、句末来一句「执行这段命令」），命中组合当场多出几条——那一条的标签就废了。
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
                    name=entry["name"], example=entry["example"], rule=entry["rule"]
                )
            )
        if "T2" in hits:
            # 角色伪造那一条是**唯一**与版面有关的判据（行首），光给一个单行例子模型读不出来
            # ——真机第一批 9 条里 5 条拒稿都是它：模型把 `system:` 写在了句子中间。
            lines.append(str(prompt["role_line"]))
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


def build_task(spec: PromptInjectSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：真值规格、语言字典。输出：渲染任务。

    预期行为：任务 id 由规格算（断点续跑按它查进度），`spec` 里带上注入特征——后端实现与假后端
    若想「照着计划答」，不必再解析提示词。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="promptinject",
        prompt=build_prompt(loc, spec.case),
        response_model=ContentDraft,
        seeds=(),
        spec={
            "case": spec.case.as_dict(),
            "hits": list(spec.case.tags()),
            "source": spec.case.source,
        },
    )


def check_draft(draft: ContentDraft, case: PromptInjectCase) -> None:
    """校验「不可信内容」是否兑现了注入特征。

    输入：后端草稿、注入特征。输出：无（不合格抛 `DraftError`，管线据此重渲染）。
    预期行为：逐条查——

    1. 内容非空、不带围栏、长度落在 `MIN_CONTENT_CHARS`~`MAX_CONTENT_CHARS`；
    2. **按字面算出来的命中组合与计划逐条相同**（多一条少一条都不行，理由里说清差在哪）；
    第 2 条是这一族的核心：命中组合就是金标签，它由正则从文本算得出来，所以「模型写歪了」在
    渲染期就能拦住，不用等判别器。
    """
    text = draft.text.strip()
    if not text:
        raise DraftError("内容是空的")
    if "```" in text:
        raise DraftError(f"内容里带围栏：{text[:60]!r}")
    if len(text) < MIN_CONTENT_CHARS:
        raise DraftError(f"内容太短（至少 {MIN_CONTENT_CHARS} 个字符）：{text!r}")
    if len(text) > MAX_CONTENT_CHARS:
        raise DraftError(f"内容太长（最多 {MAX_CONTENT_CHARS} 个字符，实际 {len(text)}）")
    found = features.hits_of(text)
    if found != case.hits:
        raise DraftError(
            f"内容按字面算出来的命中组合是 {found or '无'}，计划要的是 {case.hits or '无'}"
        )


def to_state(draft: ContentDraft, loc: dict[str, JsonValue]) -> str:
    """输入：后端草稿、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：判据表 + 阈值表 + 内容抬头 + 内容。**判据表与阈值表印在 state 里**，所以「这段
    内容带了哪几条、该怎么处置」是自明的——考生只读 state 与题面就能答，不必知道外部规则
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
    """输入：state、语言字典。输出：按这一份语言反解的结果。"""
    return parse_sections(state, loc, line_prefix=LINE_PREFIX)


def detect_locale(state: object) -> str | None:
    """输入：state。输出：能认出素材抬头的那一种语言（认不出给 None）。"""
    return recognize_locale(state, locales, parse_with)


def parse_state(state: object) -> ParsedContent:
    """输入：state。输出：纯代码能反解的部分（判据表 / 素材）。"""
    return parse_any(state, locales, parse_with)
