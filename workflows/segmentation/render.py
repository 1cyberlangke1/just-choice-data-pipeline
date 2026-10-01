"""文档包切分的文本外壳：种类表、切分规则与页序抬头由代码写，页面正文由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **种类表、切分规则、页序抬头由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**每一页的正文**（申请表、信函、收据、合同读起来都是长尾的，模板拼出来的脱离现实分布）；
- **每一页属于哪一类由代码按字面算**（`features.PATTERNS` 那四条正则），模型碰不到：提示词里
  逐页写明「这一页属于哪一类、必须出现哪一类抬头、不许出现别的类抬头」，`check_draft` 在渲染期
  按同一份判据复核一遍（不合格就重渲染），`judges.HygieneJudge` / `FrameJudge` 再各查一遍，
  兜住不是本 workflow 渲染出来的 state。

**判据原文也印进 state**（范式 §9.44 的教训，这一族一开始就照做）：反解模型要判「这一页属于
哪一类」，而种类名是**压缩过的标签**——「表单」推不出「页面里出现『表单编号 / 申请表』这类抬头」。
`memorygate` 上量过一次：只印判据名时反解 1/8，补上原文 8/8。

**来源（`source`）为什么要写进提示词**：同一包页从扫描仪、传真机或手机来，页面上的痕迹不一样
（扫描件带页码戳、传真件带传真抬头、手机拍照带文件名时间戳）。点明之后，这一维就成了一条
**只影响渲染、不影响标签**的留出维（`test_ood` 侧是训练侧一次都没见过的 `photo`）。

输入/输出：提示词与草稿模型；除 `parse_state` 外不发网络请求。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, Field

from jc.backend.types import GenerationTask
from jc.jsontypes import JsonValue
from jc.state_parse import recognize_locale
from jc.workflow import DraftError

from . import features, locales
from .features import SegmentationPlan
from .spec import SegmentationSpec

#: 种类表与切分规则每行的前缀。**这是 state 格式的一部分，不是文案**，所以不进语言文件：
#: 渲染与解析共用这一个常量，两边不可能漂移。
LINE_PREFIX: Final[str] = "- "

class PacketDraft(BaseModel):
    """后端要填的草稿：逐页的正文。

    `pages` 是**一页一段文字**的数组，长度必须与计划里的页数一致（`check_draft` 会当场核）；
    结构化事实（种类表、切分规则、页序抬头）全由组装器渲染，模型碰不到。
    """

    pages: list[str] = Field(
        description="逐页的正文，按页序排列；每页一段文字，不要标题、不要围栏、不要页码行"
    )


@dataclass(frozen=True)
class ParsedPacket:
    """从 state 反解出来的、纯代码看得见的部分。

    字段：`classes` 种类表里列出的种类编号；`pages` 逐页正文（认不出来给空元组）。
    """

    classes: tuple[str, ...]
    pages: tuple[str, ...]

    @property
    def found(self) -> bool:
        """输入：无。输出：页序那一块认出来了没有。"""
        return bool(self.pages)


def class_copy(loc: dict[str, JsonValue], cls: str) -> dict[str, JsonValue]:
    """输入：语言字典、种类编号。输出：该种类的文案（name / rule / avoid）。

    预期行为：缺字段就当场报错——种类表与提示词都靠它，缺一个字段模型就会拿到半句话。
    """
    table = loc.get("class")
    entry = table.get(cls) if isinstance(table, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"语言文件里缺种类 {cls} 的文案")
    for key in ("name", "rule", "avoid"):
        if key not in entry:
            raise ValueError(f"种类 {cls} 的文案缺字段 {key}")
    return entry


def class_line(loc: dict[str, JsonValue], cls: str) -> str:
    """输入：语言字典、种类编号。输出：种类表里那一行（不含行前缀）。

    预期行为：`编号 + 分隔符 + 种类名 + 规则分隔符 + 判据原文`。分隔符按语言取
    （`state.class_sep`），解析时按同一份分隔符切回编号——**编号是结构键**，所以它排在分隔符
    前面，与语言的书写顺序无关。
    """
    entry = class_copy(loc, cls)
    sep = str(loc["state"]["class_sep"])
    rule_sep = str(loc["state"]["rule_sep"])
    return f"{cls}{sep}{entry['name']}{rule_sep}{entry['rule']}"


def rule_lines(loc: dict[str, JsonValue]) -> list[str]:
    """输入：语言字典。输出：切分规则那几行（不含行前缀）。

    预期行为：三条——跨类一定切、可并的类整段并成一份、不可并的类一页一份。可并 / 不可并的
    分组**由 `features.MERGEABLE` 现算**，不在语言文件里另抄一份，两边不可能各说各的。
    """
    spec = loc["state"]["rules"]
    joiner = str(spec["joiner"])
    mergeable = [cls for cls in features.CLASS_IDS if cls in features.MERGEABLE]
    each = [cls for cls in features.CLASS_IDS if cls not in features.MERGEABLE]

    def names(group: list[str]) -> str:
        """输入：种类编号列表。输出：这一组的种类名（按语言）。"""
        return joiner.join(str(loc["class"][cls]["name"]) for cls in group)

    return [
        str(spec["cross"]),
        str(spec["merge"]).format(names=names(mergeable)),
        str(spec["each"]).format(names=names(each)),
    ]


def page_mark(loc: dict[str, JsonValue], index: int) -> str:
    """输入：语言字典、页号（从 1 起）。输出：页序抬头那一行。"""
    marks = loc["state"]
    return f"{marks['page_prefix']}{index}{marks['page_suffix']}"


def page_marker_re(loc: dict[str, JsonValue]) -> re.Pattern[str]:
    """输入：语言字典。输出：匹配「页序抬头」的正则（只认整行）。

    预期行为：由该语言的 `page_prefix` / `page_suffix` 现拼（转义后），所以抬头换了写法也不会
    出现「渲染用一套、解析认另一套」的漂移。
    """
    marks = loc["state"]
    prefix = re.escape(str(marks["page_prefix"]))
    suffix = re.escape(str(marks["page_suffix"]))
    return re.compile(rf"^{prefix}(\d+){suffix}$")


def page_bounds(loc: dict[str, JsonValue]) -> tuple[int, int]:
    """输入：语言字典。输出：单页正文的 (下限, 上限) 字符数。

    预期行为：从 `state.page_min` / `state.page_max` 取，**按语言校准**——同样一页纸，英文的
    字符数天然比中文多（真机实测 zh 中位数 131、en 中位数 212），一份共用上下限会把英文顶在
    上限上（首轮 en 侧 1 条 295 字符被拒，而 zh 侧最大才 186）。缺字段或下界不小于上界都当场
    报错——这个数同时喂给提示词与渲染期校验，两边必须是同一个。
    """
    marks = loc["state"]
    low, high = marks.get("page_min"), marks.get("page_max")
    if not isinstance(low, int) or not isinstance(high, int) or isinstance(low, bool):
        raise ValueError(f"语言文件里的单页长度上下限不是整数：{low!r} / {high!r}")
    if low >= high:
        raise ValueError(f"单页长度下界必须小于上界：{low} >= {high}")
    return low, high


def source_copy(loc: dict[str, JsonValue], source: str) -> dict[str, JsonValue]:
    """输入：语言字典、来源编号。输出：该来源的文案（desc / trace）。预期行为：缺字段当场报错。"""
    table = loc.get("source")
    entry = table.get(source) if isinstance(table, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"语言文件里缺来源 {source} 的文案")
    for key in ("desc", "trace"):
        if key not in entry:
            raise ValueError(f"来源 {source} 的文案缺字段 {key}")
    return entry


def build_prompt(
    loc: dict[str, JsonValue], plan: SegmentationPlan, source: str
) -> str:
    """组装提示词。

    输入：语言字典、切分计划、来源编号。输出：完整提示词（不含答案）。
    预期行为：**每一条硬性要求都写清楚**——逐页印上它属于哪一类、该类的判据原文、不许出现别的
    类的抬头、来源痕迹、单页长度区间、输出前自查。这些就是「受控合成」的规格，判据全在
    `check_draft` 里机械复核，模型写歪了当场重渲染。

    为什么**一个示例页面都不给**（范式 §9.45）：只给一个具体页面当例子，模型会把它整段照抄进
    正文。判据原文本身就是完整规格（`rule` 把要匹配的词逐个列了出来），所以提示词只印判据原文、
    不给可抄的成品页；`pasted_rules` 再兜住「把判据原文整段贴进正文」这条残余捷径。
    """
    prompt = loc["prompt"]
    src = source_copy(loc, source)
    low, high = page_bounds(loc)
    lines = [
        str(prompt["intro"]).format(
            pages=len(plan.shape), min=low, max=high
        ),
        str(prompt["lang_rule"]).format(language=loc["language_name"]),
        str(prompt["source_line"]).format(source=src["desc"], trace=src["trace"]),
        str(prompt["assign_header"]),
    ]
    for index, cls in enumerate(plan.shape, start=1):
        entry = class_copy(loc, cls)
        lines.append(
            str(prompt["assign_line"]).format(
                index=index, name=entry["name"], rule=entry["rule"]
            )
        )
    others = [cls for cls in features.CLASS_IDS if cls not in set(plan.shape)]
    if others:
        lines.append(str(prompt["forbid_header"]))
        for cls in others:
            entry = class_copy(loc, cls)
            lines.append(
                str(prompt["forbid_line"]).format(name=entry["name"], avoid=entry["avoid"])
            )
    lines.append(str(prompt["vary_rule"]))
    lines.append(
        str(prompt["length_rule"]).format(
            pages=len(plan.shape), min=low, max=high
        )
    )
    lines.append(str(prompt["self_check"]))
    lines.append(str(prompt["output"]).format(pages=len(plan.shape)))
    return "\n".join(lines)


def build_task(spec: SegmentationSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：真值规格、语言字典。输出：渲染任务。

    预期行为：任务 id 由规格算（断点续跑按它查进度），`spec` 里带上注入特征——后端实现与假后端
    若想「照着计划答」，不必再解析提示词。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="segmentation",
        prompt=build_prompt(loc, spec.plan, spec.case.source),
        response_model=PacketDraft,
        seeds=(),
        spec={
            "case": spec.case.as_dict(),
            "shape": list(spec.plan.shape),
            "source": spec.case.source,
        },
    )


def pasted_rules(text: str, loc: dict[str, JsonValue]) -> list[str]:
    """输入：正文、语言字典。输出：被**逐字贴进**正文的判据原文（按 `CLASS_IDS` 顺序）。

    预期行为：只认**完全相同的整段**判据原文（`class_copy(...)["rule"]`）——改一个词就不再算。

    为什么要有这条：提示词里给模型看的**唯一**一段具体文字就是判据原文，所以模型能「抄」的只剩
    它——把判据的词表整段贴进页面，页面照样算命中（正则只看子串），但那段文字读起来不是材料，
    而是判据本身。这条把这种残余捷径变成机械可查：贴了就重渲染。真实材料不会包含整段判据原文。
    """
    return [
        entry
        for entry in (str(class_copy(loc, cls)["rule"]) for cls in features.CLASS_IDS)
        if entry in text
    ]


def check_draft(
    draft: PacketDraft, plan: SegmentationPlan, source: str, loc: dict[str, JsonValue]
) -> None:
    """校验「这一包材料」是否兑现了切分计划。

    输入：后端草稿、切分计划、来源编号、语言字典。输出：无（不合格抛 `DraftError`，管线据此重渲染）。
    预期行为：逐条查——

    1. 页数恰好等于计划里的页数；
    2. 每一页非空、不带围栏、长度落在该语言的 `state.page_min`~`state.page_max`（`page_bounds`）；
    3. **每一页按字面算出来的种类与计划逐页相同**（多一类少一类、或抬头混着写都算不合格）；
    4. 每一页都带本来源的痕迹（`features.SOURCE_TRACES`）；
    5. **不许把判据原文原样贴进正文**（`pasted_rules` 逐字比对）。
    第 3 条是这一族的核心：逐页种类就是金标签，它由正则从正文算得出来，所以「模型写歪了」在
    渲染期就能拦住，不用等判别器。
    """
    pages = [text.strip() for text in draft.pages]
    low, high = page_bounds(loc)
    if len(pages) != len(plan.shape):
        raise DraftError(f"页数不对：计划要 {len(plan.shape)} 页，草稿给了 {len(pages)} 页")
    for index, (text, want) in enumerate(zip(pages, plan.shape, strict=True), start=1):
        if not text:
            raise DraftError(f"第 {index} 页是空的")
        if "```" in text:
            raise DraftError(f"第 {index} 页里带围栏：{text[:60]!r}")
        if len(text) < low:
            raise DraftError(
                f"第 {index} 页太短（至少 {low} 个字符）：{text!r}"
            )
        if len(text) > high:
            raise DraftError(
                f"第 {index} 页太长（最多 {high} 个字符，实际 {len(text)}）"
            )
        hits = features.matches(text)
        if hits != (want,):
            raise DraftError(
                f"第 {index} 页按字面算出来属于 {list(hits) or '哪一类都不属于'}，"
                f"计划要的是 {want}"
            )
        if not features.source_trace(text, source):
            raise DraftError(f"第 {index} 页没有带来源 {source!r} 的痕迹")
    pasted = pasted_rules("\n".join(pages), loc)
    if pasted:
        raise DraftError(f"正文把判据原文原样贴了进去：{pasted}")


def to_state(draft: PacketDraft, loc: dict[str, JsonValue]) -> str:
    """输入：后端草稿、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：种类表 + 切分规则 + 页序抬头 + 逐页正文。**种类表与切分规则印在 state 里**，所以
    「这一包里有哪几类、该切成几份」是自明的——考生只读 state 与题面就能答，不必知道外部规则
    （范式 §9.8）。
    """
    marks = loc["state"]
    parts = [str(marks["table_mark"])]
    parts.extend(f"{LINE_PREFIX}{class_line(loc, cls)}" for cls in features.CLASS_IDS)
    parts.append(str(marks["rule_mark"]))
    parts.extend(f"{LINE_PREFIX}{line}" for line in rule_lines(loc))
    parts.append(str(marks["order_mark"]))
    for index, text in enumerate(draft.pages, start=1):
        parts.append(page_mark(loc, index))
        parts.append(text.strip())
    return "\n".join(parts)


def parse_with(state: str, loc: dict[str, JsonValue]) -> ParsedPacket:
    """输入：state、语言字典。输出：按这一份语言反解的结果。

    预期行为：认种类表抬头、切分规则抬头、页序抬头；**页序抬头之后的内容按抬头切成页**——
    抬头之外的内容一律不当结构，否则正文里出现一行以 `- ` 开头的清单就会把种类表数歪。
    """
    marks = loc["state"]
    table_mark = str(marks["table_mark"])
    rule_mark = str(marks["rule_mark"])
    order_mark = str(marks["order_mark"])
    sep = str(marks["class_sep"])
    marker = page_marker_re(loc)
    classes: list[str] = []
    chunks: list[list[str]] = []
    section = ""
    for raw in state.splitlines():
        text = raw.strip()
        if marker.match(text):
            section = "page"
            chunks.append([])
            continue
        if text.startswith(order_mark):
            section = "order"
            continue
        if text.startswith(table_mark):
            section = "table"
            continue
        if text.startswith(rule_mark):
            section = "rule"
            continue
        if section == "table" and text.startswith(LINE_PREFIX):
            classes.append(text[len(LINE_PREFIX) :].partition(sep)[0].strip())
            continue
        if section == "page" and chunks:
            chunks[-1].append(raw.strip())
    pages = tuple("\n".join(lines).strip() for lines in chunks)
    return ParsedPacket(classes=tuple(classes), pages=pages)


def detect_locale(state: object) -> str | None:
    """输入：state。输出：能认出页序抬头的那一种语言（认不出给 None）。

    预期行为：审计拿它按 state **自己**的语言取规范文案——一次 run 里混了 zh / en 时，用调用方
    给的那一种去比会平白报一堆「文案对不上」。

    为什么认的是**页序抬头**而不是种类编号：编号（form / letter…）是结构键，与语言无关，拿它当
    判据会把别的语言的 state 也抢过来（这条坑在 lint 上踩过一次，见范式 §9.22）；页序抬头是渲染
    时按语言写死的、整段最后那一块的开头，才分得开。
    """
    return recognize_locale(state, locales, parse_with)


def parse_state(state: object) -> ParsedPacket:
    """输入：state。输出：纯代码能反解的部分（种类表 / 逐页正文）。

    预期行为：与语言无关——把各语言的标记都试一遍，认出页序抬头的那一份就用它；一份都认不出来
    时返回空结果（`pages` 为空）。
    """
    if not isinstance(state, str):
        return ParsedPacket(classes=(), pages=())
    lang = detect_locale(state)
    if lang is None:
        return ParsedPacket(classes=(), pages=())
    return parse_with(state, locales.load(lang))
