"""检索重排的文本外壳：排序口径与候选编号由代码写，查询与候选原文由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **查询抬头、关键词抬头、候选抬头、排序口径由代码渲染**——它们是结构化事实，让模型去写
  只会写歪；
- 后端写**查询原文、关键词表、那 5 条候选原文**（真实检索结果里的句子是长尾的，模板拼的
  脱离现实）；
- 写出来的候选必须**兑现将计划**：计划说沾上几个词就得沾上几个词。`check_draft` 在渲染期
  查一遍是为了**重试**，`judges.HygieneJudge` 再查一遍是为了兜住不是本 workflow 渲染出来的
  state。

提示词里**逐条写明每条候选该沾上几个关键词**——这是「受控合成」的规格，不是让模型自由
发挥；关键词表由模型自己挑（长尾归它），但挑完**由代码逐个复核**。

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
from .features import RerankPlan
from .spec import RerankSpec

#: 候选行的编号前缀。**这是 state 格式的一部分，不是文案**，所以不进语言文件：渲染与解析
#: 共用这一个常量，两边不可能漂移。
LINE_TEMPLATE: Final[str] = "{n}. {text}"
LINE_PREFIX: Final[re.Pattern[str]] = re.compile(r"^\s*\d+\.\s?")

#: 关键词最短长度的**默认值**（语言文件里可以用 `meta.min_term_chars` 覆盖）。
#
#: 为什么是「按语言给」而不是写死：真机第一批（zh）里模型挑了 `推荐`、`北京` 这种**两个字**
#: 的中文词——那是最正常的中文词长，却被写死的 3 字下限判成「太短」白拒了四条。这个下限的
#: 用意是挡住 `a`、`of`、`的` 这类几乎每条候选都会出现的词，而**中文一个字就是词**，所以
#: 中文的下限只能是 2。判据带语言偏置时，改判据，别怪模型。
MIN_TERM_CHARS: Final[int] = 3


class RerankDraft(BaseModel):
    """后端要填的草稿：查询 + 关键词表 + 候选原文。

    三个字段都是**自然语言**（查询与候选是长尾的、由模型写）；结构化事实（抬头、编号、
    排序口径）全由组装器渲染，模型碰不到。
    """

    query: str = Field(description="一条检索查询，必须包含 terms 里的每个关键词")
    terms: list[str] = Field(description="查询里的关键词，个数由提示词给定，两两不互为子串")
    candidates: list[str] = Field(
        min_length=features.CANDIDATES,
        max_length=features.CANDIDATES,
        description="候选原文，按提示词要求的顺序，每条一行、不要编号前缀、不要围栏",
    )


@dataclass(frozen=True)
class ParsedRerank:
    """从 state 反解出来的、纯代码看得见的部分。"""

    query: str | None
    terms: tuple[str, ...]
    candidates: tuple[str, ...]
    rule: tuple[str, ...]

    @property
    def found(self) -> bool:
        """输入：无。输出：查询抬头认出来了没有。预期行为：就是 `query` 在不在。"""
        return self.query is not None


def cand_ref(loc: dict[str, JsonValue], index: int) -> str:
    """输入：语言字典、候选下标（0 起）。输出：候选的可读指代（`候选 3`）。"""
    return str(loc["meta"]["cand"]).format(n=index + 1)


def rule_rows(loc: dict[str, JsonValue]) -> tuple[str, ...]:
    """输入：语言字典。输出：排序口径那两行（规范形态）。

    预期行为：整段由代码渲染，模型碰不到；判别器拿它逐行比，渲染抄错口径也在这里暴露。
    """
    return tuple(str(row) for row in loc["state"]["rule_rows"])


def _requirement_lines(plan: RerankPlan, loc: dict[str, JsonValue]) -> list[str]:
    """输入：名次计划、语言字典。输出：逐条候选的写作要求（五行）。

    预期行为：部分覆盖的那几条**指名道姓**地说该写哪一个关键词（`terms` 里第几个），而不是
    含糊的「其中 1 个」——真机第一批（zh 7/10、en 4/10）里六条拒稿有五条是同一个理由：模型
    对着「包含其中 1 个」无从下手，干脆写了一句一个关键词都不含的话。规格说得越具体，
    它越没机会自由发挥；哪一个词由代码按顺序轮着派，与标签无关。
    """
    out: list[str] = []
    partial_rank = 0
    for index, value in enumerate(plan.overlaps):
        if value == plan.terms:
            out.append(str(loc["prompt"]["req_full"]).format(pos=index + 1, n=plan.terms))
        elif value > 0:
            out.append(
                str(loc["prompt"]["req_part"]).format(pos=index + 1, idx=partial_rank % plan.terms + 1)
            )
            partial_rank += 1
        else:
            out.append(str(loc["prompt"]["req_none"]).format(pos=index + 1))
    return out


def coverage_vector(plan: RerankPlan, loc: dict[str, JsonValue]) -> str:
    """输入：名次计划、语言字典。输出：覆盖率清单（`3、0、1、1、0`）。

    预期行为：纯机械地把计划里的覆盖率排成一行，给模型当**输出前自查**用——真机第一批
    （zh 7/10、en 4/10）里六条拒稿有五条是同一个理由：模型给「该沾上 1 个词」的候选写了
    一句泛化的话。逐条要求它写了没照做，改成「写完照这个清单核一遍」，是把规格再摆一遍，
    不是放宽判据（判据一个字没动）。
    """
    sep = str(loc["meta"]["list_sep"])
    return sep.join(str(value) for value in plan.overlaps)


def build_prompt(plan: RerankPlan, loc: dict[str, JsonValue]) -> str:
    """组装提示词。

    输入：名次计划、语言字典。输出：完整提示词（不含答案）。
    预期行为：关键词个数、逐条候选的覆盖率要求写在最前面；关键词的挑选规则（够长、两两不
    互为子串）与「一条都不许出现」的那些候选该怎么写，各有一句——不写清楚，模型就会拿
    `the`、`and` 这类词充数，覆盖率当场退化成常量。
    **语言约束紧跟开场白**：真机实测 zh 侧 3/3、4/6、1/4 的产物把查询与候选写成了英文
    （汉字占比 0.081~0.120），en 侧反过来有 2/9 写成中文——提示词通篇是目标语言并不足以
    让模型不换语言，得有一句明说的。这一条与 `jc.language` 的判据是同一件事的两端：这里
    要求，那里复核。
    """
    lines = [
        str(loc["prompt"]["intro"]).format(n=features.CANDIDATES),
        str(loc["prompt"]["lang_rule"]).format(language=loc["language_name"]),
        str(loc["prompt"]["terms_rule"]).format(k=plan.terms, min=min_term_chars(loc)),
        str(loc["prompt"]["req_header"]),
        *_requirement_lines(plan, loc),
        str(loc["prompt"]["none_hint"]),
        str(loc["prompt"]["self_check"]).format(vec=coverage_vector(plan, loc)),
        str(loc["prompt"]["output"]).format(n=features.CANDIDATES),
    ]
    return "\n".join(lines)


def min_term_chars(loc: dict[str, JsonValue]) -> int:
    """输入：语言字典。输出：该语言下关键词的最短长度。

    预期行为：读 `meta.min_term_chars`，缺省回落到 `MIN_TERM_CHARS`。
    """
    value = loc["meta"].get("min_term_chars", MIN_TERM_CHARS)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"meta.min_term_chars 必须是整数：{value!r}")
    return value


def build_task(spec: RerankSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：真值规格、语言字典。输出：渲染任务。

    预期行为：任务 id 由规格算（断点续跑按它查进度），`spec` 里带上计划里的覆盖率——后端
    实现若想看「该写几条」，不必再解析提示词。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="rerank",
        prompt=build_prompt(spec.plan, loc),
        response_model=RerankDraft,
        spec={"case": spec.case.as_dict(), "overlaps": list(spec.plan.overlaps)},
    )


def check_parts(
    query: str,
    terms: list[str],
    candidates: list[str],
    plan: RerankPlan,
    loc: dict[str, JsonValue],
) -> None:
    """校验「查询 + 关键词表 + 候选」是否兑现了名次计划。

    输入：三段文本、名次计划、语言字典。输出：无（不合格抛 `DraftError`，管线据此重渲染）。
    预期行为：逐条查——关键词个数与两两不互为子串、查询里出现每个词、候选正好 5 条且互不
    相同、每条候选的覆盖率正好等于计划。**覆盖率按大小写不敏感的子串算**，与 `features.
    coverage` 是同一份实现，所以构造侧与读法侧不可能漂移。
    """
    normalized = [term.strip() for term in terms]
    if len(normalized) != plan.terms:
        raise DraftError(f"关键词该有 {plan.terms} 个，实际 {len(normalized)} 个：{terms!r}")
    if any(not term for term in normalized):
        raise DraftError(f"关键词里有空串：{terms!r}")
    floor = min_term_chars(loc)
    if any(len(term) < floor for term in normalized):
        raise DraftError(f"关键词太短（至少 {floor} 个字符）：{normalized!r}")
    lowered = [term.lower() for term in normalized]
    if len(set(lowered)) != len(lowered):
        raise DraftError(f"关键词重复：{normalized!r}")
    for left in lowered:
        for right in lowered:
            if left != right and left in right:
                raise DraftError(f"关键词互为子串（覆盖率会算不清）：{normalized!r}")
    text = query.strip()
    if not text:
        raise DraftError("查询是空的")
    for term in lowered:
        if term not in text.lower():
            raise DraftError(f"查询里没出现关键词 {term!r}：{text!r}")
    if len(candidates) != features.CANDIDATES:
        raise DraftError(f"候选该有 {features.CANDIDATES} 条，实际 {len(candidates)} 条")
    cleaned = [item.strip() for item in candidates]
    if any(not item for item in cleaned):
        raise DraftError(f"候选里有空串：{candidates!r}")
    if any("\n" in item for item in cleaned):
        raise DraftError("候选必须是单行（出现换行说明模型把整段塞进了一条）")
    if any("```" in item for item in cleaned):
        raise DraftError(f"候选里带围栏：{cleaned!r}")
    if len(set(cleaned)) != len(cleaned):
        raise DraftError(f"候选重复：{cleaned!r}")
    for index, item in enumerate(cleaned):
        got = features.coverage(item, normalized)
        if got != plan.overlaps[index]:
            raise DraftError(
                f"第 {index + 1} 条候选该沾上 {plan.overlaps[index]} 个关键词，"
                f"实际沾上 {got} 个：{item!r}"
            )


def check_draft(draft: RerankDraft, plan: RerankPlan, loc: dict[str, JsonValue]) -> None:
    """输入：后端草稿、名次计划、语言字典。输出：无（不合格抛 `DraftError`）。"""
    check_parts(draft.query, list(draft.terms), list(draft.candidates), plan, loc)


def to_state(draft: RerankDraft, plan: RerankPlan, loc: dict[str, JsonValue]) -> str:
    """输入：后端草稿、名次计划、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：查询 + 关键词 + 编号候选 + 排序口径；**排序口径印在 state 里**，所以「谁排
    前面」是自明的（范式 §9.8）。
    """
    sep = str(loc["meta"]["list_sep"])
    terms = [term.strip() for term in draft.terms]
    candidates = [item.strip() for item in draft.candidates]
    state = loc["state"]
    parts = [
        f"{state['query_mark']}{draft.query.strip()}",
        f"{state['terms_mark']}{sep.join(terms)}",
        f"{state['cands_mark']}{str(state['cands_suffix']).format(n=features.CANDIDATES)}",
    ]
    parts.extend(
        LINE_TEMPLATE.format(n=index + 1, text=text)
        for index, text in enumerate(candidates)
    )
    parts.append(str(state["rule_mark"]))
    parts.extend(rule_rows(loc))
    return "\n".join(parts)


def parse_with(state: str, loc: dict[str, JsonValue]) -> ParsedRerank:
    """输入：state、语言字典。输出：按这一份语言反解的结果。

    预期行为：认查询标记、关键词标记、候选标记、排序口径标记四种行；候选是候选段里的编号
    行，前缀之后的部分。标记之外的行（例如排序口径那两行之外的东西）一律不当候选——否则
    口径本身会被数成一条候选。
    """
    marks = loc["state"]
    query_mark = str(marks["query_mark"])
    terms_mark = str(marks["terms_mark"])
    cands_mark = str(marks["cands_mark"])
    rule_mark = str(marks["rule_mark"])
    sep = str(loc["meta"]["list_sep"])
    query: str | None = None
    terms: list[str] = []
    candidates: list[str] = []
    rule: list[str] = []
    in_cands = False
    in_rule = False
    for raw in state.splitlines():
        text = raw.strip()
        if text.startswith(query_mark):
            query = text[len(query_mark) :].strip()
            in_cands = False
            in_rule = False
            continue
        if text.startswith(terms_mark):
            body = text[len(terms_mark) :].strip()
            terms = [part.strip() for part in body.split(sep) if part.strip()]
            in_cands = False
            in_rule = False
            continue
        if text.startswith(cands_mark):
            in_cands = True
            in_rule = False
            continue
        if text.startswith(rule_mark):
            in_rule = True
            in_cands = False
            continue
        if in_rule:
            if text:
                rule.append(text)
            continue
        if in_cands:
            found = LINE_PREFIX.match(raw)
            if found:
                candidates.append(raw[found.end() :])
    return ParsedRerank(
        query=query,
        terms=tuple(terms),
        candidates=tuple(candidates),
        rule=tuple(rule),
    )


def detect_locale(state: object) -> str | None:
    """输入：state。输出：能认出查询抬头的那一种语言（认不出给 None）。

    预期行为：审计拿它按 state **自己**的语言取规范表——一次 run 里混了 zh / en 时，用调用方
    给的那一种去比会平白报一堆「排序口径对不上」。

    为什么认的是**查询抬头**而不是编号行：编号行（`1. …`）与语言无关，任何一种语言都能把它
    认下来，于是「先试的那一种」会抢走所有 state。抬头是渲染时按语言写死的，拿它当判据才
    分得开（这条坑在 lint 上踩过一次，见范式 §9.22）。
    """
    return recognize_locale(state, locales, parse_with)


def parse_state(state: object) -> ParsedRerank:
    """输入：state。输出：纯代码能反解的部分（查询 / 关键词 / 候选 / 排序口径）。

    预期行为：与语言无关——把各语言的标记都试一遍，认出查询抬头的那一份就用它；一份都认不
    出来时返回空结果（`query` 为 None）。
    """
    lang = detect_locale(state)
    if lang is None:
        return ParsedRerank(query=None, terms=(), candidates=(), rule=())
    return parse_with(state, locales.load(lang))
