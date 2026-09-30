"""报销收据归类的文本外壳：归类表、上限表与档位表由代码写，收据行由后端写。

对应 PLAN 3.4 文本来源的第 ③ 档「LLM 自由生成 + 反向校验」：

- **归类表、上限表、档位表由代码渲染**——它们是结构化事实，让模型去写只会写歪；
- 后端只写**一条收据行**（真实收据上的商户名与条目是长尾的，模板拼出来的脱离现实分布）；
- **命中哪几类、金额落哪一档由代码按字面算**（`features.categories_hit` + `band_of`），模型
  碰不到：提示词里逐条写明「必须出现哪些词与金额、不许出现哪些」，`check_draft` 在渲染期按
  同一份判据复核一遍（不合格就重渲染），`judges.HygieneJudge` / `FrameJudge` 再各查一遍，
  兜住不是本 workflow 渲染出来的 state。

**为什么把「不许出现类别名本身」也写进契约**：归类表印在 state 里，类别名一出现在条目里，
这道题就从「按表判」退化成「抄表」——样本没错，但在训练里是废的（范式 §9.25）。

**为什么诱饵词要逐字给出**：模型不会自发写出「明显像某一类、但不在表里」的说法——含糊地说
「写个像餐饮的条目」，它会直接写表里的词，计划当场落空。所以提示词把
`render.decoy_word_for` 挑出来的那个词原样摆出来，并明说「不许换成表里的词」。

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
from .features import ReceiptCase
from .spec import ReceiptCatSpec

#: 归类表 / 上限表 / 档位表每行的前缀。**这是 state 格式的一部分，不是文案**，所以不进语言
#: 文件：渲染与解析共用这一个常量，两边不可能漂移。
LINE_PREFIX: Final[str] = "- "

#: 正文长度上下限（与 `features` 共用同一份常量，改一处两边都跟着改）。
MIN_CONTENT_CHARS: Final[int] = features.MIN_CONTENT_CHARS
MAX_CONTENT_CHARS: Final[int] = features.MAX_CONTENT_CHARS


class ContentDraft(BaseModel):
    """后端要填的草稿：收据行那一段。

    `text` 是**自然语言**（收据行是长尾的、由模型写）；结构化事实（归类表、上限表、档位表）
    全由组装器渲染，模型碰不到。
    """

    text: str = Field(
        description="收据行：商户或条目名 + 金额，一行写完，不要标题、不要分点、不要围栏"
    )


@dataclass(frozen=True)
class ParsedContent:
    """从 state 反解出来的、纯代码看得见的部分。

    字段：`categories` 归类表里列出的类别编号；`caps` 上限表里列出的类别编号；`tiers` 档位表
    的行数；`text` 收据行（认不出来给 None）。
    """

    categories: tuple[str, ...]
    caps: tuple[str, ...]
    tiers: int
    text: str | None

    @property
    def found(self) -> bool:
        """输入：无。输出：收据行那一块认出来了没有。"""
        return self.text is not None


def category_name(loc: dict[str, JsonValue], cid: str) -> str:
    """输入：语言字典、类别编号。输出：该类别在这一语言里的名字。

    预期行为：缺项当场报错——归类表、题面与提示词都靠它，缺一个字段模型就会拿到半句话。
    """
    table = loc.get("category")
    entry = table.get(cid) if isinstance(table, dict) else None
    name = entry.get("name") if isinstance(entry, dict) else None
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"语言文件里缺类别 {cid} 的名字")
    return name


def category_names(loc: dict[str, JsonValue]) -> dict[str, str]:
    """输入：语言字典。输出：{类别编号: 名字}。预期行为：四个类别一个都不能少。"""
    return {cid: category_name(loc, cid) for cid in features.CATEGORY_IDS}


def cap_text(cid: str) -> str:
    """输入：类别编号。输出：该类单笔上限的金额字符串（分转成两位小数）。"""
    cents = features.CAPS_CENTS[cid]
    return f"{cents // 100}.{cents % 100:02d}"


def decoy_word_for(case: ReceiptCase, loc: dict[str, JsonValue]) -> str:
    """输入：注入特征、语言字典。输出：这一条要照抄的那个诱饵词。

    预期行为：取**下一个类别**（按 `CATEGORY_IDS` 环形）的第一个诱饵词——确定性、可重算，
    而且一定不是本条真值那一类的词（否则读者分不清「真命中」与「诱饵」）。
    """
    index = features.CATEGORY_IDS.index(case.category)
    other = features.CATEGORY_IDS[(index + 1) % len(features.CATEGORY_IDS)]
    return features.decoy_table(loc)[other][0]


def _tier_copy(loc: dict[str, JsonValue], tier: str) -> tuple[str, str]:
    """输入：语言字典、档位键。输出：(档位名, 档位规则)。预期行为：缺字段当场报错。"""
    table = loc.get("tier")
    entry = table.get(tier) if isinstance(table, dict) else None
    if not isinstance(entry, dict):
        raise ValueError(f"语言文件里缺档位 {tier} 的文案")
    name = entry.get("name")
    rule = entry.get("rule")
    if not isinstance(name, str) or not isinstance(rule, str):
        raise ValueError(f"档位 {tier} 的文案缺 name / rule 字段")
    return name, rule


def tier_name(loc: dict[str, JsonValue], tier: str) -> str:
    """输入：语言字典、档位键。输出：该档位的名字（题面与审计共用）。"""
    return _tier_copy(loc, tier)[0]


def build_prompt(loc: dict[str, JsonValue], case: ReceiptCase) -> str:
    """组装提示词。

    输入：语言字典、注入特征。输出：完整提示词（不含答案）。
    预期行为：**每一条硬性要求都写清楚**——必须出现哪一个类别的哪个关键词（逐个列出来，
    原样照抄）、必须原样写出哪个金额、不许出现别的类别关键词、不许出现类别名本身、
    `near` 时把诱饵词原样给出、长度区间、输出前自查。这些就是「受控合成」的规格，判据全在
    `check_draft` 里机械复核，模型写歪了当场重渲染。

    为什么必须逐个列出关键词：它们**就是政策**（哪些说法算哪一类是表定的），模型自己编一个
    同义词不算命中——把表列出来，它才有得抄。
    """
    prompt = loc["prompt"]
    keywords = features.keyword_table(loc)
    names = category_names(loc)
    lines = [
        str(prompt["intro"]).format(min=MIN_CONTENT_CHARS, max=MAX_CONTENT_CHARS),
        str(prompt["lang_rule"]).format(language=loc["language_name"]),
        str(prompt["cat_header"]).format(category=names[case.category]),
    ]
    lines.extend(str(prompt["cat_line"]).format(word=word) for word in keywords[case.category])
    lines.append(str(prompt["amount_line"]).format(amount=case.amount))
    lines.append(str(prompt["miss_header"]))
    for cid in features.CATEGORY_IDS:
        if cid == case.category:
            continue
        lines.extend(
            str(prompt["miss_line"]).format(word=word, category=names[cid])
            for word in keywords[cid]
        )
    lines.append(str(prompt["name_rule"]).format(names="、".join(names.values())))
    if case.decoy == features.DECOY_NEAR:
        lines.append(str(prompt["decoy_line"]).format(decoy=decoy_word_for(case, loc)))
    lines.append(str(prompt["extra_rule"]))
    lines.append(str(prompt["length_rule"]).format(min=MIN_CONTENT_CHARS, max=MAX_CONTENT_CHARS))
    lines.append(str(prompt["self_check"]))
    if case.decoy == features.DECOY_NEAR:
        # 诱饵词放在**最后一句**再点一遍：它是模型唯一「没有语义动机去写」的词，实测 en 侧漏抄
        # 是这一族最常见的渲染期拒稿原因；末尾重述比中段那句提示的召回高（同一段提示词里，
        # 越靠后的要求越容易被当成本次的硬约束）。
        lines.append(str(prompt["decoy_check"]).format(decoy=decoy_word_for(case, loc)))
    lines.append(str(prompt["output"]))
    return "\n".join(lines)


def build_task(spec: ReceiptCatSpec, loc: dict[str, JsonValue]) -> GenerationTask:
    """输入：真值规格、语言字典。输出：渲染任务。

    预期行为：任务 id 由规格算（断点续跑按它查进度），`spec` 里带上注入特征——后端实现与假
    后端若想「照着计划答」，不必再解析提示词。
    """
    return GenerationTask(
        task_id=spec.render_task_id,
        kind="choice",
        domain="receiptcat",
        prompt=build_prompt(loc, spec.case),
        response_model=ContentDraft,
        seeds=(),
        spec={
            "case": spec.case.as_dict(),
            "amount": spec.case.amount,
            "category": spec.case.category,
            "band": spec.case.band,
            "decoy": spec.case.decoy,
        },
    )


def check_draft(draft: ContentDraft, case: ReceiptCase, loc: dict[str, JsonValue]) -> None:
    """校验「收据行」是否兑现了注入特征。

    输入：后端草稿、注入特征、语言字典。输出：无（不合格抛 `DraftError`，管线据此重渲染）。
    预期行为：逐条查——

    1. 正文非空、不带围栏、长度落在 `MIN_CONTENT_CHARS`~`MAX_CONTENT_CHARS`；
    2. **按字面算出来的命中类别恰好是真值那一类**（0 个或 2 个以上都不行，理由里说清差在哪）；
    3. **注入金额原样出现且唯一**——档位就是拿它与上限比出来的，另编一个金额推不出计划；
    4. 诱饵：`near` 时正文里必须真有那个诱饵词，`none` 时一处诱饵词都不许有；
    5. 正文里不许出现**任何类别名**（否则这道题退化成抄表）。

    第 2 条是这一族的核心：命中类别就是金标签，它由关键词表从文本算得出来，所以「模型写歪了」
    在渲染期就能拦住，不用等判别器。
    """
    text = draft.text.strip()
    if not text:
        raise DraftError("收据行是空的")
    if "```" in text:
        raise DraftError(f"收据行里带围栏：{text[:60]!r}")
    if len(text) < MIN_CONTENT_CHARS:
        raise DraftError(f"收据行太短（至少 {MIN_CONTENT_CHARS} 个字符）：{text!r}")
    if len(text) > MAX_CONTENT_CHARS:
        raise DraftError(f"收据行太长（最多 {MAX_CONTENT_CHARS} 个字符，实际 {len(text)}）")
    hits = features.categories_hit(text, features.keyword_table(loc))
    if hits != (case.category,):
        names = category_names(loc)
        got = "、".join(names[cid] for cid in hits) or "哪一类都不算"
        raise DraftError(f"收据行按字面算归到「{got}」，计划要的是「{names[case.category]}」")
    amounts = features.amounts_in(text)
    if amounts != (case.amount,):
        raise DraftError(f"收据行里的金额是 {list(amounts) or '一个都没有'}，计划要的是 {case.amount}")
    near = features.decoy_hit(text, features.decoy_table(loc))
    if case.decoy == features.DECOY_NEAR:
        word = decoy_word_for(case, loc)
        if word not in text:
            raise DraftError(f"计划要埋诱饵（原样照抄 {word!r}），收据行里没有")
    elif near:
        raise DraftError("收据行里出现了诱饵词，但计划没要")
    for cid in features.CATEGORY_IDS:
        name = category_name(loc, cid)
        if name in text:
            raise DraftError(f"收据行里出现了类别名 {name!r}——那这道题就成了抄表")


def to_state(draft: ContentDraft, loc: dict[str, JsonValue]) -> str:
    """输入：后端草稿、语言字典。输出：契约允许的 state（多行文本）。

    预期行为：归类表 + 上限表 + 档位表 + 收据行。**三张表都印在 state 里**，所以「这条条目
    归哪一类、金额落哪一档、该走哪一档审核」是自明的——考生只读 state 与题面就能答，不必
    知道外部政策（范式 §9.8）。
    """
    marks = loc["state"]
    names = category_names(loc)
    keywords = features.keyword_table(loc)
    parts = [str(marks["cat_mark"])]
    for cid in features.CATEGORY_IDS:
        parts.append(
            f"{LINE_PREFIX}{names[cid]}{marks['cat_sep']}{marks['word_sep'].join(keywords[cid])}"
        )
    parts.append(str(marks["cap_mark"]))
    parts.extend(
        f"{LINE_PREFIX}{names[cid]}{marks['cap_sep']}{cap_text(cid)}"
        for cid in features.CATEGORY_IDS
    )
    parts.append(str(marks["tier_mark"]))
    for tier in features.TIERS:
        name, rule = _tier_copy(loc, tier)
        parts.append(f"{LINE_PREFIX}{name}{marks['tier_sep']}{rule}")
    parts.append(str(marks["content_mark"]))
    parts.append(draft.text.strip())
    return "\n".join(parts)


def parse_with(state: str, loc: dict[str, JsonValue]) -> ParsedContent:
    """输入：state、语言字典。输出：按这一份语言反解的结果。

    预期行为：认归类表抬头、上限表抬头、档位表抬头、收据行抬头；归类表与上限表里带行前缀的
    那些行取**类别名**（按各自的分隔符切），再按名字换回类别编号——**编号是结构键**，所以
    换语言不会改编号；收据行抬头之后的所有行原样拼回收据行，否则正文里提到「归类表」三个字
    就会被数成表。
    """
    marks = loc["state"]
    cat_mark = str(marks["cat_mark"])
    cap_mark = str(marks["cap_mark"])
    tier_mark = str(marks["tier_mark"])
    content_mark = str(marks["content_mark"])
    by_name = {name: cid for cid, name in category_names(loc).items()}
    categories: list[str] = []
    caps: list[str] = []
    tiers = 0
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
        if text.startswith(cat_mark):
            section = "cat"
            continue
        if text.startswith(cap_mark):
            section = "cap"
            continue
        if text.startswith(tier_mark):
            section = "tier"
            continue
        if not text.startswith(LINE_PREFIX):
            continue
        rest = text[len(LINE_PREFIX) :]
        if section == "cat":
            name = rest.partition(str(marks["cat_sep"]))[0].strip()
            if name in by_name:
                categories.append(by_name[name])
        elif section == "cap":
            name = rest.partition(str(marks["cap_sep"]))[0].strip()
            if name in by_name:
                caps.append(by_name[name])
        elif section == "tier":
            tiers += 1
    content = "\n".join(body).strip() if body else None
    return ParsedContent(categories=tuple(categories), caps=tuple(caps), tiers=tiers, text=content)


def detect_locale(state: object) -> str | None:
    """输入：state。输出：能认出收据行抬头的那一种语言（认不出给 None）。

    预期行为：审计拿它按 state **自己**的语言取规范文案——一次 run 里混了 zh / en 时，用调用方
    给的那一种去比会平白报一堆「文案对不上」。

    为什么认的是**收据行抬头**而不是类别编号：编号（C1…C4）是结构键，与语言无关，拿它当判据会
    把别的语言的 state 也抢过来（这条坑在 lint 上踩过一次，见范式 §9.22）；收据行抬头是渲染时
    按语言写死的、整段最后那一块的开头，才分得开。
    """
    if not isinstance(state, str):
        return None
    for lang in locales.available():
        if parse_with(state, locales.load(lang)).found:
            return lang
    return None


def parse_state(state: object) -> ParsedContent:
    """输入：state。输出：纯代码能反解的部分（三张表 + 收据行）。

    预期行为：与语言无关——把各语言的标记都试一遍，认出收据行抬头的那一份就用它；一份都认不
    出来时返回空结果（`text` 为 None）。
    """
    if not isinstance(state, str):
        return ParsedContent(categories=(), caps=(), tiers=0, text=None)
    lang = detect_locale(state)
    if lang is None:
        return ParsedContent(categories=(), caps=(), tiers=0, text=None)
    return parse_with(state, locales.load(lang))
