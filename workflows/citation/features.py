"""引文核查（citation）的特征层：文档结构 + 引文构造计划 + 核查结论（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条「这条引文能在文档里核实吗」的样本由四个**注入特征**描述
——`form` 引文是怎么造出来的、`where` 声称的出处对不对、`place` 那句话落在哪一节、
`target` 落在该节的第几句。**标签先行**：先定这四个取值，再由本模块的纯函数算出结论。

链条（对应官方 `citation_check` cookbook，但**每一步都机械可判**）：

1. **引文在不在文档里**：字符串整句匹配——`altered` 造出来的引文一定不在，另外两种一定在；
2. **在不在它声称的那一节里**：定位——引文真正出自的那一节与声称的那一节比；
3. **结论**：`verified` / `mislocated` / `fabricated`。

引文的**原文**是代码从后端写的文档里**剪**出来的（整句照抄 / 截前半句 / 改一个数字），所以
「引文到底在不在文档里」这件事是**算**出来的，不靠模型自己说；模型只负责写一篇读得通的
短文。文档长什么样（3 节、每节 2 句、每句带一个具体数字）由提示词规定、由
`render.check_draft` 逐条复核。

三种造法：

- `exact`：整句照抄（含句末标点）；
- `partial`：截前半句，切在字符边界上——**是原句的真子串**，所以还在文档里；
- `altered`：把原句里的一个数字换成另一个——**换完一定不在文档里**，这才是「查无此文」。

为什么「改数字」而不是「换一个词」：数字是跨语言都一样的东西，判据（换完之后整句在不在
文档里）也就跨语言都一样；换词要先分词，中文分词的边界会变成判据的一部分。

本模块有**两条独立路径**：`label_for` 从计划推标签（构造侧），`observed_labels` 从文本重推
标签（审计侧）。两者对不上就说明渲染没兑现计划——这正是审计要抓的东西。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Final

from jc.cover import Fingerprinted, cover_dims
from jc.jsontypes import JsonValue

#: 引文的三种造法。
FORM_EXACT: Final[str] = "exact"
FORM_PARTIAL: Final[str] = "partial"
FORM_ALTERED: Final[str] = "altered"
FORMS: Final[tuple[str, ...]] = (FORM_EXACT, FORM_PARTIAL, FORM_ALTERED)

#: 声称的出处与真正出处的关系：`same` 对得上、`other` 对不上。
WHERE_SAME: Final[str] = "same"
WHERE_OTHER: Final[str] = "other"
WHERES: Final[tuple[str, ...]] = (WHERE_SAME, WHERE_OTHER)

#: 那句话落在哪一节（三节的固定顺序）。
PLACES: Final[tuple[str, ...]] = ("first", "middle", "last")
PLACE_INDEX: Final[dict[str, int]] = {"first": 0, "middle": 1, "last": 2}

#: 那句话落在该节的第几句（每节两句）。
TARGETS: Final[tuple[str, ...]] = ("first", "last")
TARGET_INDEX: Final[dict[str, int]] = {"first": 0, "last": 1}

#: 文档的小节数（固定 3）、每节句数（固定 2）与总句数。
SECTIONS: Final[int] = 3
SENTENCES_PER_SECTION: Final[int] = 2
SENTENCE_COUNT: Final[int] = SECTIONS * SENTENCES_PER_SECTION

#: 写短文时的「题材方向」池：spec 只存**下标**（与问法编号同一套约定），每种语言各有一份等长的
#: 词表（`test_locales.py` 锁住）。
#:
#: **来由**（真机扫出来的，范式 §9.57 的同一类坑）：`build_prompt` 原本**一个样本级变量都没有**，
#: 同一切分的每一条提示词**逐字相同**——于是 9 条 en 产物的标题全是同一个字符串
#: （`System Log Report 2026`），另一批 78% 也是同一个；短字段（标题）被模型的先验直接压平。
#: 给每条样本抽一个题材方向，标题与六句话就都有了各自的落点。
TOPIC_POOL: Final[int] = 12
TOPIC_IDS: Final[tuple[int, ...]] = tuple(range(TOPIC_POOL))

#: 三种结论。
VERDICT_VERIFIED: Final[str] = "verified"
VERDICT_MISLOCATED: Final[str] = "mislocated"
VERDICT_FABRICATED: Final[str] = "fabricated"
VERDICTS: Final[tuple[str, ...]] = (
    VERDICT_VERIFIED,
    VERDICT_MISLOCATED,
    VERDICT_FABRICATED,
)

#: 注入维度表（`cover_dims` 的入参）。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {
    "form": FORMS,
    "where": WHERES,
    "place": PLACES,
    "target": TARGETS,
}

#: 每句话里必须有的「一个具体数字」——`altered` 要拿它换掉。
DIGITS: Final[re.Pattern[str]] = re.compile(r"\d+")

#: 截前半句时的最短长度（太短的片段几乎必然在别的小节里也出现，一撞就得重渲染）。
MIN_PARTIAL: Final[int] = 4

#: 每句话的最短长度：太短的句子截不出有意义的「半句」，`partial` 会退化成整句。
MIN_SENTENCE_CHARS: Final[int] = 16


@dataclass(frozen=True)
class CitationCase(Fingerprinted):
    """一条样本的注入特征：造法、出处对不对、那句话落在哪一节的第几句。"""

    form: str
    where: str
    place: str
    target: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {
            "form": self.form,
            "where": self.where,
            "place": self.place,
            "target": self.target,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.form}-{self.where}-{self.place}-{self.target}"


@dataclass(frozen=True)
class CitationPlan:
    """一条样本的引文计划：造法 + 真正出处 + 声称出处。

    字段：`form` 造法；`place` 真正落在哪一节；`target` 落在该节的第几句；
    `true_section` / `claimed` 都是 0 起的节下标。**引文的文本不在这里**——它是 `render`
    阶段从后端写的文档里剪出来的，所以计划与文本、语言都无关。
    """

    form: str
    place: str
    target: str
    true_section: int
    claimed: int

    @property
    def sentence_index(self) -> int:
        """输入：无。输出：被引用的那句话在 6 句里的下标（0 起）。"""
        return PLACE_INDEX[self.place] * SENTENCES_PER_SECTION + TARGET_INDEX[self.target]


def plan_for(case: CitationCase, phrasing: int) -> CitationPlan:
    """输入：case、问法编号。输出：引文计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里
    都得到同一份计划；`where = other` 时「声称出自哪一节」由指纹从另外两节里掷出——不同问法
    拿到不同的落点，两种问法就不会渲染出逐字节相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    rng = random.Random(seed)
    true_section = PLACE_INDEX[case.place]
    if case.where == WHERE_SAME:
        claimed = true_section
    else:
        others = [index for index in range(SECTIONS) if index != true_section]
        claimed = rng.choice(others)
    return CitationPlan(
        form=case.form,
        place=case.place,
        target=case.target,
        true_section=true_section,
        claimed=claimed,
    )


@dataclass(frozen=True)
class CitationLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`verdict` 核查结论；`section` 引文真正出自哪一节（不在文档里时是 None）；
    `in_doc` 引文是否原样出现在文档里；`claimed` 声称出自哪一节。
    """

    verdict: str
    section: int | None
    in_doc: bool
    claimed: int

    @property
    def at_claimed(self) -> bool:
        """输入：无。输出：引文是否**出现在它声称的那一节里**。

        预期行为：这是「引文核查」那一问的判据——既要在文档里，又要落在声称的那一节。
        引文根本不在文档里（`fabricated`）时，声称的出处无从谈起，一律算「不在」。
        """
        return self.in_doc and self.section == self.claimed


def label_for(plan: CitationPlan) -> CitationLabels:
    """输入：引文计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`render.check_draft`
    负责保证文本真的兑现这份计划（引文只出在那一节、`altered` 的引文一处都不出现）。
    """
    if plan.form == FORM_ALTERED:
        return CitationLabels(
            verdict=VERDICT_FABRICATED, section=None, in_doc=False, claimed=plan.claimed
        )
    verdict = VERDICT_VERIFIED if plan.claimed == plan.true_section else VERDICT_MISLOCATED
    return CitationLabels(
        verdict=verdict, section=plan.true_section, in_doc=True, claimed=plan.claimed
    )


def locate(sections: tuple[tuple[str, ...], ...], quote: str) -> int | None:
    """输入：按节切好的句子、引文。输出：引文出自哪一节（不在文档里给 None）。

    预期行为：逐节找「引文是某一句话的子串」；命中多节时取最靠前的那一节——`check_draft`
    已经在渲染期保证了只会命中一节，这里取最靠前只是为了让审计**不炸**。
    """
    hits = locate_sections(sections, quote)
    return hits[0] if hits else None


def locate_sections(sections: tuple[tuple[str, ...], ...], quote: str) -> tuple[int, ...]:
    """输入：按节切好的句子、引文。输出：引文出现过的**全部**节号（按顺序）。

    预期行为：返回元组而不是单个数，是因为「引文只出在那一节」是判据的一部分：半句引文完全
    可能在别的小节里也出现，一旦撞上，那一条的 `section` 答案就没有唯一解。`check_draft`
    拿它当判据；`locate` 拿它的第一个当「出自哪一节」。
    """
    return tuple(
        index
        for index, group in enumerate(sections)
        if any(quote in sentence for sentence in group)
    )


def observed_labels(
    sections: tuple[tuple[str, ...], ...], quote: str, claimed: int
) -> CitationLabels:
    """输入：按节切好的句子、引文、声称的节下标。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认文本，不认计划。两者对不上就说明渲染
    没兑现计划，审计当场就能看见。
    """
    index = locate(sections, quote)
    if index is None:
        return CitationLabels(
            verdict=VERDICT_FABRICATED, section=None, in_doc=False, claimed=claimed
        )
    verdict = VERDICT_VERIFIED if index == claimed else VERDICT_MISLOCATED
    return CitationLabels(verdict=verdict, section=index, in_doc=True, claimed=claimed)


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[CitationCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`CitationCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [
        CitationCase(
            form=str(row["form"]),
            where=str(row["where"]),
            place=str(row["place"]),
            target=str(row["target"]),
        )
        for row in rows
    ]
