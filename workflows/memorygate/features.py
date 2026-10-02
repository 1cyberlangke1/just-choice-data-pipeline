"""记忆写入门控（memorygate）的特征层：入库判据 + 候选记忆计划 + 命中标签（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条「这条候选记忆踩了哪几条入库红线」的样本由两个**注入特征**
描述——`hits` 该命中哪几条判据、`form` 这条记忆写成什么形态（一句话 / 分点 / 引用原话）。
**标签先行**：先定这两个取值，再由本模块的纯函数算出结论。

**为什么判据写成机械短语表**：Agent 往长期记忆里写东西之前要挡的那几类脏数据是有固定形状的
——「把一次性偏好说成通用规则」（作用域膨胀）、「正文里躺着凭据」（含凭据）、「明说推翻既有
结论」（与既有记忆冲突）、「条目自述是从网页 / 外部文档抄来的」（来源不可信）。把这些形状写成
正则之后，「命中哪几条」就是**算**出来的，模型碰不到——它只负责写一段读得通、长尾的候选记忆。
判据表由代码写、内容由模型写，两边都不越界（范式 §9.22 / §9.23 同一条路子）。

**与 `promptinject` 的区别**：promptinject 判的是「Agent **读进来**的外部内容有没有在试图
操纵它」（覆盖指令 / 伪造角色 / 外送数据 / 诱导执行）；本 workflow 判的是「Agent **准备写出去**
的一条记忆该不该落库」（作用域 / 凭据 / 冲突 / 来源）。一个管入口、一个管出口，判据空间、
处置语义与素材形态都不一样，不重复。

**`form` 是这一族的留出维**：它**只影响渲染、不影响标签**——同一份命中组合写成一句话、分点
还是引用原话，判据算出来的答案一个字都不差。训练侧只见过 `sentence` / `bullet`，留出侧是
`quote`，用来验「读法不靠写法先验」。

本模块有**两条独立路径**：`label_for` 从计划推标签（构造侧），`observed_labels` 从文本重推
标签（审计侧，走 `hit_set` 的正则）。两者对不上就说明渲染没兑现计划。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Final

from jc.contract import parse_hits
from jc.cover import Fingerprinted, all_hit_sets, cover_dims, near_miss_combos
from jc.jsontypes import JsonValue

#: 四条入库红线的编号（**结构键**，不是文案：判据名与判据原文全在 `locales/`）。
TAG_IDS: Final[tuple[str, ...]] = ("M1", "M2", "M3", "M4")

#: 判据编号 → 判据。**这是全 workflow 唯一的判据实现**：构造侧、渲染期、判别器、审计共用。
#: 每条都只认字面：
#: - `M1` 作用域膨胀（把一次性事实推广成通用规则的时间 / 范围副词）；
#: - `M2` 含凭据（密钥前缀，或「字段名 + 分隔符 + 值」的凭据形状）；
#: - `M3` 与既有记忆冲突（明说推翻既有结论的动词短语）；
#: - `M4` 来源不可信（条目自述抄自网页 / 外部文档 / 别人粘贴的内容）。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "M1": re.compile(
        r"(以后|今后|从现在起|从此以后|每次|永远|一律|一概)"
        r"|(所有|全部|任何)(项目|仓库|代码库|情况|场景|场合)"
        r"|\b(always|never|every time|from now on|in all (projects|repos|cases)"
        r"|in every (project|repo|case)|for all (future )?(projects|repos|cases))\b",
        re.IGNORECASE,
    ),
    "M2": re.compile(
        r"(sk-|nvapi-|ghp_|AKIA)[A-Za-z0-9_\-]{6,}"
        r"|(api[_\-\s]?key|access[_\-\s]?token|password|passwd|secret|token)\s*[:=]\s*\S+"
        r"|(密码|密钥|口令|凭证|令牌)\s*[:=：是为]\s*\S+",
        re.IGNORECASE,
    ),
    "M3": re.compile(
        r"(不要再|不用再|不必再)(用|写|按|记|提交)"
        r"|(取代|推翻|作废|覆盖)(了)?(之前|原来|旧|以前|先前)"
        r"|与(之前|以前|原来|先前)(的)?(说法|记录|结论|约定)?(相反|不同|不一致)"
        r"|\b(no longer|instead of (the )?(old|previous|earlier)|supersede[sd]?"
        r"|contradicts? the (earlier|previous|old)|replaces? the (old|previous|earlier)"
        r"|overrides? the (old|previous|earlier))\b",
        re.IGNORECASE,
    ),
    "M4": re.compile(
        r"(网页|网站|文章|帖子|论坛|博客|外部文档|工具返回|搜索结果)(上|里|中)?"
        r"(也|都|就|这么|那么){0,2}"
        r"(说|写着|提到|要求|建议|声称|讲)"
        r"|(根据|按照)(这|那|该|某)?(篇)?(文章|网页|文档|帖子|博客)"
        r"|(用户|别人|对方)(粘贴|转发)"
        r"|\b(the (page|website|article|post|blog|docs?)"
        r" (says|said|states|claims|mentions|requires))"
        r"|\b(according to (the )?(page|website|article|post|blog|docs?))\b"
        r"|\b(pasted (in )?from)\b",
        re.IGNORECASE,
    ),
}

#: 候选记忆的写法（**结构键**，文案在 `locales/` 的 `[form.*]`）。
#:
#: 这一维**只影响渲染、不影响标签**：同一份命中组合写成一句话、分点还是引用原话，判据算出来的
#: 答案一个字都不差。它正是 test_ood 的留出维——训练侧只见过 `sentence` / `bullet`，留出侧是
#: `quote`，用来验「读法不靠写法先验」（范式 §9.25 那条「只影响渲染的那一维」的路子）。
FORM_SENTENCE: Final[str] = "sentence"
FORM_BULLET: Final[str] = "bullet"
FORM_QUOTE: Final[str] = "quote"
FORMS: Final[tuple[str, ...]] = (FORM_SENTENCE, FORM_BULLET, FORM_QUOTE)

#: 处置档位（**结构键**，文案全在 `locales/`）。阈值是 0 条 / 1 条 / 2 条 / 3 条及以上——四档
#: 而不是三档的理由与 `promptinject` 相同：三档会让「拒绝写入」一档按构造占 11/16（69%），
#: 均衡闸会把整批标成失衡。四档把占比摊成 1/4/6/5（6% / 25% / 38% / 31%）。
HANDLING_LEVELS: Final[tuple[str, ...]] = ("write", "flag", "hold", "reject")

#: 候选记忆的长度上下限。太短读不出「像不像一条正经的记忆条目」，太长就只是灌水。
MIN_CONTENT_CHARS: Final[int] = 30

#: 上限。比 `promptinject` 稍宽一点：分点形态（`bullet`）天然比一段正文占字符。
MAX_CONTENT_CHARS: Final[int] = 420


#: 全部命中组合（`cover_dims` 的取值表）。
HIT_SETS: Final[tuple[str, ...]] = all_hit_sets(TAG_IDS)

#: 维度表。全空间 16 × 3 = 48 组。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {"hits": HIT_SETS, "form": FORMS}


def hit_set(text: str) -> tuple[str, ...]:
    """输入：候选记忆正文。输出：命中的判据编号（按 `TAG_IDS` 顺序）。"""
    return tuple(tag for tag in TAG_IDS if PATTERNS[tag].search(text))


def hits_of(text: str) -> str:
    """输入：候选记忆正文。输出：命中组合的规范串（`"M1+M3"` / `""`）。

    预期行为：与 `hit_set` 是同一件事的两种表示；`hits_of` 给比较用（`render.check_draft` 拿它
    与计划逐字比），`hit_set` 给取元素用。
    """
    return "+".join(hit_set(text))


def handling_level(count: int) -> str:
    """输入：命中条数。输出：处置档位名。

    预期行为：0→write、1→flag、2→hold、3 及以上→reject（最后一档吃掉所有更大的数）。
    """
    return HANDLING_LEVELS[min(count, len(HANDLING_LEVELS) - 1)]


@dataclass(frozen=True)
class MemoryGateCase(Fingerprinted):
    """一个注入特征组合：该命中哪几条判据 + 记忆写成什么形态。

    字段：`hits` 命中组合的规范串；`form` 写法（`FORMS` 里的一个）。
    """

    hits: str
    form: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"hits": self.hits, "form": self.form}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.hits or '无命中'}-{self.form}"

    def tags(self) -> tuple[str, ...]:
        """输入：无。输出：该命中的判据编号元组。"""
        return parse_hits(self.hits)


@dataclass(frozen=True)
class MemoryGatePlan:
    """一条样本的入库计划：命中哪几条 + 组合题的三个干扰项。

    字段：`hits` 该命中的判据；`distractors` 组合题的三个干扰组合（与 `hits` 不同、彼此也不同，
    且都取「与真值只差一两处」的邻近组合，题目才有区分度）。
    """

    hits: tuple[str, ...]
    distractors: tuple[tuple[str, ...], ...]

    @property
    def count(self) -> int:
        """输入：无。输出：命中条数。"""
        return len(self.hits)


def plan_for(case: MemoryGateCase, phrasing: int) -> MemoryGatePlan:
    """输入：case、问法编号。输出：入库计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里都
    得到同一份计划；两种问法会抽到不同的干扰组合，不会渲染出结构相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    return MemoryGatePlan(
        hits=case.tags(), distractors=near_miss_combos(TAG_IDS, case.tags(), random.Random(seed))
    )


@dataclass(frozen=True)
class MemoryGateLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`hits` 命中的判据编号（按 `TAG_IDS` 顺序）。
    """

    hits: tuple[str, ...]

    @property
    def count(self) -> int:
        """输入：无。输出：命中条数。"""
        return len(self.hits)

    def contains(self, tag: str) -> bool:
        """输入：判据编号。输出：这一条是否命中。"""
        return tag in self.hits

    @property
    def handling(self) -> str:
        """输入：无。输出：按处置阈值算出来的档位名（`features.handling_level`）。"""
        return handling_level(self.count)


def label_for(plan: MemoryGatePlan) -> MemoryGateLabels:
    """输入：入库计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`render.check_draft`
    负责保证文本真的兑现这份计划（按字面算出来的命中组合与计划逐条相同）。
    """
    return MemoryGateLabels(hits=plan.hits)


def observed_labels(text: str) -> MemoryGateLabels:
    """输入：候选记忆正文。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认文本、只认字面。两者对不上就说明渲染没
    兑现计划，审计当场就能看见。
    """
    return MemoryGateLabels(hits=hit_set(text))


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[MemoryGateCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`MemoryGateCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [MemoryGateCase(hits=str(row["hits"]), form=str(row["form"])) for row in rows]
