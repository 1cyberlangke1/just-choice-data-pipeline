"""数据库变更评审（sqlreview）的特征层：四条判据 + 变更场景 + 命中标签（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条「这次数据库变更能不能直接执行」的样本由两个**注入特征**描述
——`hits` 该命中哪几条判据、`workload` 这次变更发生在哪一类业务上（事务型 / 分析型 / 迁移）。
**标签先行**：先定这两个取值，再由本模块的纯函数算出结论。

**为什么判据写成机械短语表**：数据库变更评审在真实流水线里有固定形状——`statement:` 那一行的
写操作没有 `WHERE`（无条件改整张表）、`plan:` 那一行是大表的全表扫描、`ddl:` 那一行的结构变更
不可重入（没有 `IF EXISTS`）、`note:` 那一行的评审备注里留着「待确认」这类没写完的话。把这些
形状写成正则之后，「命中哪几条」就是**算**出来的，模型碰不到——它只负责写一屏读得通的评审记录。

**为什么四条判据各占一整行**（范式 §9.51）：一屏评审记录是**分行**写的（语句 / 执行计划 /
结构变更 / 备注），四条判据各自钉在自己那一行上——`Q1` 只看 `statement:` 行、`Q2` 只看 `plan:`
行、`Q3` 只看 `ddl:` 行、`Q4` 只看 `note:` 行。`adblock` 上量过一次「两条判据落在同一段文字上」
的代价（写手与读者一起丢内层那条），这里从设计上避开：**判据之间连一行都不共用**。

**为什么要这一族**：SQL 是 JEV 生态里最厚的一簇——`pg-jev`（★224）把自然语言查询落到 Postgres
表上、`jevql`（★7）提供原生 SQL 函数 `jev()` / `jev_choice` / `jev_score`、`pg_typesafe`（★79）
与 `vgi-typesafe`（★3）把 choice / noul / score 做成可 join 的表函数。它们共同的落地前提是
**先把变更评审一遍**：这一次 `UPDATE` 会不会改到整张表、这条 `ALTER` 能不能重跑、这份计划里
有没有大表全扫。

**与 `lint` / `review` / `logtriage` 的区别**：`lint` 判的是代码片段的行级规则、`review` 判的是
一次改动清单、`logtriage` 判的是日志行——三者都不看 SQL 语句、执行计划与结构变更。本家判的是
一屏**数据库变更评审记录**，四条判据认的是 SQL 关键字、估算行数的位数与结构变更的可重入性。

**`workload` 是这一族的留出维**：它**只影响渲染、不影响标签**——同一份命中组合写在事务型、分析型
还是迁移场景的评审记录里，判据算出来的答案一个字都不差（四条判据认的全是关键字、数字位数与
符号，与业务无关）。训练侧只见过 `oltp` / `analytics`，留出侧是 `migration`。

本模块有**两条独立路径**：`label_for` 从计划推标签（构造侧），`observed_labels` 从文本重推标签
（审计侧，走 `hit_set` 的正则）。两者对不上就说明渲染没兑现计划。

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

#: 四条「这次数据库变更能不能直接执行」判据的编号（**结构键**，不是文案：判据名与判据原文全在
#: `locales/`）。
TAG_IDS: Final[tuple[str, ...]] = ("Q1", "Q2", "Q3", "Q4")

#: 一屏评审记录的**结构键**：这四个英文键名与它们的冒号是 state 格式的一部分，**不进语言文件**——
#: 渲染、校验、解析、判别器共用这一份清单，四边不可能漂移。键名用英文是因为它们本来就是格式的
#: 一部分（SQL 与执行计划本来就用英文写），只有 `note` 那一行的取值是自然语言。
KEY_LINES: Final[tuple[str, ...]] = ("statement", "plan", "ddl", "note")

#: 判据编号 → 判据。**这是全 workflow 唯一的判据实现**：构造侧、渲染期、判别器、审计共用。
#: 每条都只认字面，而且**各占一整行**（见模块文档）：
#: - `Q1` `statement:` 行里是 `UPDATE` / `DELETE` 且**通篇没有 `WHERE`**（无条件改整张表）；
#: - `Q2` `plan:` 行里是 `Seq Scan`（全表扫描）**且估算行数 `rows=` 至少七位**（大表全扫）；
#: - `Q3` `ddl:` 行里是 `ALTER` / `DROP` 但**没有 `IF EXISTS` / `IF NOT EXISTS`**（不可重入）；
#: - `Q4` `note:` 行里留着没写完的痕迹（TODO / FIXME / 待确认 / 待定 / 存疑 / 需要复核 / 核对）。
#:
#: `Q2` 的「七位」不是拍出来的：`rows=` 后面跟的是十进制整数，没有前导零，「至少七位」与
#: 「≥ 1,000,000」是同一件事——写成 `\d{7,}` 就不用把数字解析出来再比大小。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "Q1": re.compile(
        r"^[^\n]*\bstatement\s*:[^\n]*\b(?:UPDATE|DELETE)\b(?![^\n]*\bWHERE\b)",
        re.IGNORECASE | re.MULTILINE,
    ),
    "Q2": re.compile(
        r"^[^\n]*\bplan\s*:(?=[^\n]*\bSeq\s+Scan\b)(?=[^\n]*\brows=\d{7,}\b)",
        re.IGNORECASE | re.MULTILINE,
    ),
    "Q3": re.compile(
        r"^[^\n]*\bddl\s*:(?=[^\n]*\b(?:ALTER|DROP)\b)(?![^\n]*\bIF\s+(?:NOT\s+)?EXISTS\b)",
        re.IGNORECASE | re.MULTILINE,
    ),
    "Q4": re.compile(
        r"^[^\n]*\bnote\s*:[^\n]*?(?:\bTODO\b|\bFIXME\b|待确认|待定|存疑|需要复核|核对)",
        re.IGNORECASE | re.MULTILINE,
    ),
}

#: 变更场景（**结构键**，文案在 `locales/` 的 `[workload.*]`）。
#:
#: 这一维**只影响渲染、不影响标签**：同一份命中组合写在事务型、分析型还是迁移场景的评审记录里，
#: 判据算出来的答案一个字都不差——四条判据认的是关键字、数字位数与符号，与业务无关。它正是
#: test_ood 的留出维：训练侧只见过 `oltp` / `analytics`，留出侧是 `migration`（范式 §9.25 那条
#: 「只影响渲染的那一维」）。
WORKLOAD_OLTP: Final[str] = "oltp"
WORKLOAD_ANALYTICS: Final[str] = "analytics"
WORKLOAD_MIGRATION: Final[str] = "migration"
WORKLOADS: Final[tuple[str, ...]] = (WORKLOAD_OLTP, WORKLOAD_ANALYTICS, WORKLOAD_MIGRATION)

#: 处置档位（**结构键**，文案全在 `locales/`）。阈值是 0 条 / 1 条 / 2 条 / 3 条及以上——语义是
#: 「这条变更可疑的地方越多，下一步要花的力气越大」：0 条直接执行、1 条先看一遍执行计划、
#: 2 条交 DBA 人工复核、3 条及以上打回重写。四档而不是三档的理由与 `subqa` 相同：三档会让最后
#: 一档按构造占 11/16（69%），均衡闸会把整批标成失衡。四档把占比摊成 1/4/6/5。
ACTS: Final[tuple[str, ...]] = ("apply", "explain_first", "dba_review", "reject")

#: 一屏评审记录的字符数上下限。太短装不下四行（语句 + 计划 + 结构变更 + 一句备注），太长就不是
#: 「一屏」了。
MIN_CONTENT_CHARS: Final[int] = 80

#: 上限。四行材料加各自的取值，再长就不是一次变更的评审记录了。
MAX_CONTENT_CHARS: Final[int] = 600


#: 全部命中组合（`cover_dims` 的取值表）。
HIT_SETS: Final[tuple[str, ...]] = all_hit_sets(TAG_IDS)

#: 维度表。全空间 16 × 3 = 48 组。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {"hits": HIT_SETS, "workload": WORKLOADS}


def hit_set(text: str) -> tuple[str, ...]:
    """输入：一屏评审记录。输出：命中的判据编号（按 `TAG_IDS` 顺序）。"""
    return tuple(tag for tag in TAG_IDS if PATTERNS[tag].search(text))


def hits_of(text: str) -> str:
    """输入：一屏评审记录。输出：命中组合的规范串（`"Q1+Q4"` / `""`）。

    预期行为：与 `hit_set` 是同一件事的两种表示；`hits_of` 给比较用（`render.check_draft` 拿它
    与计划逐字比），`hit_set` 给取元素用。
    """
    return "+".join(hit_set(text))


def missing_keys(text: str) -> tuple[str, ...]:
    """输入：一屏评审记录。输出：缺掉的**结构键**（四个键都在就给空元组）。

    预期行为：键必须**单独占一行**、后面跟冒号（`^\\s*键\\s*:`）。为什么要单占一行：判据是钉在
    行上的（见 `PATTERNS`），键不单独成行的话那几条正则连定位都定不到。渲染期用它拦「少写一行」，
    审计期 `FrameJudge` 再用同一份清单复核一遍。
    """
    lowered = text.lower()
    return tuple(key for key in KEY_LINES if not re.search(rf"(?m)^\s*{key}\s*:", lowered))


def line_value(text: str, key: str) -> str:
    """输入：一屏评审记录、结构键。输出：那一行冒号后面的内容（没有那一行就给空串）。

    预期行为：只取**第一个**匹配到的键行；行内的空白原样保留（取值里可能有空格）。
    """
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*:(.*)$", text)
    return match.group(1).strip() if match else ""


def act_level(count: int) -> str:
    """输入：命中条数。输出：处置档位名。

    预期行为：0→apply、1→explain_first、2→dba_review、3 及以上→reject（最后一档吃掉所有更大的数）。
    """
    return ACTS[min(count, len(ACTS) - 1)]


@dataclass(frozen=True)
class SqlReviewCase(Fingerprinted):
    """一个注入特征组合：该命中哪几条判据 + 这次变更发生在哪一类业务上。

    字段：`hits` 命中组合的规范串；`workload` 变更场景（`WORKLOADS` 里的一个）。
    """

    hits: str
    workload: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"hits": self.hits, "workload": self.workload}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.hits or '无命中'}-{self.workload}"

    def tags(self) -> tuple[str, ...]:
        """输入：无。输出：该命中的判据编号元组。"""
        return parse_hits(self.hits)


@dataclass(frozen=True)
class SqlReviewPlan:
    """一条样本的判定计划：命中哪几条 + 组合题的三个干扰项。

    字段：`hits` 该命中的判据；`distractors` 组合题的三个干扰组合（与 `hits` 不同、彼此也不同，
    且都取「与真值只差一两处」的邻近组合，题目才有区分度）。
    """

    hits: tuple[str, ...]
    distractors: tuple[tuple[str, ...], ...]

    @property
    def count(self) -> int:
        """输入：无。输出：命中条数。"""
        return len(self.hits)


def plan_for(case: SqlReviewCase, phrasing: int) -> SqlReviewPlan:
    """输入：case、问法编号。输出：判定计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里都
    得到同一份计划；两种问法会抽到不同的干扰组合，不会渲染出结构相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    return SqlReviewPlan(
        hits=case.tags(), distractors=near_miss_combos(TAG_IDS, case.tags(), random.Random(seed))
    )


@dataclass(frozen=True)
class SqlReviewLabels:
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
    def act(self) -> str:
        """输入：无。输出：按阈值算出来的档位名（`features.act_level`）。"""
        return act_level(self.count)


def label_for(plan: SqlReviewPlan) -> SqlReviewLabels:
    """输入：判定计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`render.check_draft` 负责保证
    文本真的兑现这份计划（按字面算出来的命中组合与计划逐条相同）。
    """
    return SqlReviewLabels(hits=plan.hits)


def observed_labels(text: str) -> SqlReviewLabels:
    """输入：一屏评审记录。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认文本、只认字面。两者对不上就说明渲染没兑现
    计划，审计当场就能看见。
    """
    return SqlReviewLabels(hits=hit_set(text))


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[SqlReviewCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`SqlReviewCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [SqlReviewCase(hits=str(row["hits"]), workload=str(row["workload"])) for row in rows]
