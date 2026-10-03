"""日志行分诊（logtriage）的特征层：噪声判据 + 日志行计划 + 命中标签（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条「这条日志行为什么不值得打扰人」的样本由两个**注入特征**
描述——`hits` 该命中哪几条判据、`source` 这条日志从哪种通道来（应用 / 基础设施 / 访问日志）。
**标签先行**：先定这两个取值，再由本模块的纯函数算出结论。

**为什么判据写成机械短语表**：真实日志里「不值得进告警通道」的形状是有固定形状的——心跳与
健康检查、生命周期与配置播报、调试与追踪级、已自愈的重试与降级。把这些形状写成正则之后，
「命中哪几条」就是**算**出来的，模型碰不到——它只负责写一行读得通、长尾的日志。判据表由代码
写、内容由模型写，两边都不越界（范式 §9.22 / §9.23 同一条路子）。

**为什么要这一族**：社区里 `jevlogs`（OpenTelemetry 的 Jev 日志分诊）把它明写成「在昂贵的 LLM
分析**之前**先评估信号」，`Stratum` 用 Jev 做日志异常检测与根因分析，`agent-jev` /
`claude-code-trace` 则把「日志 / trace」直接当作喂给 Jev 的非结构化状态。共同点是一条**单行**
日志进来、要一个**立刻能落地**的处置（静默 / 聚合 / 暂存 / 告警），而不是让 LLM 去通读全量。

**与 `guardrail` / `mail` 的区别**：guardrail 判的是「一条用户发的帖子有没有踩政策条款」、
mail 判的是「一封邮件是不是钓鱼」，两者的判据都是**人际内容**的形状（联系方式、外链、@、
发件人身份）。本 workflow 判的是**机器日志**的形状（级别词、探测路径、生命周期动词、重试与
退避），判据空间、处置语义与素材形态都不一样，不重复。

**`source` 是这一族的留出维**：它**只影响渲染、不影响标签**——同一份命中组合写在应用日志、
容器编排日志还是访问日志里，判据算出来的答案一个字都不差。训练侧只见过 `app` / `infra`，
留出侧是 `access`，用来验「读法不靠来源先验」。

本模块有**两条独立路径**：`label_for` 从计划推标签（构造侧），`observed_labels` 从文本重推
标签（审计侧，走 `hit_set` 的正则）。两者对不上就说明渲染没兑现计划。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Final

from jc.contract import parse_hits
from jc.cover import Fingerprinted, all_hit_sets, cover_dims, near_miss_combos
from jc.jsontypes import JsonValue

#: 四条噪声判据的编号（**结构键**，不是文案：判据名与判据原文全在 `locales/`）。
TAG_IDS: Final[tuple[str, ...]] = ("N1", "N2", "N3", "N4")

#: 判据编号 → 判据。**这是全 workflow 唯一的判据实现**：构造侧、渲染期、判别器、审计共用。
#: 每条都只认字面：
#: - `N1` 心跳与健康检查（心跳 / 探活 / `healthz` / `readiness` 这类探测形状）；
#: - `N2` 生命周期与配置播报（正在启动 / 开始监听 / 正在关闭 / 配置已加载）；
#: - `N3` 调试与追踪级（`DEBUG` / `TRACE` / `verbose` / 调试）；
#: - `N4` 已自愈的重试与降级（重试中 / 退避 / 降级到 / `retrying` / `fell back` / 缓存未命中）。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "N1": re.compile(
        r"(心跳|健康检查|探活|存活探测|就绪探测|保活)"
        r"|\b(heartbeat|health[ _-]?check|healthz|readiness|liveness|keep-?alive)\b",
        re.IGNORECASE,
    ),
    "N2": re.compile(
        r"(正在启动|启动完成|已就绪|开始监听|已停止|正在关闭|优雅退出|加载配置|配置已加载)"
        r"|\b(starting|started|listening on|shutting down|graceful shutdown|loaded config"
        r"|config loaded)\b",
        re.IGNORECASE,
    ),
    "N3": re.compile(
        r"(调试|详细日志|追踪)"
        r"|\b(debug|trace|verbose)\b",
        re.IGNORECASE,
    ),
    "N4": re.compile(
        r"(重试中|正在重试|退避|降级到|已降级|回退到|缓存未命中|已恢复)"
        r"|\b(retrying|retry attempt|backoff|fell back|falling back|degraded to|cache miss)\b",
        re.IGNORECASE,
    ),
}

#: 日志通道（**结构键**，文案在 `locales/` 的 `[source.*]`）。
#:
#: 这一维**只影响渲染、不影响标签**：同一份命中组合写在应用日志、编排日志还是访问日志里，
#: 判据算出来的答案一个字都不差。它正是 test_ood 的留出维——训练侧只见过 `app` / `infra`，
#: 留出侧是 `access`，用来验「读法不靠来源先验」（范式 §9.25 那条「只影响渲染的那一维」）。
SOURCE_APP: Final[str] = "app"
SOURCE_INFRA: Final[str] = "infra"
SOURCE_ACCESS: Final[str] = "access"
SOURCES: Final[tuple[str, ...]] = (SOURCE_APP, SOURCE_INFRA, SOURCE_ACCESS)

#: 处置档位（**结构键**，文案全在 `locales/`）。阈值是 0 条 / 1 条 / 2 条 / 3 条及以上——
#: 语义是「噪声理由越多越不该打扰人」：0 条立刻告警、1 条聚合后告警、2 条暂存人工抽查、
#: 3 条及以上直接静默。四档而不是三档的理由与 `promptinject` 相同：三档会让「静默」一档按构造
#: 占 11/16（69%），均衡闸会把整批标成失衡。四档把占比摊成 1/4/6/5。
HANDLING_LEVELS: Final[tuple[str, ...]] = ("alert", "aggregate", "hold", "silence")

#: 日志行的长度上下限。太短读不出「像不像一行真日志」，太长就只是灌水。
MIN_CONTENT_CHARS: Final[int] = 30

#: 上限。比 `promptinject` 稍宽：访问日志那一档自带时间戳、路径与状态码，天然更长。
MAX_CONTENT_CHARS: Final[int] = 420


#: 全部命中组合（`cover_dims` 的取值表）。
HIT_SETS: Final[tuple[str, ...]] = all_hit_sets(TAG_IDS)

#: 维度表。全空间 16 × 3 = 48 组。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {"hits": HIT_SETS, "source": SOURCES}


def hit_set(text: str) -> tuple[str, ...]:
    """输入：日志行。输出：命中的判据编号（按 `TAG_IDS` 顺序）。"""
    return tuple(tag for tag in TAG_IDS if PATTERNS[tag].search(text))


def hits_of(text: str) -> str:
    """输入：日志行。输出：命中组合的规范串（`"N1+N4"` / `""`）。

    预期行为：与 `hit_set` 是同一件事的两种表示；`hits_of` 给比较用（`render.check_draft` 拿它
    与计划逐字比），`hit_set` 给取元素用。
    """
    return "+".join(hit_set(text))


def handling_level(count: int) -> str:
    """输入：命中条数。输出：处置档位名。

    预期行为：0→alert、1→aggregate、2→hold、3 及以上→silence（最后一档吃掉所有更大的数）。
    """
    return HANDLING_LEVELS[min(count, len(HANDLING_LEVELS) - 1)]


@dataclass(frozen=True)
class LogTriageCase(Fingerprinted):
    """一个注入特征组合：该命中哪几条判据 + 日志从哪种通道来。

    字段：`hits` 命中组合的规范串；`source` 通道（`SOURCES` 里的一个）。
    """

    hits: str
    source: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"hits": self.hits, "source": self.source}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.hits or '无命中'}-{self.source}"

    def tags(self) -> tuple[str, ...]:
        """输入：无。输出：该命中的判据编号元组。"""
        return parse_hits(self.hits)


@dataclass(frozen=True)
class LogTriagePlan:
    """一条样本的分诊计划：命中哪几条 + 组合题的三个干扰项。

    字段：`hits` 该命中的判据；`distractors` 组合题的三个干扰组合（与 `hits` 不同、彼此也不同，
    且都取「与真值只差一两处」的邻近组合，题目才有区分度）。
    """

    hits: tuple[str, ...]
    distractors: tuple[tuple[str, ...], ...]

    @property
    def count(self) -> int:
        """输入：无。输出：命中条数。"""
        return len(self.hits)


def plan_for(case: LogTriageCase, phrasing: int) -> LogTriagePlan:
    """输入：case、问法编号。输出：分诊计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里都
    得到同一份计划；两种问法会抽到不同的干扰组合，不会渲染出结构相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    return LogTriagePlan(
        hits=case.tags(), distractors=near_miss_combos(TAG_IDS, case.tags(), random.Random(seed))
    )


@dataclass(frozen=True)
class LogTriageLabels:
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


def label_for(plan: LogTriagePlan) -> LogTriageLabels:
    """输入：分诊计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`render.check_draft`
    负责保证文本真的兑现这份计划（按字面算出来的命中组合与计划逐条相同）。
    """
    return LogTriageLabels(hits=plan.hits)


def observed_labels(text: str) -> LogTriageLabels:
    """输入：日志行。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认文本、只认字面。两者对不上就说明渲染没
    兑现计划，审计当场就能看见。
    """
    return LogTriageLabels(hits=hit_set(text))


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[LogTriageCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`LogTriageCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [LogTriageCase(hits=str(row["hits"]), source=str(row["source"])) for row in rows]
