"""社交帖子分诊的特征层：信号判据 + 帖子计划 + 命中标签（纯函数，零依赖）。

社区里 TradingAgents（★108,478）「Jev 筛查社交帖子与公司的相关性及情绪」、sift（★0）「重排
搜索结果并折叠销售页面」都是同一个形状：**一条用户产出的短文本进来 → 判它发出了哪些可机械
核对的信号**。这里把「信号」做成可按字面判的四条判据（`SIGNAL_IDS`）——**标签先行**：先定
命中哪几条，再由本模块的纯函数算出结论。

对应 PLAN 3.2「特征注入式」：一条样本由两个**注入特征**描述——`signals` 该命中哪几条判据、
`decoy` 帖子里要不要埋一个「像代码、按字面不算」的小写 cashtag。判据表由代码写、帖子正文由
模型写，两边都不越界（范式 §9.22 / §9.23 同一条路子）。

**为什么判据写成正则**：官方形态就是「有没有给代码、有没有给价位、有没有给涨跌幅、有没有外链」
这种按字面能判的规则。写成正则之后，「命中哪几条」就是**算**出来的，模型碰不到——它只负责写
一条读得通、长尾的帖子。真要做语义（情绪是看涨还是看跌），那是另一族 workflow 的事，不该混进
一条机械判据里（范式 §9.13：写不出强校验就别硬编一个看起来像校验的东西）。

**诱饵（`decoy=lower`）是这一族最要紧的设计**：`$acme` 看着就是一只股票的代码，可按判据
**只认大写字母**，所以不算命中 S1。这正是「见 `$` 就当代码」的捷径要压掉的东西，也是 test_ood
的留出取值（训练侧一次都见不到诱饵）。

本模块有**两条独立路径**：`label_for` 从计划推标签（构造侧），`observed_labels` 从文本重推
标签（审计侧，走 `signal_set` 的正则）。两者对不上就说明渲染没兑现计划。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Final

from jc.cover import cover_dims
from jc.jsontypes import JsonValue

#: 四条信号判据的编号（**结构键**，不是文案：判据名与判据原文全在 `locales/`）。
SIGNAL_IDS: Final[tuple[str, ...]] = ("S1", "S2", "S3", "S4")

#: 判据编号 → 正则。**这是全 workflow 唯一的判据实现**：构造侧、渲染期、判别器、审计共用。
#: 每条都只认字面，四条互不包含（`S2` 的两位小数不会被 `S3` 的百分号认走，反之亦然）。
#: `S1` 尾部那个否定前瞻是必须的：正则不锚定尾部时，`$ABCDEF` 会靠前五个字母蹭成命中，而判据
#: 原文说的是「2~5 个」——口径对不上，模型按判据写反而被判错。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    # S1：股票代码（cashtag）——**只认大写字母**，小写那种是诱饵。
    "S1": re.compile(r"\$[A-Z]{2,5}(?![A-Za-z])"),
    # S2：价位——一到五位整数 + 两位小数（像 123.45、1200.00）。
    "S2": re.compile(r"[0-9]{1,5}\.[0-9]{2}"),
    # S3：涨跌幅——百分号，允许一位以上小数（像 12%、12.5%）。
    "S3": re.compile(r"[0-9]{1,3}(\.[0-9]+)?%"),
    # S4：外链。
    "S4": re.compile(r"https?://|www\.", re.IGNORECASE),
}

#: 帖子里要不要埋诱饵。`lower` 的那个 cashtag **是小写的**，按判据（只认大写）不算命中 S1。
DECOY_NONE: Final[str] = "none"
DECOY_LOWER: Final[str] = "lower"
DECOYS: Final[tuple[str, ...]] = (DECOY_NONE, DECOY_LOWER)

#: 诱饵的形状：`$` + 2~5 个小写字母。**只认小写**，所以任何大写 cashtag 都不会被它误认；
#: 尾部的否定前瞻与 `S1` 用的是同一把尺子——诱饵与真命中只差大小写这一件事，长度口径必须一致。
DECOY_PATTERN: Final[re.Pattern[str]] = re.compile(r"\$[a-z]{2,5}(?![A-Za-z])")

#: 提示词里**原样给出**的诱饵写法。模型不会自发写出「小写 cashtag」这种形状（它会写 `$ACME`），
#: 所以逐字给一个；它必须同时满足两件事：匹配 `DECOY_PATTERN`（算是诱饵），且**不触发 S1**
#: （没有大写字母）。`tests/workflows/posttriage/test_features.py` 把这两条钉在数据上。
DECOY_EXAMPLE: Final[str] = "$acme"

#: 处置档位（**结构键**，文案全在 `locales/`）。阈值是 0 / 1 / 2 / 3 条及以上——为什么是四档而
#: 不是三档：三档（0 / 1 / 2 以上）会让「升级」一档按构造占 11/16，答案分布天然偏斜到 69%，
#: 均衡闸会把整批标成失衡。四档把占比摊成 1/4/6/5，最大值 37.5%（范式 §9.25 关于「结构性偏斜」
#: 的那一段）。
ROUTE_LEVELS: Final[tuple[str, ...]] = ("ignore", "watch", "report", "escalate")

#: 帖子的最短长度（太短读不出「像不像一条正经帖子」）。
MIN_CONTENT_CHARS: Final[int] = 20

#: 帖子的最长长度（再长就只是灌水，题面也没必要）。
MAX_CONTENT_CHARS: Final[int] = 400


def _all_signal_sets() -> tuple[str, ...]:
    """输入：无。输出：16 个命中组合的规范串（空串在前，然后单条、两条……）。

    预期行为：顺序稳定（case 矩阵逐字节可重建），每条判据在 16 组里各出现 8 次——这是 `flag`
    那一族的答案占比按构造就是 50% 的原因。
    """
    out: list[str] = []
    for size in range(len(SIGNAL_IDS) + 1):
        for combo in combinations(SIGNAL_IDS, size):
            out.append("+".join(combo))
    return tuple(out)


#: 全部命中组合（`cover_dims` 的取值表）。
SIGNAL_SETS: Final[tuple[str, ...]] = _all_signal_sets()

#: 注入维度表。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {"signals": SIGNAL_SETS, "decoy": DECOYS}


def signal_set(text: str) -> tuple[str, ...]:
    """输入：帖子正文。输出：命中的判据编号（按 `SIGNAL_IDS` 顺序）。

    预期行为：逐条按字面判，不做任何语义推断——这正是这一族要训的东西。
    """
    return tuple(signal for signal in SIGNAL_IDS if PATTERNS[signal].search(text))


def signals_of(text: str) -> str:
    """输入：帖子正文。输出：命中组合的规范串（与 `SIGNAL_SETS` 里的写法一致）。"""
    return "+".join(signal_set(text))


def parse_signals(value: str) -> tuple[str, ...]:
    """输入：命中组合的规范串。输出：判据编号元组。预期行为：空串给空元组。"""
    return tuple(part for part in value.split("+") if part)


def has_decoy(text: str) -> bool:
    """输入：帖子正文。输出：正文里有没有那个「小写 cashtag」形状的诱饵。"""
    return DECOY_PATTERN.search(text) is not None


def route_level(count: int) -> str:
    """输入：命中条数。输出：处置档位名。

    预期行为：0 → `ignore`、1 → `watch`、2 → `report`、3 条及以上 → `escalate`。阈值表印在
    state 里（`render.to_state`），题目选项与 state 里那张表取的是**同一份文案**，两边不可能漂移。
    """
    return ROUTE_LEVELS[min(count, len(ROUTE_LEVELS) - 1)]


@dataclass(frozen=True)
class PostTriageCase:
    """一条样本的注入特征：该命中哪几条判据、要不要埋诱饵。"""

    signals: str
    decoy: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"signals": self.signals, "decoy": self.decoy}

    def fingerprint(self) -> str:
        """输入：无。输出：case 指纹。预期行为：同一 case 任何进程都得到同一串。"""
        return "|".join(f"{key}={value}" for key, value in self.as_dict().items())

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.signals or '无信号'}-{self.decoy}"

    def hits(self) -> tuple[str, ...]:
        """输入：无。输出：该命中的判据编号元组。"""
        return parse_signals(self.signals)


@dataclass(frozen=True)
class PostTriagePlan:
    """一条样本的分诊计划：命中哪几条 + 组合题的三个干扰项。

    字段：`signals` 该命中的判据；`distractors` 组合题的三个干扰组合（与 `signals` 不同、彼此
    也不同，且都取「与真值只差一两处」的邻近组合，题目才有区分度）。
    """

    signals: tuple[str, ...]
    distractors: tuple[tuple[str, ...], ...]

    @property
    def count(self) -> int:
        """输入：无。输出：命中条数。"""
        return len(self.signals)


def _distractors(truth: tuple[str, ...], rng: random.Random) -> tuple[tuple[str, ...], ...]:
    """输入：真值命中组合、随机源。输出：三个干扰组合。

    预期行为：先按「与真值只差几处」排序，再从最近的一批里抽三个——差得越少越难分辨，题目才
    有区分度；抽完按 (条数, 编号) 排好，保证同一入参必得同一结果。
    """
    truth_set = set(truth)
    near = sorted(
        (
            combo
            for size in range(len(SIGNAL_IDS) + 1)
            for combo in combinations(SIGNAL_IDS, size)
            if set(combo) != truth_set
        ),
        key=lambda combo: (len(set(combo) ^ truth_set), combo),
    )
    picked = rng.sample(near[:8], 3)
    return tuple(sorted(picked, key=lambda combo: (len(combo), combo)))


def plan_for(case: PostTriageCase, phrasing: int) -> PostTriagePlan:
    """输入：case、问法编号。输出：分诊计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里都
    得到同一份计划；两种问法会抽到不同的干扰组合，不会渲染出结构相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    return PostTriagePlan(
        signals=case.hits(), distractors=_distractors(case.hits(), random.Random(seed))
    )


@dataclass(frozen=True)
class PostTriageLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`signals` 命中的判据编号（按 `SIGNAL_IDS` 顺序）。
    """

    signals: tuple[str, ...]

    @property
    def count(self) -> int:
        """输入：无。输出：命中条数。"""
        return len(self.signals)

    def contains(self, signal: str) -> bool:
        """输入：判据编号。输出：这一条是否命中。"""
        return signal in self.signals

    @property
    def route(self) -> str:
        """输入：无。输出：按处置阈值算出来的档位名（`features.route_level`）。"""
        return route_level(self.count)


def label_for(plan: PostTriagePlan) -> PostTriageLabels:
    """输入：分诊计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`render.check_draft` 负责
    保证文本真的兑现这份计划（按字面算出来的命中组合与计划逐条相同）。
    """
    return PostTriageLabels(signals=plan.signals)


def observed_labels(text: str) -> PostTriageLabels:
    """输入：帖子正文。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认文本、只认字面。两者对不上就说明渲染没
    兑现计划，审计当场就能看见。
    """
    return PostTriageLabels(signals=signal_set(text))


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[PostTriageCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`PostTriageCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [
        PostTriageCase(signals=str(row["signals"]), decoy=str(row["decoy"])) for row in rows
    ]
