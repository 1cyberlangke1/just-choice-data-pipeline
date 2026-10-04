"""代理轨迹自评（tracescore）的特征层：信号表 + 评分档 + 轨迹标签（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条「这段代理轨迹推进得怎么样」的样本由两个**注入特征**描述——
`hits` 该命中哪几条行为信号、`source` 这条轨迹从哪种入口来（命令行代理 / 编辑器内代理 /
对话式代理）。**标签先行**：先定这两个取值，再由本模块的纯函数算出结论（该打几分、该走哪档）。

**为什么把信号写成机械词表**：真实轨迹里「推进得好不好」是有固定形状的——跑没跑验证、有没有原地
打转、有没有按观察调整、有没有反复翻旧账。把这些形状写成词表之后，「命中哪几条」就是**算**出来的，
模型碰不到——它只负责写一段读得通、长尾的轨迹。表由代码写、内容由模型写（范式 §9.22 / §9.23）。

**为什么要这一族**：社区里 `claude-code-trace`（★370）拿 Jev 从 Claude Code 会话日志里给「代理
进度、工具使用、焦点、探索、恢复与令牌效率」逐维打分；`summarize`（★223）用 Jev 当**摘要之前的
预过滤器**给转录片段打分；`jevmeter`（★70）逐句评分并渲染成片。共同点是一段**已经发生过的执行
轨迹**进来、要一个**立刻能落地**的结论（继续 / 换方案 / 停下问人），而不是让人把轨迹从头读到尾
再拍脑袋。

**与 `rubric` 的区别**：`rubric` 判的是「一份材料按清单逐条打分」（清单由代码写、材料是任意文本）；
本 workflow 判的是**执行轨迹**这一种材料，计分口径是四条行为信号的加权（跑过验证、按观察调整是
加分项，原地打转、反复翻旧账是扣分项），分数直接决定处置档位。材料形态、计分口径与结论语义都
不一样，不重复。

**`source` 是这一族的留出维**：它**只影响渲染、不影响标签**——同一份命中组合写在命令行代理、
编辑器内代理还是对话式代理的轨迹里，信号算出来的分数一个字都不差。训练侧只见过 `cli` /
`editor`，留出侧是 `chat`，用来验「读法不靠入口先验」。

本模块有**两条独立路径**：`label_for` 从计划推标签（构造侧），`observed_labels` 从文本重推标签
（审计侧，走 `PATTERNS` 的词表）。两者对不上就说明渲染没兑现计划。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations
from typing import Final

from jc.contract import parse_hits
from jc.cover import Fingerprinted, all_hit_sets, cover_dims
from jc.jsontypes import JsonValue

#: 四条行为信号的编号（**结构键**，不是文案：信号名与判据原文全在 `locales/`）。
TAG_IDS: Final[tuple[str, ...]] = ("S1", "S2", "S3", "S4")

#: 判据编号 → 判据。**这是全 workflow 唯一的判据实现**：构造侧、渲染期、判别器、审计共用。
#: 每条都只认字面：
#: - `S1` 跑过验证（跑了测试 / 构建 / lint 这类**验证动作**）；
#: - `S2` 原地打转（又试一次 / 重试 / 还是失败 / 重复上一步）；
#: - `S3` 按观察调整（输出显示 / 报错表明 / 从输出看这类**读了结果**的形状）；
#: - `S4` 反复翻旧账（重新读一遍 / 再次打开 / 回到之前这类**回到已看过的东西**的形状）。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "S1": re.compile(
        r"(跑了测试|运行测试|跑一遍测试|执行测试|跑用例|测试通过|构建通过|lint 通过|验证通过)"
        r"|\b(pytest|npm test|make test|cargo test|go test|npm run build|eslint)\b"
        # 英文散文形态：`locales/en.toml` 的判据原文把它们逐个列了出来，规格里列了就**必须**算命中，
        # 否则模型照规格写反而被判成没命中（en 侧首轮 10 条里 3 条就是这么被拒的）。
        r"|\b(ran the tests|run the tests|run the test suite|executed the tests"
        r"|the tests pass|the build passes|lint passes|verification passed)\b",
        re.IGNORECASE,
    ),
    "S2": re.compile(
        r"(又试了一次|再试一遍|又跑了一次|重试|还是失败|仍然报同样的错|重复了上一步)"
        r"|\b(retry|retried|tried again|same error|repeated the|ran it once more)\b",
        re.IGNORECASE,
    ),
    "S3": re.compile(
        r"(输出显示|日志显示|报错表明|错误信息显示|从输出看|看到输出后|读到输出后|提示信息显示)"
        r"|\b(the output shows|the error says|the log shows|reading the output)\b",
        re.IGNORECASE,
    ),
    "S4": re.compile(
        r"(重新读了一遍|又看了一遍|再次打开|回到之前|重新检查同一个文件|重复读取|又翻回)"
        r"|\b(re-?read|reopened|read the same file|went back to)\b",
        re.IGNORECASE,
    ),
}

#: 轨迹入口（**结构键**，文案在 `locales/` 的 `[source.*]`）。
#:
#: 这一维**只影响渲染、不影响标签**：同一份命中组合写在命令行代理、编辑器内代理还是对话式代理的
#: 轨迹里，分数一个字都不差。它正是 test_ood 的留出维——训练侧只见过 `cli` / `editor`，留出侧是
#: `chat`，用来验「读法不靠入口先验」（范式 §9.25 那条「只影响渲染的那一维」）。
SOURCE_CLI: Final[str] = "cli"
SOURCE_EDITOR: Final[str] = "editor"
SOURCE_CHAT: Final[str] = "chat"
SOURCES: Final[tuple[str, ...]] = (SOURCE_CLI, SOURCE_EDITOR, SOURCE_CHAT)

#: 处置档位（**结构键**，文案全在 `locales/`）。三档按分数切，在 16 组命中组合上占比 5:6:5，
#: 摊得很平——三档而不是四档，是因为分档口径只有「继续 / 换方案 / 停下问人」这三种真实动作。
ROUTE_LEVELS: Final[tuple[str, ...]] = ("proceed", "replan", "escalate")

#: 继续的分界（4 分及以上继续）。
PROCEED_SCORE: Final[int] = 4

#: 换方案的分界（正好 3 分换方案）。
REPLAN_SCORE: Final[int] = 3

#: 轨迹的长度上下限。太短读不出「像不像一段真轨迹」，太长就只是灌水。
MIN_CONTENT_CHARS: Final[int] = 120

#: 上限。轨迹要容得下四五步（命令行代理那一档还带命令与输出片段），所以比日志行宽得多。
MAX_CONTENT_CHARS: Final[int] = 800


#: 全部命中组合（`cover_dims` 的取值表）。
HIT_SETS: Final[tuple[str, ...]] = all_hit_sets(TAG_IDS)

#: 维度表。全空间 16 × 3 = 48 组。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {"hits": HIT_SETS, "source": SOURCES}


def hit_set(text: str) -> tuple[str, ...]:
    """输入：轨迹正文。输出：命中的信号编号（按 `TAG_IDS` 顺序）。"""
    return tuple(tag for tag in TAG_IDS if PATTERNS[tag].search(text))


def hits_of(text: str) -> str:
    """输入：轨迹正文。输出：命中组合的规范串（`"S1+S4"` / `""`）。

    预期行为：与 `hit_set` 是同一件事的两种表示；`hits_of` 给比较用（`render.check_draft` 拿它
    与计划逐字比），`hit_set` 给取元素用。
    """
    return "+".join(hit_set(text))


def score_for(hits: Iterable[str]) -> int:
    """输入：命中的信号编号。输出：这段轨迹的推进分（1~5）。

    预期行为：**基础 1 分，四条计分条件各加 1 分**——`S1`（跑了验证）命中加 1、`S3`（按观察调整）
    命中加 1、`S2`（原地打转）**未**命中加 1、`S4`（反复翻旧账）**未**命中加 1。四条在 16 组命中
    组合上各占一半，所以五档的占比是 1:4:6:4:1（与命中条数同分布，与 `logtriage` 的 `count` 同款）。
    """
    ids = set(hits)
    return (
        1
        + int("S1" in ids)
        + int("S3" in ids)
        + int("S2" not in ids)
        + int("S4" not in ids)
    )


def route_for(score: int) -> str:
    """输入：推进分。输出：处置档位名。预期行为：4 分及以上继续、正好 3 分换方案、2 分及以下问人。"""
    if score >= PROCEED_SCORE:
        return "proceed"
    if score >= REPLAN_SCORE:
        return "replan"
    return "escalate"


@dataclass(frozen=True)
class TraceScoreCase(Fingerprinted):
    """一个注入特征组合：该命中哪几条信号 + 轨迹从哪种入口来。

    字段：`hits` 命中组合的规范串；`source` 入口（`SOURCES` 里的一个）。
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
        """输入：无。输出：该命中的信号编号元组。"""
        return parse_hits(self.hits)


@dataclass(frozen=True)
class TraceScorePlan:
    """一条样本的自评计划：命中哪几条 + 组合题的三个干扰项。

    字段：`hits` 该命中的信号；`distractors` 组合题的三个干扰组合（与 `hits` 不同、彼此也不同，
    且都取「与真值只差一两处」的邻近组合，题目才有区分度）。
    """

    hits: tuple[str, ...]
    distractors: tuple[tuple[str, ...], ...]

    @property
    def count(self) -> int:
        """输入：无。输出：命中条数。"""
        return len(self.hits)

    @property
    def score(self) -> int:
        """输入：无。输出：这段轨迹的推进分。"""
        return score_for(self.hits)

    @property
    def route(self) -> str:
        """输入：无。输出：处置档位名。"""
        return route_for(self.score)


def _distractors(truth: tuple[str, ...], rng: random.Random) -> tuple[tuple[str, ...], ...]:
    """输入：真值命中组合、随机源。输出：三个干扰组合。

    预期行为：只取**与真值相差不超过两处**的集合（差满四处 = 与真值完全不相交，等于把答案摆到
    另一个角落，反而好猜），按「差得越少越靠前」排序后抽三个；抽完按 (条数, 编号) 排好，保证
    同一入参必得同一结果。

    为什么这个池子恒有 6 个：四条信号一共 16 种组合，去掉真值自己剩 15 种，其中与真值相差三处或
    四处的有 9 种（差 4 处 1 种、差 3 处 4 种），所以池子恒为 6 个，抽三个不会撞车。
    """
    truth_set = set(truth)
    near = sorted(
        (
            combo
            for size in range(len(TAG_IDS) + 1)
            for combo in combinations(TAG_IDS, size)
            if set(combo) != truth_set and len(set(combo) ^ truth_set) <= 2
        ),
        key=lambda combo: (len(set(combo) ^ truth_set), combo),
    )
    picked = rng.sample(near, 3)
    return tuple(sorted(picked, key=lambda combo: (len(combo), combo)))


def plan_for(case: TraceScoreCase, phrasing: int) -> TraceScorePlan:
    """输入：case、问法编号。输出：自评计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里都得到
    同一份计划；两种问法会抽到不同的干扰组合，不会渲染出结构相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    return TraceScorePlan(
        hits=case.tags(), distractors=_distractors(case.tags(), random.Random(seed))
    )


@dataclass(frozen=True)
class TraceScoreLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`hits` 命中的信号编号（按 `TAG_IDS` 顺序）。
    """

    hits: tuple[str, ...]

    @property
    def count(self) -> int:
        """输入：无。输出：命中条数。"""
        return len(self.hits)

    def contains(self, tag: str) -> bool:
        """输入：信号编号。输出：这一条是否命中。"""
        return tag in self.hits

    @property
    def score(self) -> int:
        """输入：无。输出：按计分口径算出来的推进分。"""
        return score_for(self.hits)

    @property
    def route(self) -> str:
        """输入：无。输出：按分数算出来的处置档位名。"""
        return route_for(self.score)


def label_for(plan: TraceScorePlan) -> TraceScoreLabels:
    """输入：自评计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`render.check_draft` 负责
    保证文本真的兑现这份计划（按字面算出来的命中组合与计划逐条相同）。
    """
    return TraceScoreLabels(hits=plan.hits)


def observed_labels(text: str) -> TraceScoreLabels:
    """输入：轨迹正文。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认文本、只认字面。两者对不上就说明渲染没兑现
    计划，审计当场就能看见。
    """
    return TraceScoreLabels(hits=hit_set(text))


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[TraceScoreCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`TraceScoreCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [
        TraceScoreCase(hits=str(row["hits"]), source=str(row["source"])) for row in rows
    ]
