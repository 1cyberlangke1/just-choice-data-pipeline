"""内容审核（护栏）的特征层：政策条款 + 待审内容计划 + 命中标签（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条「这条内容踩了哪几条政策」的样本由两个**注入特征**描述
——`hits` 该命中哪几条条款、`decoy` 正文里要不要埋一个「像违规、按字面不算」的诱饵。
**标签先行**：先定这两个取值，再由本模块的纯函数算出结论。

**为什么政策条款写成正则**：官方 `llm_guardrails` 那类任务里，条款是「不许出现联系方式 /
外链 / 邮箱 / 情绪化喊话」这种**可按字面判**的规则。把条款写成机械判据之后，「命中哪几条」
就是**算**出来的，模型碰不到——它只负责写一条读得通、长尾的待审内容。判据表由代码写、正文
由模型写，两边都不越界（范式 §9.22 / §9.23 同一条路子）。

**诱饵（`decoy`）是这一族最要紧的设计**：`dashed` 要求在正文里埋一个**用连字符分段的电话
号码**（`138-1234-5678`）。它看起来就是泄露联系方式，可按政策字面**每一段都不足 7 位**，
所以不算命中 P1。这正是「见关键词就报警」的捷径要压掉的东西，也是 test_ood 的留出取值
（训练侧一次都见不到诱饵）。

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
from jc.cover import cover_dims
from jc.jsontypes import JsonValue

#: 四条政策条款的编号（**结构键**，不是文案：条款名与判据原文全在 `locales/`）。
CLAUSE_IDS: Final[tuple[str, ...]] = ("P1", "P2", "P3", "P4")

#: 条款编号 → 判据。**这是全 workflow 唯一的判据实现**：构造侧、渲染期、判别器、审计共用。
#: 每条都只认字面：半角 0-9、`http(s)://` 或 `www.`、`@`、连续三个以上感叹号（半角或全角）。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "P1": re.compile(r"[0-9]{7,}"),
    "P2": re.compile(r"https?://|www\.", re.IGNORECASE),
    "P3": re.compile(r"@"),
    "P4": re.compile(r"[!！]{3,}"),
}

#: 正文里要不要埋诱饵。`dashed` 的那个电话号码**每一段都不足 7 位**，按字面不算命中 P1。
DECOY_NONE: Final[str] = "none"
DECOY_DASHED: Final[str] = "dashed"
DECOYS: Final[tuple[str, ...]] = (DECOY_NONE, DECOY_DASHED)

#: 诱饵的形状：三段数字、每段 3~4 位、用连字符分开。**要求三段**是为了把日期（`2026-09-29`
#: 的第二段只有 2 位）排除在外——只写 `\d+-\d+` 的话，一个日期就能冒充诱饵。
DECOY_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9]{3,4}[-‑][0-9]{3,4}[-‑][0-9]{3,4}")

#: 提示词里**原样给出**的诱饵写法。模型不会自发写出「分段号码」这种形状，所以逐字给一个；
#: 它必须同时满足两件事：匹配 `DECOY_PATTERN`（算是诱饵），且**不触发 P1**（每段都不足 7 位）。
#: `tests/workflows/guardrail/test_features.py` 把这两条钉在数据上。
DECOY_EXAMPLE: Final[str] = "138-1234-5678"

#: 处置档位（**结构键**，文案全在 `locales/`）。阈值是 0 条 / 1 条 / 2 条 / 3 条及以上——
#: 为什么是四档而不是三档：三档（0 / 1 / 2 以上）会让「拦截」一档按构造占 11/16，答案分布
#: 天然偏斜到 69%，均衡闸会把整批标成失衡。四档把占比摊成 1/4/6/5，最大值 37.5%，
#: 落在噪声地板之内（见范式 §9.25 关于「结构性偏斜」的那一段）。
ACTION_LEVELS: Final[tuple[str, ...]] = ("pass", "flag", "review", "block")

#: 正文的最短长度（太短的内容读不出「像不像一条正经帖子」）。
MIN_CONTENT_CHARS: Final[int] = 20

#: 正文的最长长度（再长就只是灌水，题面也没必要）。
MAX_CONTENT_CHARS: Final[int] = 400


def _all_hit_sets() -> tuple[str, ...]:
    """输入：无。输出：16 个命中组合的规范串（空串在前，然后单条、两条……）。

    预期行为：顺序稳定（case 矩阵逐字节可重建），每条条款在 16 组里各出现 8 次——这是
    `clause` 那一族的答案占比按构造就是 50% 的原因。
    """
    out: list[str] = []
    for size in range(len(CLAUSE_IDS) + 1):
        for combo in combinations(CLAUSE_IDS, size):
            out.append("+".join(combo))
    return tuple(out)


#: 全部命中组合（`cover_dims` 的取值表）。
HIT_SETS: Final[tuple[str, ...]] = _all_hit_sets()

#: 注入维度表。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {"hits": HIT_SETS, "decoy": DECOYS}


def hit_set(text: str) -> tuple[str, ...]:
    """输入：待审正文。输出：命中的条款编号（按 `CLAUSE_IDS` 顺序）。

    预期行为：逐条按字面判，不做任何语义推断——这正是这一族要训的东西。
    """
    return tuple(clause for clause in CLAUSE_IDS if PATTERNS[clause].search(text))


def hits_of(text: str) -> str:
    """输入：待审正文。输出：命中组合的规范串（与 `HIT_SETS` 里的写法一致）。"""
    return "+".join(hit_set(text))


def has_decoy(text: str) -> bool:
    """输入：待审正文。输出：正文里有没有那个「分段电话号码」形状的诱饵。"""
    return DECOY_PATTERN.search(text) is not None


def action_level(count: int) -> str:
    """输入：命中条数。输出：处置档位名。

    预期行为：0 → `pass`、1 → `flag`、2 → `review`、3 条及以上 → `block`。阈值表印在 state 里
    （`render.to_state`），题目选项与 state 里那张表取的是**同一份文案**，两边不可能漂移。
    """
    return ACTION_LEVELS[min(count, len(ACTION_LEVELS) - 1)]


@dataclass(frozen=True)
class GuardrailCase:
    """一条样本的注入特征：该命中哪几条条款、要不要埋诱饵。"""

    hits: str
    decoy: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"hits": self.hits, "decoy": self.decoy}

    def fingerprint(self) -> str:
        """输入：无。输出：case 指纹。预期行为：同一 case 任何进程都得到同一串。"""
        return "|".join(f"{key}={value}" for key, value in self.as_dict().items())

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.hits or '无命中'}-{self.decoy}"

    def clauses(self) -> tuple[str, ...]:
        """输入：无。输出：该命中的条款编号元组。"""
        return parse_hits(self.hits)


@dataclass(frozen=True)
class GuardrailPlan:
    """一条样本的审核计划：命中哪几条 + 组合题的三个干扰项。

    字段：`hits` 该命中的条款；`distractors` 组合题的三个干扰组合（与 `hits` 不同、彼此也不同，
    且都取「与真值只差一两处」的邻近组合，题目才有区分度）。
    """

    hits: tuple[str, ...]
    distractors: tuple[tuple[str, ...], ...]

    @property
    def count(self) -> int:
        """输入：无。输出：命中条数。"""
        return len(self.hits)


def _distractors(truth: tuple[str, ...], rng: random.Random) -> tuple[tuple[str, ...], ...]:
    """输入：真值命中组合、随机源。输出：三个干扰组合。

    预期行为：先按「与真值只差几处」排序，再从最近的一批里抽三个——差得越少越难分辨，题目才
    有区分度；抽完按 (条数, 编号) 排好，保证同一入参必得同一结果。
    """
    truth_set = set(truth)
    near = sorted(
        (
            combo
            for size in range(len(CLAUSE_IDS) + 1)
            for combo in combinations(CLAUSE_IDS, size)
            if set(combo) != truth_set
        ),
        key=lambda combo: (len(set(combo) ^ truth_set), combo),
    )
    picked = rng.sample(near[:8], 3)
    return tuple(sorted(picked, key=lambda combo: (len(combo), combo)))


def plan_for(case: GuardrailCase, phrasing: int) -> GuardrailPlan:
    """输入：case、问法编号。输出：审核计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里都
    得到同一份计划；两种问法会抽到不同的干扰组合，不会渲染出结构相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    return GuardrailPlan(
        hits=case.clauses(), distractors=_distractors(case.clauses(), random.Random(seed))
    )


@dataclass(frozen=True)
class GuardrailLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`hits` 命中的条款编号（按 `CLAUSE_IDS` 顺序）。
    """

    hits: tuple[str, ...]

    @property
    def count(self) -> int:
        """输入：无。输出：命中条数。"""
        return len(self.hits)

    def contains(self, clause: str) -> bool:
        """输入：条款编号。输出：这一条是否命中。"""
        return clause in self.hits

    @property
    def action(self) -> str:
        """输入：无。输出：按处置阈值算出来的档位名（`features.action_level`）。"""
        return action_level(self.count)


def label_for(plan: GuardrailPlan) -> GuardrailLabels:
    """输入：审核计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`render.check_draft`
    负责保证文本真的兑现这份计划（按字面算出来的命中组合与计划逐条相同）。
    """
    return GuardrailLabels(hits=plan.hits)


def observed_labels(text: str) -> GuardrailLabels:
    """输入：待审正文。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认文本、只认字面。两者对不上就说明渲染没
    兑现计划，审计当场就能看见。
    """
    return GuardrailLabels(hits=hit_set(text))


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[GuardrailCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`GuardrailCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [GuardrailCase(hits=str(row["hits"]), decoy=str(row["decoy"])) for row in rows]
