"""回合结束判定（turntake）的特征层：未完判据 + 输入来源 + 命中标签（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条「这段输入到底说完了没有」的样本由两个**注入特征**描述——
`hits` 该命中哪几条「还没说完」的判据、`channel` 这段输入从哪种通道来（聊天 / 语音转写 / 口述
备忘）。**标签先行**：先定这两个取值，再由本模块的纯函数算出结论。

**为什么判据写成机械短语表**：真实对话里「用户还没说完」是有固定形状的——话头悬在连接词上、
标点断在半路、括号开了没关、明说「等我一下」。把这些形状写成正则之后，「命中哪几条」就是
**算**出来的，模型碰不到——它只负责写一段读得通、长尾的用户输入。判据表由代码写、内容由模型写
（范式 §9.22 / §9.23 同一条路子）。

**为什么要这一族**：社区里 `aiavatarkit`（★678）把它明写成「带有 Jev 门，可判断说话者是否已完成
回合」，`jev-chat-windows`（★467）/ `jev-chat-jarvis-mac`（★248）则在助手给出回复之前先用 Jev
判意图与风险——共同点是**每一轮对话都要先决定「现在该不该接话」**，而这个决定必须在模型开口
之前做完（接早了打断人、接晚了显得迟钝）。

**与 `routing` / `mail` / `logtriage` 的区别**：routing 判的是「这条工单该归谁」，mail 判的是
「这封邮件是不是钓鱼」，logtriage 判的是「这行日志该不该进告警通道」——判据空间、处置语义与
素材形态都不一样（那三家看的是**已经完整**的材料，本 workflow 看的是**可能还没写完**的材料）。

**`channel` 是这一族的留出维**：它**只影响渲染、不影响标签**——同一份命中组合写在聊天记录、
语音转写还是口述备忘里，判据算出来的答案一个字都不差。训练侧只见过 `im` / `voice`，留出侧是
`note`，用来验「读法不靠来源先验」。

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

#: 四条「还没说完」判据的编号（**结构键**，不是文案：判据名与判据原文全在 `locales/`）。
TAG_IDS: Final[tuple[str, ...]] = ("T1", "T2", "T3", "T4")

#: 判据编号 → 判据。**这是全 workflow 唯一的判据实现**：构造侧、渲染期、判别器、审计共用。
#: 每条都只认字面：
#: - `T1` 话头悬在连接词上（整段以「然后 / 还有 / and / so」这类词收尾）；
#: - `T2` 标点断在半路（整段以逗号 / 顿号 / 省略号收尾）；
#: - `T3` 括号开了没关（出现左括号，之后到结尾都没有对应的右括号）；
#: - `T4` 明说「等我一下」（等一下 / 让我想想 / wait / hold on 这类标记）。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "T1": re.compile(
        r"(然后|还有|而且|以及|所以|但是|另外|就是|那个|因为|接着|再说|还有呢|然后呢)"
        r"[，,。.！!？?…]*\s*$"
        r"|\b(and|also|so|but|because|then|plus|or)\b[\s.,…]*$",
        re.IGNORECASE,
    ),
    "T2": re.compile(r"(?:[，,、]|…|\.\.\.)\s*$"),
    "T3": re.compile(r"[（(【\[](?=[^）)】\]]*$)"),
    "T4": re.compile(
        r"(等一下|等等|让我想想|我想想|先别急|还没说完|补充一下|补充一点|先听我说)"
        r"|\b(wait|hold on|hang on|let me think|one more thing|give me a second)\b",
        re.IGNORECASE,
    ),
}

#: 输入通道（**结构键**，文案在 `locales/` 的 `[channel.*]`）。
#:
#: 这一维**只影响渲染、不影响标签**：同一份命中组合写在聊天记录、语音转写还是口述备忘里，判据
#: 算出来的答案一个字都不差。它正是 test_ood 的留出维——训练侧只见过 `im` / `voice`，留出侧是
#: `note`，用来验「读法不靠来源先验」（范式 §9.25 那条「只影响渲染的那一维」）。
CHANNEL_IM: Final[str] = "im"
CHANNEL_VOICE: Final[str] = "voice"
CHANNEL_NOTE: Final[str] = "note"
CHANNELS: Final[tuple[str, ...]] = (CHANNEL_IM, CHANNEL_VOICE, CHANNEL_NOTE)

#: 处置档位（**结构键**，文案全在 `locales/`）。阈值是 0 条 / 1 条 / 2 条 / 3 条及以上——
#: 语义是「还没说完的理由越多，越不该接话」：0 条立刻接话、1 条等一拍、2 条继续等、
#: 3 条及以上等着并轻问一句（等太久会让人以为助手没反应）。四档而不是三档的理由与 `logtriage`
#: 相同：三档会让「继续等」一档按构造占 11/16（69%），均衡闸会把整批标成失衡。四档摊成 1/4/6/5。
ACTS: Final[tuple[str, ...]] = ("answer", "pause", "wait", "check")

#: 输入的字符数上下限。太短读不出「像不像一段真的用户输入」，太长就只是灌水。
MIN_CONTENT_CHARS: Final[int] = 20

#: 上限。用户一口气说的话可以很长，但不能长到变成一篇文档。
MAX_CONTENT_CHARS: Final[int] = 300


#: 全部命中组合（`cover_dims` 的取值表）。
HIT_SETS: Final[tuple[str, ...]] = all_hit_sets(TAG_IDS)

#: 维度表。全空间 16 × 3 = 48 组。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {"hits": HIT_SETS, "channel": CHANNELS}


def hit_set(text: str) -> tuple[str, ...]:
    """输入：一段用户输入。输出：命中的判据编号（按 `TAG_IDS` 顺序）。"""
    return tuple(tag for tag in TAG_IDS if PATTERNS[tag].search(text))


def hits_of(text: str) -> str:
    """输入：一段用户输入。输出：命中组合的规范串（`"T1+T4"` / `""`）。

    预期行为：与 `hit_set` 是同一件事的两种表示；`hits_of` 给比较用（`render.check_draft` 拿它
    与计划逐字比），`hit_set` 给取元素用。
    """
    return "+".join(hit_set(text))


def act_level(count: int) -> str:
    """输入：命中条数。输出：处置档位名。

    预期行为：0→answer、1→pause、2→wait、3 及以上→check（最后一档吃掉所有更大的数）。
    """
    return ACTS[min(count, len(ACTS) - 1)]


@dataclass(frozen=True)
class TurnTakeCase(Fingerprinted):
    """一个注入特征组合：该命中哪几条判据 + 输入从哪种通道来。

    字段：`hits` 命中组合的规范串；`channel` 通道（`CHANNELS` 里的一个）。
    """

    hits: str
    channel: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"hits": self.hits, "channel": self.channel}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.hits or '无命中'}-{self.channel}"

    def tags(self) -> tuple[str, ...]:
        """输入：无。输出：该命中的判据编号元组。"""
        return parse_hits(self.hits)


@dataclass(frozen=True)
class TurnTakePlan:
    """一条样本的接话计划：命中哪几条 + 组合题的三个干扰项。

    字段：`hits` 该命中的判据；`distractors` 组合题的三个干扰组合（与 `hits` 不同、彼此也不同，
    且都取「与真值只差一两处」的邻近组合，题目才有区分度）。
    """

    hits: tuple[str, ...]
    distractors: tuple[tuple[str, ...], ...]

    @property
    def count(self) -> int:
        """输入：无。输出：命中条数。"""
        return len(self.hits)


def plan_for(case: TurnTakeCase, phrasing: int) -> TurnTakePlan:
    """输入：case、问法编号。输出：接话计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里都
    得到同一份计划；两种问法会抽到不同的干扰组合，不会渲染出结构相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    return TurnTakePlan(
        hits=case.tags(), distractors=near_miss_combos(TAG_IDS, case.tags(), random.Random(seed))
    )


@dataclass(frozen=True)
class TurnTakeLabels:
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


def label_for(plan: TurnTakePlan) -> TurnTakeLabels:
    """输入：接话计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`render.check_draft`
    负责保证文本真的兑现这份计划（按字面算出来的命中组合与计划逐条相同）。
    """
    return TurnTakeLabels(hits=plan.hits)


def observed_labels(text: str) -> TurnTakeLabels:
    """输入：一段用户输入。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认文本、只认字面。两者对不上就说明渲染没
    兑现计划，审计当场就能看见。
    """
    return TurnTakeLabels(hits=hit_set(text))


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[TurnTakeCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`TurnTakeCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [TurnTakeCase(hits=str(row["hits"]), channel=str(row["channel"])) for row in rows]
