"""聊天回复门控（chatgate）的特征层：消息形状判据 + 候选毛病判据 + 计划与标签（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条「对方发来一条消息、我该回哪条」的样本由三个**注入特征**描述
——`degree`（这条消息踩中几条「要小心的形状」）、`scene`（消息长在哪种聊天场景里）、
`slot`（三条候选回复里哪一条一个毛病都不踩）。**标签先行**：先定这三个取值，再由本模块的纯函数
算出结论（该命中哪几条判据 + 该发哪条候选），模型碰不到。

**为什么要这一族**：社区清单里最大的一支应用是**聊天助手**——`jev-chat-jarvis`（★1696，在用户
插入回复之前对候选回复排序）、`jev-chat-windows`（★467，先分意图、再给三条候选回复）、
`crush-monitor`（★183，情绪 / 意图 / 回复表现）、`Jev-X-Sentiment-Analysis`（★162）、
`yanwai`（★15，情绪概率 + 沟通建议）、`wechat-jev-assistant`（★13）。它们的共同形状是：
**一条进来的消息 → 先判「对方在要什么、有多需要谨慎」→ 再决定「三条候选里该发哪一条」**，
而且**绝不让模型直接生成要发出去的那句话**——先给候选、再判该发哪条。

**与 `mail` / `guardrail` / `logtriage` 的区别**：那三家的判据空间是**内容安全 / 运维**的形状
（发件人身份、外链、政策条款、日志级别词）。本 workflow 判的是**人际沟通**的形状（要不要答复、
有没有明确时限、要不要我担义务、涉不涉及隐私），而且多一层**候选回复的取舍**：`reply` 那一族
问的是「三条候选里该发哪一条」，答案由**候选自己的毛病**算出来，与消息那几条判据互不决定
（两张表、两个计划）。

**判据写成机械短语表**：人际消息里「要小心」的形状是有固定写法的（「你定 / 行不行」是推决定、
「今天之内 / 半小时内」是明确时限、「借我 / 替我签」是担义务、「身份证 / 验证码」是隐私）。
写成正则之后，「踩中哪几条」就是**算**出来的，模型只负责写一条读得通、长尾的消息与三条候选。

**为什么 `degree` 是主维度，而不是「命中组合」**：命中条数决定 `count`（一共踩几条）与
`route`（该走哪一档）两族的**金标签**。把 16 组命中组合原样当维度铺开，条数分布就是
1:4:6:4:1（2 条那一档占 37.5%），也就是说这批数据里将近四成的「一共踩几条」答案是同一个
——那是结构性偏斜，`logtriage` / `guardrail` 也有，只是它们的 case 空间小、没显出来。
把**条数**当维度（每档 1/5）、具体哪几条由 case 指纹派生（`tags_for`），条数就是均匀的
（45 个 case 里每档恰好 9 个，`tests/workflows/chatgate/test_features.py` 把它钉住）；
而 `tag` 那一族仍是近似 50/50——`degree=k` 的 case 里每条判据被选中的概率都是 k/4，
五档加权后正好一半。

**为什么这条要自己钉、不能指望通用均衡闸**：通用闸看的是 `answer["choice"]`，也就是**打乱之后
的选项位置**（见 `jc.assemble`），而 choice 的选项每道题都重新打乱，所以它量到的是「正确答案
有没有总落在同一个位置」，不是标签分布。标签分布这一条得由 workflow 自己量。

**为什么 `slot` 是维度而不是随机数**：`reply` 那一族的答案是「三条候选里该发哪条」。若这个位置
由随机数派生，同一个命中组合在训练集里就**总是落在同一个位置**——模型只要认出消息踩了哪几条
（那是 `tag` 那一族在问的），就能把 `reply` 的答案背出来，答案成了消息的函数。把位置当 case 的
一维、与 `degree` / `scene` 正交铺开，这条捷径在数据里就不存在。

**`scene` 是这一族的留出维**：它**只影响渲染、不影响标签**——同一份命中组合写在同事私聊、
工作群还是陌生人私信里，判据算出来的答案一个字都不差。训练侧只见过 `colleague` / `group`，
留出侧是 `stranger`，用来验「读法不靠场景先验」（范式 §9.25 那条「只影响渲染的那一维」）。

本模块有**两条独立路径**：`label_for` 从计划推标签（构造侧），`observed_labels` 从文本重推
标签（审计侧，走 `hit_set` / `clean_indexes` 的正则）。两者对不上就说明渲染没兑现计划。

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

#: 四条「要小心的形状」判据的编号（**结构键**，不是文案：判据名与判据原文全在 `locales/`）。
TAG_IDS: Final[tuple[str, ...]] = ("C1", "C2", "C3", "C4")

#: 判据编号 → 判据。**这是全 workflow 唯一的判据实现**：构造侧、渲染期、判别器、审计共用。
#: 每条都只认字面，且**刻意收窄**（宁可漏，不可宽）：
#: - `C1` 把决定推给我（「你定 / 行不行 / 帮我确认」这类把选择权交出来的写法）；
#: - `C2` 带**明确时限**（「今天之内 / 半小时内 / by tomorrow」）——「尽快 / 马上 / 赶紧」这类
#:   催促语**不算**（同 `mail` 的教训：催促语与时限是两回事，混在一起那一档就没法判了）；
#: - `C3` 要我做一件**我得担下来**的事（借、垫、担保、署名、顶班、替你出面）；
#: - `C4` 要隐私或凭据（证件号、卡号、密码、验证码、住址、手机号、病历这类）。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "C1": re.compile(
        r"(你定|你选|你说吧|听你的|你决定|定一下|要不要|行不行|可以吗|可以不|成不成|"
        r"帮我确认|麻烦确认|确认一下|选哪个|哪一个|给个说法|给个准话)"
        r"|\b(your call|you decide|which one|should we|can you confirm|let me know|"
        r"your choice|up to you|is that ok)\b",
        re.IGNORECASE,
    ),
    "C2": re.compile(
        r"(今天|明天|今晚|本周|这周|这个月|月底|三点|五点|下班)\s*(之内|内|之前|前)"
        r"|\d+\s*(分钟|小时|天|个工作日)\s*(之内|内|以内)"
        r"|截止"
        r"|\b(by|before|within)\s+\d+\s*(minutes?|hours?|days?|weeks?)\b"
        r"|\bby (today|tomorrow|tonight|eod)\b"
        r"|\bend of (day|week)\b|\bdeadline\b",
        re.IGNORECASE,
    ),
    "C3": re.compile(
        r"(答应我|你得答应|保证|发誓|包在我身上|借我|垫一下|垫付|担保|署名|顶班|顶一下|"
        r"替我签|替我出面|你负责|你来承担|就说你同意)"
        r"|\b(promise me|you must promise|guaranteed|vouch for me|lend me|cover for me|"
        r"sign for me|take the blame|commit to)\b",
        re.IGNORECASE,
    ),
    "C4": re.compile(
        r"(身份证|银行卡|卡号|密码|验证码|短信码|工牌号|住址|家庭住址|手机号|护照号|社保号|"
        r"工资条|病历|体检报告)"
        r"|\b(id number|identity card|bank card|card number|password|passcode|"
        r"verification code|security code|badge number|home address|phone number|"
        r"passport number|social security|medical record|payslip)\b",
        re.IGNORECASE,
    ),
}

#: 三条「候选毛病」的编号（**结构键**，文案在 `locales/`）。这三条判的是**我自己要发出去的话**，
#: 与上面四条判「对方那条消息」的判据是两张互不相关的表。
#:
#: 为什么单独一张表：`reply` 那一族的答案必须**由候选自己的文字**决定，而不是由消息决定——否则
#: 「该发哪条」就成了「这条消息踩了几条」的函数，模型不用读候选也能答对，那一族白出。
FLAW_IDS: Final[tuple[str, ...]] = ("F1", "F2", "F3")

#: 毛病编号 → 判据。同样只认字面：
#: - `F1` 作出承诺（「保证 / 我答应 / 包在我身上 / 放心 / 没问题 / 我来搞定」这类把话说死的话）；
#:   **英文侧刻意不收 `sure thing`**：它在英文里就是一句普通的应声（相当于中文的「行」），
#:   而中文表里并没有把「行」算作把话说死——收了它，同一个语义在 zh 侧干净、在 en 侧算踩中，
#:   两边判据就不对称了（实测真机 en 侧因此把「干净的那条候选」判成 F1，10 条里废掉 3 条）。
#: - `F2` 追问隐私或凭据（把 `C4` 那张词表**再问回去**——对方问隐私已经够糟，我再追问就更糟）；
#: - `F3` 空话敷衍（「再说吧 / 看情况 / 有空再说」这类把人晾着的话）。
FLAW_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "F1": re.compile(
        r"(保证|我答应|包在我身上|没问题|放心|我来搞定|说到做到|绝不|肯定给你|我承诺)"
        r"|\b(i promise|guaranteed|consider it done|you have my word|"
        r"no problem at all|i'?ll make sure)\b",
        re.IGNORECASE,
    ),
    "F2": re.compile(
        r"(身份证|银行卡|卡号|密码|验证码|工牌|住址|护照|社保|手机号|工资条)"
        r"|\b(id number|bank card|card number|password|verification code|badge|"
        r"home address|passport|social security|phone number)\b",
        re.IGNORECASE,
    ),
    "F3": re.compile(
        r"(再说吧|再说|看情况|有空再说|回头说|回头聊|嗯嗯|emmm|不好说|到时候看|改天|先这样|随便)"
        r"|\b(we'?ll see|maybe later|not sure yet|hmm+|let'?s see|later on|depends|whatever)\b",
        re.IGNORECASE,
    ),
}

#: 聊天场景（**结构键**，文案在 `locales/` 的 `[scene.*]`）。
#:
#: 这一维**只影响渲染、不影响标签**：同一份命中组合写在同事私聊、工作群还是陌生人私信里，判据
#: 算出来的答案一个字都不差。它正是 `test_ood` 的留出维——训练侧只见过 `colleague` / `group`，
#: 留出侧是 `stranger`。
SCENE_COLLEAGUE: Final[str] = "colleague"
SCENE_GROUP: Final[str] = "group"
SCENE_STRANGER: Final[str] = "stranger"
SCENES: Final[tuple[str, ...]] = (SCENE_COLLEAGUE, SCENE_GROUP, SCENE_STRANGER)

#: 命中条数（**结构键**，文案在 `locales/`）。这是这一族的**主维度**：条数在 case 空间里均匀，
#: `count` 与 `route` 两族的答案分布才是均匀的（见模块开头「为什么 `degree` 是主维度」）。
DEGREES: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)

#: 三条候选里「一个毛病都不踩」的那一条落在第几位（0 起）。**这是 case 的一维**，不是随机数：
#: 它与 `degree` / `scene` 正交，所以「该发哪条」不可能被当成「消息踩了几条」的函数学走。
SLOTS: Final[tuple[int, ...]] = (0, 1, 2)

#: 处置档位（**结构键**，文案全在 `locales/`）。阈值是 0 条 / 1 条 / 2 条 / 3 条及以上——语义是
#: 「要小心的形状越多，越不该当场把话说死」：0 条直接回、1 条先确认、2 条先搁置、3 条以上转人工。
#: 四档而不是三档的理由与 `logtriage` / `guardrail` 相同：三档会让最后一档按构造吃掉一大半。
HANDLING_LEVELS: Final[tuple[str, ...]] = ("direct", "confirm", "hold", "escalate")

#: 对方那条消息的长度上下限。太短读不出「像不像一条真消息」，太长就只是灌水。
MIN_MESSAGE_CHARS: Final[int] = 12

#: 上限。聊天消息天然短，比 `logtriage` 的日志行紧得多。
MAX_MESSAGE_CHARS: Final[int] = 120

#: 单条候选回复的长度上下限。一句话的量级。
MIN_CANDIDATE_CHARS: Final[int] = 6

#: 候选上限。
MAX_CANDIDATE_CHARS: Final[int] = 90

#: 三条候选的位置（0 / 1 / 2）——渲染与解析共用，别处不许写死这个数。
CANDIDATE_SLOTS: Final[tuple[int, ...]] = (0, 1, 2)

#: 两条「不干净」候选的毛病组合表。每一对都是**两个互不相同的非空子集**：让两条候选的毛病
#: 长得不一样，读者才不能靠「哪条被写得最像模板」去猜。
FLAW_PAIRS: Final[tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]] = (
    (("F1",), ("F2",)),
    (("F1",), ("F3",)),
    (("F2",), ("F3",)),
    (("F1",), ("F2", "F3")),
    (("F2",), ("F1", "F3")),
    (("F3",), ("F1", "F2")),
)

#: 维度表。全空间 5 × 3 × 3 = 45 组。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {
    "degree": DEGREES,
    "scene": SCENES,
    "slot": SLOTS,
}


def hit_set(text: str) -> tuple[str, ...]:
    """输入：对方那条消息。输出：命中的判据编号（按 `TAG_IDS` 顺序）。"""
    return tuple(tag for tag in TAG_IDS if PATTERNS[tag].search(text))


def hits_of(text: str) -> str:
    """输入：对方那条消息。输出：命中组合的规范串（`"C1+C4"` / `""`）。

    预期行为：与 `hit_set` 是同一件事的两种表示；`hits_of` 给比较用（`render.check_draft` 拿它
    与计划逐字比），`hit_set` 给取元素用。
    """
    return "+".join(hit_set(text))


def tags_for(degree: int, scene: str, slot: int) -> tuple[str, ...]:
    """输入：命中条数、场景、该发候选的位置。输出：该命中的判据编号（按 `TAG_IDS` 顺序）。

    预期行为：从 `TAG_IDS` 里**无放回抽** `degree` 条，抽法由 `(degree, scene, slot)` 的指纹
    决定——所以同一个 case 在任何进程、任何切分里都得到同一组判据（case 矩阵逐字节可重建），
    且不是「前 degree 条」那种常量（那样 `degree=2` 永远只踩 C1+C2，模型不用读消息就能答）。
    """
    seed = int(
        hashlib.sha256(f"tags|{degree}|{scene}|{slot}".encode("utf-8")).hexdigest()[:16], 16
    )
    return tuple(sorted(random.Random(seed).sample(list(TAG_IDS), degree)))


def flaw_set(text: str) -> tuple[str, ...]:
    """输入：一条候选回复。输出：它踩中的毛病编号（按 `FLAW_IDS` 顺序）。

    预期行为：只认字面。**空元组就是「这条可以发」**——`reply` 那一族的答案由它算出来。
    """
    return tuple(flaw for flaw in FLAW_IDS if FLAW_PATTERNS[flaw].search(text))


def clean_indexes(candidates: tuple[str, ...]) -> tuple[int, ...]:
    """输入：三条候选回复。输出：一个毛病都不踩的那几条的**下标**（按原顺序）。

    预期行为：返回元组而不是单个下标——「一条都不干净」或「干净的不止一条」都是构造 bug，
    让审计能把这两种情形都看见，而不是硬凑一个答案出来。
    """
    return tuple(index for index, text in enumerate(candidates) if not flaw_set(text))


def handling_level(count: int) -> str:
    """输入：命中条数。输出：处置档位名。

    预期行为：0→direct、1→confirm、2→hold、3 及以上→escalate（最后一档吃掉所有更大的数）。
    """
    return HANDLING_LEVELS[min(count, len(HANDLING_LEVELS) - 1)]


@dataclass(frozen=True)
class ChatCase(Fingerprinted):
    """一条样本的注入特征。

    字段：`degree` 这条消息该踩几条判据；`scene` 聊天场景；`slot` 该发的那条候选的下标。
    """

    degree: int
    scene: str
    slot: int

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"degree": self.degree, "scene": self.scene, "slot": self.slot}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"degree{self.degree}-{self.scene}-slot{self.slot + 1}"

    def tags(self) -> tuple[str, ...]:
        """输入：无。输出：该命中的判据编号元组（由指纹派生，见 `tags_for`）。"""
        return tags_for(self.degree, self.scene, self.slot)


@dataclass(frozen=True)
class ReplyPlan:
    """三条候选的毛病计划：哪一条该发、另外两条各踩哪些毛病。

    字段：`clean` 该发的候选下标；`flaws` 三条候选各自的毛病元组（`flaws[clean]` 必须是空元组）。
    """

    clean: int
    flaws: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        """输入：无。输出：无。

        预期行为：构造期就把坏计划挡掉——三条候选、恰好一条干净、另外两条的毛病互不相同且非空。
        这些不是「顺手查一下」：任何一条不成立，`reply` 那一族的答案就不唯一，题目直接作废。
        """
        if len(self.flaws) != len(CANDIDATE_SLOTS):
            raise ValueError(f"候选毛病计划要有 {len(CANDIDATE_SLOTS)} 条，实际 {len(self.flaws)}")
        if self.clean not in CANDIDATE_SLOTS:
            raise ValueError(f"该发的候选下标越界：{self.clean}")
        if self.flaws[self.clean]:
            raise ValueError(f"该发的那条候选不许有毛病：{self.flaws[self.clean]}")
        dirty = [flaws for index, flaws in enumerate(self.flaws) if index != self.clean]
        if any(not flaws for flaws in dirty):
            raise ValueError(f"另外两条候选必须各踩至少一个毛病：{self.flaws}")
        if len(set(dirty)) != len(dirty):
            raise ValueError(f"另外两条候选的毛病不许一样：{self.flaws}")
        unknown = sorted({flaw for flaws in self.flaws for flaw in flaws} - set(FLAW_IDS))
        if unknown:
            raise ValueError(f"计划里出现不认识的毛病编号：{unknown}")

    @property
    def clean_answer(self) -> int:
        """输入：无。输出：该发的那条候选的下标。"""
        return self.clean


@dataclass(frozen=True)
class ChatPlan:
    """一条样本的完整计划：消息该踩哪几条 + 三条候选的毛病怎么分。

    字段：`hits` 该命中的判据；`reply` 候选毛病计划。
    """

    hits: tuple[str, ...]
    reply: ReplyPlan

    @property
    def count(self) -> int:
        """输入：无。输出：命中条数。"""
        return len(self.hits)


def _reply_plan(clean: int, seed: int) -> ReplyPlan:
    """输入：该发的候选下标、随机种子。输出：候选毛病计划。

    预期行为：干净的那条固定在 `clean`（case 的一维），另外两条从 `FLAW_PAIRS` 里抽一对——种子
    由 case 指纹派生，所以同一个 case 在任何进程里都得到同一份计划。
    """
    rng = random.Random(seed)
    first, second = rng.choice(FLAW_PAIRS)
    others = [index for index in CANDIDATE_SLOTS if index != clean]
    flaws: list[tuple[str, ...]] = [() for _ in CANDIDATE_SLOTS]
    flaws[others[0]] = first
    flaws[others[1]] = second
    return ReplyPlan(clean=clean, flaws=tuple(flaws))


def plan_for(case: ChatCase, phrasing: int) -> ChatPlan:
    """输入：case、问法编号。输出：完整计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里都得到
    同一份计划；两种问法会抽到不同的毛病分配，不会渲染出结构相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    return ChatPlan(hits=case.tags(), reply=_reply_plan(case.slot, seed))


@dataclass(frozen=True)
class ChatLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`hits` 命中的判据编号（按 `TAG_IDS` 顺序）；`clean` 一个毛病都不踩的候选下标（按原顺序）。
    """

    hits: tuple[str, ...]
    clean: tuple[int, ...]

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


def label_for(plan: ChatPlan) -> ChatLabels:
    """输入：完整计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`render.check_draft`
    负责保证文本真的兑现这份计划（消息按字面算出来的命中组合、三条候选各自的毛病都与计划相同）。
    """
    return ChatLabels(hits=plan.hits, clean=(plan.reply.clean_answer,))


def observed_labels(message: str, candidates: tuple[str, ...]) -> ChatLabels:
    """输入：对方那条消息、三条候选回复。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认文本、只认字面。两者对不上就说明渲染没兑现
    计划，审计当场就能看见。
    """
    return ChatLabels(hits=hit_set(message), clean=clean_indexes(candidates))


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[ChatCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`ChatCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [
        ChatCase(
            degree=int(str(row["degree"])),
            scene=str(row["scene"]),
            slot=int(str(row["slot"])),
        )
        for row in rows
    ]
