"""招聘初筛门禁（hirecheck）的特征层：四条初筛判据 + 定级规则 + 处置档位（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条「这份候选人材料要不要进下一轮」的样本由两个**注入特征**描述
——`flags` 哪几条初筛判据命中了、`channel` 这份材料从哪条渠道进来（内推 / 招聘网站 / 校招）。
**标签先行**：先定这两个取值，再由本模块的纯函数算出结论。

**为什么判据写成机械短语表**：首轮筛选在真实 ATS 里也是固定形状——年限够不够、岗位的硬性技能
在不在、有没有工作许可、学历符不符合。把四条形状写成正则之后，「命中了哪几条」就是**算**出来的，
模型碰不到；它只负责写一屏读得通的候选人摘要。

**为什么有两条判据是「不出现」**（`H2` / `H4`）：准入类判据天然有正反两面——「技能表里有
`kubernetes`」是反面，「技能表里**没有**它」才是这条判据要考的东西。两条都用负向前瞻（`(?!...)`）
写成正则，所以 `PATTERNS` 仍然是一张「命中 = 正则匹配」的表：构造侧、渲染期、判别器、审计四处
共用同一份实现（范式 §9.51）。负向前瞻只在**那一行之内**生效（`[^\n]*` 跨不过换行），这正是
「每条判据各占一整行」要的性质。

**为什么等级是加权和而不是数条数**：四条判据的后果本来就不等重——缺硬性技能（`H2`）与没有工作
许可（`H3`）各值 2 分，年限不足（`H1`）与学历不符（`H4`）各值 1 分。等级 = `1 + 权重和`、封顶
5 级，所以「两个轻判据」与「一个重判据」同档（都是 3 级）。这条规则**印在 state 里**（见
`render`），考生只读 state 就能自己算出来（范式 §9.8）。

**为什么要这一族**：社区里 `hiresignal` 明写「用 TypeSafe Jev 做简历首轮匹配和面试筛选」，
`Open Medical Jev` 那一簇（先逐项评分、再用门控决定放行还是升级给人工）是同一形状的另一个落地：
机器把一份材料摆出来，**要不要进下一轮**这个判断交给类型化决策。与 `depcheck` / `alertrisk` 不
重复：那两家判的是一个依赖的准入与一条安全警报的定级，本家判的是**一份候选人材料的首轮去留**。

**`channel` 是这一族的留出维**：它**只影响渲染、不影响标签**——同一份材料从内推、招聘网站还是
校招进来，四条判据算出来的等级一个字都不差。训练侧只见过 `referral` / `board`，留出侧是 `campus`。

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
from jc.cover import Fingerprinted, all_hit_sets, cover_dims
from jc.jsontypes import JsonValue

#: 四条初筛判据的编号（**结构键**，不是文案：判据名与判据原文全在 `locales/`）。
TAG_IDS: Final[tuple[str, ...]] = ("H1", "H2", "H3", "H4")

#: 候选人材料的**结构键**：这六个英文键名与它们的冒号是 state 格式的一部分，**不进语言文件**——
#: 渲染、校验、解析、判别器共用这一份清单，四边不可能漂移。键名用英文是因为它们本来就是格式的
#: 一部分（年限、技能表、许可状态、学历都是机器写的），只有 `note` 那一行是自然语言。
KEY_LINES: Final[tuple[str, ...]] = ("candidate", "years", "skills", "permit", "degree", "note")

#: 判据编号 → 它钉在哪一个结构键那一行上。四条判据**各占一整行**（范式 §9.51）：同一段文字上挂
#: 两条判据时，写手与读者会一起丢掉内层那一条，这里从设计上避开。
LINE_KEYS: Final[dict[str, str]] = {
    "H1": "years",
    "H2": "skills",
    "H3": "permit",
    "H4": "degree",
}

#: 岗位的**硬性技能**：`H2` 判的就是「技能表里有没有它」。它是**结构键**（与判据一起印在 state
#: 的口径行里），所以不进语言文件——换掉这一个常量就等于换掉整族的岗位要求，两语言同时生效。
REQUIRED_SKILL: Final[str] = "kubernetes"

#: 判据编号 → 判据。**这是全 workflow 唯一的判据实现**：构造侧、渲染期、判别器、审计共用。
#: 每条都只认字面、只认自己那一行：
#: - `H1` `years:` 那一行的数字**小于 3**（年限不够）；
#: - `H2` `skills:` 那一行里**没有** `kubernetes` 这个词（缺硬性技能）；
#: - `H3` `permit:` 那一行写着 `none`（没有工作许可）；
#: - `H4` `degree:` 那一行不是 `bs` / `ms` / `phd` 之一（学历不符）。
#:
#: 三处细节都不是拍出来的：
#: - `H1` 用 `[0-2](?!\d)` 而不是「把数字解析出来比大小」：`(?!\d)` 挡住 `12` / `20` 这种多位数
#:   （它们的首位落在 0~2 里），所以这条正则只认 0 / 1 / 2 三个一位数；
#: - `H2` / `H4` 是**负向前瞻**：`(?![^\n]*\bkubernetes\b)` 读作「这一行剩下的部分里没有它」，
#:   `[^\n]*` 保证只看本行；
#: - `H4` 用整词比对：`bsc` 里没有 `bs` 这个词，所以它算「学历不符」（提示词里明说只认那四个
#:   写法，写歪了渲染期会打回）。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "H1": re.compile(r"^[^\n]*\byears\s*:\s*[0-2](?!\d)", re.IGNORECASE | re.MULTILINE),
    "H2": re.compile(
        rf"^[^\n]*\bskills\s*:(?![^\n]*\b{REQUIRED_SKILL}\b)", re.IGNORECASE | re.MULTILINE
    ),
    "H3": re.compile(r"^[^\n]*\bpermit\s*:(?=[^\n]*\bnone\b)", re.IGNORECASE | re.MULTILINE),
    "H4": re.compile(
        r"^[^\n]*\bdegree\s*:(?![^\n]*\b(?:bs|ms|phd)\b)", re.IGNORECASE | re.MULTILINE
    ),
}

#: 判据编号 → 权重。等级 = `1 + 命中判据的权重和`（封顶 `MAX_LEVEL`）。
LEVEL_WEIGHTS: Final[dict[str, int]] = {"H1": 1, "H2": 2, "H3": 2, "H4": 1}

#: 等级上限（**结构键**：等级名与处置动作的文案在 `locales/`）。
MAX_LEVEL: Final[int] = 5

#: 全部等级（1~5）。
LEVELS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)

#: 处置档位（**结构键**，文案全在 `locales/`）。四档由等级经 `act_for` 映射：1 级直接排面试、
#: 2 级电话初筛、3 级人工复核、4 级及以上淘汰。四档而不是五档的理由与 `alertrisk` / `subqa`
#: 相同：档位太多会让小批量的答案分布被噪声吞掉（`pipeline.balance_report` 的噪声地板是 3 倍
#: 标准误）。
ACTS: Final[tuple[str, ...]] = ("interview", "screen", "hold", "reject")

#: 等级 → 处置档位。**印在 state 里的档位表就是按它渲染的**，所以这张表与题面不可能各说各的。
ACT_TABLE: Final[dict[int, str]] = {1: "interview", 2: "screen", 3: "hold", 4: "reject", 5: "reject"}

#: 候选人材料的字符数上下限。太短装不下六行，太长就不是「一屏材料」了。
MIN_CONTENT_CHARS: Final[int] = 100

#: 上限。六行材料加各自的取值，再长就不是一屏摘要了。
MAX_CONTENT_CHARS: Final[int] = 700

#: 写文本时的「岗位方向」池：spec 只存**下标**（与问法编号同一套约定），每种语言各有一份等长的
#: 词表（`test_locales.py` 锁住）。
#:
#: **来由**（范式 §9.57 / §9.58 的同一类坑）：一屏材料里唯一的自由文字是 `note:` 那一行，而提示词
#: 里只有「哪条渠道」一个变量——不加样本级变量的话，同一渠道的每条材料都会塌成同一句。给每条样本
#: 抽一个岗位方向，提示词就不再逐字相同了。
#:
#: 方向是**跨渠道通用**的（哪个渠道都可能投这些岗），所以不会与 `channel` 打架；它也只影响
#: `note:` 怎么写与技能表怎么列，一个字符都不参与判据。
ROLE_POOL: Final[int] = 6
ROLE_IDS: Final[tuple[int, ...]] = tuple(range(ROLE_POOL))

#: 姓名池的长度（**结构键**：池子本身是领域词表，按语言放在 `locales/` 里；spec 里存的是下标）。
#: 为什么要这一族旋钮：`candidate:` 是这一族最后一个没有样本级变量的自由字段——真机 `hire-10b`
#: 的 zh 侧 10 条里「张伟」出现了 9 次（范式 §9.60 / §9.61）。它只改 `candidate:` 那一行怎么写，
#: 与 case、标签、切分都无关。
NAME_POOL: Final[int] = 12
NAME_IDS: Final[tuple[int, ...]] = tuple(range(NAME_POOL))

#: 写文本时的「年限」池：spec 里存的是**取值本身**（数字，与语言无关）。
#:
#: **来由**（范式 §9.60 / §9.61）：`years:` 是这一族另一个「本来应该自由」的格式字段，不钉住的话
#: 模型会落回自己的先验（「三年」），整批压平。池子**按 `H1` 分两半**：命中侧只能取 0~2（年限
#: 不足那一档），不命中侧取 3 以上——两侧都不与判据打架（见 `year_pool`）。
YEAR_HINTS_SHORT: Final[tuple[int, ...]] = (0, 1, 2)
YEAR_HINTS_LONG: Final[tuple[int, ...]] = (3, 4, 5, 6, 7, 8, 9, 11, 13, 15)

#: 渠道（**结构键**，文案在 `locales/` 的 `[channel.*]`）。
CHANNEL_REFERRAL: Final[str] = "referral"
CHANNEL_BOARD: Final[str] = "board"
CHANNEL_CAMPUS: Final[str] = "campus"
CHANNELS: Final[tuple[str, ...]] = (CHANNEL_REFERRAL, CHANNEL_BOARD, CHANNEL_CAMPUS)


#: 全部命中组合（`cover_dims` 的取值表）。
HIT_SETS: Final[tuple[str, ...]] = all_hit_sets(TAG_IDS)

#: 维度表。全空间 16 × 3 = 48 组。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {"flags": HIT_SETS, "channel": CHANNELS}


def hit_set(text: str) -> tuple[str, ...]:
    """输入：一屏候选人材料。输出：命中的判据编号（按 `TAG_IDS` 顺序）。"""
    return tuple(tag for tag in TAG_IDS if PATTERNS[tag].search(text))


def hits_of(text: str) -> str:
    """输入：一屏候选人材料。输出：命中组合的规范串（`H1+H3` / 空串）。

    预期行为：与 `hit_set` 是同一件事的两种表示；`hits_of` 给比较用（`render.check_draft` 拿它
    与计划逐字比），`hit_set` 给取元素用。
    """
    return "+".join(hit_set(text))


def missing_keys(text: str) -> tuple[str, ...]:
    """输入：一屏候选人材料。输出：缺掉的**结构键**（六个键都在就给空元组）。

    预期行为：键必须**单独占一行**、后面跟冒号。为什么要单占一行：判据是钉在行上的（见
    `PATTERNS`），键不单独成行的话那四条正则连定位都定不到。渲染期用它拦「少写一行」，审计期
    `FrameJudge` 再用同一份清单复核一遍。
    """
    lowered = text.lower()
    return tuple(key for key in KEY_LINES if not re.search(rf"(?m)^\s*{key}\s*:", lowered))


def line_value(text: str, key: str) -> str:
    """输入：一屏候选人材料、结构键。输出：那一行冒号后面的内容（没有那一行就给空串）。

    预期行为：只取**第一个**匹配到的键行；行内的空白原样保留（取值里可能有空格）。
    """
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*:(.*)$", text)
    return match.group(1).strip() if match else ""


def level_of(hits: tuple[str, ...]) -> int:
    """输入：命中组合。输出：初筛等级（1~5）。

    预期行为：`1 + 命中判据的权重和`，封顶 `MAX_LEVEL`。加法形式让**单调性**是构造出来的（每个
    权重都是正数），`test_features` 里那条单调性测试就是钉这件事。
    """
    total = 1 + sum(LEVEL_WEIGHTS[tag] for tag in TAG_IDS if tag in hits)
    return min(MAX_LEVEL, total)


def act_for(level: int) -> str:
    """输入：初筛等级。输出：处置档位名。预期行为：越界等级按最高档处理（`ACT_TABLE` 收全 1~5）。"""
    return ACT_TABLE[level]


def year_pool(tags: tuple[str, ...]) -> tuple[int, ...]:
    """输入：命中组合。输出：这条样本可用的年限池。

    预期行为：`H1` 在命中组合里就给「不足 3 年」那一半，否则给 3 年以上那一半——两半都**不与判据
    打架**，所以旋钮抽到的值一定兑现得了计划（兑现不了的样本会被 `render.check_draft` 打回重渲染）。
    """
    return YEAR_HINTS_SHORT if "H1" in tags else YEAR_HINTS_LONG


def pick_years(rng: random.Random, pool: tuple[int, ...], tags: tuple[str, ...]) -> int:
    """输入：随机源、配置的年限池、命中组合。输出：这一条要写的年限。

    预期行为：只从池子里**与判据相容的那一半**取（`H1` 命中取 0~2、否则取 3 以上）；相容的那一半
    空着就报人话——配置层已经拦过一道（`HireCheckConfig.__post_init__`），这里是采样期的兜底。
    """
    half = year_pool(tags)
    allowed = tuple(item for item in pool if item in half)
    if not allowed:
        side = "命中" if "H1" in tags else "不命中"
        raise ValueError(
            f"年限池里没有与这条 case 相容的取值（H1 {side}，要从 {half} 里取）：{pool!r}"
        )
    return rng.choice(allowed)


def note_leaks_skill(note: str) -> bool:
    """输入：`note:` 那一行的内容。输出：它有没有提到岗位的硬性技能。

    预期行为：提到了就给 True。为什么要这条：`H2` 判的是「技能表里有没有 `kubernetes`」，备注里
    再提一次技能名，材料就自相矛盾了（技能表里没有、备注里却有）——那是把答案的另一半写进了自由
    文字，与范式 §9.59 那条「标签泄漏」同源。`render.check_draft` 与 `judges.HygieneJudge` 共用。
    """
    return bool(re.search(rf"\b{REQUIRED_SKILL}\b", note, re.IGNORECASE))


@dataclass(frozen=True)
class HireCheckCase(Fingerprinted):
    """一个注入特征组合：命中组合 + 渠道。

    字段：`flags` 命中组合的规范串（`+` 连接，空串表示一条都没命中）；`channel` 渠道。
    """

    flags: str
    channel: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"flags": self.flags, "channel": self.channel}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.flags or '无判据命中'}-{self.channel}"

    def tags(self) -> tuple[str, ...]:
        """输入：无。输出：命中的判据编号元组。"""
        return parse_hits(self.flags)


@dataclass(frozen=True)
class HireCheckPlan:
    """一条样本的判定计划：命中哪几条判据 + 组合题的三个干扰项。

    字段：`hits` 命中组合；`distractors` 组合题的三个干扰组合（与 `hits` 不同、彼此也不同，且都取
    「与真值只差一两处」的邻近组合，题目才有区分度）。
    """

    hits: tuple[str, ...]
    distractors: tuple[tuple[str, ...], ...]

    @property
    def level(self) -> int:
        """输入：无。输出：按定级规则算出来的初筛等级。"""
        return level_of(self.hits)


def _distractors(truth: tuple[str, ...], rng: random.Random) -> tuple[tuple[str, ...], ...]:
    """输入：真值命中组合、随机源。输出：三个干扰组合。

    预期行为：先按「与真值只差几处」排序，再从最近的一批里抽三个——差得越少越难分辨，题目才有
    区分度；抽完按 (条数, 编号) 排好，保证同一入参必得同一结果。
    """
    truth_set = set(truth)
    near = sorted(
        (
            combo
            for size in range(len(TAG_IDS) + 1)
            for combo in combinations(TAG_IDS, size)
            if set(combo) != truth_set
        ),
        key=lambda combo: (len(set(combo) ^ truth_set), combo),
    )
    picked = rng.sample(near[:8], 3)
    return tuple(sorted(picked, key=lambda combo: (len(combo), combo)))


def plan_for(case: HireCheckCase, phrasing: int) -> HireCheckPlan:
    """输入：case、问法编号。输出：判定计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里都得到
    同一份计划；两种问法会抽到不同的干扰组合，不会渲染出结构相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    return HireCheckPlan(
        hits=case.tags(), distractors=_distractors(case.tags(), random.Random(seed))
    )


@dataclass(frozen=True)
class HireCheckLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`hits` 命中的判据编号（按 `TAG_IDS` 顺序）。
    """

    hits: tuple[str, ...]

    @property
    def level(self) -> int:
        """输入：无。输出：初筛等级（按定级规则算）。"""
        return level_of(self.hits)

    @property
    def act(self) -> str:
        """输入：无。输出：按等级算出来的处置档位名。"""
        return act_for(self.level)

    def contains(self, tag: str) -> bool:
        """输入：判据编号。输出：这一条是不是命中了它。"""
        return tag in self.hits


def label_for(plan: HireCheckPlan) -> HireCheckLabels:
    """输入：判定计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`render.check_draft` 负责保证
    文本真的兑现这份计划（按字面算出来的命中组合与计划逐条相同）。
    """
    return HireCheckLabels(hits=plan.hits)


def observed_labels(text: str) -> HireCheckLabels:
    """输入：一屏候选人材料。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认文本、只认字面。两者对不上就说明渲染没兑现
    计划，审计当场就能看见。
    """
    return HireCheckLabels(hits=hit_set(text))


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[HireCheckCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`HireCheckCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [HireCheckCase(flags=str(row["flags"]), channel=str(row["channel"])) for row in rows]
