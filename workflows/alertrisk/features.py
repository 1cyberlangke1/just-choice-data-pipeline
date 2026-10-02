"""安全警报风险定级（alertrisk）的特征层：三个风险因子 + 定级矩阵 + 处置档位（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条「这条安全警报该定几级、该怎么处置」的样本由两个**注入特征**
描述——`highs` 哪几个风险因子取到了高危那一档、`source` 这条警报从哪种通道报上来（终端检测 /
Web 应用防火墙 / 云配置审计）。**标签先行**：先定这两个取值，再由本模块的纯函数算出结论。

**为什么风险因子写成机械短语表**：安全警报的风险分级在真实流水线里也是固定形状——暴露面是
不是对公网可达、利用需不需要凭据、影响范围是不是十台以上。把这三条形状写成正则之后，「哪几个
因子取到高危档」就是**算**出来的，模型碰不到——它只负责写一屏读得通、长尾的警报记录。

**为什么是矩阵而不是数条数**：这一族的中心题型是 `score`（1~5 级），而等级**不是**「命中几条」
——它是三因子经 `LEVEL_TABLE` 查表得来的。三者权重不同：对公网可达单独就能把等级顶到 3 级，
而「无需凭据」单独只到 2 级。表印在 state 里（见 `render`），考生只读 state 就能查出来。

**为什么要这一族**：社区里 `triagedy`（安全警报分类过滤器，★1）明写「模型返回类型化配置、
**策略路由保留在代码里**」——本 workflow 的定级矩阵与处置档位表就是那份「保留在代码里」的策略；
`typesafe-triage-guard`（可观测性告警分诊 + 部署风险门控）与 `captaincore`（★71，把恶意软件
扫描结果做成 noul / choice / score 三类决策）是同一形状的另外两个落地。

**与 `logtriage` / `guardrail` 的区别**：`logtriage` 判的是**一条日志行值不值得进告警通道**
（噪声判据），`guardrail` 判的是一条内容有没有踩政策条款；本家判的是**一条已经报上来的警报**
该定几级、走哪一档处置，判据是暴露面 / 利用条件 / 影响范围，素材形态与结论空间都不一样。

**`source` 是这一族的留出维**：它**只影响渲染、不影响标签**——同一组因子写在终端检测、WAF 还是
云配置审计的警报里，矩阵算出来的等级一个字都不差。训练侧只见过 `edr` / `waf`，留出侧是 `cloud`。

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

#: 三个风险因子的编号（**结构键**，不是文案：因子名与判据原文全在 `locales/`）。
TAG_IDS: Final[tuple[str, ...]] = ("R1", "R2", "R3")

#: 警报记录的**结构键**：这五个英文键名与它们的冒号是 state 格式的一部分，**不进语言文件**——
#: 渲染、校验、解析、判别器共用这一份清单，四边不可能漂移。键名用英文是因为它们本来就是格式的
#: 一部分（规则名、主机数、可达性、凭据要求都是机器写的），只有 `detail` 那一行是自然语言。
KEY_LINES: Final[tuple[str, ...]] = ("rule", "impact", "exposure", "exploit", "detail")

#: 因子编号 → 它钉在哪一个结构键那一行上。三条判据**各占一整行**（范式 §9.51）：同一段文字上
#: 挂两条判据时，写手与读者会一起丢掉内层那一条，这里从设计上避开——三条判据连一行都不共用。
LINE_KEYS: Final[dict[str, str]] = {"R1": "exposure", "R2": "exploit", "R3": "impact"}

#: 因子编号 → 判据。**这是全 workflow 唯一的判据实现**：构造侧、渲染期、判别器、审计共用。
#: 每条都只认字面、只认自己那一行：
#: - `R1` `exposure:` 行里写着 `reachable=public`（对公网可达）；
#: - `R2` `exploit:` 行里写着 `auth=none`（无需凭据就能利用）；
#: - `R3` `impact:` 行里 `hosts=` 后面**至少两位数字**（十台以上）。
#:
#: `R3` 的「两位」不是拍出来的：`hosts=` 后面跟的是十进制整数、没有前导零，「至少两位」与
#: 「不小于 10」是同一件事——写成 `\d{2,}` 就不用把数字解析出来再比大小（`sqlreview` 的
#: `rows=` 同款）。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "R1": re.compile(
        r"^[^\n]*\bexposure\s*:(?=[^\n]*\breachable\s*=\s*public\b)",
        re.IGNORECASE | re.MULTILINE,
    ),
    "R2": re.compile(
        r"^[^\n]*\bexploit\s*:(?=[^\n]*\bauth\s*=\s*none\b)",
        re.IGNORECASE | re.MULTILINE,
    ),
    "R3": re.compile(
        r"^[^\n]*\bimpact\s*:(?=[^\n]*\bhosts\s*=\s*\d{2,}\b)",
        re.IGNORECASE | re.MULTILINE,
    ),
}

#: 警报来源（**结构键**，文案在 `locales/` 的 `[source.*]`）。
#:
#: 这一维**只影响渲染、不影响标签**：同一组因子写在终端检测、WAF 还是云配置审计的警报里，矩阵
#: 算出来的等级一个字都不差——三条判据认的是 `reachable=` / `auth=` / `hosts=` 这三个取值，与
#: 报警通道无关。它正是 test_ood 的留出维：训练侧只见过 `edr` / `waf`，留出侧是 `cloud`。
SOURCE_EDR: Final[str] = "edr"
SOURCE_WAF: Final[str] = "waf"
SOURCE_CLOUD: Final[str] = "cloud"
SOURCES: Final[tuple[str, ...]] = (SOURCE_EDR, SOURCE_WAF, SOURCE_CLOUD)

#: 定级矩阵：三个因子（按 `TAG_IDS` 顺序取「是否取到高危档」）→ 风险等级。
#:
#: 权重不是均等的：`R1`（对公网可达）单独就能到 3 级、`R2` / `R3` 单独只到 2 级，三者齐了才是
#: 5 级。**八格取遍之后五档的分布是 1 / 2 / 2 / 2 / 1**，而且每一维都单调（把任一因子从低换成
#: 高，等级不会降）——单调性有测试钉着，改表时会被当场拦下。
LEVEL_TABLE: Final[dict[tuple[bool, bool, bool], int]] = {
    (False, False, False): 1,
    (False, True, False): 2,
    (False, False, True): 2,
    (True, False, False): 3,
    (False, True, True): 3,
    (True, True, False): 4,
    (True, False, True): 4,
    (True, True, True): 5,
}

#: 全部风险等级（1~5，**结构键**：等级名与处置动作的文案在 `locales/`）。
LEVELS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)

#: 处置档位（**结构键**，文案全在 `locales/`）。四档由等级经 `act_for` 映射：1 级记入清单、
#: 2 级开单排期、3 级当天出补丁、4 级及以上先隔离再修。四档而不是五档的理由与 `subqa` 相同：
#: 档位太多会让小批量的答案分布被噪声吞掉（`pipeline.balance_report` 的噪声地板是 3 倍标准误）。
ACTS: Final[tuple[str, ...]] = ("monitor", "ticket", "patch_now", "isolate")

#: 警报记录的字符数上下限。太短装不下五行（规则名 + 三个因子 + 一句描述），太长就不是「一屏」了。
MIN_CONTENT_CHARS: Final[int] = 80

#: 上限。五行材料加各自的取值，再长就不是一条警报记录了。
MAX_CONTENT_CHARS: Final[int] = 600

#: 写文本时的「手法方向」池：spec 只存**下标**（与问法编号同一套约定），每种语言各有一份等长的
#: 词表（`test_locales.py` 锁住）。
#:
#: **来由**（真机实测，范式 §9.7 那个坑）：一屏记录里唯一的自由文字是 `detail:` 那一行，而提示词里
#: 只有「哪条通道」一个变量——于是**同一通道的每一条记录都塌成同一句**：9 条中文产物里 5 条 EDR
#: 全写「PowerShell 脚本」、4 条 WAF 全写「SQL 注入」，规则名与描述一起塌。给每条样本抽一个手法
#: 方向，提示词就不再逐字相同了。
#:
#: 方向是**跨通道通用**的（凭据窃取 / 数据外传 / 加密勒索…哪条通道都报得出来），所以不会与
#: `source` 打架；它也只影响 `detail:` 怎么写，一个字符都不参与判据。
DIRECTION_POOL: Final[int] = 8
DIRECTION_IDS: Final[tuple[int, ...]] = tuple(range(DIRECTION_POOL))


#: 全部命中组合（`cover_dims` 的取值表）。
HIT_SETS: Final[tuple[str, ...]] = all_hit_sets(TAG_IDS)

#: 维度表。全空间 8 × 3 = 24 组。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {"highs": HIT_SETS, "source": SOURCES}


def hit_set(text: str) -> tuple[str, ...]:
    """输入：一屏警报记录。输出：取到高危档的因子编号（按 `TAG_IDS` 顺序）。"""
    return tuple(tag for tag in TAG_IDS if PATTERNS[tag].search(text))


def hits_of(text: str) -> str:
    """输入：一屏警报记录。输出：命中组合的规范串（`R1+R3` / 空串）。

    预期行为：与 `hit_set` 是同一件事的两种表示；`hits_of` 给比较用（`render.check_draft` 拿它
    与计划逐字比），`hit_set` 给取元素用。
    """
    return "+".join(hit_set(text))


def missing_keys(text: str) -> tuple[str, ...]:
    """输入：一屏警报记录。输出：缺掉的**结构键**（五个键都在就给空元组）。

    预期行为：键必须**单独占一行**、后面跟冒号。为什么要单占一行：判据是钉在行上的
    （见 `PATTERNS`），键不单独成行的话那三条正则连定位都定不到。渲染期用它拦「少写一行」，
    审计期 `FrameJudge` 再用同一份清单复核一遍。
    """
    lowered = text.lower()
    return tuple(key for key in KEY_LINES if not re.search(rf"(?m)^\s*{key}\s*:", lowered))


def line_value(text: str, key: str) -> str:
    """输入：一屏警报记录、结构键。输出：那一行冒号后面的内容（没有那一行就给空串）。

    预期行为：只取**第一个**匹配到的键行；行内的空白原样保留（取值里可能有空格）。
    """
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*:(.*)$", text)
    return match.group(1).strip() if match else ""


def flags_of(highs: tuple[str, ...]) -> tuple[bool, bool, bool]:
    """输入：命中组合。输出：三个因子（按 `TAG_IDS` 顺序）取没取到高危档。

    预期行为：顺序固定，所以矩阵查表的键与 `LEVEL_TABLE` 的键是同一种形状。
    """
    return (TAG_IDS[0] in highs, TAG_IDS[1] in highs, TAG_IDS[2] in highs)


def level_of(highs: tuple[str, ...]) -> int:
    """输入：命中组合。输出：风险等级（1~5）。预期行为：查 `LEVEL_TABLE`，不做别的推理。"""
    return LEVEL_TABLE[flags_of(highs)]


def act_for(level: int) -> str:
    """输入：风险等级。输出：处置档位名。

    预期行为：1→monitor、2→ticket、3→patch_now、4 及以上→isolate（最后一档吃掉所有更高的级）。
    """
    return ACTS[min(level - 1, len(ACTS) - 1)]


def combo_text_key(highs: tuple[str, ...]) -> str:
    """输入：命中组合。输出：矩阵那一行的因子取值键（如 `R1=high R2=low R3=low`）。

    预期行为：**这是 state 格式的一部分**（矩阵表的行首），不进语言文件；三个因子按 `TAG_IDS`
    顺序排，高低用 ASCII 的 high / low 写——它们是格式，不是文案。
    """
    marks = ("high" if tag in highs else "low" for tag in TAG_IDS)
    return " ".join(f"{tag}={mark}" for tag, mark in zip(TAG_IDS, marks, strict=True))


@dataclass(frozen=True)
class AlertRiskCase(Fingerprinted):
    """一个注入特征组合：哪几个因子取到高危档 + 这条警报从哪种通道来。

    字段：`highs` 命中组合的规范串；`source` 报警通道（`SOURCES` 里的一个）。
    """

    highs: str
    source: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"highs": self.highs, "source": self.source}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.highs or '无高危因子'}-{self.source}"

    def tags(self) -> tuple[str, ...]:
        """输入：无。输出：取到高危档的因子编号元组。"""
        return parse_hits(self.highs)


@dataclass(frozen=True)
class AlertRiskPlan:
    """一条样本的判定计划：命中哪几个因子 + 组合题的三个干扰项。

    字段：`highs` 命中组合；`distractors` 组合题的三个干扰组合（与 `highs` 不同、彼此也不同，
    且都取「与真值只差一两处」的邻近组合，题目才有区分度）。
    """

    highs: tuple[str, ...]
    distractors: tuple[tuple[str, ...], ...]

    @property
    def level(self) -> int:
        """输入：无。输出：按矩阵算出来的风险等级。"""
        return level_of(self.highs)


def _distractors(truth: tuple[str, ...], rng: random.Random) -> tuple[tuple[str, ...], ...]:
    """输入：真值命中组合、随机源。输出：三个干扰组合。

    预期行为：先按「与真值只差几处」排序，再从最近的一批里抽三个——差得越少越难分辨，题目才
    有区分度；抽完按 (条数, 编号) 排好，保证同一入参必得同一结果。
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
    picked = rng.sample(near[:6], 3)
    return tuple(sorted(picked, key=lambda combo: (len(combo), combo)))


def plan_for(case: AlertRiskCase, phrasing: int) -> AlertRiskPlan:
    """输入：case、问法编号。输出：判定计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里都
    得到同一份计划；两种问法会抽到不同的干扰组合，不会渲染出结构相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    return AlertRiskPlan(
        highs=case.tags(), distractors=_distractors(case.tags(), random.Random(seed))
    )


@dataclass(frozen=True)
class AlertRiskLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`highs` 取到高危档的因子编号（按 `TAG_IDS` 顺序）。
    """

    highs: tuple[str, ...]

    @property
    def level(self) -> int:
        """输入：无。输出：风险等级（查矩阵）。"""
        return level_of(self.highs)

    @property
    def act(self) -> str:
        """输入：无。输出：按等级算出来的处置档位名。"""
        return act_for(self.level)

    def contains(self, tag: str) -> bool:
        """输入：因子编号。输出：这一条是不是取到了高危档。"""
        return tag in self.highs


def label_for(plan: AlertRiskPlan) -> AlertRiskLabels:
    """输入：判定计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`render.check_draft` 负责保证
    文本真的兑现这份计划（按字面算出来的命中组合与计划逐条相同）。
    """
    return AlertRiskLabels(highs=plan.highs)


def observed_labels(text: str) -> AlertRiskLabels:
    """输入：一屏警报记录。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认文本、只认字面。两者对不上就说明渲染没兑现
    计划，审计当场就能看见。
    """
    return AlertRiskLabels(highs=hit_set(text))


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[AlertRiskCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`AlertRiskCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [AlertRiskCase(highs=str(row["highs"]), source=str(row["source"])) for row in rows]
