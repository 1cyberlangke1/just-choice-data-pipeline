"""模型路由与降级 workflow 的真值规格：一张档位表 + 一组请求画像。

输入/输出：不可变数据类与纯函数；这里只有**键**（档位名 / 任务名 / 各档取值），
中英文案全在 `locales/`。
预期行为：答案只从渲染出来的那屏画像反推——档位表（成本 / 能力 / 速度 / 窗口）、任务能力
需求表、升级链、推理深度挪档表、成本与缓存规则都是已知的**纯函数**，构造侧与判别侧共用
同一套，所以「表写错」与「渲染写歪」会变成同一类错误，不会两边各错一半还互相印证。

覆盖的决策形状（社区用例地图里最大的缺口：模型路由与降级，86 条第三方项目）：
- **选档**（`tier`）：这条请求的首选档位 = 能力需求对应的最低档；
- **链**（`next`）：升级链上首选档之后的**一站**（两阶段依赖：max 之上没有档，不出这题）；
- **检测**（`is_first`）：这条请求是不是首次请求；
- **定档**（`effort`）：推理深度档（复杂度档按延迟预算挪一档）；
- **门控**（`degrade`）：成本敏感时是否允许降一档能力；
- **门控**（`cache`）：私有数据不进缓存。

六条题族的答案分布**全由构造保证**均衡：`tier` 四个档各 1/4、`next` 三个档各 1/3、
`effort` 三档各 1/3（复杂度三档 × 延迟三档的挪档表恰好三等分）、三个 noul 族各 0.5。
均衡是**构造出来的**，不靠事后过滤（范式 §5.7）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import product
from typing import Any, Final

from jc.knobs import require_pool

# ---------- 档位表 ----------

#: 档位：从最便宜到最强。名字只是键，文案在 locales/。
TIERS: Final[tuple[str, ...]] = ("nano", "mini", "core", "max")

#: 每档的属性（成本 / 能力 / 速度 / 窗口，1~4 分；**速度越大越快**）。
#: 数值是挑过的：速度与成本**不**同序（mini 最快、nano 次之），否则「延迟紧时取最快的
#: 够用档」会退化成「取最便宜的档」，那一条规则就白写了。
PROFILE: Final[dict[str, dict[str, int]]] = {
    "nano": {"cost": 1, "power": 1, "speed": 3, "window": 1},
    "mini": {"cost": 2, "power": 2, "speed": 4, "window": 2},
    "core": {"cost": 3, "power": 3, "speed": 2, "window": 3},
    "max": {"cost": 4, "power": 4, "speed": 1, "window": 3},
}

# ---------- 任务类型 ----------

#: 任务类型：按**能力需求**从低到高排。名字只是键，文案在 locales/。
TASKS: Final[tuple[str, ...]] = ("classify", "extract", "codefix", "reason")

#: 任务 → 能力需求（1~4）。四个任务恰好把四个需求档各占一次，`tier` 因此天然均匀。
POWER_NEED: Final[dict[str, int]] = {
    "classify": 1,
    "extract": 2,
    "codefix": 3,
    "reason": 4,
}

#: 能力需求档 → 档位（能力分恰好等于需求分的那个档）。四档一一对应，所以是满射。
TIER_FOR_POWER: Final[dict[int, str]] = {
    profile["power"]: tier for tier, profile in PROFILE.items()
}

# ---------- 画像维度 ----------

#: 请求复杂度信号（决定推理深度基准档）。
COMPLEXITIES: Final[tuple[str, ...]] = ("low", "mid", "high")
#: 延迟预算：宽松 / 常规 / 紧。
LATENCIES: Final[tuple[str, ...]] = ("relaxed", "normal", "strict")
#: 成本敏感度。
COSTS: Final[tuple[str, ...]] = ("normal", "tight")
#: 尝试记录：首次请求 / 已经失败过一次。
RETRIES: Final[tuple[str, ...]] = ("first", "retry")
#: 数据分级：公开 / 私有。
DATAS: Final[tuple[str, ...]] = ("public", "private")
#: 句面（只影响措辞，不影响答案）。
PHRASINGS: Final[tuple[int, ...]] = (0, 1)

#: 推理深度的三个档（从浅到深）。
STEPS: Final[tuple[str, ...]] = ("low", "mid", "high")
#: 延迟预算 → 挪档量：宽松上挪一档、常规不动、紧下挪一档。
SHIFT: Final[dict[str, int]] = {"relaxed": 1, "normal": 0, "strict": -1}

# ---------- 规则函数（构造与判别共用） ----------


def tier_of(task: str) -> str:
    """输入：任务类型。输出：首选档位。

    预期行为：查「能力需求 → 档位」表；任务名不认识直接报错（不静默给个默认档——
    那会把「任务表改了、读法忘了改」变成一堆答案对不上的样本）。
    """
    if task not in POWER_NEED:
        raise KeyError(f"没有任务类型 {task!r}（可用：{', '.join(TASKS)}）")
    return TIER_FOR_POWER[POWER_NEED[task]]


def has_next(task: str) -> bool:
    """输入：任务类型。输出：这条请求有没有升级链。

    预期行为：首选档不是最高档就有链。`max` 之上没有档，所以 `reason` 没有升级链——
    这正是 `next` 那一族的两阶段依赖。
    """
    return POWER_NEED[task] < max(POWER_NEED.values())


def next_tier_of(task: str) -> str | None:
    """输入：任务类型。输出：升级链的下一站；没有链时给 None。"""
    if not has_next(task):
        return None
    return TIER_FOR_POWER[POWER_NEED[task] + 1]


def effort_of(complexity: str, latency: str) -> str:
    """输入：复杂度档、延迟预算。输出：推理深度档。

    预期行为：以复杂度档为基准，按 `SHIFT` 挪档，越界就停在端点上（不报错——端点钳位
    是这条规则的一部分，题面里写明了「到顶或触底不再挪」）。
    """
    index = STEPS.index(complexity) + SHIFT[latency]
    return STEPS[max(0, min(len(STEPS) - 1, index))]


def is_first_of(retry: str) -> bool:
    """输入：尝试记录。输出：是不是首次请求。预期行为：查值（首次请求为真）。"""
    return retry == "first"


def degrade_of(cost: str) -> bool:
    """输入：成本敏感度。输出：是否允许降一档能力。预期行为：成本敏感时为真。"""
    return cost == "tight"


def cache_of(data: str) -> bool:
    """输入：数据分级。输出：能不能进缓存。预期行为：公开数据可以、私有数据不行。"""
    return data == "public"


# ---------- case ----------


@dataclass(frozen=True)
class RouteCase:
    """一个请求画像：七个维度的组合。"""

    task: str
    complexity: str
    latency: str
    cost: str
    retry: str
    data: str
    phrasing: int

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：{维度名: 取值}（七个维度全摊平）。"""
        return {
            "task": self.task,
            "complexity": self.complexity,
            "latency": self.latency,
            "cost": self.cost,
            "retry": self.retry,
            "data": self.data,
            "phrasing": self.phrasing,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return (
            f"{self.task}/{self.complexity}/{self.latency}/"
            f"{self.cost}/{self.retry}/{self.data}/{self.phrasing}"
        )

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这个组合里「训练侧没见过」的维度名（按固定顺序）。

        预期行为：留出两处——`task` 的 `reason`（答案相关的值留出）与 `phrasing` 的第二套
        句面（措辞轴）。另外几个维度**不能**留出：`cost` / `data` / `retry` 各自是某个 noul
        族的唯一判据，留出一个取值就会让那一族在训练侧退化成常量；`complexity` / `latency`
        留出会把 `effort` 的三档分布压歪（见 PLAN 10.61）。
        """
        out: list[str] = []
        if self.task not in SEEN_TASKS:
            out.append("task")
        if self.phrasing not in SEEN_PHRASINGS:
            out.append("phrasing")
        return tuple(out)


def material_of(case: RouteCase) -> str:
    """输入：case。输出：素材指纹（确定性；切分分桶与内容种子都由它派生）。"""
    return (
        f"{case.task}|{case.complexity}|{case.latency}|{case.cost}"
        f"|{case.retry}|{case.data}|{case.phrasing}"
    )


# ---------- 值留出（训练侧见过哪些取值） ----------

#: 训练侧见过的任务类型：`reason` 整个留出（连带「多步推理 → max」这一格与整条
#: 「max 之上没有升级链」的判断一起没见过）。
SEEN_TASKS: Final[frozenset[str]] = frozenset(task for task in TASKS if task != "reason")
#: 训练侧见过的句面：第二套句面整个留出（措辞轴）。
#: 它**不影响任何答案**，所以留出它不会把哪一族的分布压歪——顺带把训练侧的 `tier`
#: 留在了噪声地板之内（训练侧少一个任务档，四档里有一档取不到）。
SEEN_PHRASINGS: Final[frozenset[int]] = frozenset({PHRASINGS[0]})


# ---------- 采样参数 ----------


@dataclass(frozen=True)
class RouteConfig:
    """一个切分的采样参数：六张取值表 + 是否只留 OOD。"""

    tasks: tuple[str, ...] = TASKS
    complexities: tuple[str, ...] = COMPLEXITIES
    latencies: tuple[str, ...] = LATENCIES
    costs: tuple[str, ...] = COSTS
    retries: tuple[str, ...] = RETRIES
    datas: tuple[str, ...] = DATAS
    phrasings: tuple[int, ...] = PHRASINGS
    split: str = "train"
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：任一取值表为空就报人话（点名是哪个字段）。"""
        require_pool("tasks", self.tasks)
        require_pool("complexities", self.complexities)
        require_pool("latencies", self.latencies)
        require_pool("costs", self.costs)
        require_pool("retries", self.retries)
        require_pool("datas", self.datas)
        require_pool("phrasings", self.phrasings)


def build_cases(config: RouteConfig) -> list[RouteCase]:
    """输入：采样参数。输出：case 矩阵（顺序稳定）。

    预期行为：七张表取全组合（**不做成立性过滤**——七个维度两两不矛盾）；
    `ood_only` 时只留「至少含一个训练侧没见过的取值」的组合，所以 OOD 与同分布侧
    按构造就不重叠。
    """
    out: list[RouteCase] = []
    for task, complexity, latency, cost, retry, data, phrasing in product(
        config.tasks,
        config.complexities,
        config.latencies,
        config.costs,
        config.retries,
        config.datas,
        config.phrasings,
    ):
        case = RouteCase(task, complexity, latency, cost, retry, data, phrasing)
        if config.ood_only and not case.unseen_dims():
            continue
        out.append(case)
    return out


# ---------- 真值规格 ----------

REQUEST_PREFIX: Final[str] = "R-"


@dataclass(frozen=True)
class RouteSpec:
    """一条请求的完整真值：case + 由规则算出来的一切。"""

    case: RouteCase
    split: str
    seq: int
    phrasing: int
    request_id: str
    tier: str
    next_tier: str | None
    effort: str
    is_first: bool
    degrade: bool
    cache: bool


def request_id_of(case: RouteCase) -> str:
    """输入：case。输出：请求号（形如 `R-48213`）。

    预期行为：由素材指纹的 sha256 派生——确定性，而且**不泄露切分**：用序号或切分名
    拼请求号会让同一条请求在不同切分里拿到不同请求号，等于把切分信息写进 state。
    """
    digest = hashlib.sha256(material_of(case).encode("utf-8")).hexdigest()
    return f"{REQUEST_PREFIX}{10000 + int(digest[:6], 16) % 90000}"


def sample_spec(case: RouteCase, split: str, seq: int, phrasing: int) -> RouteSpec:
    """输入：case、切分、序号、问法。输出：真值规格。

    预期行为：**不掷骰子**——同一 case 必得同一份规格（请求号也由 case 指纹派生），
    这样「同素材同题必须同答」这条一致性校验才成立。
    """
    return RouteSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        request_id=request_id_of(case),
        tier=tier_of(case.task),
        next_tier=next_tier_of(case.task),
        effort=effort_of(case.complexity, case.latency),
        is_first=is_first_of(case.retry),
        degrade=degrade_of(case.cost),
        cache=cache_of(case.data),
    )


def spec_to_dict(spec: RouteSpec) -> dict[str, Any]:
    """输入：规格。输出：可 JSON 序列化的摘要（写进 manifest 与调试，不进样本行）。"""
    return {
        "case": spec.case.describe(),
        "unseen_dims": list(spec.case.unseen_dims()),
        "split": spec.split,
        "seq": spec.seq,
        "request_id": spec.request_id,
        "tier": spec.tier,
        "next_tier": spec.next_tier,
        "effort": spec.effort,
        "is_first": spec.is_first,
        "degrade": spec.degrade,
        "cache": spec.cache,
    }
