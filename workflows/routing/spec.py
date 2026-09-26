"""工单分诊 workflow 的真值规格：一条工单 + 一组路由规则表。

输入/输出：不可变数据类与纯函数；**不含任何领域词表**（产品名、组名、影响面 / 紧急度 /
合规标记的文案、正文模板、选项描述与判据原文全在 `locales/`）。
预期行为：所有答案都**只从渲染出来的那屏工单文本反推**——严重度矩阵、派单表、升级表、
优先级表、SLA 时限表都是已知的表，构造侧与判别侧**共用同一套规则函数**，
所以「表格写错」与「渲染写歪」会变成同一类错误，不会两边各错一半还互相印证。

为什么不做成 LLM 生成：工单是**结构化的**（产品域 / 等级 / 影响面 / 紧急度 / 合规标记 /
时间），派单规则也是已知的表。让模型去编只会引入抖动；这里要训的是
「按规则读一条工单并派单」，不是「编一条工单」。
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from typing import Any, Final

from jc.knobs import require_pool

# ---------- 产品域 ----------

AREAS: Final[tuple[str, ...]] = ("payments", "storage", "analytics", "identity")
#: 关键产品域：要额外挂观察（`critical_area` 那一族的判据就是「在不在这个集合里」）。
CRITICAL_AREAS: Final[frozenset[str]] = frozenset({"payments", "identity"})

# ---------- 问题类型 ----------

ISSUES: Final[tuple[str, ...]] = ("outage", "data_loss", "bug", "howto", "feature")

# ---------- 客户等级 ----------

TIERS: Final[tuple[str, ...]] = ("free", "pro", "enterprise")

# ---------- 影响面 / 紧急度 ----------

IMPACTS: Final[tuple[str, ...]] = ("single_user", "team", "all_users")
URGENCIES: Final[tuple[str, ...]] = ("can_wait", "blocked", "deadline")

# ---------- 合规标记 ----------

COMPLIANCES: Final[tuple[str, ...]] = ("none", "gdpr", "security")

# ---------- 已等待时长 ----------

WAITS: Final[tuple[str, ...]] = ("short", "medium", "long")
#: 已等待档 → 分钟数（渲染时换算成「创建时间 / 当前时间」两格）。
WAIT_MINUTES: Final[dict[str, int]] = {"short": 8, "medium": 45, "long": 200}

# ---------- 严重度 ----------

SEVERITIES: Final[tuple[str, ...]] = ("s1", "s2", "s3", "s4")
#: 严重度矩阵：影响面 × 紧急度 → 严重度（九格全写，判据原文照抄进题面）。
SEVERITY_TABLE: Final[dict[tuple[str, str], str]] = {
    ("single_user", "can_wait"): "s4",
    ("single_user", "blocked"): "s4",
    ("single_user", "deadline"): "s3",
    ("team", "can_wait"): "s3",
    ("team", "blocked"): "s2",
    ("team", "deadline"): "s2",
    ("all_users", "can_wait"): "s3",
    ("all_users", "blocked"): "s1",
    ("all_users", "deadline"): "s1",
}

# ---------- 队列 ----------

QUEUES: Final[tuple[str, ...]] = (
    "support_l1",
    "engineering",
    "incident_response",
    "privacy_ops",
    "security_incident",
    "product_backlog",
)

# ---------- 升级 ----------

ESCALATIONS: Final[tuple[str, ...]] = ("none", "security_lead", "privacy_lead")
#: 合规标记 → 升级对象（没有标记就不升级）。
ESCALATION_BY_COMPLIANCE: Final[dict[str, str]] = {
    "security": "security_lead",
    "gdpr": "privacy_lead",
}

# ---------- 优先级 ----------

PRIORITY_LEVELS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
#: 优先级表：严重度 × 客户等级 → 0~4 档。**最高与最低两档不看客户等级**，
#: 中间两档（s2 / s3）合并成同一行——这张表是「查表」，不是「加权求和」，
#: 别拿严重度去线性外推（外推出来的分布会一头沉，见 PLAN 10.51）。
PRIORITY_TABLE: Final[dict[tuple[str, str], int]] = {
    ("s1", "free"): 4,
    ("s1", "pro"): 4,
    ("s1", "enterprise"): 4,
    ("s2", "free"): 1,
    ("s2", "pro"): 2,
    ("s2", "enterprise"): 3,
    ("s3", "free"): 1,
    ("s3", "pro"): 2,
    ("s3", "enterprise"): 3,
    ("s4", "free"): 0,
    ("s4", "pro"): 0,
    ("s4", "enterprise"): 0,
}

# ---------- SLA ----------

#: SLA 时限表：严重度 × 客户等级 → 分钟（首次响应时限）。
#: 数值是**挑过**的：它决定 `sla_breach` 的真值占比，见 PLAN 10.51 的均衡账。
SLA_TABLE: Final[dict[tuple[str, str], int]] = {
    ("s1", "free"): 10,
    ("s1", "pro"): 5,
    ("s1", "enterprise"): 3,
    ("s2", "free"): 30,
    ("s2", "pro"): 15,
    ("s2", "enterprise"): 10,
    ("s3", "free"): 120,
    ("s3", "pro"): 60,
    ("s3", "enterprise"): 30,
    ("s4", "free"): 480,
    ("s4", "pro"): 300,
    ("s4", "enterprise"): 200,
}

# ---------- 工单号与时间基准 ----------

TICKET_PREFIX: Final[str] = "T-"
#: 创建时间的基准（08:00 起，按产品域序号每档错开 10 分钟）。
BASE_HOUR: Final[int] = 8
AREA_MINUTE_STEP: Final[int] = 10


# ---------- 规则引擎（构造与判别共用） ----------


def severity_of(impact: str, urgency: str) -> str:
    """输入：影响面、紧急度。输出：严重度（s1~s4）。

    预期行为：查矩阵；表里没有的组合直接报错（矩阵漏格比静默给错答案好）。
    """
    try:
        return SEVERITY_TABLE[(impact, urgency)]
    except KeyError as exc:
        raise ValueError(f"严重度矩阵里没有这个组合：{impact!r} × {urgency!r}") from exc


def queue_of(issue: str, compliance: str) -> str:
    """输入：问题类型、合规标记。输出：队列。

    预期行为：**按顺序**取第一个命中的分支（顺序本身是判据的一部分）：
    ① 咨询类 → 一线支持组；② 数据删除请求且问题类型是数据丢失 / 需求 → 隐私合规组；
    ③ 安全报告且问题类型是故障 / 数据丢失 / 缺陷 → 安全事件组；
    ④ 故障或数据丢失 → 事故响应组；⑤ 缺陷 → 工程组；⑥ 其余 → 产品待办组。
    """
    if issue == "howto":
        return "support_l1"
    if compliance == "gdpr" and issue in ("data_loss", "feature"):
        return "privacy_ops"
    if compliance == "security" and issue in ("outage", "data_loss", "bug"):
        return "security_incident"
    if issue in ("outage", "data_loss"):
        return "incident_response"
    if issue == "bug":
        return "engineering"
    return "product_backlog"


def escalation_of(compliance: str) -> str:
    """输入：合规标记。输出：升级对象。

    预期行为：安全报告叫安全负责人、数据删除请求叫隐私负责人、没有标记就不升级——
    合规标记是**唯一**判据；严重度只影响优先级与时限，不改变叫谁。
    """
    return ESCALATION_BY_COMPLIANCE.get(compliance, "none")


def priority_of(severity: str, tier: str) -> int:
    """输入：严重度、客户等级。输出：优先级 0~4。预期行为：查表（不做加权外推）。"""
    return PRIORITY_TABLE[(severity, tier)]


def critical_area_of(area: str) -> bool:
    """输入：产品域。输出：是不是关键产品域。预期行为：查集合。"""
    return area in CRITICAL_AREAS


def needs_human_of(severity: str, compliance: str, tier: str) -> bool:
    """输入：严重度、合规标记、客户等级。输出：是否必须人工介入。

    预期行为：两条命中任意一条即为真——严重度 s1/s2；或者企业版的**安全报告**。
    这两条是「不许自动关单」的理由。
    """
    return severity in ("s1", "s2") or (compliance == "security" and tier == "enterprise")


def data_risk_of(issue: str, compliance: str) -> bool:
    """输入：问题类型、合规标记。输出：是否要走数据事故流程。

    预期行为：问题类型是数据丢失，或者合规标记是数据删除请求——任一条成立即为真。
    """
    return issue == "data_loss" or compliance == "gdpr"


def sla_limit_of(severity: str, tier: str) -> int:
    """输入：严重度、客户等级。输出：SLA 时限（分钟）。预期行为：查表。"""
    return SLA_TABLE[(severity, tier)]


def elapsed_of(wait: str) -> int:
    """输入：已等待档。输出：已等待分钟数。预期行为：查表（短 8 / 中 45 / 长 200）。"""
    return WAIT_MINUTES[wait]


def sla_breach_of(elapsed: int, limit: int) -> bool:
    """输入：已等待分钟、SLA 时限。输出：是否超时。

    预期行为：**严格大于**才算超时——刚好卡在时限上不算（判据原文写在题面里）。
    """
    return elapsed > limit


def times_of(area: str, wait: str) -> tuple[str, str]:
    """输入：产品域、已等待档。输出：(创建时间, 当前时间)，都是 `HH:MM`。

    预期行为：创建时间 = 08:00 + 产品域序号 × 10 分钟，当前时间 = 创建时间 + 已等待分钟。
    只渲染这两格、**不渲染分钟数**——「已等待多久」由判别器从这两格相减反推。
    """
    start = dt.datetime(2026, 1, 1, BASE_HOUR, 0) + dt.timedelta(
        minutes=AREA_MINUTE_STEP * AREAS.index(area)
    )
    return (
        start.strftime("%H:%M"),
        (start + dt.timedelta(minutes=elapsed_of(wait))).strftime("%H:%M"),
    )


# ---------- case 与采样参数 ----------


@dataclass(frozen=True)
class TicketCase:
    """一条工单的七个维度（顺序与 `as_dict` 一致，哈希分桶按它算，必须稳定）。"""

    area: str
    issue: str
    tier: str
    impact: str
    urgency: str
    compliance: str
    wait: str

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：{维度名: 取值}（顺序稳定）。"""
        return {
            "area": self.area,
            "issue": self.issue,
            "tier": self.tier,
            "impact": self.impact,
            "urgency": self.urgency,
            "compliance": self.compliance,
            "wait": self.wait,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return (
            f"{self.area}/{self.issue}/{self.tier}/{self.impact}×{self.urgency}"
            f"/合规={self.compliance}/等={self.wait}"
        )

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这个 case 里训练侧没见过的维度名（空 = 全部见过）。"""
        checks: tuple[tuple[str, Any, frozenset[Any]], ...] = (
            ("area", self.area, SEEN_AREAS),
            ("issue", self.issue, SEEN_ISSUES),
            ("compliance", self.compliance, SEEN_COMPLIANCES),
        )
        return tuple(name for name, value, seen in checks if value not in seen)


def material_of(case: TicketCase) -> str:
    """输入：case。输出：素材指纹——决定**工单文本**的那几维拼成的串。

    预期行为：本 workflow 没有「只改题面、不改工单」的维度（题面点名的对象就是工单本身），
    所以指纹 = 全部维度。切分按它分桶（`workflow._slice`），同一屏工单整组落在同一档。
    """
    return "|".join(f"{key}={value}" for key, value in case.as_dict().items())


@dataclass(frozen=True)
class RoutingConfig:
    """一个切分的采样参数：各维度取值表 + 问法池 + `ood_only`。"""

    split: str = "train"
    areas: tuple[str, ...] = AREAS
    issues: tuple[str, ...] = ISSUES
    tiers: tuple[str, ...] = TIERS
    impacts: tuple[str, ...] = IMPACTS
    urgencies: tuple[str, ...] = URGENCIES
    compliances: tuple[str, ...] = COMPLIANCES
    waits: tuple[str, ...] = WAITS
    phrasings: tuple[int, ...] = (0, 1, 2)
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在构造期就报人话，不拖到采样阶段。"""
        pools: tuple[tuple[str, tuple[Any, ...]], ...] = (
            ("areas", self.areas),
            ("issues", self.issues),
            ("tiers", self.tiers),
            ("impacts", self.impacts),
            ("urgencies", self.urgencies),
            ("compliances", self.compliances),
            ("waits", self.waits),
            ("phrasings", self.phrasings),
        )
        for field, values in pools:
            require_pool(field, values)
        known: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
            ("areas", self.areas, AREAS),
            ("issues", self.issues, ISSUES),
            ("tiers", self.tiers, TIERS),
            ("impacts", self.impacts, IMPACTS),
            ("urgencies", self.urgencies, URGENCIES),
            ("compliances", self.compliances, COMPLIANCES),
            ("waits", self.waits, WAITS),
        )
        for field, values, allowed in known:
            unknown = [name for name in values if name not in allowed]
            if unknown:
                raise ValueError(f"{field} 里有不存在的取值：{unknown}（可用：{list(allowed)}）")


def build_cases(config: RoutingConfig) -> list[TicketCase]:
    """枚举 case 矩阵。

    输入：采样参数。输出：`TicketCase` 列表（顺序稳定、可复算）。
    预期行为：全枚举（**不做成立性过滤**——这七个维度两两不矛盾，不像检索页那样
    需要剔掉「干扰项比正解还相关」的组合）；`ood_only` 时只留「至少含一个训练侧
    没见过的取值」的 case，所以 OOD 与同分布侧**按构造就不重叠**。
    """
    out: list[TicketCase] = []
    for area in config.areas:
        for issue in config.issues:
            for tier in config.tiers:
                for impact in config.impacts:
                    for urgency in config.urgencies:
                        for compliance in config.compliances:
                            for wait in config.waits:
                                case = TicketCase(
                                    area, issue, tier, impact, urgency, compliance, wait
                                )
                                if config.ood_only and not case.unseen_dims():
                                    continue
                                out.append(case)
    return out


# ---------- 真值规格 ----------


@dataclass(frozen=True)
class TicketSpec:
    """一条工单的完整真值：case + 由规则算出来的一切。"""

    case: TicketCase
    split: str
    seq: int
    phrasing: int
    ticket_id: str
    severity: str
    queue: str
    escalation: str
    priority: int
    critical_area: bool
    needs_human: bool
    data_risk: bool
    sla_limit: int
    elapsed: int
    sla_breach: bool
    created: str
    current: str


def ticket_id_of(case: TicketCase) -> str:
    """输入：case。输出：工单号（形如 `T-48213`）。

    预期行为：由素材指纹的 sha256 派生——确定性，而且**不泄露切分**：
    用序号或切分名拼工单号会让同一屏工单在不同切分里拿到不同工单号，
    等于把切分信息写进 state。
    """
    digest = hashlib.sha256(material_of(case).encode("utf-8")).hexdigest()
    return f"{TICKET_PREFIX}{10000 + int(digest[:6], 16) % 90000}"


def sample_spec(rng: Any, case: TicketCase, split: str, seq: int, phrasing: int) -> TicketSpec:
    """输入：RNG（**不用**）、case、切分、序号、问法。输出：真值规格。

    预期行为：不掷骰子——同一 case 必得同一份规格（工单号也由 case 指纹派生），
    这样「同素材同题必须同答」这条一致性校验才成立。
    """
    del rng  # 规格完全由 case 决定；签名里的 RNG 只为与协议一致
    severity = severity_of(case.impact, case.urgency)
    limit = sla_limit_of(severity, case.tier)
    elapsed = elapsed_of(case.wait)
    created, current = times_of(case.area, case.wait)
    return TicketSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        ticket_id=ticket_id_of(case),
        severity=severity,
        queue=queue_of(case.issue, case.compliance),
        escalation=escalation_of(case.compliance),
        priority=priority_of(severity, case.tier),
        critical_area=critical_area_of(case.area),
        needs_human=needs_human_of(severity, case.compliance, case.tier),
        data_risk=data_risk_of(case.issue, case.compliance),
        sla_limit=limit,
        elapsed=elapsed,
        sla_breach=sla_breach_of(elapsed, limit),
        created=created,
        current=current,
    )


def spec_to_dict(spec: TicketSpec) -> dict[str, Any]:
    """输入：规格。输出：可 JSON 序列化的摘要（写进 manifest 与调试，不进样本行）。"""
    return {
        "case": spec.case.describe(),
        "unseen_dims": list(spec.case.unseen_dims()),
        "split": spec.split,
        "seq": spec.seq,
        "ticket_id": spec.ticket_id,
        "severity": spec.severity,
        "queue": spec.queue,
        "escalation": spec.escalation,
        "priority": spec.priority,
        "critical_area": spec.critical_area,
        "needs_human": spec.needs_human,
        "data_risk": spec.data_risk,
        "sla_limit": spec.sla_limit,
        "elapsed": spec.elapsed,
        "sla_breach": spec.sla_breach,
        "created": spec.created,
        "current": spec.current,
    }


# ---------- 值留出（训练侧见过哪些取值） ----------

#: 训练侧见过的产品域：`identity` 整个留出（产品名与正文模板一起没见过）。
SEEN_AREAS: Final[frozenset[str]] = frozenset(area for area in AREAS if area != "identity")
#: 训练侧见过的问题类型：`feature` 留出（连带整条「产品待办组」分支没见过）。
SEEN_ISSUES: Final[frozenset[str]] = frozenset(issue for issue in ISSUES if issue != "feature")
#: 训练侧见过的合规标记：`security` 留出（连带「安全事件组」与「安全负责人」没见过）。
SEEN_COMPLIANCES: Final[frozenset[str]] = frozenset(
    value for value in COMPLIANCES if value != "security"
)

