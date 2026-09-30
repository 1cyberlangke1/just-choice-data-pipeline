"""工单分诊（routing）的特征层：注入维度 + 派单表 / 优先级表 / 附加规则（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一次「这条工单该怎么分诊」的判断由 5 个**注入特征**描述，
**标签先行**——先定这 5 维取值，再由本模块的纯函数算出全部金标签。

五维分成两拨（这条分法是设计的第一步，见范式 §9.18）：
- **只由工单正文承载**（提示词给、属性行不给）：这条工单**是什么问题** `issue`、它**有多急**
  `urgency`——真实工单里这两件事本来就要读了那段话才知道；
- **只由「工单属性」那一行承载**（提示词不给）：客户等级 `tier`、合规标记 `compliance`、
  产品域 `area`——这三件事在真实分诊台里是工单系统里的字段，不是用户话里的措辞。

六个题族里两个由正文承载（`issue` / `urgency`），交给 `judges.TraceJudge` 反解校验；另外
四个（`rule` / `queue` / `priority` / `needs_human`）由这五维经纯函数算出，`audit.py` 拿
产物里记下的答案做交叉校验。

与旧版（`9bae6ba` 删掉的那个构造式 routing）的根本区别：旧版把**工单正文**也拿模板拼
（每个问题类型一句固定话），那正是被判为垃圾的一类；新版只把正文交给后端写，属性行与
三张表留在代码里。

输入/输出：只有常量与纯函数；不碰网络、不碰文件、**绝不执行任何东西**。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jc.cover import cover_dims, row_allowed

# ---------- 维度表 ----------

#: 这条工单是什么问题。**只由正文承载**（也是 `issue` 题族的选项顺序）。
ISSUES: Final[tuple[str, ...]] = (
    "outage",  # 故障：某条线整体不可用
    "data_loss",  # 数据丢失：记录少了 / 对不上
    "bug",  # 缺陷：行为与文档不符
    "howto",  # 咨询：问正确用法
    "feature",  # 需求：想要一种新做法
)

#: 有多急。**只由正文承载**。
URGENCIES: Final[tuple[str, ...]] = (
    "can_wait",  # 可以等
    "blocked",  # 业务受阻
    "deadline",  # 有截止时间
)

#: 客户等级。**只由「工单属性」那一行承载**。
TIERS: Final[tuple[str, ...]] = ("free", "pro", "enterprise")

#: 合规标记。**只由「工单属性」那一行承载**。
COMPLIANCES: Final[tuple[str, ...]] = ("none", "gdpr", "security")

#: 产品域。**只由「工单属性」那一行承载**（同时决定正文里提到哪条线）。
AREAS: Final[tuple[str, ...]] = ("payments", "storage", "analytics", "identity")

#: 关键产品域：附加规则里那条「关键域 + 企业版要人工」就查这个集合。
CRITICAL_AREAS: Final[frozenset[str]] = frozenset({"payments", "identity"})

DIMS: Final[dict[str, tuple[str, ...]]] = {
    "issue": ISSUES,
    "urgency": URGENCIES,
    "tier": TIERS,
    "compliance": COMPLIANCES,
    "area": AREAS,
}

#: 派单表。顺序就是**从上往下第一条命中**的顺序，改顺序等于改语义——
#: `tests/workflows/routing/test_features.py` 拿派单表的文本行顺序钉住它。
QUEUE_RULES: Final[tuple[str, ...]] = (
    "security_report",
    "privacy_request",
    "outage_response",
    "loss_response",
    "defect",
    "howto",
    "feature_request",
)

#: 每条派单规则给出的队列。
RAW_QUEUES: Final[dict[str, str]] = {
    "security_report": "security_incident",
    "privacy_request": "privacy_ops",
    "outage_response": "incident_response",
    "loss_response": "incident_response",
    "defect": "engineering",
    "howto": "support_l1",
    "feature_request": "product_backlog",
}

#: 队列。`rule → queue` 是多对一：`outage_response` 与 `loss_response` 都进事故响应组。
QUEUES: Final[tuple[str, ...]] = (
    "support_l1",
    "engineering",
    "incident_response",
    "privacy_ops",
    "security_incident",
    "product_backlog",
)

#: 优先级档数（0 ~ 3）。
PRIORITY_LEVELS: Final[int] = 4

#: 优先级表：紧急度 × 客户等级 → 档位。**查表，不是加权求和**——表里缺格直接抛
#: KeyError（见 `priority_for`）。
PRIORITY_TABLE: Final[dict[tuple[str, str], int]] = {
    ("can_wait", "free"): 0,
    ("can_wait", "pro"): 0,
    ("can_wait", "enterprise"): 1,
    ("blocked", "free"): 1,
    ("blocked", "pro"): 2,
    ("blocked", "enterprise"): 3,
    ("deadline", "free"): 2,
    ("deadline", "pro"): 3,
    ("deadline", "enterprise"): 3,
}

#: 天生不能同时出现的取值对（四元组 `(左维, 左值, 右维, 右值)`）。三条都是「这一格没有
#: 唯一答案」，不是「物理上不可能」：
#:
#: - `故障 × 可以等`：某条线整体不可用却「不急」，读的人分不出这是故障还是缺陷；
#: - `数据丢失 × 可以等`：记录都少了还说「等有空再看」，同样分不出是数据丢失还是缺陷；
#: - `咨询 × 业务受阻`：咨询类工单天生就是「问一句」，写不出「业务停住了」那种话——
#:   真这么写，读的人只会判成缺陷或故障。
#:
#: 这三格由 `tests/workflows/routing/test_features.py` 钉住；真机批次若暴露别的格子，
#: 按同样口径补进来（范式 §9.15 第二条教训：能禁就禁，不能禁才改提示词）。
FORBIDDEN: Final[frozenset[tuple[str, str, str, str]]] = frozenset(
    {
        ("issue", "outage", "urgency", "can_wait"),
        ("issue", "data_loss", "urgency", "can_wait"),
        ("issue", "howto", "urgency", "blocked"),
    }
)


@dataclass(frozen=True)
class Labels:
    """一条样本的全部金标签。字段名与题族名一一对应。"""

    issue: str
    urgency: str
    rule: str
    queue: str
    priority: int
    needs_human: bool


@dataclass(frozen=True)
class TagSet:
    """一组注入特征。字段名与 `DIMS` 的键一一对应。"""

    issue: str
    urgency: str
    tier: str
    compliance: str
    area: str

    def as_dict(self) -> dict[str, str]:
        """输入：无。输出：`{维度名: 取值}`（顺序按 `DIMS`）。"""
        return {name: getattr(self, name) for name in DIMS}

    def label(self) -> Labels:
        """输入：无。输出：这组特征对应的全部金标签。"""
        return label_for(self)


def allow_pair(left: str, left_value: str, right: str, right_value: str) -> bool:
    """输入：两个维度名与各自取值。输出：这一对写得出来吗。

    预期行为：与顺序无关——`(a, x, b, y)` 与 `(b, y, a, x)` 都查同一张禁用表。
    """
    if left > right:
        left, left_value, right, right_value = right, right_value, left, left_value
    return (left, left_value, right, right_value) not in FORBIDDEN


def rule_for(tags: TagSet) -> str:
    """输入：一组注入特征。输出：命中的派单规则名（第一条命中者）。

    预期行为：严格按 `QUEUE_RULES` 的顺序判。这张表就是 state 里渲染给考生的那张表，
    所以这里的顺序必须与 locale 里 `rules` 的书写顺序一致。
    """
    if tags.compliance == "security" and tags.issue in {"outage", "data_loss", "bug"}:
        return "security_report"
    if tags.compliance == "gdpr" and tags.issue in {"data_loss", "feature"}:
        return "privacy_request"
    if tags.issue == "outage":
        return "outage_response"
    if tags.issue == "data_loss":
        return "loss_response"
    if tags.issue == "bug":
        return "defect"
    if tags.issue == "howto":
        return "howto"
    return "feature_request"


def queue_for(tags: TagSet) -> str:
    """输入：一组注入特征。输出：最终队列（等于命中规则给出的那一档）。"""
    return RAW_QUEUES[rule_for(tags)]


def priority_for(tags: TagSet) -> int:
    """输入：一组注入特征。输出：优先级档（0~3）。

    预期行为：查 `PRIORITY_TABLE`（紧急度 × 客户等级）；表里缺这一格直接抛 KeyError：
    缺格意味着有取值组合压根没定义，那这条样本就不该生成，静默兜底会把「没定义」伪装成
    「有定义」。
    """
    return PRIORITY_TABLE[(tags.urgency, tags.tier)]


def needs_human_for(tags: TagSet) -> bool:
    """输入：一组注入特征。输出：这条工单要不要人工介入。

    预期行为：附加规则里那两条任一成立即为真——① 队列是事故响应组 / 安全事件组 /
    隐私合规组；② 产品域是关键域（支付 / 身份），而且客户等级是企业版。
    """
    if queue_for(tags) in {"incident_response", "security_incident", "privacy_ops"}:
        return True
    return tags.area in CRITICAL_AREAS and tags.tier == "enterprise"


def label_for(tags: TagSet) -> Labels:
    """输入：一组注入特征。输出：全部金标签。

    预期行为：**纯函数、只看特征、不看文本**——文本是后写的，标签先定。

    全空间（5×3×3×3×4 = 540 组，剔除 3 对禁用对后 432 组）上的分布：`issue` / `urgency`
    受禁用对影响（`outage` / `data_loss` 各少一档 `can_wait`，`howto` 少一档 `blocked`），
    其余各维均匀；`rule` 七条都到得了（`outage_response` 与 `loss_response` 各占一片，
    `security_report` / `privacy_request` 靠合规标记覆盖的那几个问题类型）；
    `queue` 六档都到得了（`incident_response` 由两条规则共占，所以它最多）；
    `priority` 0~3 四档都到得了；`needs_human` 真假都到得了（假要靠「非事故队列 +
    非关键域，或非企业版」）。要的是**每一档都到得了**，不是每档一样多。
    """
    return Labels(
        issue=tags.issue,
        urgency=tags.urgency,
        rule=rule_for(tags),
        queue=queue_for(tags),
        priority=priority_for(tags),
        needs_human=needs_human_for(tags),
    )


def build_cases(max_full: int, dims: dict[str, tuple[str, ...]] | None = None) -> list[TagSet]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖；LLM workflow 要把它压到几十）、
    可选的维度表覆盖（留出某几个取值时用，例如 OOD 的「值留出」）。
    输出：`TagSet` 列表，顺序稳定。
    预期行为：同一入参必得同一输出；成对覆盖保证任意两维的取值组合至少出现一次；
    被 `FORBIDDEN` 否掉的取值对既不出现、也不算没覆盖。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full, allow_pair=allow_pair)
    return [TagSet(**row) for row in rows]


def compatible(tags: TagSet) -> bool:
    """输入：一组特征。输出：这组特征写得出来吗（没有任何被禁的取值对）。"""
    return row_allowed(list(DIMS), tags.as_dict(), allow_pair)


#: choice 题族允许出现的选项名（题面只收本题允许的，别的取值留给反解提示词用）。
CHOICE_OPTIONS: Final[dict[str, tuple[str, ...]]] = {
    "issue": ISSUES,
    "urgency": URGENCIES,
    "rule": QUEUE_RULES,
    "queue": QUEUES,
}
