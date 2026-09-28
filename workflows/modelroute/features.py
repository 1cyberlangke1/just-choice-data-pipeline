"""模型路由与降级的特征层：注入维度 + 候选目录 + 标签规则（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」与 3.3「tag 组合」：
- 一条路由请求由 5 个**注入特征**描述（任务类型 / 规模 / 时延要求 / 预算档 / 提供方健康）；
- **标签先行**：先定这 5 维取值，再由本模块的纯函数在**候选目录**上算出全部金标签；
- 候选目录固定 4 行，由代码渲染进 state——所以「该选哪个」是表上的一次可复核筛选，
  不是模型的自由发挥（模型只写那句用户请求，见 `render.py`）。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jc.cover import cover_dims


@dataclass(frozen=True)
class Candidate:
    """候选目录里的一行。

    字段：`name` 模型名（也是 choice 的选项名）；`provider` 提供方；`window` 上下文窗口
    （千 token）；`ability` / `price` / `latency` 都是 1~3 档，越大分别表示越强 / 越贵 / 越慢。
    """

    name: str
    provider: str
    window: int
    ability: int
    price: int
    latency: int


#: 候选目录。四行的 (能力, 价格, 时延) 三元组**两两不同**——不然「按能力最高、再按便宜、
#: 再按快」这条规则会退化成「看谁排在前面」，答案就不再是表上可复核的事实了。
CATALOG: Final[tuple[Candidate, ...]] = (
    Candidate("atlas-xl", "atlas", 200, 3, 3, 3),
    Candidate("atlas-lite", "atlas", 32, 1, 1, 1),
    Candidate("borealis-pro", "borealis", 128, 2, 2, 2),
    Candidate("cinder-mini", "cinder", 64, 2, 1, 3),
)

TASK_KINDS: Final[tuple[str, ...]] = (
    "code_gen",  # 写 / 改代码
    "long_summary",  # 把长材料压成摘要
    "extraction",  # 从材料里抽字段
    "chat",  # 多轮闲聊问答
    "math_reason",  # 多步数学推理
    "vision",  # 看图
)

SIZES: Final[tuple[str, ...]] = ("small", "medium", "huge")
LATENCIES: Final[tuple[str, ...]] = ("interactive", "batch")
BUDGETS: Final[tuple[str, ...]] = ("tight", "normal")
HEALTHS: Final[tuple[str, ...]] = ("all_ok", "atlas_down", "borealis_down")

DIMS: Final[dict[str, tuple[str, ...]]] = {
    "task_kind": TASK_KINDS,
    "size": SIZES,
    "latency": LATENCIES,
    "budget": BUDGETS,
    "health": HEALTHS,
}

#: 规模 → 请求需要的上下文窗口（千 token）。
WINDOW_NEED: Final[dict[str, int]] = {"small": 8, "medium": 32, "huge": 96}

#: 任务类型 → 要求的最低能力档。**最高只到 3 档**：目录里只有 atlas-xl 是 3 档。
ABILITY_NEED: Final[dict[str, int]] = {
    "code_gen": 2,
    "long_summary": 2,
    "extraction": 1,
    "chat": 1,
    "math_reason": 3,
    "vision": 2,
}

#: 时延要求 → 允许的最高时延档（3 档等于「不限」）。
LATENCY_LIMIT: Final[dict[str, int]] = {"interactive": 2, "batch": 3}

#: 预算档 → 允许的最高价格档（3 档等于「不限」）。
PRICE_LIMIT: Final[dict[str, int]] = {"tight": 2, "normal": 3}

#: 健康档 → 挂掉的提供方。
DOWN_PROVIDERS: Final[dict[str, tuple[str, ...]]] = {
    "all_ok": (),
    "atlas_down": ("atlas",),
    "borealis_down": ("borealis",),
}

#: 「目录里第一行这次为什么不能用」的四种答案。**没有「窗口」这一档**——第一行的窗口
#: （200k）比任何规模的需求都大，列一个永远不会出现的选项只会诱导人去猜一个不可能的答案。
BLOCKED_REASONS: Final[tuple[str, ...]] = ("usable", "provider_down", "latency", "budget")

#: 窗口富余的判据：选中模型的窗口至少是需求的几倍。
SLACK_FACTOR: Final[int] = 2

#: 没有候选可用时的两个占位取值（score 的 0 档、route 的 none）。
NO_CANDIDATE: Final[str] = "none"


@dataclass(frozen=True)
class Needs:
    """一条请求筛候选时的四个门槛 + 挂掉的提供方。

    这是标签计算与**事后审计**共用的中间形态：渲染器把它写进 state，审计再从 state 里
    解析出同构的 `Needs`，两边各算一次「该选哪个」——不一致就说明文本与标签脱节了。
    """

    window: int
    ability: int
    latency: int
    price: int
    down: tuple[str, ...]


def needs_for(tags: TagSet) -> Needs:
    """输入：一组注入特征。输出：四个门槛 + 挂掉的提供方。**不看文本**。"""
    return Needs(
        window=WINDOW_NEED[tags.size],
        ability=ABILITY_NEED[tags.task_kind],
        latency=LATENCY_LIMIT[tags.latency],
        price=PRICE_LIMIT[tags.budget],
        down=DOWN_PROVIDERS[tags.health],
    )


def survivors(needs: Needs) -> list[Candidate]:
    """输入：门槛。输出：全部可用的候选（按目录顺序）。

    预期行为：四道门槛全过才算可用——提供方没挂、窗口够大、能力够高、时延与价格不超标。
    """
    return [
        cand
        for cand in CATALOG
        if cand.provider not in needs.down
        and cand.window >= needs.window
        and cand.ability >= needs.ability
        and cand.latency <= needs.latency
        and cand.price <= needs.price
    ]


def route_from(needs: Needs) -> str:
    """输入：门槛。输出：该路由到哪个候选（没有可用候选时是 `none`）。

    预期行为：在可用的候选里取「能力最高 → 更便宜 → 更快 → 目录更靠前」的第一个。
    目录里四行的三元组两两不同，所以前三条就能定出唯一赢家，最后一条只是保底。
    """
    alive = survivors(needs)
    if not alive:
        return NO_CANDIDATE
    return max(
        alive, key=lambda cand: (cand.ability, -cand.price, -cand.latency, -CATALOG.index(cand))
    ).name


def blocked_from(needs: Needs) -> str:
    """输入：门槛。输出：目录里**第一行**这次不能用的原因（能用就是 `usable`）。

    预期行为：按「提供方挂了 → 时延超标 → 价格超标」的顺序判；窗口不判，因为第一行的窗口
    比任何规模的需求都大（见 `BLOCKED_REASONS` 的说明）。
    """
    head = CATALOG[0]
    if head.provider in needs.down:
        return "provider_down"
    if head.latency > needs.latency:
        return "latency"
    if head.price > needs.price:
        return "budget"
    return "usable"


def cost_from(needs: Needs) -> int:
    """输入：门槛。输出：选中模型的价格档（没有可用候选时是 0 档）。"""
    chosen = route_from(needs)
    if chosen == NO_CANDIDATE:
        return 0
    return next(cand.price for cand in CATALOG if cand.name == chosen)


def slack_from(needs: Needs) -> bool:
    """输入：门槛。输出：选中模型的窗口有没有 `SLACK_FACTOR` 倍富余（没有可用候选时否）。"""
    chosen = route_from(needs)
    if chosen == NO_CANDIDATE:
        return False
    cand = next(cand for cand in CATALOG if cand.name == chosen)
    return cand.window >= SLACK_FACTOR * needs.window


@dataclass(frozen=True)
class TagSet:
    """一组注入特征（= PLAN 3.3 的一个 tag 组合）。

    字段：`task_kind` / `size` / `latency` / `budget` / `health`。
    """

    task_kind: str
    size: str
    latency: str
    budget: str
    health: str

    def as_dict(self) -> dict[str, str]:
        """输入：无。输出：`{维度名: 取值}`，顺序与 `DIMS` 一致。"""
        return {
            "task_kind": self.task_kind,
            "size": self.size,
            "latency": self.latency,
            "budget": self.budget,
            "health": self.health,
        }

    @property
    def needs(self) -> Needs:
        """输入：无。输出：这组特征对应的门槛。"""
        return needs_for(self)

    def label(self) -> Labels:
        """输入：无。输出：这一组特征对应的全部金标签。"""
        return label_for(self)


@dataclass(frozen=True)
class Labels:
    """一条样本的全部金标签：字段名就是题族名，`questions.py` 直接按名字取。

    `kind` 是唯一一个**由文本承载**的标签（模型写的那句请求必须看得出来是什么任务，
    由 `judges.TraceJudge` 反解校验）；其余四个都能从 state 里那张表与门槛行重算出来。
    """

    kind: str
    route: str
    blocked: str
    cost: int
    slack: bool


def label_for(tags: TagSet) -> Labels:
    """输入：一组注入特征。输出：全部金标签。

    预期行为：**纯函数、只看特征、不看文本**——文本是后写的，标签先定。
    """
    needs = needs_for(tags)
    return Labels(
        kind=tags.task_kind,
        route=route_from(needs),
        blocked=blocked_from(needs),
        cost=cost_from(needs),
        slack=slack_from(needs),
    )


def build_cases(max_full: int, dims: dict[str, tuple[str, ...]] | None = None) -> list[TagSet]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖；LLM workflow 要把它压到几十）、
    可选的维度表覆盖（留出某几个取值时用，例如 OOD 的「值留出」）。
    输出：`TagSet` 列表，顺序稳定。
    预期行为：同一入参必得同一输出；成对覆盖保证任意两维的取值组合至少出现一次。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [TagSet(**row) for row in rows]


def label_flip_axes(tags: TagSet) -> list[TagSet]:
    """列出「只改一维」的邻居组合（用来查最小对覆盖率）。

    输入：一组特征。输出：把每个维度换成同维其它取值后的组合列表（不含自己）。
    """
    values = tags.as_dict()
    out: list[TagSet] = []
    for name, items in DIMS.items():
        for value in items:
            if value != values[name]:
                out.append(TagSet(**{**values, name: value}))
    return out


#: choice 题族允许出现的选项名（题面只收本题允许的，别的取值留给反解提示词用）。
CHOICE_OPTIONS: Final[dict[str, tuple[str, ...]]] = {
    "kind": TASK_KINDS,
    "route": tuple(cand.name for cand in CATALOG) + (NO_CANDIDATE,),
    "blocked": BLOCKED_REASONS,
}
