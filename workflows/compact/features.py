"""会话上下文压缩（compact）的特征层：注入维度 + 标签规则（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一次「这一块上下文该怎么处置」的判断由 5 个**注入特征**
描述，**标签先行**——先定这 5 维取值，再由本模块的纯函数算出全部金标签。

五维分成两拨（这条分法是设计的第一步，见范式 §9.13）：
- **由文本承载**（提示词给、状态行不给）：这一块**是什么**`kind`、它**有没有承载一条
  持久事实**`fact`——真实会话里这两件事本来就要读了那一段才知道；
- **由代码渲染的「本块位置」那一行承载**（提示词不给）：轮次位置 `age`、有没有被钉住
  `pins`、剩余预算 `budget`——这三件事在真实 agent 里是运行时元数据，不是正文里的措辞。

六个题族里两个由文本承载（`kind` / `fact`），交给 `judges.TraceJudge` 反解校验；另外四个
（`rule` / `disposition` / `pin_overrode` / `priority`）由这五维经纯函数算出，`audit.py`
拿产物里记下的答案做交叉校验。

输入/输出：只有常量与纯函数；不碰网络、不碰文件、**绝不执行任何东西**。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final

from jc.cover import cover_dims

# ---------- 维度表 ----------

#: 这一块是什么。**只由正文承载**。
KINDS: Final[tuple[str, ...]] = (
    "system",  # 系统提示词
    "user",  # 用户这一轮说的话
    "assistant",  # 助手这一轮的回复
    "tool_output",  # 工具返回的一大段结果
    "note",  # 随手记下的备注
)

#: 这一块有没有承载一条**持久事实**（一个决定、一个配置值、一个结论）。**只由正文承载**。
FACTS: Final[tuple[str, ...]] = (
    "fact",  # 承载持久事实
    "chatter",  # 只是闲聊 / 过渡话
)

#: 轮次位置。**只由「本块位置」那一行承载**，正文里不许写。
AGES: Final[tuple[str, ...]] = (
    "current",  # 就是这一轮
    "recent",  # 最近几轮
    "old",  # 很早以前
)

#: 有没有被钉住（钉住 = 用户或系统要求无论如何都留着）。**只由那一行承载**。
PINS: Final[tuple[str, ...]] = (
    "pinned",  # 钉住了
    "unpinned",  # 没钉
)

#: 剩余预算。**只由那一行承载**。
BUDGETS: Final[tuple[str, ...]] = (
    "ample",  # 宽裕
    "tight",  # 紧张
    "critical",  # 快满了
)

DIMS: Final[dict[str, tuple[str, ...]]] = {
    "kind": KINDS,
    "fact": FACTS,
    "age": AGES,
    "pins": PINS,
    "budget": BUDGETS,
}

#: 处置表。顺序就是**从上往下第一条命中**的顺序，改顺序等于改语义——
#: `tests/workflows/compact/test_features.py` 拿处置表的文本行顺序钉住它。
RULES: Final[tuple[str, ...]] = (
    "pinned_keep",
    "system_keep",
    "fact_keep",
    "budget_tool",
    "critical_old",
    "tight_old",
    "old_truncate",
    "none",
)

#: 每条规则给出的处置。
RAW_DISPOSITIONS: Final[dict[str, str]] = {
    "pinned_keep": "keep",
    "system_keep": "keep",
    "fact_keep": "keep",
    "budget_tool": "summarize",
    "critical_old": "drop",
    "tight_old": "summarize",
    "old_truncate": "truncate",
    "none": "keep",
}

#: 处置。`keep` = 原样留；`truncate` = 掐头去尾留一段；`summarize` = 压成摘要；
#: `drop` = 整块丢掉。
DISPOSITIONS: Final[tuple[str, ...]] = ("keep", "truncate", "summarize", "drop")

#: 优先级档数（0 ~ 3）。
PRIORITY_LEVELS: Final[int] = 4

#: 「这一块是什么 × 轮次位置」的底分表。表里缺格直接抛 KeyError（见 `priority_for`）。
BASE_PRIORITY: Final[dict[tuple[str, str], int]] = {
    ("system", "current"): 3,
    ("system", "recent"): 3,
    ("system", "old"): 3,
    ("user", "current"): 3,
    ("user", "recent"): 2,
    ("user", "old"): 1,
    ("assistant", "current"): 2,
    ("assistant", "recent"): 2,
    ("assistant", "old"): 1,
    ("tool_output", "current"): 1,
    ("tool_output", "recent"): 1,
    ("tool_output", "old"): 0,
    ("note", "current"): 2,
    ("note", "recent"): 1,
    ("note", "old"): 0,
}

#: 天生不能同时出现的取值对（四元组 `(左维, 左值, 右维, 右值)`）。两条都是「这一格
#: 没有唯一答案」，不是「物理上不可能」：
#:
#: - `工具输出 × 持久事实`：工具返回的是数据，不是一条持久事实——这一格真写出来，处置表里
#:   「事实必须留」与「预算吃紧时工具输出压成摘要」两条会同时讲得通，读的人分不出该答哪一档。
#: - `系统提示词 × 没有持久事实`：系统提示词**天生就是一串常驻规矩**，而「规矩」与「能长期
#:   复用的决定」在文本里分不开。真机第一批两条这样写的数据（「…必须将全局超时阈值统一配置
#:   为 45s」「…所有技术指令必须以代码块形式呈现」）都被反解读成「承载了持久事实」，而它读得
#:   并不算错——这条边界画不出来，于是这一格不该生成（范式 §9.15 第二条教训）。
FORBIDDEN: Final[frozenset[tuple[str, str, str, str]]] = frozenset(
    {
        ("fact", "fact", "kind", "tool_output"),
        ("fact", "chatter", "kind", "system"),
    }
)


@dataclass(frozen=True)
class Labels:
    """一条样本的全部金标签。字段名与题族名一一对应。"""

    kind: str
    fact: str
    rule: str
    disposition: str
    pin_overrode: bool
    priority: int


@dataclass(frozen=True)
class TagSet:
    """一组注入特征。字段名与 `DIMS` 的键一一对应。"""

    kind: str
    fact: str
    age: str
    pins: str
    budget: str

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
    """输入：一组注入特征。输出：命中的规则名（第一条命中者）。

    预期行为：严格按 `RULES` 的顺序判。这张表就是 state 里渲染给考生的那张表，
    所以这里的顺序必须与 locale 里 `rules` 的书写顺序一致。
    """
    if tags.pins == "pinned":
        return "pinned_keep"
    if tags.kind == "system":
        return "system_keep"
    if tags.fact == "fact":
        return "fact_keep"
    if tags.budget != "ample" and tags.kind == "tool_output":
        return "budget_tool"
    if tags.budget == "critical" and tags.age == "old":
        return "critical_old"
    if tags.budget == "tight" and tags.age == "old":
        return "tight_old"
    if tags.age == "old":
        return "old_truncate"
    return "none"


def disposition_for(tags: TagSet) -> str:
    """输入：一组注入特征。输出：最终处置（等于命中规则给出的那一档）。"""
    return RAW_DISPOSITIONS[rule_for(tags)]


def pin_overrode_for(tags: TagSet) -> bool:
    """输入：一组注入特征。输出：钉住这一条有没有**改写**了本该有的处置。

    预期行为：反事实推断——把 `pins` 换成 `unpinned` 再算一次，如果本来会得到
    `keep` 以外的处置，就说明这一钉确实起了作用。没钉住的一律为假。
    """
    if tags.pins != "pinned":
        return False
    without_pin = replace(tags, pins="unpinned")
    return disposition_for(without_pin) != "keep"


def priority_for(tags: TagSet) -> int:
    """输入：一组注入特征。输出：优先级档（0~3）。

    预期行为：先查 `BASE_PRIORITY`（这一块是什么 × 轮次位置）；**钉住的一律抬到 3 档**
    ——被钉住的东西本来就该最先保住。表里缺这一格直接抛 KeyError：缺格意味着有取值组合
    压根没定义，那这条样本就不该生成，静默兜底会把「没定义」伪装成「有定义」。
    """
    if tags.pins == "pinned":
        return PRIORITY_LEVELS - 1
    return BASE_PRIORITY[(tags.kind, tags.age)]


def label_for(tags: TagSet) -> Labels:
    """输入：一组注入特征。输出：全部金标签。

    预期行为：**纯函数、只看特征、不看文本**——文本是后写的，标签先定。

    全空间（5×2×3×2×3 = 180 组，剔除 2 对禁用对后 144 组）上的分布实测如下：

    - `rule`：`pinned_keep` 最多（钉住那一档占了三分之一），其余每条都有若干格；
    - `disposition`：四档都到得了（`keep` 靠前三条与兜底，`truncate` 靠 `old_truncate`，
      `summarize` 靠 `budget_tool` / `tight_old`，`drop` 靠 `critical_old`）；
    - `pin_overrode`：真值偏少（要同时满足「钉住」+「不钉就会被降级」）；
    - `priority`：0~3 四档都到得了。

    偏斜来自判据本身（处置表就是这么写的），照实标出不掩盖；要的是**每一档都到得了**，
    不是每档一样多。同分布侧（留出 `age=current`）每族每档都还在。
    """
    return Labels(
        kind=tags.kind,
        fact=tags.fact,
        rule=rule_for(tags),
        disposition=disposition_for(tags),
        pin_overrode=pin_overrode_for(tags),
        priority=priority_for(tags),
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
    values = tags.as_dict()
    names = list(DIMS)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if not allow_pair(left, values[left], right, values[right]):
                return False
    return True


#: choice 题族允许出现的选项名（题面只收本题允许的，别的取值留给反解提示词用）。
CHOICE_OPTIONS: Final[dict[str, tuple[str, ...]]] = {
    "kind": KINDS,
    "fact": FACTS,
    "rule": RULES,
    "disposition": DISPOSITIONS,
}
