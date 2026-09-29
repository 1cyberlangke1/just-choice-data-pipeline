"""结构恢复（autoformat）的特征层：文档形状 + 块计划 + 金标签（纯函数，零依赖）。

对应官方 structure-recovery 那类用例：给一份**丢掉了 Markdown 标记、正文被硬换行折断**的纯文本，
让读者把「内容块」的切分与类型读回来。

本模块只放**结构键**（块类型名 / 形状名 / 变体名 / 命令族名 / 主题名）与字符预算，一句话都不写：
所有文案在 `locales/`，正文本身由后端写（范式 §9.22 / §9.23 / §9.25 同一条路子）。
**标签先行**：块的顺序、类型、列表有没有序、提示框是哪一类，都由 `build_plan` 定死；
后端只负责把每一块的文字写出来，碰不到标签。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from itertools import product
from typing import Final

from jc.jsontypes import JsonValue
from jc.knobs import require_pool

# ---------- 块类型（键，不是文案） ----------

KIND_HEADING: Final[str] = "heading"
KIND_PARAGRAPH: Final[str] = "paragraph"
KIND_LIST_ITEM: Final[str] = "list_item"
KIND_QUOTE: Final[str] = "quote"
KIND_CODE: Final[str] = "code"
KIND_CALLOUT: Final[str] = "callout"

#: 块类型顺序：题面选项与统计都按它排（稳定顺序，便于逐字节复现）。
BLOCK_KINDS: Final[tuple[str, ...]] = (
    KIND_HEADING,
    KIND_PARAGRAPH,
    KIND_LIST_ITEM,
    KIND_QUOTE,
    KIND_CODE,
    KIND_CALLOUT,
)

# ---------- 文档形状 ----------

SHAPE_PLAIN: Final[str] = "plain"
SHAPE_LIST: Final[str] = "list"
SHAPE_CODE: Final[str] = "code"
SHAPE_QUOTE: Final[str] = "quote"
SHAPE_CALLOUT: Final[str] = "callout"
SHAPE_MIXED: Final[str] = "mixed"

SHAPES: Final[tuple[str, ...]] = (
    SHAPE_PLAIN,
    SHAPE_LIST,
    SHAPE_CODE,
    SHAPE_QUOTE,
    SHAPE_CALLOUT,
    SHAPE_MIXED,
)

#: 每种形状允许的 `variant`（`na` = 该形状没有变体）。
SHAPE_VARIANTS: Final[dict[str, tuple[str, ...]]] = {
    SHAPE_PLAIN: ("na",),
    SHAPE_LIST: ("ordered", "bullet"),
    SHAPE_CODE: ("na",),
    SHAPE_QUOTE: ("na",),
    SHAPE_CALLOUT: ("note", "tip", "warning"),
    SHAPE_MIXED: ("note", "tip", "warning"),
}

#: 每种形状允许的命令族（`na` = 该形状不带代码块）。
SHAPE_CODE_FAMILIES: Final[dict[str, tuple[str, ...]]] = {
    SHAPE_PLAIN: ("na",),
    SHAPE_LIST: ("na",),
    SHAPE_CODE: ("build", "vcs", "container", "package"),
    SHAPE_QUOTE: ("na",),
    SHAPE_CALLOUT: ("na",),
    SHAPE_MIXED: ("build", "vcs", "container", "package"),
}

#: 提示框的三种种类（键；文案在 `locales/`）。
CALLOUT_KINDS: Final[tuple[str, ...]] = ("note", "tip", "warning")

#: 全部变体 / 全部命令族（默认配置用）。`dict.fromkeys` 去重且保序。
ALL_VARIANTS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(value for shape in SHAPES for value in SHAPE_VARIANTS[shape])
)
ALL_CODE_FAMILIES: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(value for shape in SHAPES for value in SHAPE_CODE_FAMILIES[shape])
)

# ---------- 主题与行宽 ----------

TOPICS: Final[tuple[str, ...]] = ("build", "release", "onboarding", "incident", "migration")

#: 行宽（字符）：反解靠「短行」认标题与列表项，所以行宽必须**明显大于**短行上限。
WRAPS: Final[tuple[int, ...]] = (56, 64, 72)

# ---------- 值留出（训练侧见过哪些取值） ----------

#: 训练侧见过的主题：`incident` / `migration` 整个留出（连它们的词一起没见过）。
SEEN_TOPICS: Final[frozenset[str]] = frozenset({"build", "release", "onboarding"})
#: 训练侧见过的行宽：`56` 留出（折行位置整个换一种节奏）。
SEEN_WRAPS: Final[frozenset[int]] = frozenset({64, 72})
#: 训练侧见过的变体：`tip` 留出（提示框的第三种种类没见过）。
SEEN_VARIANTS: Final[frozenset[str]] = frozenset({"na", "ordered", "bullet", "note", "warning"})
#: 训练侧见过的命令族：`container` 留出。
SEEN_CODE_FAMILIES: Final[frozenset[str]] = frozenset({"na", "build", "vcs", "package"})

# ---------- 字符预算 ----------

#: 「短行」上限（字符）。折行后的**非末行**一定比它长，所以短行只可能是标题或列表项。
#: **两语言共用**：state 里没有语言标记，反解只能拿同一个阈值去判（`locales.merged()`）；
#: 各语言自己的字符区间写在 `[limit]` 表里（`parse.limits` 读并校验），且必须落在它之内。
#: 取 44 是给英文留的余量：英文标题的自然长度是 30~40 个字符，卡在 36 会把正常标题判成正文
#: （真机上 10 条里 3 条栽在这）。它仍远小于最小行宽（56），判据不会因此变糊。
SHORT_LIMIT: Final[int] = 44
#: 代码块的行数区间（按行数算，与语言无关）。
MIN_CODE_LINES: Final[int] = 2
MAX_CODE_LINES: Final[int] = 4
#: 列表项的项数区间。
MIN_LIST_ITEMS: Final[int] = 2
MAX_LIST_ITEMS: Final[int] = 3


def _check_knobs(
    topics: tuple[str, ...],
    wraps: tuple[int, ...],
    shapes: tuple[str, ...],
    variants: tuple[str, ...],
    code_families: tuple[str, ...],
    phrasings: tuple[int, ...],
) -> None:
    """输入：六个旋钮。输出：无。预期行为：任一为空表就报人话（点名是哪个字段）。"""
    require_pool("topics", topics)
    require_pool("wraps", wraps)
    require_pool("shapes", shapes)
    require_pool("variants", variants)
    require_pool("code_families", code_families)
    require_pool("phrasing_pool", phrasings)


@dataclass(frozen=True)
class DocConfig:
    """一个切分的采样参数：六张取值表 + 是否只留 OOD。

    预期行为：构造期就把空表拦下来（`jc.knobs`），别让它一路活到采样阶段才炸。
    """

    split: str = "train"
    topics: tuple[str, ...] = TOPICS
    wraps: tuple[int, ...] = WRAPS
    shapes: tuple[str, ...] = SHAPES
    variants: tuple[str, ...] = ALL_VARIANTS
    code_families: tuple[str, ...] = ALL_CODE_FAMILIES
    phrasing_pool: tuple[int, ...] = (0, 1)
    #: True 时只留「至少含一个训练侧没见过的取值」的组合，保证 OOD 与同分布侧不重叠。
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在构造期就报人话。"""
        _check_knobs(
            self.topics,
            self.wraps,
            self.shapes,
            self.variants,
            self.code_families,
            self.phrasing_pool,
        )


@dataclass(frozen=True)
class DocCase:
    """一个文档组合：主题 × 行宽 × 形状 × 变体 × 命令族。"""

    topic: str
    wrap: int
    shape: str
    variant: str
    code_family: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：{维度名: 取值}（五个维度全摊平）。"""
        return {
            "topic": self.topic,
            "wrap": self.wrap,
            "shape": self.shape,
            "variant": self.variant,
            "code_family": self.code_family,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.topic}/{self.wrap}/{self.shape}/{self.variant}/{self.code_family}"

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这个组合里「训练侧没见过」的维度名（按固定顺序）。"""
        out: list[str] = []
        if self.topic not in SEEN_TOPICS:
            out.append("topic")
        if self.wrap not in SEEN_WRAPS:
            out.append("wrap")
        if self.variant not in SEEN_VARIANTS:
            out.append("variant")
        if self.code_family not in SEEN_CODE_FAMILIES:
            out.append("code_family")
        return tuple(out)


def material_of(case: DocCase) -> str:
    """输入：case。输出：素材指纹（确定性；切分分桶与块计划都由它派生）。"""
    return f"{case.topic}|{case.wrap}|{case.shape}|{case.variant}|{case.code_family}"


def build_cases(config: DocConfig) -> list[DocCase]:
    """输入：采样参数。输出：case 矩阵（顺序稳定）。

    预期行为：五张表取全组合（按形状过滤出该形状真正允许的变体与命令族）；`ood_only` 时只留
    「至少含一个训练侧没见过的取值」的组合，所以 OOD 与同分布侧**按构造就不重叠**。某形状在当前
    旋钮下一个变体都不剩时**跳过**它——不静默改写成别的形状（那会让「形状」这一维的覆盖报告撒谎）。
    """
    out: list[DocCase] = []
    for topic, wrap, shape in product(config.topics, config.wraps, config.shapes):
        variants = tuple(value for value in SHAPE_VARIANTS[shape] if value in config.variants)
        families = tuple(value for value in SHAPE_CODE_FAMILIES[shape] if value in config.code_families)
        if not variants or not families:
            continue
        for variant in variants:
            for family in families:
                case = DocCase(topic, wrap, shape, variant, family)
                if config.ood_only and not case.unseen_dims():
                    continue
                out.append(case)
    return out


# ---------- 块计划（还没填文案） ----------


@dataclass(frozen=True)
class Slot:
    """一个内容块的**槽位**：后端要为它写一段文字。

    字段：`kind` 块类型；`callout_kind` 提示框种类（只有提示框用）；`code_family` 命令族（只有
    代码块用）；`list_ordered` / `list_id` 只有列表项用。
    """

    kind: str
    callout_kind: str = ""
    code_family: str = ""
    list_ordered: bool = False
    list_id: int = 0


@dataclass(frozen=True)
class DocPlan:
    """一份文档的**块计划**：有序的槽位表。标签全部由它推出来，后端碰不到。"""

    case: DocCase
    slots: tuple[Slot, ...]

    def kinds(self) -> tuple[str, ...]:
        """输入：无。输出：各块的类型（按块序）。"""
        return tuple(slot.kind for slot in self.slots)

    def callout_kinds(self) -> tuple[str, ...]:
        """输入：无。输出：全部提示框的种类（按块序）。"""
        return tuple(slot.callout_kind for slot in self.slots if slot.kind == KIND_CALLOUT)

    def list_flags(self) -> tuple[bool, ...]:
        """输入：无。输出：全部列表项的有序性（按块序）。"""
        return tuple(slot.list_ordered for slot in self.slots if slot.kind == KIND_LIST_ITEM)


@dataclass(frozen=True)
class DocLabels:
    """一份文档的金标签：块类型 / 提示框种类 / 列表有序性。"""

    kinds: tuple[str, ...]
    callout_kinds: tuple[str, ...]
    list_flags: tuple[bool, ...]


def label_for(plan: DocPlan) -> DocLabels:
    """输入：块计划。输出：金标签（**构造侧**那条路径）。"""
    return DocLabels(plan.kinds(), plan.callout_kinds(), plan.list_flags())


def _ordered_for(case: DocCase, rng: random.Random) -> bool:
    """输入：case、RNG。输出：列表要不要排成步骤列表。

    预期行为：`list` 形状用 `variant` 决定（这一维可以留出）；`mixed` 的 `variant` 已经被提示框
    种类占着，所以另抽一次——两个形状一起才能让「有序 / 无序」这一族不至于一边倒。
    """
    if case.shape == SHAPE_LIST:
        return case.variant == "ordered"
    return rng.random() < 0.5


def build_plan(case: DocCase) -> DocPlan:
    """输入：case。输出：块计划（顺序稳定、与语言无关的随机）。

    预期行为：随机种子由**素材指纹**派生，所以同一个 case 在任何切分、任何进程里都得到同一份计划
    （「同素材同题必须同答」这条一致性校验才成立）。

    文档骨架：标题 + 一段正文 +（按形状追加的块）+ 收尾段落。标题**总是紧贴**它后面那段正文
    （中间不空行）——这正是「短行后面跟着长行」那种需要读者自己判断「标题到此为止」的难例。
    """
    rng = random.Random(int(hashlib.sha256(material_of(case).encode("utf-8")).hexdigest()[:16], 16))
    slots: list[Slot] = [Slot(KIND_HEADING), Slot(KIND_PARAGRAPH)]

    if case.shape in (SHAPE_LIST, SHAPE_MIXED):
        slots.append(Slot(KIND_PARAGRAPH))
        ordered = _ordered_for(case, rng)
        count = rng.randint(MIN_LIST_ITEMS, MAX_LIST_ITEMS)
        slots.extend(
            Slot(KIND_LIST_ITEM, list_ordered=ordered, list_id=1) for _ in range(count)
        )

    if case.shape in (SHAPE_CODE, SHAPE_MIXED):
        slots.append(Slot(KIND_PARAGRAPH))
        slots.append(Slot(KIND_CODE, code_family=case.code_family))

    if case.shape in (SHAPE_QUOTE, SHAPE_MIXED):
        slots.append(Slot(KIND_QUOTE))

    if case.shape in (SHAPE_CALLOUT, SHAPE_MIXED):
        slots.append(Slot(KIND_CALLOUT, callout_kind=case.variant))

    slots.append(Slot(KIND_PARAGRAPH))
    return DocPlan(case=case, slots=tuple(slots))


def code_families_in(plan: DocPlan) -> tuple[str, ...]:
    """输入：块计划。输出：这份文档里用到的命令族（没有代码块就是空表）。"""
    return tuple(dict.fromkeys(slot.code_family for slot in plan.slots if slot.code_family))
