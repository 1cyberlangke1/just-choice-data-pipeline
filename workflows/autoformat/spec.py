"""结构恢复（autoformat）workflow 的真值规格：一份丢掉标记的纯文本文档 + 它的块结构。

输入/输出：不可变数据类与纯函数；**不含任何领域词表**（句子、标题、列表项、命令行、
引用、提示框、问法全在 `locales/`；这里只放「槽位名 / 形状名 / 块类型名」这类键）。
预期行为：答案**只从渲染出来的文本反推**——块的切分、块的类型、列表是否有序、提示框
属于哪一类，都由 `parse` 与判别器**共用同一套读法**，所以「构造写歪」与「渲染写歪」
会变成同一类错误，不会两边各错一半还互相印证。

为什么不做成 LLM 生成：这是一份**已知结构的文档**被剥掉标记再硬换行，让模型去写只会
引入抖动；这里要训的是「按文本把结构读回来」，不是「编一份文档」。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from itertools import product
from typing import Any, Final

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

#: 全部变体 / 全部命令族（默认配置用）。用 `dict.fromkeys` 去重且保序。
ALL_VARIANTS: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(value for shape in SHAPES for value in SHAPE_VARIANTS[shape])
)
ALL_CODE_FAMILIES: Final[tuple[str, ...]] = tuple(
    dict.fromkeys(value for shape in SHAPES for value in SHAPE_CODE_FAMILIES[shape])
)

# ---------- 主题与行宽 ----------

TOPICS: Final[tuple[str, ...]] = ("build", "release", "onboarding", "incident", "migration")
#: 行宽（字符）：`parse` 靠「短行」认标题与列表项，所以行宽必须**明显大于**短行上限。
WRAPS: Final[tuple[int, ...]] = (56, 64, 72)

# ---------- 值留出（训练侧见过哪些取值） ----------

#: 训练侧见过的主题：`incident` / `migration` 整个留出（连它们的句子与标题一起没见过）。
SEEN_TOPICS: Final[frozenset[str]] = frozenset({"build", "release", "onboarding"})
#: 训练侧见过的行宽：`56` 留出（折行位置整个换一种节奏）。
SEEN_WRAPS: Final[frozenset[int]] = frozenset({64, 72})
#: 训练侧见过的变体：`tip` 留出（提示框的第三种种类没见过）。
SEEN_VARIANTS: Final[frozenset[str]] = frozenset({"na", "ordered", "bullet", "note", "warning"})
#: 训练侧见过的命令族：`container` 留出。
SEEN_CODE_FAMILIES: Final[frozenset[str]] = frozenset({"na", "build", "vcs", "package"})


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
    require_pool("phrasings", phrasings)


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
    phrasings: tuple[int, ...] = (0, 1)
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
            self.phrasings,
        )


@dataclass(frozen=True)
class DocCase:
    """一个文档组合：主题 × 行宽 × 形状 × 变体 × 命令族。"""

    topic: str
    wrap: int
    shape: str
    variant: str
    code_family: str

    def as_dict(self) -> dict[str, Any]:
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
    """输入：case。输出：素材指纹（确定性；切分分桶与内容种子都由它派生）。"""
    return f"{case.topic}|{case.wrap}|{case.shape}|{case.variant}|{case.code_family}"


def build_cases(config: DocConfig) -> list[DocCase]:
    """输入：采样参数。输出：case 矩阵（顺序稳定）。

    预期行为：五张表取全组合（按形状过滤出该形状真正允许的变体与命令族）；
    `ood_only` 时只留「至少含一个训练侧没见过的取值」的组合，所以 OOD 与同分布侧
    **按构造就不重叠**。某形状在当前旋钮下一个变体都不剩时**跳过**它——不静默改写成
    别的形状（那会让「形状」这一维的覆盖报告撒谎）。
    """
    out: list[DocCase] = []
    for topic, wrap, shape in product(config.topics, config.wraps, config.shapes):
        variants = tuple(v for v in SHAPE_VARIANTS[shape] if v in config.variants)
        families = tuple(f for f in SHAPE_CODE_FAMILIES[shape] if f in config.code_families)
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

_PLACEHOLDER: Final[re.Pattern[str]] = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True)
class Part:
    """一个模板片段：位置 `(区域, 键, 下标)` + 抽好的槽位取值。"""

    region: str
    key: str
    index: int
    slots: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class BlockPlan:
    """一个块的**计划**：类型 + 若干模板片段（或代码块的行）+ 伴随信息。

    为什么不直接存文本：`sample` 阶段拿不到语言文件（协议里没有 locale），
    所以这里只定「用哪些模板、槽位取哪些词」，真正的文本在 `render` 阶段填。
    这样同一份 spec 在任何语言下都能渲染，而题目与答案两边用的是同一份计划。

    一段正文由**多个片段**拼成（2~3 句），这样它才会被硬换行折成好几行——
    否则「并段」那一族永远只有「不是同一块」这一种答案。
    """

    kind: str
    parts: tuple[Part, ...] = ()
    #: 代码块专用：直接存要敲的命令行（代码块不折行，也不需要槽位）。
    lines: tuple[str, ...] = ()
    callout_kind: str = ""
    list_ordered: bool = False
    list_id: int = 0


@dataclass(frozen=True)
class DocSpec:
    """一条文档样本的真值：case + 块计划（文案在渲染阶段才填）。"""

    case: DocCase
    split: str
    seq: int
    phrasing: int
    plan: tuple[BlockPlan, ...]


#: 槽位取值表：构造计划时就要用，所以固定从**默认语言**的 `[slot]` 读一次。
#: 槽位名与语言无关（两种语言用同一批名字，只是文案不同），所以这样是安全的。
_SLOT_TABLE: dict[str, list[str]] = {}


def _slot_table() -> dict[str, list[str]]:
    """输入：无。输出：槽位取值表（幂等、带缓存）。

    预期行为：语言文件里没有 `[slot]` 表就直接报错——否则抽槽位时会静默留下 `{xxx}`。
    """
    if not _SLOT_TABLE:
        from . import locales

        for name, pool in locales.load(locales.DEFAULT).get("slot", {}).items():
            _SLOT_TABLE[name] = list(pool)
        if not _SLOT_TABLE:
            raise RuntimeError("语言文件里没有 [slot] 表，构造计划时无法抽槽位")
    return _SLOT_TABLE


def _part(region: str, key: str, index: int, template: str, rng: random.Random) -> Part:
    """输入：模板位置、模板原文、RNG。输出：抽好槽位的片段。

    预期行为：槽位**在这里就抽定**（不是渲染时才抽），所以同一份 spec 渲染两次逐字节一致；
    模板里出现语言文件没有的槽位直接报错，不静默留下 `{xxx}`。
    """
    table = _slot_table()
    slots: list[tuple[str, str]] = []
    for name in dict.fromkeys(_PLACEHOLDER.findall(template)):
        if name not in table:
            raise KeyError(f"模板里有未知槽位 {{{name}}}：{template!r}")
        slots.append((name, rng.choice(table[name])))
    return Part(region, key, index, tuple(slots))


def _pick(rng: random.Random, pool: list[str]) -> tuple[int, str]:
    """输入：RNG、候选列表。输出：(下标, 原文)。预期行为：下标与原文一定对应。"""
    index = rng.randrange(len(pool))
    return index, pool[index]


def _paragraph(rng: random.Random, region: str, key: str, pool: list[str], count: int) -> BlockPlan:
    """输入：RNG、模板位置、句子池、句子数。输出：一段正文的计划。

    预期行为：抽**互不相同**的句子（同一段里不重复同一句），顺序按抽中的先后。
    """
    indices = rng.sample(range(len(pool)), min(count, len(pool)))
    return BlockPlan(
        kind=KIND_PARAGRAPH,
        parts=tuple(_part(region, key, index, pool[index], rng) for index in indices),
    )


def build_plan(case: DocCase) -> tuple[BlockPlan, ...]:
    """输入：case。输出：块计划（顺序稳定、与语言无关的随机）。

    预期行为：随机种子由**素材指纹**派生，所以同一个 case 在任何切分、任何进程里都得到
    同一份计划（「同素材同题必须同答」这条一致性校验才成立）。

    文档骨架（每份都一样）：标题 + 一段正文（2~3 句）+ 可选块 + 收尾段落。
    标题**总是紧贴**它后面那段正文（中间不空行）——这正是「短行后面跟着长行」那种
    需要读者自己判断「标题到此为止」的难例。
    """
    from . import locales

    loc = locales.load(locales.DEFAULT)
    rng = random.Random(int(hashlib.sha256(material_of(case).encode("utf-8")).hexdigest()[:16], 16))
    blocks: list[BlockPlan] = []

    head_index, head_text = _pick(rng, loc["head"][case.topic])
    blocks.append(BlockPlan(kind=KIND_HEADING, parts=(_part("head", case.topic, head_index, head_text, rng),)))

    blocks.append(_paragraph(rng, "sent", case.topic, loc["sent"][case.topic], rng.randint(4, 5)))

    if case.shape in (SHAPE_LIST, SHAPE_MIXED):
        # `list` 形状用 variant 决定有序性（可以留出）；`mixed` 的 variant 已经被提示框种类占着，
        # 所以它另抽一次——两个形状一起才能让「有序 / 无序」这一族不至于一边倒。
        ordered = case.variant == "ordered" if case.shape == SHAPE_LIST else rng.random() < 0.5
        index, text = _pick(rng, loc["intro"]["list"])
        blocks.append(BlockPlan(kind=KIND_PARAGRAPH, parts=(_part("intro", "list", index, text, rng),)))
        pool_key = "step" if ordered else "plain"
        pool = loc["item"][pool_key]
        for item_index in sorted(rng.sample(range(len(pool)), rng.randint(2, min(3, len(pool))))):
            blocks.append(
                BlockPlan(
                    kind=KIND_LIST_ITEM,
                    parts=(_part("item", pool_key, item_index, pool[item_index], rng),),
                    list_ordered=ordered,
                    list_id=1,
                )
            )

    if case.shape in (SHAPE_CODE, SHAPE_MIXED):
        index, text = _pick(rng, loc["intro"]["code"])
        blocks.append(BlockPlan(kind=KIND_PARAGRAPH, parts=(_part("intro", "code", index, text, rng),)))
        pool = loc["code"][case.code_family]
        picks = sorted(rng.sample(range(len(pool)), rng.randint(1, min(2, len(pool)))))
        blocks.append(BlockPlan(kind=KIND_CODE, lines=tuple(pool[item] for item in picks)))

    if case.shape in (SHAPE_QUOTE, SHAPE_MIXED):
        keys = sorted(loc["quote"])
        key = keys[rng.randrange(len(keys))]
        blocks.append(BlockPlan(kind=KIND_QUOTE, parts=(_part("quote", key, 0, loc["quote"][key], rng),)))

    if case.shape in (SHAPE_CALLOUT, SHAPE_MIXED):
        kind = case.variant
        pool = loc["callout"][kind]
        index, text = _pick(rng, pool)
        blocks.append(
            BlockPlan(
                kind=KIND_CALLOUT,
                parts=(_part("callout", kind, index, text, rng),),
                callout_kind=kind,
            )
        )

    index, text = _pick(rng, loc["intro"]["close"])
    blocks.append(BlockPlan(kind=KIND_PARAGRAPH, parts=(_part("intro", "close", index, text, rng),)))
    return tuple(blocks)


def sample_spec(case: DocCase, split: str, seq: int, phrasing: int) -> DocSpec:
    """输入：case、切分、序号、问法。输出：真值规格。

    预期行为：**不掷骰子**——计划完全由 case 指纹决定，这样「同素材同题必须同答」成立。
    """
    return DocSpec(case=case, split=split, seq=seq, phrasing=phrasing, plan=build_plan(case))
