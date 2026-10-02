"""片段筛选与去冗余（clipcut）的特征层：主题序列、高亮序列与「先到先得」的保留规则。

社区依据：`summarize`（★223）在把转录片段送去摘要器之前先给片段打分筛选、`jevmeter`（★70）
逐句给视频评分、`jevclip`（★21）判断字幕片段以筛选高亮内容并删去冗余。它们要解决的是同一
件事：**从一串片段里挑出该留下的那几段**。

本 workflow 把「挑哪几段」做成可判定的算术，规则只有两条，全印在 state 里：

1. 一段片段被保留，当且仅当它**含高亮点**、且它是它那个主题**第一次出现**的位置；
2. 主题重复出现时，只有第一次那一处有资格被保留，后面的即使含高亮点也不保留。

**为什么这是本仓库还没有的那个轴**：`cachereuse` 做的是两条轨迹的**逐步对齐**（成对比较），
`maze` / `invoice` 那些是**逐条独立判断**，而这一族里每一段的去留都取决于**它前面的片段**
（同一个主题有没有先出现过）——是「从前往后扫」的折叠，答案依赖前缀，而不只看本条。

**两条独立路径**：`label_for` 从注入特征推标签（构造侧），`observed_labels` 从**反解出来的**
片段表重推标签（审计侧）。两者走同一批纯函数，但输入一个是 case、一个是 state，所以
「渲染把片段表写歪了」与「生成器把标签算错了」会被分开抓到。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jc.cover import Fingerprinted, cover_dims
from jc.jsontypes import JsonValue

#: 同分布侧的主题（文案在 `locales/`，这里只有编号）。
TOPIC_IDS: Final[tuple[str, ...]] = ("T1", "T2", "T3", "T4")
#: 只在 `test_ood` 出现的主题（值留出）：同分布侧一次都见不到。
OOD_TOPIC_ID: Final[str] = "T5"

#: 一段材料固定 4 个片段。
SEGMENTS: Final[int] = 4

#: 主题序列：前四条是**同分布**用的，最后一条含留出主题 `T5`，只在 `test_ood` 出现。
#
# 为什么要有两条「全不重复」的序列（下标 2 与 3）：`dup` 那一族的 False 侧全靠它们。只留一条
# 时 `dup` 会偏到 24:8，贴着均衡闸的噪声地板（实测 dev 0.250 / noise 0.265）。
#
# 留出侧（下标 4 与 5）同理，而且更要紧：**只有一条留出序列时 `dup_any` 在留出侧是常量**。
# 下标 4 自带重复（`T5` 出现两次），所以第一版留出侧八条样本全是 True——`dup_any` 那一族在
# 留出侧一条信息都不带。补上下标 5（`T5` 只出现一次）之后，留出侧两半一半有重复、一半没有
# （范式 §9.79 第六条：留出某一维时要连着看依赖它的那一族会不会塌到单值）。
TOPIC_PATTERNS: Final[tuple[tuple[str, ...], ...]] = (
    ("T1", "T1", "T2", "T2"),
    ("T1", "T2", "T1", "T2"),
    ("T1", "T2", "T3", "T4"),
    ("T2", "T3", "T4", "T1"),
    ("T1", OOD_TOPIC_ID, OOD_TOPIC_ID, "T2"),
    ("T3", OOD_TOPIC_ID, "T1", "T2"),
)

#: 同分布侧的主题序列下标。
ID_TOPIC_PATTERNS: Final[tuple[int, ...]] = (0, 1, 2, 3)
#: 留出侧的主题序列下标（一条自带重复、一条全不重复，`dup_any` 在留出侧才有两侧）。
OOD_TOPIC_PATTERNS: Final[tuple[int, ...]] = (4, 5)

#: 高亮序列（4 段，任意子集）。**不能只取「前 k 段高亮」那种前缀形状**——那样首段几乎总被
#: 保留，`first_kept` 那一族会退化成常量。这一组覆盖了全亮 / 首段不亮 / 尾段不亮 / 中段成对 /
#: 全不亮五种形状，四个题族的答案分布才都落在噪声地板以内。
FLAG_PATTERNS: Final[tuple[tuple[bool, ...], ...]] = (
    (True, True, True, True),
    (False, True, True, True),
    (False, False, True, True),
    (False, False, False, True),
    (False, False, False, False),
    (True, True, False, False),
    (False, True, True, False),
    (True, False, True, False),
)

#: 采样维度表：主题序列 × 高亮序列 = 4 × 8 = 32 组（同分布侧）。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {
    "topics": ID_TOPIC_PATTERNS,
    "flags": tuple(range(len(FLAG_PATTERNS))),
}


def first_positions(topics: tuple[str, ...]) -> dict[str, int]:
    """输入：主题序列。输出：{主题: 它第一次出现的片段下标}。

    预期行为：纯函数，只看序列——「第一次出现」就是这一族的顺序折叠：某一段的去留取决于
    它前面有没有同主题的片段。
    """
    first: dict[str, int] = {}
    for index, topic in enumerate(topics):
        first.setdefault(topic, index)
    return first


def kept_of(topics: tuple[str, ...], flags: tuple[bool, ...]) -> tuple[int, ...]:
    """输入：主题序列、高亮序列。输出：该保留的片段下标（0 起，升序）。

    预期行为：保留 = 含高亮点 **且** 是该主题第一次出现的位置。两条都满足才留下。
    """
    first = first_positions(topics)
    return tuple(
        index
        for index, (topic, flag) in enumerate(zip(topics, flags, strict=True))
        if flag and first[topic] == index
    )


def blocked_of(topics: tuple[str, ...], flags: tuple[bool, ...]) -> tuple[int, ...]:
    """输入：主题序列、高亮序列。输出：被去冗余挡下的片段下标（含高亮但主题不是首现）。

    预期行为：这些片段本来够格（有高亮点），只因为主题在前面出现过而被丢掉——它们是
    「去冗余真的起了作用」的证据，与 `has_dup`（主题有没有重复）是两件事。
    """
    first = first_positions(topics)
    return tuple(
        index
        for index, (topic, flag) in enumerate(zip(topics, flags, strict=True))
        if flag and first[topic] != index
    )


def has_dup(topics: tuple[str, ...]) -> bool:
    """输入：主题序列。输出：有没有「主题在更靠前的片段里已经出现过」的片段。

    预期行为：只看主题序列，**不看高亮、也不看保留结果**——重复存在 ≠ 重复挡下了东西：
    重复的那一段如果本来就没有高亮点，它不会被挡下（它压根没资格）。
    """
    seen: set[str] = set()
    for topic in topics:
        if topic in seen:
            return True
        seen.add(topic)
    return False


@dataclass(frozen=True)
class ClipCase(Fingerprinted):
    """一条样本的注入特征：用哪一条主题序列 + 哪一条高亮序列。"""

    topics: int
    flags: int

    @property
    def topic_seq(self) -> tuple[str, ...]:
        """输入：无。输出：这一条的主题序列（主题编号元组）。"""
        return TOPIC_PATTERNS[self.topics]

    @property
    def flag_seq(self) -> tuple[bool, ...]:
        """输入：无。输出：这一条的高亮序列。"""
        return FLAG_PATTERNS[self.flags]

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"topics": self.topics, "flags": self.flags}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"P{self.topics}-F{self.flags}"


@dataclass(frozen=True)
class ClipLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`kept` 该保留的片段下标；`blocked` 被去冗余挡下的片段下标；`dup` 主题有没有重复。
    """

    kept: tuple[int, ...]
    blocked: tuple[int, ...]
    dup: bool

    @property
    def count(self) -> int:
        """输入：无。输出：一共保留几段（`keep_count` 那一族的答案）。"""
        return len(self.kept)

    @property
    def first(self) -> int | None:
        """输入：无。输出：保留的第一段下标；一段都没保留给 None。"""
        return self.kept[0] if self.kept else None


def label_for(case: ClipCase) -> ClipLabels:
    """输入：注入特征。输出：全部金标签（**构造侧**路径）。"""
    return observed_labels(case.topic_seq, case.flag_seq)


def observed_labels(topics: tuple[str, ...], flags: tuple[bool, ...]) -> ClipLabels:
    """输入：主题序列、高亮序列。输出：全部金标签（**审计侧**路径）。

    预期行为：构造侧与审计侧走的就是这一个函数——构造式 workflow 里 state 就是规格，
    没有「模型写的正文」可歪，所以两条路径的差别只在输入从哪来（case 还是反解结果）。
    """
    return ClipLabels(
        kept=kept_of(topics, flags),
        blocked=blocked_of(topics, flags),
        dup=has_dup(topics),
    )


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[ClipCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`ClipCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [ClipCase(topics=int(row["topics"]), flags=int(row["flags"])) for row in rows]
