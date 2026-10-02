"""情感与情绪（sentiment）的特征层：情绪弧线、词表档位与四条判读规则。

社区依据：`yanwai`（★15）在聊天模块里显示每条消息的情绪与沟通建议、`jev-dev` 比较情感变化
的偏差。它们要的是同一件事：**一串对话的情绪怎么走，哪一回合变了调**。

按本仓库第一原则（范式 §0），标签必须**算得出来**，所以情绪分值不靠模型判、而是由一张
**情绪词表**给出：每个回合的词落在 -2 ~ +2 五档上，四个答案全部由这串分值按四条规则算出来，
零 token。词表与规则都印在 state 里，只读 state 与题面的读者能自己把四个答案算出来。

四条规则（全印在 state 与题面里）：

1. 每个回合的情绪分值由词表给出（+2 最好，-2 最差，0 是不喜不悲）；
2. 整段对话的**倾向** = 各回合分值之和的符号：大于 0 偏正、小于 0 偏负、等于 0 中性；
3. 第 i 个回合（i 从 2 起）算**转折**，当且仅当它与上一回合一个为正、一个为负（0 不算）；
4. 相邻两回合的**最大落差** = 各相邻对里「上一回合减下一回合」的最大值，没有变差就是 0；
5. 第 i 个回合（i 从 2 起）算**回升**，当且仅当上一回合分值为负、这一回合分值为正。

**为什么这是本仓库还没有的那个轴**：`quoteread` 是「把读数换算成基准点再比档」（逐点独立），
`clipcut` 是线性前缀的去留，`planstep` 是偏序上的贪心推进——都是**结构**判断；这里判的是
**一串有正负号的数怎么走**：符号、符号变化的位置、变化幅度、变化的方向。

**两条独立路径**：`label_for` 从注入特征（弧线下标）推标签，`observed_labels` 从**反解出来的
分值序列**推标签（审计侧）。两者共用同一批纯函数，所以「渲染把某个词写歪了」与「生成器把
标签算错了」会被分开抓到。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jc.cover import Fingerprinted, cover_dims
from jc.jsontypes import JsonValue

#: 每段对话固定 5 个回合。
TURNS: Final[int] = 5

#: 情绪档位：+2 最好、-2 最差。词表把词映射到它。
LEVELS: Final[tuple[int, ...]] = (-2, -1, 0, 1, 2)

#: 每个档位有几个词。渲染按「弧线下标 + 回合号」轮换取词——**只换字面、不换分值**，
#: 所以字面多样性不改变任何答案（`WORD_SLOTS` 是 `locales` 里 `[word]` 每档的词数）。
WORD_SLOTS: Final[int] = 4

#: 情绪弧线表：每条是一串 5 个回合的分值。
#
# 为什么是这 15 + 4 条：四条规则把「一条弧线」压成四个答案（倾向 / 首次转折的位置 / 最大落差 /
# 有没有回升）。这 15 条同分布弧线逐族算过偏差（口径见 `tests/workflows/sentiment/test_pipeline.py`
# 的分布断言）：15 条弧线里倾向 5 正 / 6 负 / 4 中性、首次转折按五个位置铺开、最大落差 0~4
# 全取到、回升真假 8/7，四族最大偏差 0.067，远在 0.2 的退化线内。
#
# 后四条（下标 15~18）只出现在 `test_ood`：它们是同分布侧一次都没出现过的**形状**
# （来回摆、先扬后砸、先平后砸、先砸后扬）。
ARCS: Final[tuple[tuple[int, ...], ...]] = (
    (1, 1, 1, 1, 1),        # 0  一路偏好
    (2, 2, 2, -2, -2),      # 1  前半很高、后半急转直下
    (1, 1, -1, 1, 1),       # 2  中间只塌了一格
    (2, -2, 2, -2, 2),      # 3  来回摆（振幅最大）
    (1, 0, 1, 1, 1),        # 4  中间停了一拍
    (-1, -1, -1, -1, -1),   # 5  一路偏差
    (1, 1, 1, -2, -2),      # 6  先小幅偏好、末尾崩掉
    (-1, -1, 1, 1, -1),     # 7  中间回暖、末尾又回落
    (-1, -1, -2, -2, 2),    # 8  一路走低、最后一格拉回
    (-1, -1, -1, 1, 1),     # 9  后半转好
    (0, 0, 0, 0, 0),        # 10 全程中性
    (0, 1, -2, 0, 0),       # 11 中间砸了一下
    (2, -1, -1, -1, 1),     # 12 高开、低谷、末尾回升
    (0, 0, 0, -1, 1),       # 13 前三个回合中性、最后出现转折
    (1, -2, -2, 2, 1),      # 14 砸到底又拉回来
    (1, -1, 1, -1, 1),      # 15 一路来回摆（留出）
    (2, -2, -2, -1, -2),    # 16 高开之后一路偏差（留出）
    (0, 2, 2, -2, -2),      # 17 先平后高、再整体翻负（留出）
    (-2, 2, 2, 2, 2),       # 18 开局很差、之后一路偏好（留出）
)

#: 同分布侧的弧线下标。
ID_ARCS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
#: 留出侧的弧线下标（`test_ood` 用）。
OOD_ARCS: Final[tuple[int, ...]] = (15, 16, 17, 18)

#: 场景表：**只改抬头下面那一行文字与两个说话人称呼**，不改任何规则分支，所以两边共用。
TOPICS: Final[tuple[str, ...]] = ("support", "standup", "review", "interview", "neighbors")

#: 同分布侧的取值表（顺序按原表）。
ID_TOPICS: Final[tuple[str, ...]] = TOPICS

#: 采样维度表：弧线 × 场景 = 15 × 5 = 75 组（同分布侧）。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {
    "arc": ID_ARCS,
    "topic": ID_TOPICS,
}


def arc_of(arc: int) -> tuple[int, ...]:
    """输入：弧线下标。输出：这条弧线的分值序列。预期行为：越界当场报错，不静默回绕。"""
    if not 0 <= arc < len(ARCS):
        raise ValueError(f"弧线下标越界：{arc}（共 {len(ARCS)} 条）")
    return ARCS[arc]


def word_slot(arc: int, turn: int) -> int:
    """输入：弧线下标、回合下标（0 起）。输出：该回合该取词表里第几个词。

    预期行为：**只影响字面**——同一个档位取哪个词都不改变分值，所以答案与它无关；
    固定算法保证同一份 case 每次渲染出的文本逐字一致。
    """
    return (arc + turn) % WORD_SLOTS


def tone_of(levels: tuple[int, ...]) -> str:
    """输入：分值序列。输出：整段倾向（`pos` / `neg` / `flat`）。规则 2 的唯一实现。"""
    total = sum(levels)
    return "pos" if total > 0 else ("neg" if total < 0 else "flat")


def pivot_of(levels: tuple[int, ...]) -> int | None:
    """输入：分值序列。输出：**首次转折**的回合下标（0 起）；一次都没有给 None。规则 3 的唯一实现。

    预期行为：只认「一个为正、一个为负」的相邻对——0 与任何符号都不构成转折。
    """
    for index in range(1, len(levels)):
        if levels[index - 1] * levels[index] < 0:
            return index
    return None


def drop_of(levels: tuple[int, ...]) -> int:
    """输入：分值序列。输出：相邻两回合的最大落差。规则 4 的唯一实现。

    预期行为：只看**变差**的方向（上一回合减下一回合），一路不变差就是 0。
    """
    return max([levels[index - 1] - levels[index] for index in range(1, len(levels))] + [0])


def rebound_of(levels: tuple[int, ...]) -> bool:
    """输入：分值序列。输出：有没有出现「上一回合为负、这一回合为正」。规则 5 的唯一实现。"""
    return any(
        levels[index - 1] < 0 < levels[index] for index in range(1, len(levels))
    )


def pivot_key(index: int | None) -> str:
    """输入：转折回合下标（0 起）或 None。输出：`pivot` 那一族的选项名（`t2`~`t5` / `none`）。"""
    return "none" if index is None else f"t{index + 1}"


@dataclass(frozen=True)
class SentimentCase(Fingerprinted):
    """一条样本的注入特征：用哪一条情绪弧线 + 哪个场景。

    字段：`arc` 弧线下标；`topic` 场景键。
    """

    arc: int
    topic: str

    @property
    def levels(self) -> tuple[int, ...]:
        """输入：无。输出：这一条的分值序列。"""
        return arc_of(self.arc)

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"arc": self.arc, "topic": self.topic}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"A{self.arc}-{self.topic}"


@dataclass(frozen=True)
class SentimentLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`tone` 整段倾向；`pivot` 首次转折的回合下标（None = 一次都没有）；`drop` 最大落差；
    `rebound` 有没有回升。
    """

    tone: str
    pivot: int | None
    drop: int
    rebound: bool

    @property
    def pivot_key(self) -> str:
        """输入：无。输出：首次转折那一族的选项名（`t2`~`t5`；一次都没有就是 `none`）。"""
        return pivot_key(self.pivot)


def observed_labels(levels: tuple[int, ...]) -> SentimentLabels:
    """输入：分值序列。输出：全部金标签（**审计侧**路径）。

    预期行为：构造侧与审计侧走的就是这一个函数——差别只在输入从哪来（case 还是反解出来的词）。
    """
    return SentimentLabels(
        tone=tone_of(levels),
        pivot=pivot_of(levels),
        drop=drop_of(levels),
        rebound=rebound_of(levels),
    )


def label_for(case: SentimentCase) -> SentimentLabels:
    """输入：注入特征。输出：全部金标签（**构造侧**路径）。"""
    return observed_labels(case.levels)


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[SentimentCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`SentimentCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [SentimentCase(arc=int(row["arc"]), topic=str(row["topic"])) for row in rows]
