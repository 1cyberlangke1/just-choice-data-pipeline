"""行情读数判读（quoteread）的特征层：三行读数 + 单位换算 + 四个由规则算出来的标签。

社区依据：`TradingAgents`（★108478）在给出交易判断前先按行情读数做筛选、`Jev-X-Sentiment-Analysis`
（★162）对行情片段做方向判读、`jevocks`（★9）把行情表读成档位。三家做的都是同一件事：
**把一屏行情读数读成可判定的结论**。

本 workflow 把「读行情」做成可判定的算术，四条规则全印在题面里：

1. 读数表里每行的单位可能不同（百分比 `%` 或基准点 `bp`），一律**先换算成基准点**再比较，
   `1% = 100 基准点`；
2. 档位按绝对值切：`< 20` 几乎平盘、`20~49` 小幅、`50~99` 中等、`100~199` 大幅、`>= 200` 剧烈；
3. 警戒线 150 基准点：绝对值达到它就算触发；
4. 并列时取表中序号靠前的标的。

**为什么这是本仓库还没有的那个轴**：`receiptcat` 是**同量纲**比上限（金额对金额），
这里多了一步**单位换算**——先要把 `-0.45%` 读成 `-45 基准点`，才谈得上比档位与警戒线。
表里同时出现 `%` 与 `bp` 两种写法时，直接比数字会得出**与换算后不同**的答案
（`tests/workflows/quoteread/test_features.py` 里有一条断言专门钉这件事）。

**两条独立路径**：`label_for` 从注入特征推标签（构造侧），`observed_labels` 从**反解出来的**
读数重推标签（审计侧）。两者走同一批纯函数，但输入一个是 case、一个是 state，所以
「渲染把读数写歪了」与「生成器把标签算错了」会被分开抓到。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jc.cover import cover_dims
from jc.jsontypes import JsonValue

#: 读数表固定三行（三个标的）。
ROWS: Final[int] = 3

#: 同分布侧的标的名键（值留出：`test_ood` 换另一组名，规则一个字都不动）。
NAME_IDS: Final[tuple[str, ...]] = ("I1", "I2", "I3")
#: 只在 `test_ood` 出现的标的名键。
OOD_NAME_IDS: Final[tuple[str, ...]] = ("A1", "A2", "A3")

#: 标的名集：0 = 同分布侧，1 = 留出侧。
NAME_SETS: Final[tuple[tuple[str, ...], ...]] = (NAME_IDS, OOD_NAME_IDS)

#: 档位下界（单位：基准点）：下标 i 的档位覆盖 `[TIER_FLOORS[i], TIER_FLOORS[i+1])` 的**绝对值**。
#: 最后一个是开口的上界，所以它单独一档。
TIER_FLOORS: Final[tuple[int, ...]] = (0, 20, 50, 100, 200)
#: 警戒线（基准点）：绝对值达到它就算触发。**刻意不落在档位边界上**——落在边界上时
#: `alert` 就成了 `tier` 的同义词（`tier >= 3`），白出一族题。
ALERT_BP: Final[int] = 150
#: 1 个百分点 = 100 基准点（换算的唯一系数）。
BP_PER_PCT: Final[int] = 100

#: 单位键：`pct` = 百分比，`bp` = 基准点。原文写在语言文件的 `[unit]` 里。
UNITS: Final[tuple[str, ...]] = ("pct", "bp")

#: 读数模式：每一条是三个标的的**基准点数**（展示单位另算）。
#:
#: 八条候选是**先算分布再定稿**的（范式 §9.75 第二条）：四条题族的答案分布各自落在噪声
#: 地板以内，且覆盖到全部五档（`< 20` 由全零那条给、`50~99` 由 `(-20, 70, 30)` 给）。
VALUE_PATTERNS: Final[tuple[tuple[int, ...], ...]] = (
    (-150, -45, 25),
    (-10, 80, 200),
    (0, 0, 0),
    (45, 45, 45),
    (-300, 30, -20),
    (15, 155, 5),
    (-15, -30, -8),
    (-20, 70, 30),
)

#: 单位模式：每行用什么单位展示。全 `%` / 全 `bp` / 两种混排。
#:
#: 混排那两条不是装饰：`(bp, bp, pct)` 会让 `(-10, 80, 200)` 这条的**直接比数字**答案翻车
#: （显示成 `-10bp / +80bp / +2.00%`，直接比数字最大的是第二行，换算后其实是第三行）。
UNIT_PATTERNS: Final[tuple[tuple[str, ...], ...]] = (
    ("pct", "pct", "pct"),
    ("bp", "bp", "bp"),
    ("bp", "bp", "pct"),
    ("pct", "bp", "bp"),
)

#: 同分布侧的采样维度表：8 读数模式 × 4 单位模式 × 1 组名 = 32 组。
#: `names` 在同分布侧**恒为一个取值**——这就是「整个维度留出」在本格式里的落地形态
#: （字段不可能缺席，只能恒为一个值，见 `tests/workflows/test_generalization.py`）。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {
    "values": tuple(range(len(VALUE_PATTERNS))),
    "units": tuple(range(len(UNIT_PATTERNS))),
    "names": (0,),
}

#: 留出侧的采样维度表：读数与单位全开，标的名换成留出那一组。
OOD_DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {
    "values": tuple(range(len(VALUE_PATTERNS))),
    "units": tuple(range(len(UNIT_PATTERNS))),
    "names": (1,),
}


def tier_of(bp: int) -> int:
    """输入：一个基准点数。输出：它的档位下标（0~4）。

    预期行为：按**绝对值**比档位下界，取「最后一个不大于它的下界」的下标；
    `0` 落进第 0 档（几乎平盘），符号不影响档位。
    """
    size = abs(bp)
    tier = 0
    for index, floor in enumerate(TIER_FLOORS):
        if size >= floor:
            tier = index
    return tier


def leader_of(values: tuple[int, ...]) -> int:
    """输入：三个基准点读数。输出：**涨幅最大**那一行的下标。

    预期行为：带符号比大小（不是比绝对值），并列时取序号靠前的——这条并列规则印在题面里，
    否则「三行全零」这种读数就没有唯一答案。
    """
    best = 0
    for index, value in enumerate(values):
        if value > values[best]:
            best = index
    return best


def alert_of(values: tuple[int, ...]) -> bool:
    """输入：三个基准点读数。输出：有没有哪一行的绝对值达到警戒线。"""
    return any(abs(value) >= ALERT_BP for value in values)


def divergent_of(values: tuple[int, ...]) -> bool:
    """输入：三个基准点读数。输出：有没有出现「一涨一跌」。

    预期行为：只看符号，**零不算方向**（全零是「没有背离」，不是「两边都占」）。
    """
    return any(value > 0 for value in values) and any(value < 0 for value in values)


@dataclass(frozen=True)
class QuoteCase:
    """一条样本的注入特征：用哪条读数模式 + 哪条单位模式 + 哪组标的名。"""

    values: int
    units: int
    names: int

    @property
    def bp_values(self) -> tuple[int, ...]:
        """输入：无。输出：三行的基准点读数。"""
        return VALUE_PATTERNS[self.values]

    @property
    def unit_seq(self) -> tuple[str, ...]:
        """输入：无。输出：三行各自的展示单位键。"""
        return UNIT_PATTERNS[self.units]

    @property
    def name_ids(self) -> tuple[str, ...]:
        """输入：无。输出：三行的标的名键（同分布侧 / 留出侧）。"""
        return NAME_SETS[self.names]

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"values": self.values, "units": self.units, "names": self.names}

    def fingerprint(self) -> str:
        """输入：无。输出：case 指纹。预期行为：同一 case 任何进程都得到同一串。"""
        return "|".join(f"{key}={value}" for key, value in self.as_dict().items())

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"V{self.values}-U{self.units}-N{self.names}"


@dataclass(frozen=True)
class QuoteLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`tier` 最大绝对读数的档位；`leader` 涨幅最大那一行的下标；`alert` 有没有越过警戒线；
    `divergent` 有没有一涨一跌。
    """

    tier: int
    leader: int
    alert: bool
    divergent: bool


def observed_labels(values: tuple[int, ...]) -> QuoteLabels:
    """输入：三个基准点读数。输出：全部金标签（**审计侧**路径）。

    预期行为：构造侧与审计侧走的就是这一个函数——构造式 workflow 里 state 就是规格，
    没有「模型写的正文」可歪，所以两条路径的差别只在输入从哪来（case 还是反解结果）。
    """
    return QuoteLabels(
        tier=tier_of(max(values, key=abs)),
        leader=leader_of(values),
        alert=alert_of(values),
        divergent=divergent_of(values),
    )


def label_for(case: QuoteCase) -> QuoteLabels:
    """输入：注入特征。输出：全部金标签（**构造侧**路径）。"""
    return observed_labels(case.bp_values)


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[QuoteCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`QuoteCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [
        QuoteCase(values=int(row["values"]), units=int(row["units"]), names=int(row["names"]))
        for row in rows
    ]
