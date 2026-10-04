"""多批次排名聚合（siftrank）workflow 的真值规格：把重复批次的排名聚合成一条相关性顺序。

输入/输出：不可变数据类与纯函数；**不含任何领域词表**（条目名、排序提示、抬头、题面与选项
描述全在 `locales/`）。
预期行为：一条样本是「一个排序提示 + 一份条目清单 + 四个批次」，每个批次把清单里的
`BATCH_SIZE` 个条目按相关性从高到低排成一列。四个题族的答案全部由这四个批次算出来，零 token。

**为什么是「重复批次」而不是「一次排序」**：社区里 `siftrank`（★210, Go）把同一批条目
随机采样成小批次、反复让模型排序，再把各批次的排名**聚合**成一条稳定的相关性顺序
（SIFT = Stochastic / Inflective / Fixed / Trial）。它要解决的正是一次排序的三个毛病：
不确定、上下文装不下、模型漏项。所以这里考的不是「读一列排名」，而是**多列排名之间怎么
合并**——聚合规则必须写得让只读 state 的读者能自己算出来（范式 §9.8）。

**聚合口径选了 Borda 计数 + 肘部**，理由是这两条都能被「只用纸笔」复算：
- 计分：批次里第 p 名（p 从 1 起）得 `k - p` 分，k 是这个批次排了几个条目；
- 总分：条目在各批次得分之和，没出现记 0 分；
- 顺序：总分降序；**总分相同就是并列**（清单顺序只决定「怎么写出来」）；
- 肘部：相邻两项总分差最大的那一处，它左边（含左侧那个）算「相关」。

**档位与题族的分布是先算后定的**（范式 §9.70 一）：六个模板按三个答案轴各占一半/三分之一
——`cut` 取 1/2/3 各两个、`relation` 取 a_first/b_first/tie 各两个、`stable` 真伪各三个；
`agg_top` 那一族靠组装器打乱选项，答案下标天然均匀。见 `tests/workflows/siftrank/test_spec.py`
里那条把三张分布表一起钉住的尺子。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import hashlib
import random
from dataclasses import dataclass
from typing import Final

# ---------- 条目池与批次尺寸 ----------

#: 条目键（顺序稳定；清单顺序就是它，并列时的展示先后也按它）。名字写在 `locales/`。
ITEMS: Final[tuple[str, ...]] = (
    "overview",
    "runbook",
    "changelog",
    "metrics",
    "alerts",
    "postmortem",
)

#: 每个批次排几个条目。批次里第 p 名（从 1 起）得 `BATCH_SIZE - p` 分。
#: 定成 4（而不是全部 6 个）是有意的：siftrank 的批次本来就是**采样**出来的子集，
#: 而且部分批次才让「总分」与「出现次数」分得开——全量批次下每个条目每批都出现，
#: 「没出现的批次记 0 分」这条判据就成了死条文。
BATCH_SIZE: Final[int] = 4

#: 一条样本里有多少个批次（六个模板都按它写；反解也拿它当「批次齐不齐」的尺子）。
TRIALS: Final[int] = 4


# ---------- 批次模板 ----------


@dataclass(frozen=True)
class Template:
    """一个批次模板：四批各自的条目位置（指到 `ITEMS` 的下标）+ 出「谁更相关」那一族的位置对。

    输入：`batches` 是四个位置元组（每个长度 `BATCH_SIZE`），`pair` 是两个位置。
    输出：不可变数据类。
    预期行为：模板只描述**形状**；条目身份由 `RankCase.perm` 这个映射决定。于是「把映射换一换」
    只会重排总分的归属，不会改任何答案的形状——三张分布表因此只由模板决定，可离线算准。
    """

    batches: tuple[tuple[int, ...], ...]
    pair: tuple[int, int]


#: 六个模板。三条答案轴（`cut` / `relation` / `stable`）的分布由它们**按构造**决定：
#: `cut` = 1/1/2/2/3/3，`relation` = a_first/a_first/tie/tie/b_first/b_first，`stable` = 真/真/假/假/假/假。
#: 每个模板的总分向量都手算过，`tests/workflows/siftrank/test_spec.py` 把它们逐条钉在数据上。
TEMPLATES: Final[dict[str, Template]] = {
    # 一个条目领跑：总分 12/4/3/3/2/0，最大落差在第 1 与第 2 名之间 → 相关 1 个；批次彼此一致。
    "peak": Template(
        batches=((0, 1, 2, 3), (0, 2, 4, 5), (0, 3, 4, 5), (0, 1, 4, 5)),
        pair=(0, 1),
    ),
    # 领跑 + 次席咬得很紧：总分 12/5/4/2/1/0，落差仍在最前面 → 相关 1 个；批次彼此一致。
    "surge": Template(
        batches=((0, 1, 2, 3), (0, 2, 3, 4), (0, 1, 3, 5), (0, 2, 4, 5)),
        pair=(0, 1),
    ),
    # 双雄并列：总分 10/10/2/1/1/0，最大落差在第 2 与第 3 名之间 → 相关 2 个；前两名在批次里互有胜负。
    "twin": Template(
        batches=((0, 1, 2, 3), (1, 0, 4, 5), (0, 1, 4, 2), (1, 0, 3, 4)),
        pair=(0, 1),
    ),
    # 领跑 + 中段并列：总分 9/7/4/2/2/0，最大落差在第 2 与第 3 名之间 → 相关 2 个；末段两个条目互有胜负。
    "swirl": Template(
        batches=((0, 1, 2, 3), (0, 1, 4, 5), (0, 4, 2, 5), (1, 3, 4, 2)),
        pair=(2, 3),
    ),
    # 阶梯：总分 6/6/5/3/3/1，最大落差在第 3 与第 4 名之间 → 相关 3 个；末段被批次的回绕打乱。
    "chain": Template(
        batches=((0, 1, 2, 3), (1, 2, 3, 4), (2, 3, 4, 5), (3, 4, 5, 0)),
        pair=(0, 1),
    ),
    # 缓坡：总分 7/7/6/3/1/0，最大落差同样在第 3 与第 4 名之间 → 相关 3 个；四批都是一条递增序列的子列。
    "stair": Template(
        batches=((0, 1, 2, 3), (0, 1, 2, 4), (1, 2, 3, 5), (2, 3, 4, 5)),
        pair=(0, 1),
    ),
}

#: 模板名（顺序稳定）。
SHAPES: Final[tuple[str, ...]] = tuple(TEMPLATES)

#: 条目映射：模板里的位置 → `ITEMS` 下标。换映射只换「哪个条目占哪个位」，不换答案形状。
PERMS: Final[dict[str, tuple[int, ...]]] = {
    "identity": (0, 1, 2, 3, 4, 5),
    "rotate2": (2, 3, 4, 5, 0, 1),
    "reverse": (5, 4, 3, 2, 1, 0),
}

#: 映射名（顺序稳定）。
PERM_NAMES: Final[tuple[str, ...]] = tuple(PERMS)

# ---------- 排序提示（值留出维） ----------

#: 排序提示。**只改抬头下面那一行文字，不改任何规则分支**，所以是天然的留出维：
#: 最后一个（`audit`）只在 OOD 侧出现。
FLAVORS: Final[tuple[str, ...]] = ("launch", "incident", "cost", "audit")

#: 训练侧见过的排序提示。
SEEN_FLAVORS: Final[frozenset[str]] = frozenset(FLAVORS[:-1])

#: 同分布侧的取值表（顺序按原表）。
ID_FLAVORS: Final[tuple[str, ...]] = tuple(
    flavor for flavor in FLAVORS if flavor in SEEN_FLAVORS
)

# ---------- 答案的取值空间 ----------

#: 两个条目谁更相关。
A_FIRST: Final[str] = "a_first"
B_FIRST: Final[str] = "b_first"
TIE: Final[str] = "tie"
RELATIONS: Final[tuple[str, ...]] = (A_FIRST, B_FIRST, TIE)

#: 「相关条目有几个」的取值空间（`score` 族的等级表就是按它写的）。
CUTS: Final[tuple[int, ...]] = (1, 2, 3)

#: 问法池（题面措辞不是判据，换说法不影响答案）。
PHRASINGS: Final[tuple[int, ...]] = (0, 1, 2)


# ---------- 聚合（审计侧的独立路径：只看批次，不看模板） ----------


def totals_of(batches: tuple[tuple[str, ...], ...]) -> dict[str, int]:
    """输入：批次（每个是条目键的有序元组）。输出：{条目键: 总分}。

    预期行为：批次里第 p 名（p 从 1 起）得 `k - p` 分，k 是这个批次排了几个条目；
    没出现在某个批次里记 0 分。**这是审计侧的独立路径**——它只看渲染出来的批次，
    与 `RankCase` 的模板无关，两条路对不上就说明渲染没兑现计划。
    """
    totals: dict[str, int] = {key: 0 for key in ITEMS}
    for batch in batches:
        size = len(batch)
        for position, key in enumerate(batch):
            totals[key] += size - position - 1
    return totals


def order_of(totals: dict[str, int]) -> tuple[str, ...]:
    """输入：总分表。输出：条目键的**展示顺序**（总分降序，并列按清单顺序）。

    预期行为：并列时的先后只是「怎么写出来」，不表示谁更相关——「谁更相关」那一族由
    `relation_of` 单独判，它把总分相同判成并列。
    """
    return tuple(sorted(ITEMS, key=lambda key: (-totals[key], ITEMS.index(key))))


def cut_of(totals: dict[str, int]) -> int:
    """输入：总分表。输出：相关条目个数（肘部左侧的条目数）。

    预期行为：把条目按展示顺序排好后，看相邻两项的总分差；**差最大**的那一处是肘部
    （并列最大取最靠前的一处），它左边（含左侧那个）算相关。所有总分都相同时没有落差，
    这时全部算相关（返回条目总数）。
    """
    ordered = order_of(totals)
    values = [totals[key] for key in ordered]
    gaps = [values[index] - values[index + 1] for index in range(len(values) - 1)]
    if not gaps or max(gaps) == 0:
        return len(values)
    return gaps.index(max(gaps)) + 1


def relation_of(totals: dict[str, int], left: str, right: str) -> str:
    """输入：总分表、两个条目键。输出：`a_first` / `b_first` / `tie`（总分相同即并列）。"""
    if totals[left] > totals[right]:
        return A_FIRST
    if totals[left] < totals[right]:
        return B_FIRST
    return TIE


def is_stable(batches: tuple[tuple[str, ...], ...]) -> bool:
    """输入：批次。输出：这些批次彼此一致吗。

    预期行为：对每一对同时出现在两个批次里的条目，先后顺序都相同才算「一致」；只要存在
    一对条目在两个批次里的先后相反，就是「不一致」。只出现在一个批次里的对不参与判定。
    """
    signs: dict[tuple[str, str], int] = {}
    for batch in batches:
        for index, left in enumerate(batch):
            for right in batch[index + 1 :]:
                pair = (left, right) if left < right else (right, left)
                sign = 1 if left == pair[0] else -1
                known = signs.get(pair)
                if known is None:
                    signs[pair] = sign
                elif known != sign:
                    return False
    return True


def observed_labels(batches: tuple[tuple[str, ...], ...]) -> dict[str, JsonValue]:
    """输入：批次。输出：从批次**重算**出来的全部答案（外加展示顺序）。

    预期行为：构造侧（`RankCase` 的属性）与这里必须逐条一致——两条独立路径是这一族的判据
    骨架，一条歪了整族都不可信。
    """
    totals = totals_of(batches)
    ordered = order_of(totals)
    return {
        "totals": dict(totals),
        "order": list(ordered),
        "top": ordered[0],
        "cut": cut_of(totals),
        "stable": is_stable(batches),
    }


# ---------- 素材 ----------


@dataclass(frozen=True)
class RankCase:
    """一条排序请求的素材：批次形状 + 条目映射 + 排序提示。"""

    shape: str
    perm: str
    flavor: str

    def batches(self) -> tuple[tuple[str, ...], ...]:
        """输入：无。输出：四个批次（条目键的有序元组）。"""
        mapping = PERMS[self.perm]
        template = TEMPLATES[self.shape]
        return tuple(
            tuple(ITEMS[mapping[position]] for position in batch)
            for batch in template.batches
        )

    def pair(self) -> tuple[str, str]:
        """输入：无。输出：出「谁更相关」那一族用的两个条目键。"""
        mapping = PERMS[self.perm]
        left, right = TEMPLATES[self.shape].pair
        return ITEMS[mapping[left]], ITEMS[mapping[right]]

    def totals(self) -> dict[str, int]:
        """输入：无。输出：{条目键: 总分}。"""
        return totals_of(self.batches())

    def cut(self) -> int:
        """输入：无。输出：相关条目个数。"""
        return cut_of(self.totals())

    def relation(self) -> str:
        """输入：无。输出：`pair()` 里两个条目谁更相关。"""
        left, right = self.pair()
        return relation_of(self.totals(), left, right)

    def top(self) -> str:
        """输入：无。输出：展示顺序里最靠前的条目键（并列按清单顺序取靠前者）。"""
        return order_of(self.totals())[0]

    def stable(self) -> bool:
        """输入：无。输出：四个批次彼此一致吗。"""
        return is_stable(self.batches())

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这一条里训练侧没见过的维（排序提示留出时非空）。"""
        return () if self.flavor in SEEN_FLAVORS else (f"flavor={self.flavor}",)

    def material(self) -> str:
        """输入：无。输出：素材指纹（切分按它分桶，同一份请求永远落在同一档）。

        预期行为：**不含 `flavor`**——它只改抬头下面那一行文字，同一条请求的几种提示
        必须整组落在同一个切分里，否则「同素材不跨切分」这条一致性校验会失效。
        """
        return f"{self.shape}|{self.perm}"

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：{维度名: 取值}（三个维度全摊平，供统计与均衡报告）。"""
        return {"shape": self.shape, "perm": self.perm, "flavor": self.flavor}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"shape={self.shape} perm={self.perm} flavor={self.flavor}"


@dataclass(frozen=True)
class RankConfig:
    """采样参数：跑哪一档切分、每个维取哪些值。"""

    split: str
    shapes: tuple[str, ...] = SHAPES
    perms: tuple[str, ...] = PERM_NAMES
    flavors: tuple[str, ...] = FLAVORS
    phrasings: tuple[int, ...] = PHRASINGS
    ood_only: bool = False


def build_cases(config: RankConfig) -> list[RankCase]:
    """输入：采样参数。输出：case 列表（顺序稳定、可复算）。

    预期行为：三个维全枚举；`ood_only` 时只留「至少含一个训练侧没见过的取值」的 case，
    所以 OOD 与同分布侧**按构造就不重叠**。
    """
    out: list[RankCase] = []
    for shape in config.shapes:
        for perm in config.perms:
            for flavor in config.flavors:
                case = RankCase(shape, perm, flavor)
                if config.ood_only and not case.unseen_dims():
                    continue
                out.append(case)
    return out


# ---------- 真值规格 ----------


@dataclass(frozen=True)
class RankSpec:
    """一条排序请求的完整真值：case + 由批次算出来的四个答案。"""

    case: RankCase
    split: str
    seq: int
    phrasing: int
    request_id: str
    batches: tuple[tuple[str, ...], ...]
    totals: dict[str, int]
    cut: int
    stable: bool
    pair: tuple[str, str]
    relation: str
    top: str


def request_id_of(case: RankCase) -> str:
    """输入：一个 case。输出：排序请求号（由素材指纹派生，稳定可复算）。

    预期行为：同一份素材在任何切分、任何语言下都是同一个请求号——「同输入同标签」这条
    一致性校验要靠它认人。
    """
    digest = hashlib.sha256(case.material().encode("utf-8")).hexdigest()[:6].upper()
    return f"S-{digest}"


def sample_spec(
    rng: random.Random, case: RankCase, split: str, seq: int, phrasing: int
) -> RankSpec:
    """输入：RNG（**不用**）、case、切分、序号、问法。输出：真值规格。

    预期行为：不掷骰子——同一 case 必得同一份规格，这样「同素材同题必须同答」才成立。
    """
    del rng  # 规格完全由 case 决定；签名里的 RNG 只为与协议一致
    left, right = case.pair()
    return RankSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        request_id=request_id_of(case),
        batches=case.batches(),
        totals=case.totals(),
        cut=case.cut(),
        stable=case.stable(),
        pair=(left, right),
        relation=case.relation(),
        top=case.top(),
    )
