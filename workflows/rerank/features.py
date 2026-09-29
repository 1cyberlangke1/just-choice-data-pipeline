"""检索重排（rerank）的特征层：查询词覆盖率 → 名次（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条「给这批候选排个序」的样本由三个**注入特征**描述——
`terms` 查询词有几个、`relevant` 有几条候选沾上至少一个查询词、`best` 最相关的那条落在
候选列表的哪一段。**标签先行**：先定这三个取值，再由本模块的纯函数算出名次与全部金标签。

分三层，每层只干一件事：

- **判据层**（`coverage`）：一条候选沾上几个查询词由**代码**算，构造侧与读法侧共用这一份
  实现——所以「名次对不对」是机械可判的，不靠自然语言约定；
- **计划层**（`plan_for`）：每条候选的覆盖率由 case 指纹派生（`sha256` 播种），同一 case 在
  任何进程、任何切分里都得到同一份计划（「同素材同题必须同答」靠这条成立）；
- **文本层**（`render`）：查询与候选原文由后端写，代码只校验它兑现没兑现计划——模型写歪了
  就拒掉重来，而不是把名次改得跟文本一致（那才是真正的坏数据）。

**查询词由后端自己挑**（不是代码给一张领域词表）：提示词只要求「挑 `terms` 个有区分度的
词」，挑完之后**代码逐个复核**——每个词都得在查询里出现、每条候选的覆盖率得正好等于计划。
于是词表的长尾由模型负责，而「谁排第几」仍然只由代码算。

排序口径（印在 state 里，考生只读 state 就能自己答，范式 §9.8）：

1. 得分高的排前面，得分 = 这条候选沾上了几个查询词；
2. 得分相同时，**原本靠前的**排前面（稳定排序）——没有这条，「并列第三名」这种题就没有
   唯一解。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Final

from jc.cover import cover_dims
from jc.jsontypes import JsonValue

#: 候选条数（固定 5）。定成 5 是为了让「两两比较」那一族天然配平：10 对无序对**两个方向
#: 都问**（见 `PAIRS`），一问一答正好互补，noul 的 True 占比按构造就是 50%。
CANDIDATES: Final[int] = 5

#: 查询词个数（case 维度）。2 个词时「沾上一个」已经是一半相关，4 个词时只沾一个更接近
#: 噪声——同一套名次口径在三种难度上都出现。4 是 `test_ood` 的值留出（见 workflow 模块）。
TERM_COUNTS: Final[tuple[int, ...]] = (2, 3, 4)

#: 相关候选条数（case 维度，均匀取 0~4）。取 0 是「一条都不相关」这一档：那是重排的真实
#: 结局（全跑题），而且它让 `count` 的选项表 0~4 每格都到得了。
RELEVANT: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)

#: 最相关候选落在哪一段（case 维度）。位置偏置是重排最该被考的东西，所以按段注入，
#: 不靠随机掷。
BESTS: Final[tuple[str, ...]] = ("first", "middle", "last")

#: 每一段包含哪些候选下标（0 起）。`middle` 只有一格，是因为 5 条候选按三段的自然切法是
#: 2 / 1 / 2。
BEST_THIRDS: Final[dict[str, tuple[int, ...]]] = {
    "first": (0, 1),
    "middle": (2,),
    "last": (3, 4),
}

#: 注入维度表（`cover_dims` 的入参）。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {
    "terms": TERM_COUNTS,
    "relevant": RELEVANT,
    "best": BESTS,
}

#: 无序对（i < j）。两两比较那一族把**每个无序对的两个方向都出**，所以 noul 的答案按构造
#: 就是一半 True、一半 False——不需要靠采样去凑平衡。
_ASC: Final[tuple[tuple[int, int], ...]] = tuple(
    (left, right) for left in range(CANDIDATES) for right in range(left + 1, CANDIDATES)
)
PAIRS: Final[tuple[tuple[int, int], ...]] = _ASC + tuple(
    (right, left) for left, right in _ASC
)


def coverage(text: str, terms: list[str] | tuple[str, ...]) -> int:
    """输入：一段候选原文、查询词表。输出：这段原文沾上了几个查询词。

    预期行为：大小写不敏感的**子串**判断（不是分词）——判据越简单，「代码算的」与「读者
    读出来的」越不容易打架；调用方保证词表里的词两两不互为子串（`render.check_draft`）。
    """
    lowered = text.lower()
    return sum(1 for term in terms if term.lower() in lowered)


@dataclass(frozen=True)
class RerankCase:
    """一条样本的注入特征：查询词数、相关候选条数、最相关候选落在哪一段。"""

    terms: int
    relevant: int
    best: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"terms": self.terms, "relevant": self.relevant, "best": self.best}

    def fingerprint(self) -> str:
        """输入：无。输出：case 指纹。预期行为：同一 case 任何进程都得到同一串。"""
        return "|".join(f"{key}={value}" for key, value in self.as_dict().items())

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.terms}词-{self.relevant}相关-{self.best}"


@dataclass(frozen=True)
class RerankPlan:
    """一条样本的名次计划：查询词数 + 每条候选该沾上几个词。

    字段：`terms` 查询词数；`best_pos` 最相关候选的下标（`relevant` 为 0 时是 None）；
    `overlaps` 逐条候选的覆盖率（与候选顺序一一对应）。文本在 `render` 阶段由后端写。
    """

    terms: int
    best_pos: int | None
    overlaps: tuple[int, ...]

    @property
    def relevant(self) -> int:
        """输入：无。输出：沾上至少一个查询词的候选条数。"""
        return sum(1 for value in self.overlaps if value > 0)


def plan_for(case: RerankCase, phrasing: int) -> RerankPlan:
    """输入：case、问法编号。输出：名次计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里
    都得到同一份计划；最相关候选落在哪一段由 `case.best` 注入，段内具体哪一格由指纹掷出。
    问法编号也进种子，是为了让同一个 (词数, 相关条数, 段) 的不同问法拿到不同的位置排布——
    否则这几种问法会渲染出结构完全相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    rng = random.Random(seed)
    if case.relevant == 0:
        return RerankPlan(terms=case.terms, best_pos=None, overlaps=(0,) * CANDIDATES)
    best_pos = rng.choice(BEST_THIRDS[case.best])
    others = [pos for pos in range(CANDIDATES) if pos != best_pos]
    partials = set(rng.sample(others, case.relevant - 1))
    overlaps = tuple(
        case.terms if pos == best_pos else (1 if pos in partials else 0)
        for pos in range(CANDIDATES)
    )
    return RerankPlan(terms=case.terms, best_pos=best_pos, overlaps=overlaps)


@dataclass(frozen=True)
class RerankLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`scores` 逐条覆盖率；`order` 名次从高到低（元素是候选下标）；`relevant` 相关条数；
    `top` 第一名（`relevant` 为 0 时是 None——这时「谁最相关」没有唯一解）。
    """

    scores: tuple[int, ...]
    order: tuple[int, ...]
    relevant: int
    top: int | None

    def rank_of(self, index: int) -> int:
        """输入：候选下标。输出：它的名次（1 起）。"""
        return self.order.index(index) + 1

    def above(self, left: int, right: int) -> bool:
        """输入：两条候选的下标。输出：左边那条是否排在右边那条前面。"""
        return self.rank_of(left) < self.rank_of(right)


def label_for(plan: RerankPlan) -> RerankLabels:
    """输入：名次计划。输出：全部金标签。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。排序口径就是 state 里
    印着的那两条（先比得分、同分比原有先后），所以读者能自己重算。
    """
    order = tuple(sorted(range(CANDIDATES), key=lambda index: (-plan.overlaps[index], index)))
    relevant = plan.relevant
    return RerankLabels(
        scores=plan.overlaps,
        order=order,
        relevant=relevant,
        top=order[0] if relevant > 0 else None,
    )


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[RerankCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`RerankCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [
        RerankCase(terms=int(row["terms"]), relevant=int(row["relevant"]), best=str(row["best"]))
        for row in rows
    ]
