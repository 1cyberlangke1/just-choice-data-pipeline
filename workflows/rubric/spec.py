"""打分与评分 workflow 的真值规格：一张评分单 + 三张判据表。

输入/输出：不可变数据类与纯函数；这里只有**键**（判据 / 权重 / 要求 / 观测 / 档位 /
处置 / 排布 / 松紧），中英文案全在 `locales/`。
预期行为：答案只从渲染出来的那张评分单反推——达标与否按「观测 ≥ 要求」比、总分按达标
判据的权重加、档位按两条门槛切、处置按「档位 × 有没有踩红线」查表、最严重的一条按
「没达标的判据里权重最大的那条」定。这些都是已知的**纯函数**，构造侧与判别侧共用同一套，
所以「表写错」与「渲染写歪」会变成同一类错误，不会两边各错一半还互相印证。

覆盖的决策形状（社区用例地图：打分与评分，17 个第三方项目，含 `slop-grader` / `jev-seo` /
`jevmeter` / `dsh-jev` / `typesafe-jev`）：
- **每一条达没达标**（`met`，noul，逐条 6 道，总是出）：观测有没有到要求；
- **总分落哪一档**（`band`，score 3 档，总是出）：达标判据的权重之和落在低 / 中 / 高哪一档；
- **该怎么处置**（`gate`，choice 3，总是出）：按档位与红线查二维表拿一个处置；
- **最严重的是哪一条**（`top`，choice 6，只在唯一时）：没达标的判据里权重最大的那一条。

评分单固定六条判据，权重 3 / 3 / 2 / 2 / 1 / 1、要求 2 / 3 / 2 / 2 / 1 / 2。十八张
「恰好三条达标」的判据表是从 20 种三子集里**去掉两种互不相交的**（`{0,2,4}` 与
`{1,3,5}`）得来的：去掉互不相交的两张之后，每条判据恰好达标 9 次——`met` 的 50/50、
`band` 的三档各六张、`gate` 的三个处置各六张都是这么**构造**出来的，不靠事后过滤。

红线固定是权重 2 的 `original` 那条：权重 3 的两条在各档里的达标次数是 5 / 3 / 1、
权重 1 的两条是 1 / 3 / 5，凑不出「三档各 3 张」，只有权重 2 的两条是 3 / 3 / 3。
松紧两档只改**观测那一格**：`tight` 是刚好压线（达标 = 要求、没达标 = 要求 − 1），
`roomy` 是达标的高出一档（要求 + 1）、没达标的还是差一档——于是「观测等于要求」这条
捷径不再等价于达标，考生必须真的比大小。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any, Final, Protocol

from jc.knobs import require_pool

# ---------- 评分单的六条判据 ----------

#: 判据顺序。**权重 2 的那两条相邻**（顺序本身不影响任何答案，只影响渲染出来的行序）。
CRITERIA: Final[tuple[str, ...]] = (
    "example",
    "source",
    "original",
    "sections",
    "typos",
    "length",
)
#: 每条判据的权重（三档，六条判据各两条）。
WEIGHT_OF: Final[dict[str, str]] = {
    "example": "w3",
    "source": "w3",
    "original": "w2",
    "sections": "w2",
    "typos": "w1",
    "length": "w1",
}
#: 每条判据的门槛（三档，六条判据是 1 / 4 / 1 条）。
NEED_OF: Final[dict[str, str]] = {
    "example": "need2",
    "source": "need3",
    "original": "need2",
    "sections": "need2",
    "typos": "need1",
    "length": "need2",
}
#: 踩红线的那条判据：权重 2（权重 3 的两条在各档里的达标次数是 5 / 3 / 1，凑不出均分）。
REDLINE: Final[str] = "original"

# ---------- 三张判据表的取值 ----------

#: 权重三档。
WEIGHTS: Final[tuple[str, ...]] = ("w1", "w2", "w3")
#: 门槛三档。
NEEDS: Final[tuple[str, ...]] = ("need1", "need2", "need3")
#: 观测五档（0 到 4）。
COUNTS: Final[tuple[str, ...]] = ("n0", "n1", "n2", "n3", "n4")
#: 档位三档（低 / 中 / 高）。
BANDS: Final[tuple[str, ...]] = ("b1", "b2", "b3")
#: 处置三档。
GATES: Final[tuple[str, ...]] = ("pass", "fix", "reject")

#: 权重 / 门槛 / 观测三张表的数值（比大小、求和都按它）。
WEIGHT_VALUE: Final[dict[str, int]] = {"w1": 1, "w2": 2, "w3": 3}
NEED_VALUE: Final[dict[str, int]] = {"need1": 1, "need2": 2, "need3": 3}
COUNT_VALUE: Final[dict[str, int]] = {"n0": 0, "n1": 1, "n2": 2, "n3": 3, "n4": 4}
#: 档位的下界：总分 ≥ 下界就至少是这个档。
BAND_MIN: Final[dict[str, int]] = {"b3": 7, "b2": 6}
#: 处置表：档位 × 有没有踩红线 → 处置（六个格子，每个处置占两格）。
GATE_TABLE: Final[dict[tuple[str, bool], str]] = {
    ("b3", False): "pass",
    ("b2", False): "pass",
    ("b3", True): "fix",
    ("b1", False): "fix",
    ("b2", True): "reject",
    ("b1", True): "reject",
}

# ---------- 排布与松紧 ----------

#: 六种排布：`l<i>` 把判据 j 摆到第 (j + i) mod 6 个位置上——每条判据在每个位置上恰好一次。
LAYOUTS: Final[tuple[str, ...]] = ("l0", "l1", "l2", "l3", "l4", "l5")
#: 松紧两档：只改观测那一格（压线 / 高出一档）。
SLACKS: Final[tuple[str, ...]] = ("tight", "roomy")
#: 句面（只影响措辞，不影响答案）。
PHRASINGS: Final[tuple[int, ...]] = (0, 1)


# ---------- 规则函数（构造与判别共用） ----------


def _require(field: str, value: str, pool: tuple[str, ...]) -> None:
    """输入：字段名、取值、取值表。输出：无。

    预期行为：认不出的键直接报错——静默落到某个默认分支会把「表改了、读法忘了改」
    变成一堆答案对不上的样本，而不是一条能定位的错误。
    """
    if value not in pool:
        raise KeyError(f"没有 {field}={value!r}（可用：{', '.join(pool)}）")


class RowLike(Protocol):
    """评分单的一行（`spec.RubricRow` 与 `parse.ParsedRow` 都满足它）。

    为什么要这个协议：规则函数（达标、总分、档位、处置、最严重的一条）既要能吃**规格里**
    算好的行，也要能吃**从文本反解回来**的行——判别器与审计走的正是后一条路，两边共用
    同一套规则才不会各错一半还互相印证。
    """

    criterion: str
    weight: str
    need: str
    count: str


def met_of(need: str, count: str) -> bool:
    """输入：门槛键、观测键。输出：这条判据达没达标。

    预期行为：**观测 ≥ 要求**才算达标。取的是评分单里读回来的那两格（不是从判据键
    反查表），所以判别器拿文本重算时，「渲染把观测格或要求格写歪」会直接变成答案对不上。
    """
    _require("need", need, NEEDS)
    _require("count", count, COUNTS)
    return COUNT_VALUE[count] >= NEED_VALUE[need]


def count_of(need: str, met: bool, slack: str) -> str:
    """输入：门槛键、达没达标、松紧。输出：观测那一格的键。

    预期行为：`tight` 达标就刚好压线（观测 = 要求）、没达标就差一档（要求 − 1）；
    `roomy` 达标高出一档（要求 + 1）、没达标还是差一档。两档都不会让「观测 = 要求」
    等价于达标，所以「看等号」这条捷径是假的，考生必须真的比大小。
    """
    _require("need", need, NEEDS)
    _require("slack", slack, SLACKS)
    step = 1 if slack == "roomy" else 0
    value = NEED_VALUE[need] + step if met else NEED_VALUE[need] - 1
    if not 0 <= value <= len(COUNTS) - 1:
        raise ValueError(f"观测算出界了：{need} / met={met} / {slack} → {value}")
    return f"n{value}"


def weight_of(criterion: str) -> str:
    """输入：判据键。输出：它的权重键。预期行为：认不出直接报错。"""
    _require("criterion", criterion, CRITERIA)
    return WEIGHT_OF[criterion]


def need_of(criterion: str) -> str:
    """输入：判据键。输出：它的门槛键。预期行为：认不出直接报错。"""
    _require("criterion", criterion, CRITERIA)
    return NEED_OF[criterion]


def is_redline(criterion: str, met: bool) -> bool:
    """输入：判据键、达没达标。输出：这一条算不算踩红线。

    预期行为：只有 `original` **没达标**才算踩红线——红线是「不许破的那条」，破了才有事。
    """
    _require("criterion", criterion, CRITERIA)
    return criterion == REDLINE and not met


def total_of(mets: Iterable[str]) -> int:
    """输入：达标的判据键。输出：总分（这些判据的权重之和）。"""
    total = 0
    for criterion in mets:
        _require("criterion", criterion, CRITERIA)
        total += WEIGHT_VALUE[WEIGHT_OF[criterion]]
    return total


def band_of(total: int) -> str:
    """输入：总分。输出：档位键。

    预期行为：从高到低比门槛（≥ 7 高档、= 6 中档）；两条都不够就是低档——低档是地板，
    不设下界，总分再低也只会落在低档。
    """
    if total >= BAND_MIN["b3"]:
        return "b3"
    if total >= BAND_MIN["b2"]:
        return "b2"
    return "b1"


def gate_of(band: str, redline: bool) -> str:
    """输入：档位键、有没有踩红线。输出：处置键。预期行为：查处置表；认不出报人话。"""
    _require("band", band, BANDS)
    if not isinstance(redline, bool):
        raise TypeError(f"红线必须是布尔，收到 {type(redline).__name__}")
    return GATE_TABLE[(band, redline)]


# ---------- 排布 ----------


def layout_of(layout: str) -> int:
    """输入：排布名。输出：位移量。预期行为：认不出直接报错。"""
    _require("layout", layout, LAYOUTS)
    return LAYOUTS.index(layout)


def criterion_at(layout: str, slot: int) -> str:
    """输入：排布名、位置（从 1 起）。输出：摆在那里的判据。

    预期行为：环形移位——每条判据在六个位置上各出现一次，`top` 的答案位置分布因此均匀。
    """
    shift = layout_of(layout)
    size = len(CRITERIA)
    if not 1 <= slot <= size:
        raise ValueError(f"位置要在 1..{size} 之间，收到 {slot}")
    return CRITERIA[(slot - 1 - shift) % size]


def slot_of(layout: str, criterion: str) -> int:
    """输入：排布名、判据键。输出：它摆在第几个位置（从 1 起）。"""
    _require("criterion", criterion, CRITERIA)
    size = len(CRITERIA)
    return (CRITERIA.index(criterion) + layout_of(layout)) % size + 1


# ---------- 判据表（18 张「恰好三条达标」的三子集） ----------

#: 从 20 种三子集里去掉的两张：**互不相交**，于是每条判据都恰好被去掉一次。
DROPPED: Final[tuple[frozenset[int], ...]] = (
    frozenset({0, 2, 4}),
    frozenset({1, 3, 5}),
)


def _triple_total(triple: tuple[int, ...]) -> int:
    """输入：判据下标的三元组。输出：这三条都达标时的总分。"""
    return total_of(CRITERIA[index] for index in triple)


def _sheet_order(triple: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    """输入：判据下标的三元组。输出：排序键（档位序、下标元组）。

    预期行为：**按档位分组**——同一档的判据表连在一起，报告与 manifest 里一眼看得出
    每张表落在哪一档；顺序稳定（不靠集合迭代顺序）。
    """
    return (BANDS.index(band_of(_triple_total(triple))), triple)


_SHEET_TRIPLES: Final[tuple[tuple[int, ...], ...]] = tuple(
    sorted(
        (
            triple
            for triple in combinations(range(len(CRITERIA)), 3)
            if set(triple) not in DROPPED
        ),
        key=_sheet_order,
    )
)

#: 判据表的名字（`s0`..`s17`）→ 达标的判据下标集合。
MET_INDEX: Final[dict[str, frozenset[int]]] = {
    f"s{index}": frozenset(triple) for index, triple in enumerate(_SHEET_TRIPLES)
}
#: 判据表的全集（顺序稳定：按档位分组）。
SHEETS: Final[tuple[str, ...]] = tuple(MET_INDEX)


# ---------- 评分单的一行与整张单 ----------


@dataclass(frozen=True)
class RubricRow:
    """评分单的一行：判据 / 权重 / 要求 / 观测（四样都是键）。"""

    criterion: str
    weight: str
    need: str
    count: str

    def met(self) -> bool:
        """输入：无。输出：这一条达没达标。预期行为：只吃要求与观测那两格。"""
        return met_of(self.need, self.count)

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：四格（统计与调试用）。"""
        return {
            "criterion": self.criterion,
            "weight": self.weight,
            "need": self.need,
            "count": self.count,
        }

    def describe(self) -> str:
        """输入：无。输出：一行描述（报告与调试用）。"""
        return f"{self.criterion}/{self.weight}/{self.need}/{self.count}"


def rows_of(sheet: str, layout: str, slack: str) -> tuple[RubricRow, ...]:
    """输入：判据表名、排布、松紧。输出：位置 1..6 上依次是哪一行（顺序稳定）。

    预期行为：达标的那三条由**判据表**定死，观测那一格再由松紧推——所以「同一张表换个
    松紧」不会换答案，只换观测那几格（两条样本的 state 因此不同）。
    """
    _require("sheet", sheet, SHEETS)
    met_index = MET_INDEX[sheet]
    return tuple(
        RubricRow(
            criterion,
            weight_of(criterion),
            need_of(criterion),
            count_of(need_of(criterion), CRITERIA.index(criterion) in met_index, slack),
        )
        for criterion in (criterion_at(layout, slot) for slot in range(1, len(CRITERIA) + 1))
    )


def met_criteria(rows: Sequence[RowLike]) -> tuple[str, ...]:
    """输入：六行评分单（规格里的或反解回来的都行）。输出：达标的判据键（按行序）。

    预期行为：只吃**要求与观测那两格**——判别器拿从文本读回来的两格重算，
    「渲染把这两格写歪」会直接变成答案对不上。
    """
    return tuple(row.criterion for row in rows if met_of(row.need, row.count))


def top_slot(rows: Sequence[RowLike]) -> int | None:
    """输入：六行评分单。输出：最严重的那一行的位置（不唯一给 None）。

    预期行为：最严重 = **没达标的判据里权重最大的那条**；并列时「最严重的是哪一条」
    没有唯一解，这种表不出 `top` 题（出了就是坏数据，所以它是两阶段依赖）。
    """
    unmet = [
        (index, row)
        for index, row in enumerate(rows, start=1)
        if not met_of(row.need, row.count)
    ]
    if not unmet:
        return None
    best = max(WEIGHT_VALUE[row.weight] for _, row in unmet)
    tops = [index for index, row in unmet if WEIGHT_VALUE[row.weight] == best]
    return tops[0] if len(tops) == 1 else None


# ---------- case ----------


@dataclass(frozen=True)
class RubricCase:
    """一份评分单的全部自由度（四维）。"""

    sheet: str
    layout: str
    slack: str
    phrasing: int

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：任一维取值不合法直接报错。"""
        _require("sheet", self.sheet, SHEETS)
        layout_of(self.layout)
        _require("slack", self.slack, SLACKS)
        if self.phrasing not in PHRASINGS:
            raise KeyError(f"没有句面 {self.phrasing!r}（可用：{PHRASINGS}）")

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：四维取值（统计与均衡报告用）。"""
        return {
            "sheet": self.sheet,
            "layout": self.layout,
            "slack": self.slack,
            "phrasing": self.phrasing,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述。预期行为：与素材指纹同形（报告与文件名用）。"""
        return material_of(self)

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这一条里训练侧没见过的维度名（排序后）。

        预期行为：只认两个留出维——排布 `l5`、第二套句面。判据表与松紧在训练侧全开。
        """
        out: list[str] = []
        if self.layout not in SEEN_LAYOUTS:
            out.append("layout")
        if self.phrasing not in SEEN_PHRASINGS:
            out.append("phrasing")
        return tuple(sorted(out))


# ---------- 值留出（训练侧见过哪些取值） ----------

#: 训练侧见过的排布：最后一种整个留出。
SEEN_LAYOUTS: Final[frozenset[str]] = frozenset(LAYOUTS[:-1])
#: 训练侧见过的句面：第二套句面整个留出。它**不影响任何答案**，所以留出它不会把哪一族
#: 的分布压歪。
SEEN_PHRASINGS: Final[frozenset[int]] = frozenset({PHRASINGS[0]})


def material_of(case: RubricCase) -> str:
    """输入：case。输出：素材指纹（切分按它哈希分桶，同一张评分单整组不跨切分）。

    预期行为：四维全进指纹，句面也进——同一张单的两套问法是两条不同的 state，
    OOD 里才不会出现重复素材。
    """
    return f"{case.sheet}|{case.layout}|{case.slack}|{case.phrasing}"


# ---------- 采样参数 ----------


@dataclass(frozen=True)
class RubricConfig:
    """一个切分的采样参数：四张取值表 + 是否只留 OOD。"""

    sheets: tuple[str, ...] = SHEETS
    layouts: tuple[str, ...] = LAYOUTS
    slacks: tuple[str, ...] = SLACKS
    phrasings: tuple[int, ...] = PHRASINGS
    split: str = "train"
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：任一取值表为空就报人话（点名是哪个字段）。"""
        require_pool("sheets", self.sheets)
        require_pool("layouts", self.layouts)
        require_pool("slacks", self.slacks)
        require_pool("phrasings", self.phrasings)


def build_cases(config: RubricConfig) -> list[RubricCase]:
    """输入：采样参数。输出：case 矩阵（顺序稳定）。

    预期行为：四张表取全组合（**不做成立性过滤**——四个维度两两不矛盾）；`ood_only` 时
    只留「至少含一个训练侧没见过的取值」的组合，所以 OOD 与同分布侧按构造就不重叠。
    """
    out: list[RubricCase] = []
    for sheet, layout, slack, phrasing in product(
        config.sheets, config.layouts, config.slacks, config.phrasings
    ):
        case = RubricCase(sheet, layout, slack, phrasing)
        if config.ood_only and not case.unseen_dims():
            continue
        out.append(case)
    return out


# ---------- 真值规格 ----------

REQUEST_PREFIX: Final[str] = "G-"


@dataclass(frozen=True)
class RubricSpec:
    """一份评分单的完整真值：case + 由规则算出来的一切。"""

    case: RubricCase
    split: str
    seq: int
    phrasing: int
    request_id: str
    rows: tuple[RubricRow, ...]
    met: tuple[str, ...]
    total: int
    band: str
    redline: bool
    gate: str
    top_slot: int | None


def request_id_of(case: RubricCase) -> str:
    """输入：case。输出：请求号（形如 `G-48213`）。

    预期行为：由素材指纹的 sha256 派生——确定性，而且**不泄露切分**（用序号或切分名拼
    请求号会让同一张单在不同切分里拿到不同请求号，等于把切分信息写进 state）。
    """
    digest = hashlib.sha256(material_of(case).encode("utf-8")).hexdigest()
    return f"{REQUEST_PREFIX}{10000 + int(digest[:6], 16) % 90000}"


def sample_spec(case: RubricCase, split: str, seq: int, phrasing: int, rng: Any) -> RubricSpec:
    """输入：case、切分、序号、句面、RNG。输出：真值规格。

    预期行为：评分单、达标结果与三张判据表都由 case 决定（**不掷骰子**）——这个 workflow
    的每个自由度都已经在 case 里，`rng` 只为与其它 workflow 的签名一致，收下就丢。
    """
    del rng
    rows = rows_of(case.sheet, case.layout, case.slack)
    met = met_criteria(rows)
    total = total_of(met)
    band = band_of(total)
    redline = is_redline(REDLINE, REDLINE in met)
    return RubricSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        request_id=request_id_of(case),
        rows=rows,
        met=met,
        total=total,
        band=band,
        redline=redline,
        gate=gate_of(band, redline),
        top_slot=top_slot(rows),
    )


def spec_to_dict(spec: RubricSpec) -> dict[str, Any]:
    """输入：规格。输出：可 JSON 序列化的摘要（写进 manifest 与调试，不进样本行）。"""
    return {
        "case": spec.case.describe(),
        "unseen_dims": list(spec.case.unseen_dims()),
        "request_id": spec.request_id,
        "met": list(spec.met),
        "total": spec.total,
        "band": spec.band,
        "redline": spec.redline,
        "gate": spec.gate,
        "top_slot": spec.top_slot,
        "rows": [row.describe() for row in spec.rows],
    }
