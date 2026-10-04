"""评测与基准 workflow 的真值规格：一份评测记录 + 一张候选成绩表。

输入/输出：不可变数据类与纯函数；这里只有**键**（候选 / 字母 / 题号 / 集号 / 门槛 /
列序 / 句面），中英文案全在 `locales/`。
预期行为：答案只从渲染出来的那屏记录反推——某个候选在某题答没答对按「它给的答案字母与
参考答案字母是否完全相同」判，答对几道按答对的题数数，达标按「答对条数不低于门槛」判，
最好的候选按「答对最多、并列取列序靠前」定，落后幅度按「最好的那个减第 1 个」算。这些都是
已知的**纯函数**，构造侧与判别侧共用同一套，所以「表写错」与「渲染写歪」会变成同一类错误，
不会两边各错一半还互相印证。

覆盖的决策形状（社区用例地图：评测与基准，29 个第三方项目，含 `zevals` / `pi-codemode` /
`jev-pong` / `jev-align` / `Convex Decision Evals` / `decider` / `jev-mcp (arunav25)` /
`dsh-jev-verify` / `jev-scout`）：
- **第 1 个候选在第 n 题答对了吗**（`hit`，noul，逐题 5 道，总是出）；
- **第 1 个候选答对了几道**（`count`，score 6 档，总是出）；
- **第 1 个候选达标了吗**（`pass`，noul，总是出）；
- **第几个候选答对最多**（`best`，choice 3，总是出）；
- **最好的那个比第 1 个候选多答对几道**（`gap`，score 2 档，只在第 1 个候选不是最好时出）。

三十张成绩表是从（第 1 个候选的答对条数 k × 循环起点 s）= 6 × 5 张里**枚举**出来的：
三个候选的答对条数取 (k, k+2, k+4) 模 6，三个值两两不同，所以「谁最好」永远唯一。于是
`hit` 与 `pass` 恰好一半一半、`count` 六档各占六分之一、`best` 三档各占三分之一、`gap`
两档各占一半——全是构造的推论，不靠事后过滤（`test_spec.py` 把这五条写成断言）。六种排布
是三个候选的**全部列序**（位置偏差这一维），每个候选在每个列位上恰好出现两次，所以逐题、
逐列的分布也都被铺匀。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import permutations, product
from typing import Final, Protocol

from jc.knobs import require_pool

# ---------- 候选、字母与三条规则 ----------

#: 三个候选的键。列序由排布决定，每个候选在每个列位上恰好出现两次。
CANDIDATES: Final[tuple[str, ...]] = ("m0", "m1", "m2")
#: 四个选项字母。**符号不是文案**：中英共用同一套，所以它不住在 `locales/` 里，
#: 反解表也不参与「跨语言不重叠」那道检查（见 `parse` 的说明）。
LETTERS: Final[tuple[str, ...]] = ("A", "B", "C", "D")
#: 题数：一共五题，所以「答对几道」是 0 到 5 共**六档**——档数偶数才能在正中间切出 50/50。
TOTAL: Final[int] = 5
#: 达标门槛：答对条数不低于它算达标。写在抬头那一格，取值域也印在判据里。
BAR: Final[int] = 3
#: 门槛的取值域。表只用 `BAR`，但反解要认得出别的值，否则「门槛写错」会被当成认不出来。
BARS: Final[tuple[int, ...]] = (2, 3, 4)
#: 候选 i 的答对条数 = (k + 2i) 模 6。三个值两两不同，所以「谁最好」永远唯一。
OFFSETS: Final[tuple[int, ...]] = (0, 2, 4)
#: 候选 i 的答对集起点 = (s + i) 模 5。三个起点互不相同，逐题的答对率因此被铺匀。
STARTS: Final[tuple[int, ...]] = (0, 1, 2)
#: `gap` 一族的两个取值：最好的那个比第 1 个候选多答对几道。
GAP_VALUES: Final[tuple[int, ...]] = (2, 4)

# ---------- 三十张成绩表 ----------

#: 成绩表：`k<i>s<j>` = 第 1 个候选答对 i 道、循环起点 j。
TABLE_TABLE: Final[dict[str, tuple[int, int]]] = {
    f"k{k}s{s}": (k, s) for k in range(TOTAL + 1) for s in range(TOTAL)
}
#: 全部表名（顺序稳定）。
TABLES: Final[tuple[str, ...]] = tuple(TABLE_TABLE)
#: 「答对几道」的六档取值（档位就是道数本身）。
COUNTS: Final[tuple[int, ...]] = tuple(range(TOTAL + 1))

# ---------- 列序与句面 ----------

#: 六种列序名。取值就是三个候选的全部排列，顺序按 `permutations` 的字典序。
LAYOUTS: Final[tuple[str, ...]] = ("l0", "l1", "l2", "l3", "l4", "l5")
#: 列序名 → 列序（第 1 列到第 3 列依次是哪个候选）。
LAYOUT_PERMS: Final[dict[str, tuple[str, ...]]] = {
    f"l{index}": tuple(perm)
    for index, perm in enumerate(sorted(permutations(CANDIDATES)))
}
#: 两套句面：只改问法措辞，不改任何答案。
PHRASINGS: Final[tuple[int, ...]] = (0, 1)


def _require(field: str, value: object, pool: Sequence[JsonValue]) -> None:
    """输入：字段名、取值、取值域。输出：无。

    预期行为：认不出的键直接报错——静默落到某个默认分支会把「表改了、读法忘了改」
    变成一堆答案对不上的样本，而不是一条能定位的错误。
    """
    if value not in pool:
        raise KeyError(f"没有 {field}={value!r}（可用：{', '.join(map(str, pool))}）")


def table_of(name: str) -> tuple[int, int]:
    """输入：成绩表名。输出：(第 1 个候选的答对条数 k, 循环起点 s)。认不出直接报错。"""
    _require("table", name, TABLES)
    return TABLE_TABLE[name]


def layout_perm(layout: str) -> tuple[str, ...]:
    """输入：列序名。输出：列序（第 1 列到第 3 列依次是哪个候选）。认不出直接报错。"""
    _require("layout", layout, LAYOUTS)
    return LAYOUT_PERMS[layout]


def counts_of(table: tuple[int, int]) -> tuple[int, ...]:
    """输入：成绩表。输出：三个候选各自的答对条数（按 `CANDIDATES` 的固定序）。

    预期行为：三个值按构造两两不同（模 6 的错位量是 0 / 2 / 4，步长 2 恰好把 6 分成三份），
    所以「谁最好」永远唯一、`gap` 的两档也永远只落在 2 与 4 上。
    """
    k, _s = table
    return tuple((k + offset) % len(COUNTS) for offset in OFFSETS)


def correct_sets(table: tuple[int, int]) -> dict[str, frozenset[int]]:
    """输入：成绩表。输出：{候选键: 它答对的题下标集合}（题下标 0 到 4）。

    预期行为：候选 i 的答对集是从 (s + i) 起的「答对条数」个连续题下标（按 5 取模回绕）。
    条数为 0 与为 5 分别是空集与全集——不是特例，是同一段循环区间的两端。
    """
    _k, s = table
    return {
        cand: frozenset((s + start + step) % TOTAL for step in range(count))
        for cand, start, count in zip(CANDIDATES, STARTS, counts_of(table), strict=True)
    }


def letter_of(table: tuple[int, int], index: int) -> str:
    """输入：成绩表、题下标。输出：这一题的**参考答案字母**。

    预期行为：四个字母按题下标循环铺（A、B、C、D、A）。它是纯外观——「答对」是按集合判的，
    不靠这个字母算——但它必须是确定的纯函数，否则渲染与反解会各写一套。
    """
    del table
    return LETTERS[index % len(LETTERS)]


def shift_letter(letter: str, steps: int) -> str:
    """输入：字母、位移。输出：在四个字母里循环位移后的字母。认不出直接报错。"""
    _require("letter", letter, LETTERS)
    return LETTERS[(LETTERS.index(letter) + steps) % len(LETTERS)]


def answer_letter(table: tuple[int, int], cand: str, index: int) -> str:
    """输入：成绩表、候选键、题下标。输出：这个候选在这一题给的答案字母。

    预期行为：答对就给参考字母；答错就给「参考字母往后位移 1 + 候选序号」那个字母——
    三个候选的错答案因此两两不同、且都不等于参考，一眼看得出是三个不同的错。
    """
    _require("candidate", cand, CANDIDATES)
    if index in correct_sets(table)[cand]:
        return letter_of(table, index)
    return shift_letter(letter_of(table, index), 1 + CANDIDATES.index(cand))


# ---------- 记录的一行与整屏 ----------


@dataclass(frozen=True)
class BenchCell:
    """一格：候选键 + 它给的答案字母（列序已经由排布定好）。"""

    cand: str
    letter: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：两格（统计与调试用）。"""
        return {"candidate": self.cand, "letter": self.letter}

    def describe(self) -> str:
        """输入：无。输出：一行描述（报告与调试用）。"""
        return f"{self.cand}:{self.letter}"


@dataclass(frozen=True)
class BenchRow:
    """一行：参考答案字母 + 按列序排的三格。"""

    ref: str
    cells: tuple[BenchCell, ...]

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：四格（统计与调试用）。"""
        return {"ref": self.ref, "cells": [cell.as_dict() for cell in self.cells]}

    def describe(self) -> str:
        """输入：无。输出：一行描述（报告与调试用）。"""
        return "/".join([self.ref, *(cell.describe() for cell in self.cells)])


def rows_of(table: tuple[int, int], layout: str) -> tuple[BenchRow, ...]:
    """输入：成绩表、列序。输出：五题依次是哪一行（顺序稳定）。

    预期行为：行序就是题号（第 1 行是第 1 题，一共五行）；列序由**排布**定，所以同一张
    成绩表换个列序就是另一屏记录——「第 1 个候选」是谁因此随排布变。
    """
    perm = layout_perm(layout)
    return tuple(
        BenchRow(
            ref=letter_of(table, index),
            cells=tuple(BenchCell(cand, answer_letter(table, cand, index)) for cand in perm),
        )
        for index in range(TOTAL)
    )


# ---------- 只吃键的重算（判别器与审计走这条） ----------


class CellLike(Protocol):
    """判别器与审计眼里的一格：候选键 + 答案字母。"""

    cand: str
    letter: str


class RowLike(Protocol):
    """判别器与审计眼里的一行：参考字母 + 按列序排的三格。"""

    ref: str
    cells: Sequence[CellLike]


class HeadLike(Protocol):
    """判别器与审计眼里的抬头：门槛那一格。"""

    bar: int


def correct_of_cell(row: RowLike, column: int) -> bool:
    """输入：一行（键）、列号（1 起）。输出：这一列的候选在这一题答对了吗。

    预期行为：只比「这一格的字母与参考字母是否完全相同」——判别器把文本读回来重算时走的
    就是这条，所以「渲染把某一格写歪」会直接变成答案对不上。
    """
    if not 1 <= column <= len(CANDIDATES):
        raise ValueError(f"列号要在 1..{len(CANDIDATES)} 之间，收到 {column}")
    return row.cells[column - 1].letter == row.ref


def count_of_column(rows: Sequence[RowLike], column: int) -> int:
    """输入：五行（键）、列号（1 起）。输出：这一列的候选答对了几道。"""
    return sum(1 for row in rows if correct_of_cell(row, column))


def pass_of_column(rows: Sequence[RowLike], column: int, bar: int) -> bool:
    """输入：五行（键）、列号、门槛。输出：这一列的候选达标了吗。"""
    return count_of_column(rows, column) >= bar


def best_column_of_rows(rows: Sequence[RowLike]) -> int:
    """输入：五行（键）。输出：答对最多的那一列（1 起）。

    预期行为：并列取列序更靠前的那个（`max` 取第一个最大值）。三个候选的条数按构造两两
    不同，所以这条并列规则在本数据集里从不触发——但它是判据的一部分，得写着。
    """
    columns = range(1, len(CANDIDATES) + 1)
    return max(columns, key=lambda column: count_of_column(rows, column))


def gap_level_of_rows(rows: Sequence[RowLike]) -> int | None:
    """输入：五行（键）。输出：`gap` 的档位下标；第 1 个候选自己就是最好时给 None。

    预期行为：档位就是「最好的条数减第 1 列的条数」在 `GAP_VALUES` 里的下标。第 1 列已经
    是最好时没有「落后」可言，返回 None（调用方据此不出这一族）。
    """
    best = count_of_column(rows, best_column_of_rows(rows))
    first = count_of_column(rows, 1)
    if best == first:
        return None
    gap = best - first
    if gap not in GAP_VALUES:
        raise ValueError(f"落后幅度 {gap} 不在 {GAP_VALUES} 里（成绩表被改坏了）")
    return GAP_VALUES.index(gap)


# ---------- 素材、采样参数与真值规格 ----------

#: 训练侧见过的列序（`l5` 整个留出）与句面（第二套留出）。
SEEN_LAYOUTS: Final[frozenset[str]] = frozenset(LAYOUTS[:-1])
SEEN_PHRASINGS: Final[frozenset[int]] = frozenset({PHRASINGS[0]})


@dataclass(frozen=True)
class BenchCase:
    """一个 case：成绩表 + 列序 + 句面。"""

    table: str
    layout: str
    phrasing: int

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：三个维度任一个认不出就当场报人话。"""
        table_of(self.table)
        layout_perm(self.layout)
        _require("phrasing", self.phrasing, PHRASINGS)

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这条素材里训练侧没见过的维度名（排序后）。"""
        out: list[str] = []
        if self.layout not in SEEN_LAYOUTS:
            out.append("layout")
        if self.phrasing not in SEEN_PHRASINGS:
            out.append("phrasing")
        return tuple(sorted(out))

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：{维度名: 取值}（三个维度全摊平）。"""
        return {"table": self.table, "layout": self.layout, "phrasing": self.phrasing}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.table}/{self.layout}/p{self.phrasing}"


def material_of(case: BenchCase) -> str:
    """输入：case。输出：素材指纹（切分按它哈希分桶，同一屏记录整组不跨切分）。"""
    return f"{case.table}|{case.layout}|{case.phrasing}"


@dataclass(frozen=True)
class BenchConfig:
    """一个切分的采样参数：三张取值表 + 是否只留 OOD。"""

    tables: tuple[str, ...] = TABLES
    layouts: tuple[str, ...] = LAYOUTS
    phrasings: tuple[int, ...] = PHRASINGS
    split: str = "train"
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：任一取值表为空就报人话（点名是哪个字段）。"""
        require_pool("tables", self.tables)
        require_pool("layouts", self.layouts)
        require_pool("phrasings", self.phrasings)


def build_cases(config: BenchConfig) -> list[BenchCase]:
    """输入：采样参数。输出：case 矩阵（顺序稳定）。

    预期行为：三张表取全组合（**不做成立性过滤**——三个维度两两不矛盾）；`ood_only` 时
    只留「至少含一个训练侧没见过的取值」的组合，所以 OOD 与同分布侧按构造就不重叠。
    """
    out: list[BenchCase] = []
    for table, layout, phrasing in product(config.tables, config.layouts, config.phrasings):
        case = BenchCase(table, layout, phrasing)
        if config.ood_only and not case.unseen_dims():
            continue
        out.append(case)
    return out


# ---------- 真值规格 ----------

REQUEST_PREFIX: Final[str] = "E-"


@dataclass(frozen=True)
class BenchSpec:
    """一份评测记录的完整真值：case + 由规则算出来的一切。"""

    case: BenchCase
    split: str
    seq: int
    phrasing: int
    request_id: str
    bar: int
    rows: tuple[BenchRow, ...]
    counts: tuple[int, ...]
    best_column: int
    gap_level: int | None


def request_id_of(case: BenchCase) -> str:
    """输入：case。输出：请求号（形如 `E-48213`）。

    预期行为：由素材指纹的 sha256 派生——确定性，而且**不泄露切分**（用序号或切分名拼
    请求号会让同一屏记录在不同切分里拿到不同请求号，等于把切分信息写进 state）。
    """
    digest = hashlib.sha256(material_of(case).encode("utf-8")).hexdigest()
    return f"{REQUEST_PREFIX}{10000 + int(digest[:6], 16) % 90000}"


def sample_spec(case: BenchCase, split: str, seq: int, phrasing: int, rng: random.Random) -> BenchSpec:
    """输入：case、切分、序号、句面、RNG。输出：真值规格。

    预期行为：成绩表、五行与答案都由 case 决定（**不掷骰子**）——这个 workflow 的每个
    自由度都已经在 case 里，`rng` 只为与其它 workflow 的签名一致，收下就丢。
    """
    del rng
    table = table_of(case.table)
    rows = rows_of(table, case.layout)
    return BenchSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        request_id=request_id_of(case),
        bar=BAR,
        rows=rows,
        counts=counts_of(table),
        best_column=best_column_of_rows(rows),
        gap_level=gap_level_of_rows(rows),
    )
