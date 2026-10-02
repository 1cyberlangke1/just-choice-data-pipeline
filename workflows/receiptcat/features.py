"""报销收据归类（receiptcat）的特征层：归类表、单笔上限、审核档位与诱饵。

社区依据：`formanator`（★98，Rust）——提交 Forma 福利报销的 CLI / MCP 服务，用 Jev 把
**收据条目**分类到相应的报销类别。它的形状是「一条收据文本进来 → 判它该报哪一类」，与本
仓库的「特征注入式」（PLAN 3.2）天然对齐：

- **归类表、单笔上限表、审核档位表是结构化事实**，由代码渲染进 state——让模型去写只会写歪；
- 后端只写**一条收据行**（真实收据上的商户名与条目是长尾的，模板拼出来的脱离现实分布）；
- **命中哪几类、金额落哪一档由代码按字面算**（本模块的关键词匹配与金额比较），所以标签是
  注入特征经纯函数算出来的（`label_for`），模型碰不到；写出来的收据行必须兑现
  这份计划（`render.check_draft` 在渲染期查、`judges.HygieneJudge` / `FrameJudge` 再各查一遍）。

**为什么判据是「关键词表 + 字面包含」而不是语义**：这一族的官方形态就是「这条条目该报哪一
类」，而归类口径（哪些说法算哪一类）是**政策**，不是联想——政策表印在 state 里，按字面判。
于是「命中哪几类」是**算**出来的，模型只负责写一条读得通的条目（范式 §9.13：写不出强校验
就别硬编一个看起来像校验的东西）。

**诱饵（`decoy=near`）是这一族最要紧的设计**：正文里必须出现一个「明显像某一类条目、但不
在归类表里」的说法（如「加班餐」对表里的「工作餐」）。按字面算它**不算**命中，可只要按语义
联想就会把它当成命中——这正是「见关键词就归类」的捷径要压掉的东西，也是 `test_ood` 的留出
取值（同分布侧一次都见不到诱饵）。

本模块有**两条独立路径**：`label_for` 从计划推标签（构造侧），`observed_labels` 从文本重推
标签（审计侧，走同一份关键词表）。两者对不上就说明渲染没兑现计划。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jc.cover import Fingerprinted, cover_dims
from jc.jsontypes import JsonValue

#: 四个报销类别的**结构键**（类别名、关键词、诱饵词全在 `locales/`，这里只有编号）。
CATEGORY_IDS: Final[tuple[str, ...]] = ("C1", "C2", "C3", "C4")

#: 金额档位（结构键）：未超上限 / 超上限但不到两倍 / 达到两倍以上。
BANDS: Final[tuple[str, ...]] = ("under", "mild", "heavy")

#: 审核档位（结构键，文案在 `locales/`）：自动通过 / 补票据 / 主管复核 / 财务复核。
TIERS: Final[tuple[str, ...]] = ("auto", "receipt", "manager", "finance")

#: 单笔上限（**分**，整数）：金额一律用分比较——浮点在 `0.1 + 0.2` 这种地方会翻车。
CAPS_CENTS: Final[dict[str, int]] = {"C1": 20000, "C2": 15000, "C3": 50000, "C4": 200000}

#: 每个 (类别, 档位) 对应的金额字符串。**逐条手写而不是按公式算**：1.5 倍上限那种算出来的
#: 金额都是整数，看着不像真收据。`tests/workflows/receiptcat/test_features.py` 把「每条金额
#: 真的落在它那一档」钉在数据上，手写不会写歪。
AMOUNTS: Final[dict[tuple[str, str], str]] = {
    ("C1", "under"): "168.00",
    ("C1", "mild"): "254.00",
    ("C1", "heavy"): "466.00",
    ("C2", "under"): "96.50",
    ("C2", "mild"): "178.00",
    ("C2", "heavy"): "402.60",
    ("C3", "under"): "379.90",
    ("C3", "mild"): "640.00",
    ("C3", "heavy"): "1188.00",
    ("C4", "under"): "1499.00",
    ("C4", "mild"): "2380.00",
    ("C4", "heavy"): "5160.00",
}

#: 全部金额字符串（顺序稳定）：审计按它在正文里找金额。
AMOUNT_VALUES: Final[tuple[str, ...]] = tuple(
    AMOUNTS[(cid, band)] for cid in CATEGORY_IDS for band in BANDS
)

#: 需要走**财务复核**的那一类（上限最高、单笔金额也最大）：档位表里点名了它。
SOFTWARE_CATEGORY: Final[str] = "C4"

#: 诱饵（结构键）：不埋 / 埋一个「像某一类、但不在归类表里」的说法。
DECOY_NONE: Final[str] = "none"
DECOY_NEAR: Final[str] = "near"
DECOYS: Final[tuple[str, ...]] = (DECOY_NONE, DECOY_NEAR)

#: 正文长度上下限。太短读不出「像不像一条正经收据行」，太长就只是灌水。
MIN_CONTENT_CHARS: Final[int] = 12
MAX_CONTENT_CHARS: Final[int] = 120

#: 采样维度表：类别 × 金额档位 × 诱饵 = 4 × 3 × 2 = 24 组。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {
    "category": CATEGORY_IDS,
    "band": BANDS,
    "decoy": DECOYS,
}


def cents_of(amount: str) -> int | None:
    """输入：金额字符串（形如 `168.00`）。输出：分；不是「整数 + 两位小数」就给 None。

    预期行为：只认 `AMOUNTS` 里那种形态——收据上的金额必须有两位小数，一位小数或没有小数
    点的写法不算（判据表与提示词都是这么写的）。
    """
    whole, dot, frac = amount.partition(".")
    if dot != "." or len(frac) != 2 or not whole.isdigit() or not frac.isdigit():
        return None
    return int(whole) * 100 + int(frac)


def amounts_in(text: str) -> tuple[str, ...]:
    """输入：收据行正文。输出：正文里**原样出现**的注入金额（按 `AMOUNT_VALUES` 顺序）。

    预期行为：只认那 12 个注入金额本身，不认随便一个「带两位小数的数字」——档位是拿注入金额
    与上限比出来的，正文里另编一个金额推不出计划。
    """
    return tuple(value for value in AMOUNT_VALUES if value in text)


def band_of(amount_cents: int, cap_cents: int) -> str:
    """输入：金额（分）、该类单笔上限（分）。输出：档位键。

    预期行为：分界是 `[上限, 两倍上限)`——「恰好等于上限」算超限（`mild`），与档位表里
    「未超上限 / 超过上限但不到两倍 / 达到两倍以上」三句话逐字对应。
    """
    if amount_cents < cap_cents:
        return "under"
    if amount_cents < 2 * cap_cents:
        return "mild"
    return "heavy"


def tier_of(band: str, category: str) -> str:
    """输入：金额档位、类别编号。输出：审核档位键。

    预期行为：未超限走自动通过、超限不到两倍走补票据、两倍以上走主管复核——**除了软件订阅**
    （上限最高、单笔金额也最大），它两倍以上要走财务复核。这一格交互让「档位」不再是「档位」
    的同义词（否则 `tier` 那一族与 `over` 共享同一个自由度，白出一族题）。
    """
    if band == "under":
        return "auto"
    if band == "mild":
        return "receipt"
    return "finance" if category == SOFTWARE_CATEGORY else "manager"


def _entries(loc: dict[str, JsonValue], table: str) -> dict[str, tuple[str, ...]]:
    """输入：语言字典、表名（`keyword` / `decoy`）。输出：{类别编号: 词元组}。

    预期行为：四个类别一个都不能少、词表不能为空；缺了就当场报错并点名——半张表会让
    「命中哪几类」算错，而算错的标签**看起来完全正常**（这正是 §9.32 那类静默错误）。
    """
    node = loc.get(table)
    if not isinstance(node, dict):
        raise ValueError(f"语言文件里缺 [{table}] 表")
    out: dict[str, tuple[str, ...]] = {}
    for cid in CATEGORY_IDS:
        entry = node.get(cid)
        items = entry.get("items") if isinstance(entry, dict) else None
        if not isinstance(items, list) or not items:
            raise ValueError(f"[{table}.{cid}] 缺 items，或者 items 是空的")
        out[cid] = tuple(str(item) for item in items)
    return out


def keyword_table(loc: dict[str, JsonValue]) -> dict[str, tuple[str, ...]]:
    """输入：语言字典。输出：{类别编号: 归类关键词元组}。预期行为：缺项当场报错。"""
    return _entries(loc, "keyword")


def decoy_table(loc: dict[str, JsonValue]) -> dict[str, tuple[str, ...]]:
    """输入：语言字典。输出：{类别编号: 诱饵词元组}。预期行为：缺项当场报错。"""
    return _entries(loc, "decoy")


def categories_hit(text: str, keywords: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """输入：收据行正文、关键词表。输出：按 `CATEGORY_IDS` 顺序命中的类别编号。

    预期行为：**字面包含**即命中，不做任何语义联想；一张表里的词互为子串时结果会含混，
    所以 `tests/workflows/receiptcat/test_features.py` 把「词表两两互不包含」钉住。
    """
    return tuple(cid for cid in CATEGORY_IDS if any(word in text for word in keywords[cid]))


def decoy_hit(text: str, decoys: dict[str, tuple[str, ...]]) -> bool:
    """输入：收据行正文、诱饵表。输出：正文里有没有出现诱饵词。

    预期行为：诱饵词**不是**归类关键词，所以它命中不了 `categories_hit`——「按字面算不算
    命中」与「看着像不像」在这里被拆成两件事，正是这一族要考的地方。
    """
    return any(word in text for cid in CATEGORY_IDS for word in decoys[cid])


@dataclass(frozen=True)
class ReceiptCase(Fingerprinted):
    """一条样本的注入特征：该报哪一类 + 金额落哪一档 + 要不要埋诱饵。"""

    category: str
    band: str
    decoy: str

    @property
    def amount(self) -> str:
        """输入：无。输出：这一组的注入金额字符串。"""
        return AMOUNTS[(self.category, self.band)]

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"category": self.category, "band": self.band, "decoy": self.decoy}

    @property
    def distractors(self) -> tuple[str, ...]:
        """输入：无。输出：归类题里那三个干扰类别（按 `CATEGORY_IDS` 顺序）。

        预期行为：就是**另外三类**——四个选项里三个是干扰项，干扰集恒定，所以这里没有随机
        成分，也不必按「邻近」挑（四类里没有更近的）。想加区分度只能靠类别本身，靠抽签是
        假的：抽样池只有三个元素，抽三次必然把它们全抽走。
        """
        return tuple(cid for cid in CATEGORY_IDS if cid != self.category)

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.category}-{self.band}-{self.decoy}"


@dataclass(frozen=True)
class ReceiptLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`category` 该报的类别；`band` 金额档位；`decoy` 有没有埋诱饵。
    """

    category: str
    band: str
    decoy: str

    @property
    def over(self) -> bool:
        """输入：无。输出：金额是否超了该类单笔上限。"""
        return self.band != "under"

    @property
    def tier(self) -> str:
        """输入：无。输出：该走哪一档审核。"""
        return tier_of(self.band, self.category)

    @property
    def near(self) -> bool:
        """输入：无。输出：正文里有没有「像某一类、但不在表里」的说法。"""
        return self.decoy == DECOY_NEAR


def label_for(case: ReceiptCase) -> ReceiptLabels:
    """输入：注入特征。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看 case、不看文本——文本是后写的，标签先定。`render.check_draft`
    负责保证文本真的兑现这份计划（按字面算出来的命中类别与金额与计划逐条相同）。
    """
    return ReceiptLabels(category=case.category, band=case.band, decoy=case.decoy)


def observed_labels(text: str, loc: dict[str, JsonValue]) -> ReceiptLabels | None:
    """输入：收据行正文、语言字典。输出：按字面重推的标签；认不出来给 None（**审计侧**路径）。

    预期行为：命中类别必须**恰好一个**、注入金额必须**恰好一个**——0 个或多个都算认不出来
    （那说明渲染没兑现计划，或者一条条目同时踩了两类）。两者都定下来之后，档位与审核档位由
    纯算术推出来，与构造侧走的是同一批函数（`band_of` / `tier_of`）。
    """
    hits = categories_hit(text, keyword_table(loc))
    if len(hits) != 1:
        return None
    amounts = amounts_in(text)
    if len(amounts) != 1:
        return None
    cents = cents_of(amounts[0])
    if cents is None:
        return None
    decoy = DECOY_NEAR if decoy_hit(text, decoy_table(loc)) else DECOY_NONE
    return ReceiptLabels(
        category=hits[0], band=band_of(cents, CAPS_CENTS[hits[0]]), decoy=decoy
    )


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[ReceiptCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`ReceiptCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [
        ReceiptCase(
            category=str(row["category"]), band=str(row["band"]), decoy=str(row["decoy"])
        )
        for row in rows
    ]
