"""记录匹配（match）workflow 的真值规格：两条目录记录逐字段比对。

输入/输出：不可变数据类与纯函数；**不含任何领域词表**（商品名、品牌名、规格 / 度数 /
包装的写法、栏名、题面与选项描述全在 `locales/`）。
预期行为：八个题族的答案全部由「五栏各自的关系」算出来。关系只有三种：写法相同
（`same`）、写法不同但归一化后相同（`variant`）、真的不一致（`diff`）。这一族要训的是
**先归一化、再逐栏比、最后数个数**，所以整条链是纯字符串与纯查表，零 token。

为什么是构造式：官方那篇 `entity_alignment` 拿两个啤酒目录的 450 组候选对，每对问
「1 个 Score + 3 个伴随 Noul」。这里把「哪几栏对得上」做成**可判定的文本位**——两条记录
的每一栏要么逐字相同、要么只是写法不同（大小写 / 空格 / 单位换算 / 别名）、要么真的不同，
于是「像不像同一个商品」退化成按题面给的对照表逐栏归一化再比，判别器能从 state 反解出
每一栏的原文、独立重算全部八个答案。

与 `rerank` 的区别：rerank 是**一条列表内部的次序**，这里是**两条记录之间的字段对齐**。
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from jc.knobs import require_pool

# ---------- 比较范围 ----------

#: 比较范围里的五栏（顺序固定；题面里也按这个顺序讲规则）。
FIELDS: Final[tuple[str, ...]] = ("name", "brand", "size", "abv", "pack")

#: 每一栏的三种关系。
SAME: Final[str] = "same"
VARIANT: Final[str] = "variant"
DIFF: Final[str] = "diff"
KINDS: Final[tuple[str, ...]] = (SAME, VARIANT, DIFF)

# ---------- 商品目录（键与数值；写法在 locales/） ----------

#: 商品目录：键 → (规格 ml, 度数, 包装键)。名称与品牌只在语言文件里有写法，
#: 这里只留键——`diff` 那一栏要挑「另一个商品的值」，比的是**归一化后的数值**。
CATALOG: Final[dict[str, tuple[int, float, str]]] = {
    "lager": (750, 5.0, "glass"),
    "ipa": (330, 6.5, "can"),
    "stout": (500, 7.0, "glass"),
    "cider": (330, 4.5, "can"),
    "cola": (1000, 0.0, "plastic"),
    "water": (500, 0.0, "plastic"),
    "tea": (1500, 0.0, "carton"),
    "juice": (1000, 0.0, "carton"),
}

#: 商品键（顺序稳定；留出最后一个给 OOD 侧）。
PRODUCTS: Final[tuple[str, ...]] = tuple(CATALOG)

#: 规格 / 度数 / 包装的全部取值（语言文件必须为每一个给出三种写法）。
SIZES: Final[tuple[int, ...]] = tuple(sorted({value[0] for value in CATALOG.values()}))
ABVS: Final[tuple[float, ...]] = tuple(sorted({value[1] for value in CATALOG.values()}))
PACKS: Final[tuple[str, ...]] = tuple(sorted({value[2] for value in CATALOG.values()}))

# ---------- 三种写法 ----------

#: 写法对：左记录用第一种、右记录用第二种。三对都试过一遍，留出一对给 OOD。
SPELLINGS: Final[tuple[tuple[int, int], ...]] = ((0, 1), (0, 2), (1, 2))

#: 显示顺序：五栏在 state 里的排列。题面不依赖显示顺序，这一维只改文本。
LAYOUTS: Final[tuple[str, ...]] = ("plain", "reversed", "interleaved")

#: 每栏在 state 里的行序（按 `FIELDS` 的下标）。
LAYOUT_ORDERS: Final[dict[str, tuple[int, ...]]] = {
    "plain": (0, 1, 2, 3, 4),
    "reversed": (4, 3, 2, 1, 0),
    "interleaved": (2, 0, 4, 1, 3),
}

#: 额外一栏：`none` 不写，`origin` 写一行**不在比较范围内**的产地（两条记录永远不同）。
EXTRAS: Final[tuple[str, ...]] = ("none", "origin")

# ---------- 十二种字段关系 ----------

#: 十二种字段关系（搜索脚本 `tmp/mt_design.py`，八个题族**同时**摊平）：
#: 1. 一致字段数取 {0:1, 1:3, 2:3, 3:1, 4:2, 5:2}——总数 30 = 5 栏 × 6，于是
#:    ① 三个档（0~1 / 2~3 / 4~5）各 4 条 → `score band` 精确 1/3；
#:    ② 每一栏 `diff` 恰好 6 条 → 五个 `noul same_<栏>` 精确 50/50；
#: 2. 处置档 4/4/4；`variant` 栏数 0/1/2/3 各 3 条；
#: 3. 十二种关系两两不同（「五栏全不一致」只有唯一一种写法，所以六档 score 做不到：
#:    想要两条就只能重复行，重复行就是重复样本）。
PATTERNS: Final[tuple[tuple[str, ...], ...]] = (
    # 一致 0（唯一一种写法）：五栏全不一致。
    (DIFF, DIFF, DIFF, DIFF, DIFF),
    # 一致 1：三栏全不一致的兄弟，只有一栏是「写法不同但归一化后相同」。
    (DIFF, DIFF, VARIANT, DIFF, DIFF),
    (DIFF, SAME, DIFF, DIFF, DIFF),
    (DIFF, VARIANT, DIFF, DIFF, DIFF),
    # 一致 2。
    (DIFF, DIFF, DIFF, VARIANT, VARIANT),
    (DIFF, DIFF, SAME, DIFF, VARIANT),
    (SAME, DIFF, DIFF, SAME, DIFF),
    # 一致 3。
    (VARIANT, VARIANT, DIFF, DIFF, SAME),
    # 一致 4：名称都对得上，只差一栏。
    (VARIANT, DIFF, VARIANT, VARIANT, SAME),
    (VARIANT, SAME, VARIANT, VARIANT, DIFF),
    # 一致 5：全部对得上，只是写法不同。
    (SAME, VARIANT, SAME, SAME, VARIANT),
    (VARIANT, VARIANT, SAME, VARIANT, SAME),
)

# ---------- 规则（构造侧与判别侧共用） ----------


def same_of(pattern: tuple[str, ...]) -> tuple[bool, ...]:
    """输入：一条字段关系。输出：每一栏是否一致（`diff` 之外都算一致）。"""
    return tuple(kind != DIFF for kind in pattern)


def agree_of(pattern: tuple[str, ...]) -> int:
    """输入：一条字段关系。输出：一致字段数（0~5）。"""
    return sum(1 for kind in pattern if kind != DIFF)


def variant_count_of(pattern: tuple[str, ...]) -> int:
    """输入：一条字段关系。输出：有几栏「写法不同但归一化后一致」。"""
    return sum(1 for kind in pattern if kind == VARIANT)


def band_of(count: int) -> int:
    """输入：一致字段数。输出：档位（0 = 0~1 档 / 1 = 2~3 档 / 2 = 4~5 档）。

    预期行为：这是官方那篇「四舍五入到最近档位」的落地——档位比原始计数粗，
    所以「数几条」与「落哪档」是两个不同的判断，两个题族各问一个。
    """
    if count <= 1:
        return 0
    if count <= 3:
        return 1
    return 2


def verdict_of(pattern: tuple[str, ...]) -> str:
    """输入：一条字段关系。输出：处置档（`same` / `different` / `review`）。

    预期行为：**按顺序**判，先命中先算——① 名称与品牌都不一致就判不同商品（连标识都
    对不上，不必再数）；② 否则一致字段数 ≥ 4 判同一个商品；③ 其余交人工复核。
    顺序不能换：先数后判会让「名称品牌全不一致但其余四栏相同」这种假记录被判成同一个。
    """
    if pattern[0] == DIFF and pattern[1] == DIFF:
        return "different"
    if agree_of(pattern) >= 4:
        return "same"
    return "review"


def field_value(field: str, product: str) -> Any:
    """输入：栏名、商品键。输出：这一栏**归一化后**的值（名称 / 品牌用商品键当代理）。

    预期行为：只有用来判「另一个商品在这一栏上确实不同」，所以名称与品牌拿商品键比就够
    ——每个商品的名称与品牌都是唯一的。
    """
    size, abv, pack = CATALOG[product]
    return {"name": product, "brand": product, "size": size, "abv": abv, "pack": pack}[field]


def other_product(field: str, product: str, pool: Sequence[str]) -> str:
    """输入：栏名、商品键、取值池。输出：**池子里另一个**商品键，它在这一栏上的值确实不同。

    预期行为：从目录顺序里 `product` 的下一格开始往后绕圈，**跳过不在 `pool` 里的商品**，
    第一个值不同的就用它。两条都必要：
    - 规格与度数在目录里有重复（330 出现两次、0.0 出现四次），直接取「下一个」会让
      `diff` 那一栏其实相等——那是把答案写错的硬伤，所以逐个跳过同值的；
    - 池子由 `diff_pool(切分)` 给：同分布侧不给留出商品，否则「留出商品在训练侧一个
      都不出现」只在 case 维度成立、文本里照样露脸（实测 `tea` 往后一格就是 `juice`，
      训练侧 604 条里 57 条带留出商品的名字）。
    池子里找不到不同值就直接报错，不静默少给。
    """
    order = list(PRODUCTS)
    start = order.index(product)
    for step in range(1, len(order)):
        candidate = order[(start + step) % len(order)]
        if candidate in pool and field_value(field, candidate) != field_value(field, product):
            return candidate
    raise ValueError(f"{product} 在 {field} 这一栏上找不到池子里不同的商品（池子：{list(pool)}）")


# ---------- case 与采样参数 ----------

QUERY_PREFIX: Final[str] = "M-"


@dataclass(frozen=True)
class MatchCase:
    """一条匹配请求的五个维度（顺序与 `as_dict` 一致，哈希分桶按它算，必须稳定）。"""

    product: str
    pattern: tuple[str, ...]
    spelling: tuple[int, int]
    layout: str
    extra: str

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：{维度名: 取值}（顺序稳定；字段关系摊成 `sdvdv` 这种紧凑串）。"""
        return {
            "product": self.product,
            "pattern": "".join(kind[0] for kind in self.pattern),
            "spelling": f"{self.spelling[0]}-{self.spelling[1]}",
            "layout": self.layout,
            "extra": self.extra,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        kinds = "".join(kind[0] for kind in self.pattern)
        return f"{self.product}/关系={kinds}/写法={self.spelling[0]}-{self.spelling[1]}/{self.layout}/{self.extra}"

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这个 case 里训练侧没见过的维度名（空 = 全部见过）。"""
        checks: tuple[tuple[str, Any, frozenset[Any]], ...] = (
            ("product", self.product, SEEN_PRODUCTS),
            ("spelling", self.spelling, SEEN_SPELLINGS),
        )
        return tuple(name for name, value, seen in checks if value not in seen)


def material_of(case: MatchCase) -> str:
    """输入：case。输出：素材指纹——五个维度拼成的串。

    预期行为：切分按它分桶（`workflow._slice`），同一条匹配请求整组落在同一档。
    显示顺序与额外栏只改文本，但它们确实改 state，所以也算素材。
    """
    return "|".join(f"{key}={value}" for key, value in case.as_dict().items())


def query_id_of(case: MatchCase) -> str:
    """输入：case。输出：请求号（形如 `M-48213`）。

    预期行为：由素材指纹的 sha256 派生——确定性，而且**不泄露切分**：用序号或切分名
    拼请求号会让同一条请求在不同切分里拿到不同号，等于把切分信息写进 state。
    """
    digest = hashlib.sha256(material_of(case).encode("utf-8")).hexdigest()
    return f"{QUERY_PREFIX}{10000 + int(digest[:6], 16) % 90000}"


@dataclass(frozen=True)
class MatchConfig:
    """一个切分的采样参数：各维度取值表 + 问法池 + `ood_only`。"""

    split: str = "train"
    products: tuple[str, ...] = PRODUCTS
    patterns: tuple[tuple[str, ...], ...] = PATTERNS
    spellings: tuple[tuple[int, int], ...] = SPELLINGS
    layouts: tuple[str, ...] = LAYOUTS
    extras: tuple[str, ...] = EXTRAS
    phrasings: tuple[int, ...] = (0, 1, 2)
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在**开工前**报人话（点名是哪个字段）。"""
        require_pool("products", self.products)
        require_pool("patterns", self.patterns)
        require_pool("spellings", self.spellings)
        require_pool("layouts", self.layouts)
        require_pool("extras", self.extras)
        require_pool("phrasings", self.phrasings)
        unknown_products = [key for key in self.products if key not in CATALOG]
        if unknown_products:
            raise ValueError(
                f"products 里有不存在的商品：{unknown_products}（可用：{list(PRODUCTS)}）"
            )
        unknown_layouts = [name for name in self.layouts if name not in LAYOUT_ORDERS]
        if unknown_layouts:
            raise ValueError(
                f"layouts 里有不存在的显示顺序：{unknown_layouts}（可用：{list(LAYOUTS)}）"
            )
        unknown_extras = [name for name in self.extras if name not in EXTRAS]
        if unknown_extras:
            raise ValueError(
                f"extras 里有不存在的额外栏：{unknown_extras}（可用：{list(EXTRAS)}）"
            )
        bad_patterns = [
            pattern
            for pattern in self.patterns
            if len(pattern) != len(FIELDS) or any(kind not in KINDS for kind in pattern)
        ]
        if bad_patterns:
            raise ValueError(
                f"patterns 里有不合法的字段关系：{bad_patterns}"
                f"（每条必须是 {len(FIELDS)} 个、取值在 {list(KINDS)} 里）"
            )
        bad_spellings = [
            pair
            for pair in self.spellings
            if len(pair) != 2
            or any(isinstance(index, bool) or not isinstance(index, int) for index in pair)
            or pair[0] == pair[1]
        ]
        if bad_spellings:
            raise ValueError(f"spellings 里有不合法的写法对：{bad_spellings}（两个不同的整数下标）")


def build_cases(config: MatchConfig) -> list[MatchCase]:
    """枚举 case 矩阵。

    输入：采样参数。输出：`MatchCase` 列表（顺序稳定、可复算）。
    预期行为：全枚举（**不做成立性过滤**——五个维度两两不矛盾）；`ood_only` 时只留
    「至少含一个训练侧没见过的取值」的 case，所以 OOD 与同分布侧**按构造就不重叠**。
    """
    out: list[MatchCase] = []
    for product in config.products:
        for pattern in config.patterns:
            for spelling in config.spellings:
                for layout in config.layouts:
                    for extra in config.extras:
                        case = MatchCase(product, pattern, spelling, layout, extra)
                        if config.ood_only and not case.unseen_dims():
                            continue
                        out.append(case)
    return out


# ---------- 真值规格 ----------


@dataclass(frozen=True)
class MatchSpec:
    """一条匹配请求的完整真值：case + 由规则算出来的八个答案。"""

    case: MatchCase
    split: str
    seq: int
    phrasing: int
    query_id: str
    same: tuple[bool, ...]
    agree: int
    band: int
    verdict: str
    variant_count: int

    def first_same(self) -> bool:
        """输入：无。输出：名称这一栏是否一致。"""
        return self.same[FIELDS.index("name")]

    def any_diff(self) -> bool:
        """输入：无。输出：是不是至少有一栏真的不一致。"""
        return self.agree < len(FIELDS)


def sample_spec(
    rng: Any, case: MatchCase, split: str, seq: int, phrasing: int
) -> MatchSpec:
    """输入：RNG（**不用**）、case、切分、序号、问法。输出：真值规格。

    预期行为：不掷骰子——同一 case 必得同一份规格（请求号也由 case 指纹派生），
    这样「同素材同题必须同答」这条一致性校验才成立。
    """
    del rng  # 规格完全由 case 决定；签名里的 RNG 只为与协议一致
    return MatchSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        query_id=query_id_of(case),
        same=same_of(case.pattern),
        agree=agree_of(case.pattern),
        band=band_of(agree_of(case.pattern)),
        verdict=verdict_of(case.pattern),
        variant_count=variant_count_of(case.pattern),
    )


def spec_to_dict(spec: MatchSpec) -> dict[str, Any]:
    """输入：规格。输出：可 JSON 序列化的摘要（写进 manifest 与调试，不进样本行）。"""
    return {
        "case": spec.case.describe(),
        "unseen_dims": list(spec.case.unseen_dims()),
        "split": spec.split,
        "seq": spec.seq,
        "query_id": spec.query_id,
        "same": list(spec.same),
        "agree": spec.agree,
        "band": spec.band,
        "verdict": spec.verdict,
        "variant_count": spec.variant_count,
    }


# ---------- 值留出（训练侧见过哪些取值） ----------

#: 训练侧见过的商品：最后一个留出（OOD 侧才出现那个商品）。
#: 商品只改名称 / 品牌 / 规格 / 度数的**写法**，不动任何规则分支，所以留出它是安全的。
SEEN_PRODUCTS: Final[frozenset[str]] = frozenset(
    key for key in PRODUCTS if key != PRODUCTS[-1]
)
#: 训练侧见过的写法对：最后一对留出（最绕的那种写法只出现在 OOD 侧）。
SEEN_SPELLINGS: Final[frozenset[tuple[int, int]]] = frozenset(
    pair for pair in SPELLINGS if pair != SPELLINGS[-1]
)


def seen_in_order(values: Sequence[Any], seen: frozenset[Any]) -> tuple[Any, ...]:
    """输入：完整取值表、训练侧见过的取值集合。输出：按**原表顺序**过滤后的元组。

    预期行为：不用 `sorted`——原表顺序本身有意义（商品表的下标决定 `other_product` 从哪一格
    往后绕圈，写法对的下标决定哪两种写法配在一起），排序会把「写法对 0-1」和「写法对 1-2」
    对调，切分虽然还是确定的，但报告读起来会对不上。
    """
    return tuple(value for value in values if value in seen)


#: 同分布侧的取值表：case 维度与 `diff` 取值池**共用同一份过滤结果**，避免两处各算一遍。
ID_PRODUCTS: Final[tuple[str, ...]] = seen_in_order(PRODUCTS, SEEN_PRODUCTS)
ID_SPELLINGS: Final[tuple[tuple[int, int], ...]] = seen_in_order(SPELLINGS, SEEN_SPELLINGS)


def diff_pool(split: str) -> tuple[str, ...]:
    """输入：切分名。输出：这一档 `diff` 那一栏允许从哪些商品里取值。

    预期行为：同分布侧只给训练侧见过的商品，`test_ood` 才放开全目录。不给的话
    `other_product` 会顺着目录绕到留出商品身上，留出商品的名字照样出现在训练侧文本里，
    「留出商品在训练侧一个都不出现」就只剩 case 维度成立了。
    """
    return PRODUCTS if split == "test_ood" else ID_PRODUCTS
