"""检索重排（rerank）workflow 的真值规格：一份查询 + 一个候选短名单 + 一套排序规则。

输入/输出：不可变数据类与纯函数；**不含任何领域词表**（主题词、方面词、来源名、
正文模板、题面与选项描述全在 `locales/`）。
预期行为：七个题族的答案全部由「短名单画像 + 查询主题 / 方面」算出来——相关度是
「四条必要条件满足了几条」，名次是 `(相关度, 来源权威度, 更新日期, 编号)` 的降序。
构造侧与判别侧**共用同一套规则函数**，所以「画像表写错」与「渲染写歪」不会两边
各错一半还互相印证。

为什么是构造式：官方那篇重排是「每个查询-候选对问一次模型，按分数排序」。这里把候选的
属性做成**可判定的文本位**（正文说没说自己讲的是哪个主题 / 哪个方面、有没有附示例），
于是「这一对到底多相关」退化成读四条条件，零 token 就能放量；判别器还能从 state
反解回属性、独立重算全部七个答案。

与 `search` 的区别：search 问「结果页上该点哪一条」（检索形状），这里问的是**排序**
（官方十种决策形状里我们最后补上的一种）——短名单已经给出，问的是名次、逐对先后，
以及「两条的相关度是不是一样」。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Final

from jc.knobs import require_pool

# ---------- 四条必要条件 ----------

#: 查询的四条必要条件（顺序即题面里的编号）。
CONDITIONS: Final[tuple[str, ...]] = ("topic", "aspect", "fresh", "example")

# ---------- 来源权威度 ----------

#: 来源权威度：数值越大越权威，排序规则②按它比。判据原文照抄进题面。
SOURCE_RANK: Final[dict[str, int]] = {"official": 3, "vendor": 2, "community": 1, "blog": 0}
SOURCES: Final[tuple[str, ...]] = ("official", "vendor", "community", "blog")

#: 时效线：候选更新日期不早于这一天算满足第③条。
#: ISO 日期串可以直接按字典序比大小，不用解析成日期对象（少一处能写错的地方）。
DEADLINE: Final[str] = "2024-06-01"

# ---------- 查询主题与方面 ----------

#: 查询主题（顺序稳定；留出最后一个给 OOD 侧）。
TOPICS: Final[tuple[str, ...]] = (
    "backup",
    "rotation",
    "throttle",
    "retention",
    "migrate",
    "auth",
    "cache",
    "alert",
)

#: 方面（全局三种，不随主题变）。正文里写的是方面词，判别器按**整词**比对。
#: 全局而不是每主题一套，是为了让「同主题不同方面」与「不同主题」两种干扰项
#: 都只用一张词表就能构造出来。
ASPECTS: Final[tuple[str, ...]] = ("steps", "config", "failures")

# ---------- 短名单画像 ----------


@dataclass(frozen=True)
class Cand:
    """一条候选：三条可判定的文本位 + 来源 + 更新日期。

    字段：
    - `topic_ok`：正文讲的正是查询的主题；
    - `aspect_ok`：正文讲的正是查询要问的那个方面（按构造蕴含 `topic_ok`）；
    - `example`：正文附了可照抄的示例；
    - `source` / `day`：来源与更新日期，正文之外的两格。
    """

    topic_ok: bool
    aspect_ok: bool
    example: bool
    source: str
    day: str

    def key(self) -> str:
        """输入：无。输出：一行紧凑描述（统计与指纹用，必须稳定）。"""
        bits = f"{int(self.topic_ok)}{int(self.aspect_ok)}{int(self.example)}"
        return f"{bits}/{self.source}/{self.day}"


@dataclass(frozen=True)
class Shape:
    """一个短名单画像：四条候选 + 四个被点名的位置。

    字段：`nth` 是「第几位」问的位次（1-based，取 2 或 3）；`ahead` / `tie` 是两对被点名的
    候选（0-based，前者问「谁在前」、后者问「相关度是否相同」）；`probe` 是「这条候选」
    问的那一条。四个位置都由画像本身决定，与主题 / 措辞无关——所以那两个维度只改文本。
    """

    cands: tuple[Cand, Cand, Cand, Cand]
    nth: int
    ahead: tuple[int, int]
    tie: tuple[int, int]
    probe: int

    def key(self) -> str:
        """输入：无。输出：四条候选的紧凑描述串（统计与指纹用，必须稳定）。"""
        return ";".join(cand.key() for cand in self.cands)


def _cand(raw: tuple[int, int, int, str, str]) -> Cand:
    """输入：一条候选的紧凑写法（三个 0/1 位 + 来源 + 更新日期）。输出：一条 `Cand`。

    预期行为：三个布尔位显式转 `bool`。画像表里写的是 `0`/`1`（int），直接解包会让
    `topic_ok` / `aspect_ok` / `example` 变成 int，装配层据此判「答案不是布尔」，
    把整条 case 拒掉（实测：`noul 的 answer 必须是布尔：on_topic -> 1`）。
    """
    return Cand(bool(raw[0]), bool(raw[1]), bool(raw[2]), raw[3], raw[4])


def _shape(
    cands: tuple[tuple[int, int, int, str, str], ...],
    nth: int,
    ahead: tuple[int, int],
    tie: tuple[int, int],
    probe: int,
) -> Shape:
    """输入：四条候选的紧凑写法 + 四个被点名的位置。输出：一个 `Shape`。

    预期行为：候选不是四条直接报错——画像表是手推的，写漏一条时要在**导入期**炸掉，
    而不是等到采样时抛 `IndexError`。
    """
    if len(cands) != 4:
        raise ValueError(f"画像必须有四条候选，现在有 {len(cands)} 条：{cands}")
    return Shape(
        (
            _cand(cands[0]),
            _cand(cands[1]),
            _cand(cands[2]),
            _cand(cands[3]),
        ),
        nth,
        ahead,
        tie,
        probe,
    )


#: 十个短名单画像。四个目标一起满足（复算脚本 `tmp/rr_pick2.py`，七个题族偏差 ≤ 0.05）：
#: 1. **被点名的那条候选**（`probe`）的相关度五档（0~4）各出现两次——`score relevance` 摊平；
#: 2. 被点名的那条候选「主题 + 方面都对」5 真 5 假——`noul on_topic` 摊平；
#: 3. `first` / `nth` / `last` 的答案在四条候选上都是 3/3/2/2；
#: 4. ②权威度、③日期、④编号三条破并列规则**每条都被真的用到**（见下方注释）。
#: 均衡看的是**被问到的那一条**，不是四条候选全体：全体候选的相关度分布是 3/4/6/17/10，
#: 本来就不该平——四条候选若长得一样，短名单就没有「排序」可言了。
SHAPES: Final[tuple[Shape, ...]] = (
    # ②：C2 与 C3 相关度同为 3、来源同为官方 → 靠③日期分出先后。
    _shape(
        (
            (1, 1, 1, "community", "2024-07-05"),
            (1, 1, 0, "official", "2024-08-21"),
            (1, 1, 1, "official", "2024-03-22"),
            (0, 0, 1, "blog", "2024-06-14"),
        ),
        3,
        (0, 1),
        (1, 2),
        0,
    ),
    # ④：C1 与 C4 的相关度（都是 2）、权威度（都是社区）、更新日期（都是 2024-01-07）全同
    # → 只能靠编号小的在前；这一对分出先后直接决定 `nth` 问的那一位答谁。
    _shape(
        (
            (1, 0, 1, "community", "2024-01-07"),
            (1, 1, 1, "blog", "2024-06-14"),
            (0, 0, 1, "vendor", "2023-11-18"),
            (1, 1, 0, "community", "2024-01-07"),
        ),
        3,
        (0, 1),
        (1, 0),
        3,
    ),
    # ②：C2 与 C4 相关度同为 3，来源官方 > 厂商 → 不靠日期就分出先后。
    _shape(
        (
            (0, 0, 1, "vendor", "2024-12-02"),
            (1, 1, 1, "official", "2024-04-30"),
            (1, 1, 1, "community", "2024-08-21"),
            (1, 0, 1, "vendor", "2025-01-15"),
        ),
        2,
        (2, 1),
        (1, 3),
        2,
    ),
    _shape(
        (
            (1, 1, 1, "community", "2024-08-21"),
            (0, 0, 1, "blog", "2024-06-14"),
            (1, 1, 1, "blog", "2024-03-22"),
            (1, 1, 0, "official", "2024-10-09"),
        ),
        2,
        (2, 1),
        (2, 3),
        2,
    ),
    # ④：C1 与 C2 相关度、来源、日期**全同** → 只能靠编号小的在前。
    _shape(
        (
            (1, 1, 0, "official", "2025-01-15"),
            (1, 1, 1, "vendor", "2024-04-30"),
            (0, 0, 1, "vendor", "2024-01-07"),
            (1, 1, 1, "community", "2024-10-09"),
        ),
        2,
        (3, 0),
        (0, 1),
        2,
    ),
    _shape(
        (
            (1, 0, 0, "community", "2024-07-05"),
            (1, 1, 1, "vendor", "2024-12-02"),
            (1, 1, 0, "vendor", "2025-01-15"),
            (1, 1, 1, "vendor", "2023-11-18"),
        ),
        2,
        (1, 2),
        (2, 3),
        2,
    ),
    _shape(
        (
            (1, 1, 0, "official", "2024-08-21"),
            (1, 1, 1, "community", "2024-01-07"),
            (1, 1, 1, "blog", "2024-10-09"),
            (0, 0, 0, "vendor", "2024-03-22"),
        ),
        3,
        (3, 0),
        (0, 3),
        3,
    ),
    _shape(
        (
            (1, 0, 1, "official", "2024-12-02"),
            (0, 0, 1, "blog", "2023-11-18"),
            (1, 1, 0, "community", "2024-08-21"),
            (1, 1, 1, "vendor", "2025-01-15"),
        ),
        2,
        (1, 0),
        (0, 1),
        1,
    ),
    _shape(
        (
            (1, 1, 1, "vendor", "2024-10-09"),
            (1, 1, 0, "official", "2024-07-05"),
            (0, 0, 0, "blog", "2023-11-18"),
            (1, 0, 1, "community", "2024-04-30"),
        ),
        2,
        (1, 0),
        (0, 1),
        3,
    ),
    _shape(
        (
            (0, 0, 1, "community", "2024-03-22"),
            (1, 1, 1, "official", "2024-12-02"),
            (1, 1, 0, "vendor", "2025-01-15"),
            (0, 0, 0, "blog", "2023-11-18"),
        ),
        2,
        (0, 1),
        (1, 0),
        3,
    ),
)

# ---------- 另外两维（只改文本，不改任何答案） ----------

#: 正文里多出来的「噪声句」组合。它们**不进任何判据**，只把正文撑得像真的候选片段。
TRAPS: Final[tuple[tuple[str, ...], ...]] = (
    (),
    ("forwarded",),
    ("saved", "edited"),
    ("forwarded", "saved", "edited"),
)

#: 措辞：主题 / 方面那一句的三种写法，以及示例句的三种写法。三种写法说的是同一件事，
#: 所以**不改任何答案**——这一维专门压「只认某一种模板」的捷径。
WORDINGS: Final[tuple[str, ...]] = ("plain", "titled", "terse")

# ---------- 规则（构造侧与判别侧共用） ----------


def fresh(day: str) -> bool:
    """输入：更新日期。输出：是否不早于时效线。预期行为：ISO 日期串按字典序比。"""
    return day >= DEADLINE


def relevance(cand: Cand) -> int:
    """输入：一条候选。输出：相关度 = 四条必要条件里满足的条数（0~4）。

    预期行为：四条**平权**相加，不做加权——加权会让「相关度」这个档位变得不好心算，
    而这一族要训的正是「逐条核对再数个数」。
    """
    return int(cand.topic_ok) + int(cand.aspect_ok) + int(fresh(cand.day)) + int(cand.example)


def order_of(cands: tuple[Cand, ...]) -> tuple[int, ...]:
    """输入：四条候选。输出：按题面规则排好的名次（下标序列，第一名在前）。

    预期行为：`(相关度, 来源权威度, 更新日期, -编号)` 降序。日期用字符串比（ISO 格式
    下字典序即时间序），`-编号` 配合降序让编号**小**的在前，四条规则合起来保证全序唯一。
    收的是候选元组而不是画像：判别器从 state 反解出来的只有候选，没有画像。
    """
    key = lambda index: (
        relevance(cands[index]),
        SOURCE_RANK[cands[index].source],
        cands[index].day,
        -index,
    )
    return tuple(sorted(range(4), key=key, reverse=True))


def on_topic_of(cand: Cand) -> bool:
    """输入：一条候选。输出：主题与方面是否都对。

    预期行为：只问「是不是在讲这件事」，不问时效与示例——与 `relevance` 相关但不相同
    （`relevance` 为 2 的候选里，只有「主题 + 方面都对、两条都不占」那一种为真）。
    """
    return bool(cand.topic_ok and cand.aspect_ok)


# ---------- case 与采样参数 ----------

QUERY_PREFIX: Final[str] = "Q-"


@dataclass(frozen=True)
class RerankCase:
    """一条重排请求的五个维度（顺序与 `as_dict` 一致，哈希分桶按它算，必须稳定）。"""

    topic: str
    aspect: str
    shape: Shape
    traps: tuple[str, ...]
    wording: str

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：{维度名: 取值}（顺序稳定；两个集合维度摊成 `+` 连接的串）。"""
        return {
            "topic": self.topic,
            "aspect": self.aspect,
            "shape": self.shape.key(),
            "traps": label_of(self.traps),
            "wording": self.wording,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return (
            f"{self.topic}/{self.aspect}/{self.shape.key()}"
            f"/噪声={label_of(self.traps)}/{self.wording}"
        )

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这个 case 里训练侧没见过的维度名（空 = 全部见过）。"""
        checks: tuple[tuple[str, Any, frozenset[Any]], ...] = (
            ("topic", self.topic, SEEN_TOPICS),
            ("traps", self.traps, SEEN_TRAPS),
            ("wording", self.wording, SEEN_WORDINGS),
        )
        return tuple(name for name, value, seen in checks if value not in seen)


def label_of(subset: tuple[str, ...]) -> str:
    """输入：噪声句组合。输出：可读标签（空集给 `none`）。

    预期行为：先按 `TRAPS` 里的顺序归一再用 `+` 连接——直接 `str(tuple)` 会因为顺序不同
    把同一个集合写成两种样子，统计会裂开。
    """
    if not subset:
        return "none"
    return "+".join(key for key in TRAPS_KEYS if key in subset)


#: 噪声句的全部取值（`label_of` 的归一顺序）。
TRAPS_KEYS: Final[tuple[str, ...]] = ("forwarded", "saved", "edited")


def material_of(case: RerankCase) -> str:
    """输入：case。输出：素材指纹——五个维度拼成的串。

    预期行为：切分按它分桶（`workflow._slice`），同一条重排请求整组落在同一档。
    噪声句与措辞只改文本，但它们确实改 state，所以也算素材。
    """
    return "|".join(f"{key}={value}" for key, value in case.as_dict().items())


def query_id_of(case: RerankCase) -> str:
    """输入：case。输出：查询号（形如 `Q-48213`）。

    预期行为：由素材指纹的 sha256 派生——确定性，而且**不泄露切分**：
    用序号或切分名拼查询号会让同一条请求在不同切分里拿到不同号，
    等于把切分信息写进 state。
    """
    digest = hashlib.sha256(material_of(case).encode("utf-8")).hexdigest()
    return f"{QUERY_PREFIX}{10000 + int(digest[:6], 16) % 90000}"


@dataclass(frozen=True)
class RerankConfig:
    """一个切分的采样参数：各维度取值表 + 问法池 + `ood_only`。"""

    split: str = "train"
    topics: tuple[str, ...] = TOPICS
    aspects: tuple[str, ...] = ASPECTS
    shapes: tuple[Shape, ...] = SHAPES
    traps: tuple[tuple[str, ...], ...] = TRAPS
    wordings: tuple[str, ...] = WORDINGS
    phrasings: tuple[int, ...] = (0, 1, 2)
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在**开工前**报人话（点名是哪个字段）。"""
        require_pool("topics", self.topics)
        require_pool("aspects", self.aspects)
        require_pool("shapes", self.shapes)
        require_pool("traps", self.traps)
        require_pool("wordings", self.wordings)
        require_pool("phrasings", self.phrasings)
        unknown_topics = [topic for topic in self.topics if topic not in TOPICS]
        if unknown_topics:
            raise ValueError(f"topics 里有不存在的主题：{unknown_topics}（可用：{list(TOPICS)}）")
        unknown_aspects = [aspect for aspect in self.aspects if aspect not in ASPECTS]
        if unknown_aspects:
            raise ValueError(
                f"aspects 里有不存在的方面：{unknown_aspects}（可用：{list(ASPECTS)}）"
            )
        unknown_sources = sorted(
            {
                cand.source
                for shape in self.shapes
                for cand in shape.cands
                if cand.source not in SOURCE_RANK
            }
        )
        if unknown_sources:
            raise ValueError(
                f"画像里有不存在的来源：{unknown_sources}（可用：{list(SOURCE_RANK)}）"
            )


def build_cases(config: RerankConfig) -> list[RerankCase]:
    """枚举 case 矩阵。

    输入：采样参数。输出：`RerankCase` 列表（顺序稳定、可复算）。
    预期行为：全枚举（**不做成立性过滤**——五个维度两两不矛盾）；`ood_only` 时只留
    「至少含一个训练侧没见过的取值」的 case，所以 OOD 与同分布侧**按构造就不重叠**。
    """
    out: list[RerankCase] = []
    for topic in config.topics:
        for aspect in config.aspects:
            for shape in config.shapes:
                for traps in config.traps:
                    for wording in config.wordings:
                        case = RerankCase(topic, aspect, shape, traps, wording)
                        if config.ood_only and not case.unseen_dims():
                            continue
                        out.append(case)
    return out


# ---------- 真值规格 ----------


@dataclass(frozen=True)
class RerankSpec:
    """一条重排请求的完整真值：case + 由规则算出来的七个答案。"""

    case: RerankCase
    split: str
    seq: int
    phrasing: int
    query_id: str
    order: tuple[int, ...]
    relevance: tuple[int, int, int, int]

    def first(self) -> int:
        """输入：无。输出：排第一的候选下标。"""
        return self.order[0]

    def nth(self) -> int:
        """输入：无。输出：排第 `case.shape.nth` 位的候选下标。"""
        return self.order[self.case.shape.nth - 1]

    def last(self) -> int:
        """输入：无。输出：排最后的候选下标。"""
        return self.order[-1]

    def ahead(self) -> bool:
        """输入：无。输出：被点名的那一对里，前者是否排在后者前面。"""
        left, right = self.case.shape.ahead
        return self.order.index(left) < self.order.index(right)

    def same_relevance(self) -> bool:
        """输入：无。输出：被点名的那一对的相关度是否相同。"""
        left, right = self.case.shape.tie
        return self.relevance[left] == self.relevance[right]

    def on_topic(self) -> bool:
        """输入：无。输出：被点名的那条候选的主题与方面是否都对。"""
        return on_topic_of(self.case.shape.cands[self.case.shape.probe])

    def probe_relevance(self) -> int:
        """输入：无。输出：被点名的那条候选的相关度档位。"""
        return self.relevance[self.case.shape.probe]


def sample_spec(rng: Any, case: RerankCase, split: str, seq: int, phrasing: int) -> RerankSpec:
    """输入：RNG（**不用**）、case、切分、序号、问法。输出：真值规格。

    预期行为：不掷骰子——同一 case 必得同一份规格（查询号也由 case 指纹派生），
    这样「同素材同题必须同答」这条一致性校验才成立。
    """
    del rng  # 规格完全由 case 决定；签名里的 RNG 只为与协议一致
    return RerankSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        query_id=query_id_of(case),
        order=order_of(case.shape.cands),
        relevance=tuple(relevance(cand) for cand in case.shape.cands),
    )


def spec_to_dict(spec: RerankSpec) -> dict[str, Any]:
    """输入：规格。输出：可 JSON 序列化的摘要（写进 manifest 与调试，不进样本行）。"""
    return {
        "case": spec.case.describe(),
        "unseen_dims": list(spec.case.unseen_dims()),
        "split": spec.split,
        "seq": spec.seq,
        "query_id": spec.query_id,
        "order": list(spec.order),
        "relevance": list(spec.relevance),
    }


# ---------- 值留出（训练侧见过哪些取值） ----------

#: 训练侧见过的主题：`alert` 留出（OOD 侧才出现「告警」这个主题）。
#: 主题只改正文里的主题词，不动任何规则分支，所以留出它是安全的。
SEEN_TOPICS: Final[frozenset[str]] = frozenset(
    topic for topic in TOPICS if topic != "alert"
)
#: 训练侧见过的噪声句组合：最后一种留出（OOD 侧才出现三条一起）。
SEEN_TRAPS: Final[frozenset[tuple[str, ...]]] = frozenset(
    value for value in TRAPS if value != ("forwarded", "saved", "edited")
)
#: 训练侧见过的措辞：`terse` 留出（最省字的那种写法只出现在 OOD 侧）。
#: 三条留出**都落在表面形式上**，刻意不动画像表：画像表一改，七个题族的答案分布
#: 就全跟着动，而均衡是**手推**出来的（见 `SHAPES` 的注释）——用「更难 OOD」
#: 换「七个题族全被均衡闸标红」不值。
SEEN_WORDINGS: Final[frozenset[str]] = frozenset(
    value for value in WORDINGS if value != "terse"
)
