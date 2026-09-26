"""检索 workflow 的真值规格：一次检索请求 + 一页检索结果。

输入/输出：不可变数据类与纯函数；**不含任何领域词表**（主题名、要点词、指标名、
取值、标题与摘要模板全在 `locales/`）。
预期行为：所有答案都**只从渲染出来的那页文本反推**——主题是否对上、类型是否对上、
时效是否达标、摘要覆盖了几个要点、两条摘要是否互相矛盾、两条结果是否同源，
全是文本上可观测的事实。构造器与判别器共用同一套规则，所以「表格写错」与
「渲染写歪」会变成同一类错误，不会两边各错一半还互相印证。

为什么不做成 LLM 生成：检索结果页是**结构化的**（编号 / 文档号 / 标题 / 类型 / 来源 /
更新日期 / 摘要），相关度规则也是已知的表。让模型去编只会引入抖动；这里要训的是
「按规则读一页结果」，不是「编一页结果」。
"""

from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass
from typing import Any, Final

from jc.knobs import require_pool

# ---------- 文档类型 ----------

KINDS: Final[tuple[str, ...]] = ("guide", "policy", "api", "incident", "release", "faq")


def wrong_kind_of(kind: str) -> str:
    """输入：问题要求的文档类型。输出：另一个类型（表里下一个，回绕）。

    预期行为：**必定不等于入参**（`KINDS` 长度 > 1）——「类型不对」的干扰项靠它构造，
    返回同一个值等于把干扰项变成正解。
    """
    return KINDS[(KINDS.index(kind) + 1) % len(KINDS)]


# ---------- 来源权威 ----------

AUTHORITIES: Final[tuple[str, ...]] = ("official", "team", "community", "blog")
#: 权威高低（排序规则的第二档）；渲染出来的文本靠语言文件反查回这些键。
AUTHORITY_RANK: Final[dict[str, int]] = {"official": 3, "team": 2, "community": 1, "blog": 0}


# ---------- 主题 ----------


@dataclass(frozen=True)
class Topic:
    """一个检索主题：`name` 是键，`kind` 是**该主题的正确文档类型**。

    主题的中文名、要点词、指标名与取值都在语言文件里——所以「问题问的是哪个主题」
    这件事在 `spec` 里只是一个键，渲染与判别各自按语言文件翻译。
    """

    name: str
    kind: str


TOPICS: Final[tuple[Topic, ...]] = (
    Topic("refund", "guide"),
    Topic("login", "guide"),
    Topic("ratelimit", "api"),
    Topic("retention", "policy"),
    Topic("outage", "incident"),
    Topic("rollout", "release"),
    Topic("billing", "guide"),
    Topic("webhook", "api"),
)
TOPIC_NAMES: Final[tuple[str, ...]] = tuple(topic.name for topic in TOPICS)
TOPIC_BY_NAME: Final[dict[str, Topic]] = {topic.name: topic for topic in TOPICS}


def other_topic_of(topic: Topic) -> Topic:
    """输入：主题。输出：表里下一个主题（回绕）。

    预期行为：**必定不是同一个主题**——「跑题」的干扰项靠它构造。各主题的名字与要点词
    互不包含（由 `tests/workflows/search/test_locales.py` 锁住），所以跑题项的标题里
    绝不会混进问题主题的名字。
    """
    return TOPICS[(TOPICS.index(topic) + 1) % len(TOPICS)]


# ---------- 结果的角色 ----------

#: 权威的哨兵值：这一条的来源权威**跟随最佳结果那一档**（渲染时解析成具体取值）。
FOLLOW: Final[str] = "follow"


@dataclass(frozen=True)
class Flavor:
    """一页结果里某一条的**角色**（不是内容——内容在语言文件的模板里）。

    字段：`name` 键；`relevance` 相对问题的相关度（0~3，量表见题面）；
    `covers` 摘要覆盖了问题的几个要点（0~2）；`authority` 来源权威（`FOLLOW` = 跟随最佳）；
    `fresh` 更新日期是否达标；`snippet` 摘要模板的键。

    表里的 `relevance` / `covers` 是**配方的主张**，由测试对着渲染出来的文本重算一遍——
    两边对不上就报错，所以模板改歪了不会静默产出一条错数据。
    """

    name: str
    relevance: int
    covers: int
    authority: str
    fresh: bool
    snippet: str


FLAVORS: Final[tuple[Flavor, ...]] = (
    Flavor("full", 3, 2, FOLLOW, True, "full"),
    Flavor("partial", 3, 1, FOLLOW, True, "partial"),
    Flavor("empty", 3, 0, FOLLOW, True, "empty"),
    Flavor("twin", 3, 2, FOLLOW, True, "twin"),
    Flavor("old_hit", 3, 2, "community", True, "old_hit"),
    Flavor("stale", 2, 2, "community", False, "stale"),
    Flavor("weak_hit", 2, 2, "team", False, "weak_hit"),
    Flavor("wrong_kind", 1, 1, "community", True, "wrong_kind"),
    Flavor("other_topic", 0, 0, "blog", True, "other_topic"),
)
FLAVOR_BY_NAME: Final[dict[str, Flavor]] = {flavor.name: flavor for flavor in FLAVORS}

#: 能当「最该先看的那条」的角色。四档刚好把 `next_action` 的三个分支都走到：
#: `full` → 直接回答，`partial` / `empty` → 再检一轮，`stale` → 换数据源。
BEST_FLAVORS: Final[tuple[str, ...]] = ("full", "partial", "empty", "stale")

#: 最佳结果那一档的**具体**权威（`FOLLOW` 在最佳结果自身上要落地成取值）。
#: `full` 系列给「官方文档」、`stale` 给「团队内部」——这样同权威的 `twin` 靠日期分胜负、
#: 低一档的 `old_hit` 靠权威分胜负，排序规则的两档 tie-break 都被真数据走到。
BEST_AUTHORITY: Final[dict[str, str]] = {
    "full": "official",
    "partial": "official",
    "empty": "official",
    "stale": "team",
}

#: 一页结果的干扰项配方（不含最佳结果与被点名的那两条）。长度决定页里有多少条结果。
MIXES: Final[tuple[tuple[str, ...], ...]] = (
    ("wrong_kind", "old_hit", "twin", "stale", "other_topic"),
    ("wrong_kind", "weak_hit", "stale", "other_topic"),
    ("wrong_kind", "twin", "stale", "other_topic"),
    ("wrong_kind", "stale", "weak_hit", "other_topic"),
)


def mix_fits(mix: tuple[str, ...], best_flavor: str) -> bool:
    """输入：干扰项配方、最佳结果的角色。输出：这个组合成立吗。

    预期行为：配方里任何一条的相关度都**不得高于**最佳结果，否则最佳结果不是最优，
    `best_result` 的答案就不唯一（`stale` 那一档只有不含 `old_hit` / `twin` 的配方能用）。
    """
    top = FLAVOR_BY_NAME[best_flavor].relevance
    return all(FLAVOR_BY_NAME[name].relevance <= top for name in mix)


# ---------- 版式与被点名的位置 ----------

#: (题面点名的那两条, 最佳结果的位置)，都是 1 起的编号。三种版式互不相同，
#: 所以「答案位置」不是一个常量。
LAYOUTS: Final[tuple[tuple[tuple[int, int], int], ...]] = (
    ((1, 2), 3),
    ((1, 3), 4),
    ((3, 4), 5),
    ((2, 5), 6),
)

#: 题面点名「结果 N」时 N 的取值池（问相关度 / 覆盖要点 / 时效的那三族用它）。
PROBES: Final[tuple[int, ...]] = (1, 2, 3)

#: 被点名那两条的关系：矛盾 / 同源 / 又矛盾又同源 / 都不。
PAIR_KINDS: Final[tuple[str, ...]] = ("conflict", "same_source", "both", "none")

#: 被点名那两条的角色：摘要模板键 + 标题模板键 + 是否同一份文档 + 是否跑题。
#: 只有被点名的这两条会**提到指标**，所以整页里唯一可能「互相矛盾」的就是这一对。
PAIR_PLANS: Final[dict[str, tuple[str, str, str, str, bool, bool]]] = {
    #          摘要甲        摘要乙        标题甲        标题乙        同源    跑题
    "conflict": ("conflict_a", "conflict_b", "conflict", "conflict", False, False),
    "same_source": ("plain_a", "plain_b", "plain_a", "plain_b", True, False),
    "both": ("conflict_a", "conflict_b", "conflict", "conflict", True, False),
    "none": ("other_a", "other_b", "other_a", "other_b", False, True),
}


# ---------- case ----------


def layout_key(pair: tuple[int, int], best_at: int) -> str:
    """输入：被点名的那两条、最佳结果的位置。输出：`"1-2@3"` 这种稳定字符串。"""
    return f"{pair[0]}-{pair[1]}@{best_at}"


@dataclass(frozen=True)
class SearchCase:
    """一个 case：8 个维度的取值组合。"""

    topic: str
    best_flavor: str
    mix: tuple[str, ...]
    pair: tuple[int, int]
    best_at: int
    pair_kind: str
    probe: int
    query_style: int

    @property
    def count(self) -> int:
        """输入：无。输出：这一页有几条结果（= 最佳 1 + 被点名 2 + 干扰项）。"""
        return len(self.mix) + 3

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：{维度名: 取值}，顺序与声明一致（哈希分桶按它算，必须稳定）。

        预期行为：元组维度拼成 `"1-2"` / `"a+b"` 这种稳定字符串——`as_dict` 要进哈希载荷，
        元组的 `repr` 虽然也稳定，但写成字符串后报告里更好读。
        """
        return {
            "topic": self.topic,
            "best_flavor": self.best_flavor,
            "mix": "+".join(self.mix),
            "pair": f"{self.pair[0]}-{self.pair[1]}",
            "best_at": self.best_at,
            "pair_kind": self.pair_kind,
            "probe": self.probe,
            "query_style": self.query_style,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return (
            f"{self.topic}/{self.best_flavor}@{self.best_at}/{'+'.join(self.mix)}"
            f"/对{self.pair[0]}-{self.pair[1]}({self.pair_kind})/问{self.probe}/式{self.query_style}"
        )

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这个 case 里训练侧没见过的维度名（空 = 全部见过）。"""
        checks: tuple[tuple[str, Any, frozenset[Any]], ...] = (
            ("topic", self.topic, SEEN_TOPICS),
            ("best_flavor", self.best_flavor, SEEN_BEST_FLAVORS),
            ("mix", "+".join(self.mix), SEEN_MIXES),
            ("layout", layout_key(self.pair, self.best_at), SEEN_LAYOUTS),
            ("pair_kind", self.pair_kind, SEEN_PAIR_KINDS),
            ("query_style", self.query_style, SEEN_QUERY_STYLES),
        )
        return tuple(name for name, value, seen in checks if value not in seen)


#: 只改**题面**、不改页面的维度。素材 = 除这些以外的全部维度。
#: 加新维度时**默认算进素材**——这是保守方向：分桶更细，最多是切分不够省，不会漏页。
QUESTION_ONLY_DIMS: Final[frozenset[str]] = frozenset(("probe",))


def material_of(case: SearchCase) -> str:
    """输入：case。输出：素材指纹——决定**页面文本**的那几维拼成的串。

    预期行为：`probe` 只改题面点名的编号、不改页面，所以不进指纹；同一素材必得同一指纹。
    切分按它分桶（`workflow._slice`），同一页于是整组落在同一档——否则同分布侧的页会漏进
    验证 / OOD 侧（范式 §7「切分单位是组 / 实体 / 模板，不是单条样本」）。
    """
    return "|".join(
        f"{key}={value}"
        for key, value in case.as_dict().items()
        if key not in QUESTION_ONLY_DIMS
    )


@dataclass(frozen=True)
class SearchConfig:
    """一个切分的采样参数。

    字段：各维度的取值表（留出时收窄）、问法编号池、`ood_only` 是否只留「含没见过的取值」的 case。
    """

    split: str = "train"
    topics: tuple[str, ...] = TOPIC_NAMES
    best_flavors: tuple[str, ...] = BEST_FLAVORS
    mixes: tuple[tuple[str, ...], ...] = MIXES
    layouts: tuple[tuple[tuple[int, int], int], ...] = LAYOUTS
    probes: tuple[int, ...] = PROBES
    pair_kinds: tuple[str, ...] = PAIR_KINDS
    query_styles: tuple[int, ...] = (0, 1, 2)
    phrasings: tuple[int, ...] = (0, 1, 2)
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：坏旋钮在构造期就报人话，不拖到采样阶段。"""
        for field, values in (
            ("topics", self.topics),
            ("best_flavors", self.best_flavors),
            ("mixes", self.mixes),
            ("layouts", self.layouts),
            ("probes", self.probes),
            ("pair_kinds", self.pair_kinds),
            ("query_styles", self.query_styles),
            ("phrasings", self.phrasings),
        ):
            require_pool(field, values)
        unknown = [name for name in self.topics if name not in TOPIC_BY_NAME]
        if unknown:
            raise ValueError(f"topics 里有不存在的主题：{unknown}（可用：{sorted(TOPIC_BY_NAME)}）")
        unknown_best = [name for name in self.best_flavors if name not in BEST_AUTHORITY]
        if unknown_best:
            raise ValueError(
                f"best_flavors 里不能当最佳结果的角色：{unknown_best}（可用：{list(BEST_FLAVORS)}）"
            )
        unknown_kind = [name for name in self.pair_kinds if name not in PAIR_PLANS]
        if unknown_kind:
            raise ValueError(f"pair_kinds 里有不存在的关系：{unknown_kind}（可用：{list(PAIR_PLANS)}）")
        unknown_mix = [
            name for mix in self.mixes for name in mix if name not in FLAVOR_BY_NAME
        ]
        if unknown_mix:
            raise ValueError(f"mixes 里有不存在的角色：{sorted(set(unknown_mix))}")


def build_cases(config: SearchConfig) -> list[SearchCase]:
    """枚举 case 矩阵。

    输入：采样参数。输出：`SearchCase` 列表（顺序稳定、可复算）。
    预期行为：全枚举后剔掉三类不成立的组合——「干扰项比最佳结果还相关」、
    「最佳结果落在被点名的位置上」、「位置编号超出这一页的条数」；`ood_only` 时只留
    「至少含一个训练侧没见过的取值」的 case。
    """
    out: list[SearchCase] = []
    for topic in config.topics:
        for best_flavor in config.best_flavors:
            for mix in config.mixes:
                if not mix_fits(mix, best_flavor):
                    continue
                count = len(mix) + 3
                for pair, best_at in config.layouts:
                    if best_at in pair or max(*pair, best_at) > count:
                        continue
                    for pair_kind in config.pair_kinds:
                        for probe in config.probes:
                            if probe > count:
                                continue
                            for style in config.query_styles:
                                case = SearchCase(
                                    topic=topic,
                                    best_flavor=best_flavor,
                                    mix=mix,
                                    pair=pair,
                                    best_at=best_at,
                                    pair_kind=pair_kind,
                                    probe=probe,
                                    query_style=style,
                                )
                                if config.ood_only and not case.unseen_dims():
                                    continue
                                out.append(case)
    return out


# ---------- spec（渲染前的真值） ----------

#: 问题的时效要求（更新日期不早于）。所有 case 共用一条，所以它不是维度；
#: 「新鲜 / 过期」的多样性由各条结果的更新日期承担。
SINCE: Final[str] = "2024-01-01"

#: 各角色相对 `since` 的更新日期偏移（月）。`is_best` 那一档单独给。
#: 最佳结果在同权威的一组里**必须最新**（`twin` 与它同权威，靠日期分胜负）；
#: `old_hit` 比它新但权威低一档，靠权威分胜负——排序规则的两档 tie-break 都被走到。
OFFSETS: Final[dict[str, int]] = {
    "twin": 4,
    "old_hit": 15,
    "weak_hit": -7,
    "wrong_kind": 8,
    "other_topic": 6,
    "stale": -3,
}

#: 被点名那两条的更新日期偏移（甲、乙）。两条**必须不同**——`both` 那一档里它们是
#: 同一份文档、同一个标题，日期再一样，`best_result` 的两个选项就长得一模一样了。
PAIR_OFFSETS: Final[tuple[int, int]] = (-3, -5)
#: 跑题的那一对（`none`）的偏移：两条都是新鲜结果，但主题不对，所以相关度照样是 0。
OFF_TOPIC_OFFSETS: Final[tuple[int, int]] = (6, 8)


@dataclass(frozen=True)
class ResultRow:
    """一页里的一条结果：全是**键**与日期，标题 / 摘要的文案在语言文件里。"""

    index: int
    doc_id: str
    topic: str
    title: str
    kind: str
    authority: str
    updated: str
    snippet: str
    flavor: str


@dataclass(frozen=True)
class SearchSpec:
    """一条样本的真值规格：问题要点、时效要求、整页结果。"""

    case: SearchCase
    split: str
    seq: int
    phrasing: int
    topic: Topic
    want_kind: str
    since: str
    results: tuple[ResultRow, ...]

    @property
    def best_index(self) -> int:
        """输入：无。输出：最佳结果的编号。预期行为：就是 case 里定下的那个位置。"""
        return self.case.best_at


def shift_months(day: dt.date, months: int) -> dt.date:
    """输入：日期、月份偏移（可负）。输出：偏移后的日期。

    预期行为：按月推、日号夹到该月最后一天（`date(2024,1,31)` 推一个月是 2 月 29 日），
    不借 `timedelta` 硬加 30 天——那样推出来的月份会随大小月漂。
    """
    total = day.year * 12 + (day.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    last = (dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)).day
    return dt.date(year, month, min(day.day, last))


def _day_jitter(case: SearchCase) -> int:
    """输入：case。输出：0~24 的天数抖动。

    预期行为：**整页共用一个抖动**（所有结果一起平移），所以各条之间的先后顺序不变；
    但同一角色在不同 case 里的日期不一样，考生没法靠记「日期 → 新鲜与否」蒙。
    不用内置 `hash()`（它对字符串每进程随机化），否则同一 case 两次生成会得到不同日期。
    """
    payload = "|".join(f"{key}={value}" for key, value in case.as_dict().items())
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:4], 16) % 25


def _date(since: str, months: int, jitter: int) -> str:
    """输入：基准日期、月份偏移、天数抖动。输出：ISO 日期字符串。"""
    return (shift_months(dt.date.fromisoformat(since), months) + dt.timedelta(days=jitter)).isoformat()


def _best_authority(best_flavor: str) -> str:
    """输入：最佳结果的角色。输出：它的来源权威（`FOLLOW` 在这里落地）。"""
    try:
        return BEST_AUTHORITY[best_flavor]
    except KeyError as exc:
        raise KeyError(f"角色 {best_flavor!r} 不能当最佳结果（可用：{list(BEST_AUTHORITY)}）") from exc


def _resolve_authority(authority: str, best_flavor: str) -> str:
    """输入：角色表里的权威、最佳结果的角色。输出：具体权威。

    预期行为：`FOLLOW` 解析成最佳结果那一档——`twin` 靠这一手与最佳结果同权威，
    于是「更新日期」那一档 tie-break 才有数据走到。
    """
    return _best_authority(best_flavor) if authority == FOLLOW else authority


def sample_spec(
    rng: object, case: SearchCase, split: str, seq: int, phrasing: int
) -> SearchSpec:
    """输入：RNG（不用，留着满足签名）、case、切分名、样本序号、问法编号。输出：真值规格。

    预期行为：整页结果**完全由 case 决定**，不掷骰子——所以同一 case 两次生成逐字节一致，
    「两次生成一样」这条能直接查。版式先摆好最佳结果与被点名的那两条，剩下的位置按配方
    依次填干扰项。
    """
    del rng  # 检索页不需要随机：版式与配方已经把多样性给足了
    topic = TOPIC_BY_NAME[case.topic]
    other = other_topic_of(topic)
    best_authority = _best_authority(case.best_flavor)
    snippet_a, snippet_b, title_a, title_b, same_doc, off_topic = PAIR_PLANS[case.pair_kind]
    jitter = _day_jitter(case)

    #: {位置: (角色, 摘要键, 标题键, 主题键, 文档类型, 权威, 偏移)}
    plan: dict[int, tuple[str, str, str, str, str, str, int]] = {}
    best = FLAVOR_BY_NAME[case.best_flavor]
    plan[case.best_at] = (
        best.name,
        best.snippet,
        best.name,
        topic.name,
        topic.kind,
        best_authority,
        12 if best.fresh else -1,
    )
    rest = [index for index in range(1, case.count + 1) if index not in (*case.pair, case.best_at)]
    for index, flavor_name in zip(rest, case.mix, strict=True):
        flavor = FLAVOR_BY_NAME[flavor_name]
        #: 「跑题」的换成别的主题，「类型不对」的换成别的类型，其余沿用问题主题与类型。
        row_topic = other if flavor.name == "other_topic" else topic
        row_kind = (
            other.kind
            if flavor.name == "other_topic"
            else wrong_kind_of(topic.kind)
            if flavor.name == "wrong_kind"
            else topic.kind
        )
        plan[index] = (
            flavor.name,
            flavor.snippet,
            flavor.name,
            row_topic.name,
            row_kind,
            _resolve_authority(flavor.authority, case.best_flavor),
            OFFSETS[flavor.name],
        )
    first, second = case.pair
    pair_offsets = OFF_TOPIC_OFFSETS if off_topic else PAIR_OFFSETS
    for index, snippet_key, title_key, months in (
        (first, snippet_a, title_a, pair_offsets[0]),
        (second, snippet_b, title_b, pair_offsets[1]),
    ):
        plan[index] = (
            f"pair_{case.pair_kind}",
            snippet_key,
            title_key,
            other.name if off_topic else topic.name,
            other.kind if off_topic else topic.kind,
            "blog" if off_topic else "community",
            months,
        )

    rows: list[ResultRow] = []
    for index in range(1, case.count + 1):
        flavor_name, snippet_key, title_key, topic_name, kind, authority, months = plan[index]
        doc_topic = TOPIC_BY_NAME[topic_name]
        # 同源的那两条共用一个文档号（同一份文档的两次命中），其余各用各的。
        doc_id = f"KB-{2000 + TOPIC_NAMES.index(doc_topic.name) * 100 + (first if same_doc and index == second else index)}"
        rows.append(
            ResultRow(
                index=index,
                doc_id=doc_id,
                topic=topic_name,
                title=title_key,
                kind=kind,
                authority=authority,
                updated=_date(SINCE, months, jitter),
                snippet=snippet_key,
                flavor=flavor_name,
            )
        )
    return SearchSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        topic=topic,
        want_kind=topic.kind,
        since=SINCE,
        results=tuple(rows),
    )


# ---------- 规则引擎（构造侧与判别侧共用） ----------


@dataclass(frozen=True)
class TruthRow:
    """一条结果的**可判定事实**——构造侧与判别侧共用的中间形态。

    字段：`index` 编号；`doc_id` 文档号；`topic_match` 标题讲的正是问题问的主题；
    `kind_match` 文档类型正是问题要求的类型；`fresh` 更新日期达标；`authority` 来源权威**键**；
    `updated` 更新日期（ISO 字符串，可直接比大小）；`snippet` 摘要**文本**（覆盖度靠数词）。
    """

    index: int
    doc_id: str
    topic_match: bool
    kind_match: bool
    fresh: bool
    authority: str
    updated: str
    snippet: str


def relevance_of(row: TruthRow) -> int:
    """输入：一条事实。输出：相关度 0~3。

    预期行为：就是题面那张量表——主题不对 0、类型不对 1、时效不达标 2、全对 3。
    **只看标题与元数据**，不看摘要：摘要是「覆盖了几个要点」那一族的事。
    """
    if not row.topic_match:
        return 0
    if not row.kind_match:
        return 1
    if not row.fresh:
        return 2
    return 3


def covers_of(row: TruthRow, points: tuple[str, ...]) -> int:
    """输入：一条事实、问题的要点词。输出：摘要覆盖了几个要点（原样子串计数）。"""
    return sum(1 for point in points if point in row.snippet)


def best_of(rows: tuple[TruthRow, ...]) -> TruthRow:
    """输入：整页事实。输出：最该先看的那条。

    预期行为：按题面写的排序规则取最大值——相关度 → 来源权威 → 更新日期晚者 → 编号小者。
    构造侧保证有**唯一**最优（配方里不允许出现比最佳结果更相关的干扰项，
    见 `mix_fits`；同权威的竞争者靠日期分胜负，见 `OFFSETS`）。
    """
    if not rows:
        raise ValueError("一页里一条结果都没有")
    return max(
        rows,
        key=lambda row: (relevance_of(row), AUTHORITY_RANK[row.authority], row.updated, -row.index),
    )


def action_of(rows: tuple[TruthRow, ...], points: tuple[str, ...]) -> str:
    """输入：整页事实、问题的要点词。输出：下一步动作（answer / search_more / switch_source）。

    预期行为：最相关那条**相关度 3 且要点全覆盖** → 直接回答；相关度 3 但要点不全 →
    换关键词再检一轮；相关度 ≤ 2（连类型或时效都不对）→ 换数据源。
    """
    top = best_of(rows)
    if relevance_of(top) == 3 and covers_of(top, points) == len(points):
        return "answer"
    if relevance_of(top) == 3:
        return "search_more"
    return "switch_source"


def conflict_of(left: TruthRow, right: TruthRow, metric: str, values: tuple[str, str]) -> bool:
    """输入：两条事实、问题的指标名与两个取值。输出：两条是否互相矛盾。

    预期行为：两条**都提到**这个指标，而且各自提到的取值集合不同。整页里只有被点名的
    那两条会提到指标，所以这是这一页里唯一的矛盾源（渲染模板由测试锁住）。
    """
    if metric not in left.snippet or metric not in right.snippet:
        return False
    picked = {value for value in values if value in left.snippet}
    return picked != {value for value in values if value in right.snippet}


def spec_to_dict(spec: SearchSpec) -> dict[str, Any]:
    """输入：规格。输出：可 JSON 序列化的摘要（写进 manifest 与调试，不进样本行）。"""
    return {
        "case": spec.case.describe(),
        "unseen_dims": list(spec.case.unseen_dims()),
        "split": spec.split,
        "seq": spec.seq,
        "topic": spec.topic.name,
        "want_kind": spec.want_kind,
        "since": spec.since,
        "best_at": spec.best_index,
        "results": [
            {
                "index": row.index,
                "doc_id": row.doc_id,
                "title": row.title,
                "kind": row.kind,
                "authority": row.authority,
                "updated": row.updated,
                "snippet": row.snippet,
                "flavor": row.flavor,
            }
            for row in spec.results
        ],
    }


# ---------- 值留出（训练侧见过哪些取值） ----------

#: 训练侧见过的主题：`webhook` 整个留出（主题名、要点词、指标名一起没见过）。
SEEN_TOPICS: Final[frozenset[str]] = frozenset(
    topic.name for topic in TOPICS if topic.name != "webhook"
)
#: 训练侧见过的最佳结果角色：`empty`（有命中但没有摘要）留出。
SEEN_BEST_FLAVORS: Final[frozenset[str]] = frozenset(("full", "partial", "stale"))
#: 训练侧见过的配方：留出第四条（把 `weak_hit` 与 `stale` 换个顺序）。
SEEN_MIXES: Final[frozenset[str]] = frozenset("+".join(mix) for mix in MIXES[:3])
#: 训练侧见过的版式：四种都见过——版式只决定「谁在第几位」，留出它换来的 OOD
#: 与「同一批数据换个位置」几乎等价，信息量太低，不如把预算留给主题与角色。
SEEN_LAYOUTS: Final[frozenset[str]] = frozenset(
    {layout_key(pair, best_at) for pair, best_at in LAYOUTS}
)
#: 训练侧见过的那两条的关系：`both`（又矛盾又同源）留出。
SEEN_PAIR_KINDS: Final[frozenset[str]] = frozenset(("conflict", "same_source", "none"))
#: 训练侧见过的问法：三种都见过——问法只是同一句话的措辞，留出它不算「没见过的事实」。
SEEN_QUERY_STYLES: Final[frozenset[int]] = frozenset((0, 1, 2))

#: {`"a+b"`: 配方}、{`"1-2@3"`: (被点名的那两条, 最佳位置)}——`unseen_dims` 按字符串比，
#: 而 `ID_CONFIG` 要按「见过的取值」把取值表重建回来，所以需要这两张反查表。
MIX_BY_KEY: Final[dict[str, tuple[str, ...]]] = {"+".join(mix): mix for mix in MIXES}
LAYOUT_BY_KEY: Final[dict[str, tuple[tuple[int, int], int]]] = {
    layout_key(pair, best_at): (pair, best_at) for pair, best_at in LAYOUTS
}
