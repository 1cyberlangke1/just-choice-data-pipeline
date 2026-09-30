"""广告元素判定（adblock）的特征层：广告形状判据 + 页面类型 + 命中标签（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条「这个元素是不是广告、该怎么处置」的样本由两个**注入特征**描述
——`hits` 该命中哪几条广告形状判据、`surface` 这个元素长在哪种页面上（文章正文 / 信息流 / 视频页）。
**标签先行**：先定这两个取值，再由本模块的纯函数算出结论。

**为什么判据写成机械短语表**：广告元素在真实页面里是有固定形状的——class / id 里带广告位词、
元素文字里带赞助标注、引用了第三方广告或追踪资源、做成 1×1 的隐形壳。把这些形状写成正则之后，
「命中哪几条」就是**算**出来的，模型碰不到——它只负责写一段读得通的 HTML 片段。判据表由代码写、
内容由模型写（范式 §9.22 / §9.23 同一条路子）。

**为什么四条判据分别落在四个不同的位置**（范式 §9.46 的教训）：`A1` 只看 `class` / `id` 的**取值**、
`A2` 只看标签之间的**文字**、`A3` 只看 `src` / `href` / `data-*` 这类**资源与投放属性**、`A4` 只看
`style` 与尺寸属性。四条判据各占一处，不会互相遮蔽——`turntake` 上量过一次「两条判据落在同一段
文字上」的代价（写手与读者一起丢内层那条），这里从设计上避开。

**为什么要这一族**：社区里 `typesafe-adblock`（★58）明写「让 Jev 判断 DOM 元素是否为广告并将其
移除」，`ad-radar`（★13）「识别和折叠社交平台上的广告」，`jev-shield`（★2）「按语义阻止原生广告、
赞助信息流卡片和视频广告」——共同点是**每个元素都要一次「这是不是广告」的判断，而且判断完立刻
要落地一个动作**（保留 / 折叠 / 隐藏 / 移除）。

**与 `guardrail` / `mail` / `toolgate` 的区别**：那三家判的是**人际内容**该不该放行（联系方式、
外链、发件人身份、工具策略），本家判的是**页面元素**是不是广告位——判据空间、处置语义与素材形态
（HTML 片段）都不一样。

**`surface` 是这一族的留出维**：它**只影响渲染、不影响标签**——同一份命中组合写在文章正文、信息流
还是视频页里，判据算出来的答案一个字都不差。训练侧只见过 `article` / `feed`，留出侧是 `video`，
用来验「读法不靠页面先验」。

本模块有**两条独立路径**：`label_for` 从计划推标签（构造侧），`observed_labels` 从文本重推标签
（审计侧，走 `hit_set` 的正则）。两者对不上就说明渲染没兑现计划。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Final

from jc.contract import parse_hits
from jc.cover import cover_dims, near_miss_combos
from jc.jsontypes import JsonValue

#: 四条「这是不是广告位」判据的编号（**结构键**，不是文案：判据名与判据原文全在 `locales/`）。
TAG_IDS: Final[tuple[str, ...]] = ("A1", "A2", "A3", "A4")

#: `A1` 的广告位词表。**长的排在前面**：正则的 `|` 是按顺序试的，`advert` 排在 `ad` 前面才不会
#: 在 `advertisement` 里只匹配到 `ad` 就停。
_A1_WORDS: Final[str] = (
    "advertisement|advertorial|advert|adslot|adsbygoogle|ads|ad|banner|sponsor(?:ed|ship)?|promoted|dfp"
)

#: 判据编号 → 判据。**这是全 workflow 唯一的判据实现**：构造侧、渲染期、判别器、审计共用。
#: 每条都只认字面，而且各占一处（见模块文档）：
#: - `A1` **class / id 的取值**里出现独立的广告位词（`ad` 前后是引号 / 空格 / 连字符 / 下划线 / 冒号
#:   才算一个词——`header`、`download`、`badge`、`read-more` 里的 `ad` 不算）；
#: - `A2` **标签之间的文字**里出现赞助标注（`>` 之后、`<` 之前那一块才是文字，属性里的词不算）；
#: - `A3` 引用了第三方广告或追踪资源（广告域名 / `data-ad-*` 这类投放属性 / 追踪像素与信标）；
#: - `A4` 元素被做成**隐形或极小**（1px 尺寸 / `display:none` / `visibility:hidden` / `opacity:0` /
#:   `aria-hidden="true"`）。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "A1": re.compile(
        r"(?<![\w\-])(?:class|id)\s*=\s*[\"'][^\"']*(?<![\w])(?:"
        + _A1_WORDS
        + r")(?![\w])",
        re.IGNORECASE,
    ),
    "A2": re.compile(
        r"(?:^|>)[^<>]*(?:赞助|推广|广告|品牌合作|商业内容|\bsponsored\b|\bpromoted\b"
        r"|\bpaid partnership\b|\badvertisement\b|\badvert\b)",
        re.IGNORECASE,
    ),
    "A3": re.compile(
        r"doubleclick|googlesyndication|googleadservices|adsystem|adnxs|taboola|outbrain|criteo"
        r"|adform|adserver|adservice|data-ad(?:[-_][a-z0-9]+)*\s*=|\bdata-google-query-id\b"
        r"|tracking[-_ ]?pixel|\bbeacon\b|\btracking\b",
        re.IGNORECASE,
    ),
    "A4": re.compile(
        r"(?:width|height)\s*[:=]\s*[\"']?\s*1(?:px)?\s*[\"']?(?![\w])"
        r"|display\s*:\s*none|visibility\s*:\s*hidden|opacity\s*:\s*0(?:\.0+)?(?![\d])"
        r"|aria-hidden\s*=\s*[\"']true[\"']",
        re.IGNORECASE,
    ),
}

#: 页面类型（**结构键**，文案在 `locales/` 的 `[surface.*]`）。
#:
#: 这一维**只影响渲染、不影响标签**：同一份命中组合写在文章正文、信息流还是视频页里，判据算出来的
#: 答案一个字都不差。它正是 test_ood 的留出维——训练侧只见过 `article` / `feed`，留出侧是 `video`，
#: 用来验「读法不靠页面先验」（范式 §9.25 那条「只影响渲染的那一维」）。
SURFACE_ARTICLE: Final[str] = "article"
SURFACE_FEED: Final[str] = "feed"
SURFACE_VIDEO: Final[str] = "video"
SURFACES: Final[tuple[str, ...]] = (SURFACE_ARTICLE, SURFACE_FEED, SURFACE_VIDEO)

#: 处置档位（**结构键**，文案全在 `locales/`）。阈值是 0 条 / 1 条 / 2 条 / 3 条及以上——语义是
#: 「像广告的地方越多，处置越重」：0 条保留、1 条折叠、2 条隐藏、3 条及以上移除。四档而不是三档的
#: 理由与 `logtriage` 相同：三档会让「隐藏」一档按构造占 11/16（69%），均衡闸会把整批标成失衡。
ACTS: Final[tuple[str, ...]] = ("keep", "collapse", "hide", "remove")

#: 片段的字符数上下限。太短装不下一个像样的元素（标签 + 属性 + 文字），太长就只是一整段页面。
MIN_CONTENT_CHARS: Final[int] = 60

#: 上限。一个元素加它的直接上下文，再长就不是「一个元素」了。
MAX_CONTENT_CHARS: Final[int] = 600


def _all_hit_sets() -> tuple[str, ...]:
    """输入：无。输出：四条判据的全部 16 种命中组合（`""` 是「一条都没命中」）。"""
    out: list[str] = []
    for size in range(len(TAG_IDS) + 1):
        for combo in combinations(TAG_IDS, size):
            out.append("+".join(combo))
    return tuple(out)


#: 全部命中组合（`cover_dims` 的取值表）。
HIT_SETS: Final[tuple[str, ...]] = _all_hit_sets()

#: 维度表。全空间 16 × 3 = 48 组。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {"hits": HIT_SETS, "surface": SURFACES}


def hit_set(text: str) -> tuple[str, ...]:
    """输入：一段 HTML 片段。输出：命中的判据编号（按 `TAG_IDS` 顺序）。"""
    return tuple(tag for tag in TAG_IDS if PATTERNS[tag].search(text))


def hits_of(text: str) -> str:
    """输入：一段 HTML 片段。输出：命中组合的规范串（`"A1+A4"` / `""`）。

    预期行为：与 `hit_set` 是同一件事的两种表示；`hits_of` 给比较用（`render.check_draft` 拿它
    与计划逐字比），`hit_set` 给取元素用。
    """
    return "+".join(hit_set(text))


def act_level(count: int) -> str:
    """输入：命中条数。输出：处置档位名。

    预期行为：0→keep、1→collapse、2→hide、3 及以上→remove（最后一档吃掉所有更大的数）。
    """
    return ACTS[min(count, len(ACTS) - 1)]


@dataclass(frozen=True)
class AdBlockCase:
    """一个注入特征组合：该命中哪几条判据 + 元素长在哪种页面上。

    字段：`hits` 命中组合的规范串；`surface` 页面类型（`SURFACES` 里的一个）。
    """

    hits: str
    surface: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"hits": self.hits, "surface": self.surface}

    def fingerprint(self) -> str:
        """输入：无。输出：case 指纹。预期行为：同一 case 任何进程都得到同一串。"""
        return "|".join(f"{key}={value}" for key, value in self.as_dict().items())

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.hits or '无命中'}-{self.surface}"

    def tags(self) -> tuple[str, ...]:
        """输入：无。输出：该命中的判据编号元组。"""
        return parse_hits(self.hits)


@dataclass(frozen=True)
class AdBlockPlan:
    """一条样本的判定计划：命中哪几条 + 组合题的三个干扰项。

    字段：`hits` 该命中的判据；`distractors` 组合题的三个干扰组合（与 `hits` 不同、彼此也不同，
    且都取「与真值只差一两处」的邻近组合，题目才有区分度）。
    """

    hits: tuple[str, ...]
    distractors: tuple[tuple[str, ...], ...]

    @property
    def count(self) -> int:
        """输入：无。输出：命中条数。"""
        return len(self.hits)


def plan_for(case: AdBlockCase, phrasing: int) -> AdBlockPlan:
    """输入：case、问法编号。输出：判定计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里都
    得到同一份计划；两种问法会抽到不同的干扰组合，不会渲染出结构相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    return AdBlockPlan(
        hits=case.tags(), distractors=near_miss_combos(TAG_IDS, case.tags(), random.Random(seed))
    )


@dataclass(frozen=True)
class AdBlockLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`hits` 命中的判据编号（按 `TAG_IDS` 顺序）。
    """

    hits: tuple[str, ...]

    @property
    def count(self) -> int:
        """输入：无。输出：命中条数。"""
        return len(self.hits)

    def contains(self, tag: str) -> bool:
        """输入：判据编号。输出：这一条是否命中。"""
        return tag in self.hits

    @property
    def act(self) -> str:
        """输入：无。输出：按阈值算出来的档位名（`features.act_level`）。"""
        return act_level(self.count)


def label_for(plan: AdBlockPlan) -> AdBlockLabels:
    """输入：判定计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`render.check_draft` 负责保证
    文本真的兑现这份计划（按字面算出来的命中组合与计划逐条相同）。
    """
    return AdBlockLabels(hits=plan.hits)


def observed_labels(text: str) -> AdBlockLabels:
    """输入：一段 HTML 片段。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认文本、只认字面。两者对不上就说明渲染没兑现
    计划，审计当场就能看见。
    """
    return AdBlockLabels(hits=hit_set(text))


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[AdBlockCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`AdBlockCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [AdBlockCase(hits=str(row["hits"]), surface=str(row["surface"])) for row in rows]
