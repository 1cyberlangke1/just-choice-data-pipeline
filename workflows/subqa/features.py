"""字幕对质检（subqa）的特征层：四块结构的判据 + 素材类型 + 命中标签（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条「这一对字幕能不能直接发」的样本由两个**注入特征**描述
——`hits` 该命中哪几条判据、`track` 这对字幕来自哪种素材（电影 / 剧集 / 讲座）。**标签先行**：
先定这两个取值，再由本模块的纯函数算出结论。

**为什么判据写成机械短语表**：字幕对质检在真实流水线里有固定形状——原文那一行根本不是对白
（`♪` 或 `[MUSIC]` 这类音效占位符）、译文那一行还留着没清掉的标记（`<i>` / `&amp;` /
`{\an8}`）、时间码两侧一模一样（零时长）、译者备注里留着「待确认」这类没写完的话。把这些形状
写成正则之后，「命中哪几条」就是**算**出来的，模型碰不到——它只负责写一屏读得通的双语字幕对。

**为什么四条判据各占一整行**（范式 §9.51）：一条字幕对是**分行**写的（原文 / 译文 / 时间码 /
备注），四条判据各自钉在自己那一行上——`S1` 只看 `Source:` 行、`S2` 只看 `Target:` 行、
`S3` 只看 `Timing:` 行、`S4` 只看 `Note:` 行。`adblock` 上量过一次「两条判据落在同一段文字上」
的代价（写手与读者一起丢内层那条），这里从设计上避开：**判据之间连一行都不共用**。

**为什么要这一族**：社区里 `transcript-lens`（★16）逐句分析字幕，`jev-subtitle-translator`
（★1）「用 Noul 判断标记需要人工复核的字幕对，但不改写字幕」。共同点是**每一对字幕都要一次
「能不能直接发」的判断，判断完立刻要落地一个动作**（直接发 / 抽查 / 逐句复核 / 打回重译）。

**与 `citation` / `autoformat` / `lint` 的区别**：`citation` 判的是文档正文里的引用站不站得住、
`autoformat` 判的是结构恢复得对不对、`lint` 判的是代码片段按行级规则干不干净——三者判的都是
**单语材料**。本家判的是**一对原文 / 译文**，判的是这一对的**卫生**（原文该不该翻、译文有没有
留标记、时间码合不合法、备注写没写完），素材形态与判据空间都不一样。

**`track` 是这一族的留出维**：它**只影响渲染、不影响标签**——同一份命中组合写在电影、剧集还是
讲座的字幕里，判据算出来的答案一个字都不差（四条判据认的全是符号、标记与时间码，不是题材）。
训练侧只见过 `movie` / `series`，留出侧是 `lecture`。

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

#: 四条「这一对字幕能不能直接发」判据的编号（**结构键**，不是文案：判据名与判据原文全在 `locales/`）。
TAG_IDS: Final[tuple[str, ...]] = ("S1", "S2", "S3", "S4")

#: 一条字幕对的**结构键**：这四个英文键名与它们的冒号是 state 格式的一部分，**不进语言文件**——
#: 渲染、校验、解析、判别器共用这一份清单，四边不可能漂移。键名用英文是因为它们本来就是格式的
#: 一部分（与 `.srt` 那种字幕文件同一个性质），只有 `note` 与 `target` 两行的取值是自然语言。
KEY_LINES: Final[tuple[str, ...]] = ("source", "target", "timing", "note")

#: 判据编号 → 判据。**这是全 workflow 唯一的判据实现**：构造侧、渲染期、判别器、审计共用。
#: 每条都只认字面，而且**各占一整行**（见模块文档）：
#: - `S1` `Source:` 行里有音效或占位符（成对的方括号里 2~20 个字符，或一个音符符号 `♪`）——
#:   这一行根本不该按对白翻，得人工看一眼；
#: - `S2` `Target:` 行里留着没清掉的标记（`<i>` 这类尖括号标签、`&amp;` 这类实体、`{\an8}`
#:   这类花括号指令）——说明这一对是从带标记的源文件里直出的；
#: - `S3` `Timing:` 行里 `-->` 两侧的时间码**一模一样**（零时长，字幕会在屏幕上闪一下就没了）；
#: - `S4` `Note:` 行里留着没写完的痕迹（TODO / FIXME / 待确认 / 待定 / 存疑 / 需要复核 / 核对）。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "S1": re.compile(r"^[^\n]*\bsource\s*:[^\n]*?(?:\[[^\]\n]{2,20}\]|♪)", re.IGNORECASE | re.MULTILINE),
    "S2": re.compile(
        r"^[^\n]*\btarget\s*:[^\n]*?(?:</?[a-zA-Z][^>\n]{0,20}>|&[a-zA-Z]{2,8};|\{\\[^}\n]{1,12}\})",
        re.IGNORECASE | re.MULTILINE,
    ),
    "S3": re.compile(
        r"^[^\n]*\btiming\s*:[^\n]*?(\d{1,2}:\d{2}:\d{2}[,.]\d{3})[^\n]*-->[^\n]*?\1(?![\d])",
        re.IGNORECASE | re.MULTILINE,
    ),
    "S4": re.compile(
        r"^[^\n]*\bnote\s*:[^\n]*?(?:\bTODO\b|\bFIXME\b|待确认|待定|存疑|需要复核|核对)",
        re.IGNORECASE | re.MULTILINE,
    ),
}

#: 素材类型（**结构键**，文案在 `locales/` 的 `[track.*]`）。
#:
#: 这一维**只影响渲染、不影响标签**：同一份命中组合写在电影、剧集还是讲座的字幕里，判据算出来的
#: 答案一个字都不差——四条判据认的是音效符号、标记、时间码与备注词，与题材无关。它正是
#: test_ood 的留出维：训练侧只见过 `movie` / `series`，留出侧是 `lecture`（范式 §9.25 那条
#: 「只影响渲染的那一维」）。
TRACK_MOVIE: Final[str] = "movie"
TRACK_SERIES: Final[str] = "series"
TRACK_LECTURE: Final[str] = "lecture"
TRACKS: Final[tuple[str, ...]] = (TRACK_MOVIE, TRACK_SERIES, TRACK_LECTURE)

#: 处置档位（**结构键**，文案全在 `locales/`）。阈值是 0 条 / 1 条 / 2 条 / 3 条及以上——语义是
#: 「这一对里可疑的地方越多，下一步要花的力气越大」：0 条直接发、1 条抽查一遍、2 条逐句人工复核、
#: 3 条及以上打回重译。四档而不是三档的理由与 `logtriage` 相同：三档会让最后一档按构造占
#: 11/16（69%），均衡闸会把整批标成失衡。四档把占比摊成 1/4/6/5。
ACTS: Final[tuple[str, ...]] = ("ship", "spot_check", "review", "reject")

#: 一条字幕对的字符数上下限。太短装不下四行（三行取值 + 一句备注），太长就不是「一屏」了。
MIN_CONTENT_CHARS: Final[int] = 80

#: 上限。四行字幕加各自的取值，再长就不是一对字幕了。
MAX_CONTENT_CHARS: Final[int] = 420


#: 写文本时的「开始时间码」池：spec 里存的是**取值本身**（`HH:MM:SS,mmm`，与语言无关）。
#:
#: **来由**（范式 §9.61）：`Timing:` 是这一族唯一一个**符号类**自由字段——提示词里没有任何变量
#: 去碰它，模型就落回自己的先验：`sq-10c-zh` 的 9 条产物里 4 条逐字写着同一个
#: `00:01:23,456 --> 00:01:26,789`。给每条样本钉一个开始时间码，整批就不会再共用一串数字。
#:
#: 只钉**开始**那一侧：结束那一侧要么按 `S3` 的判据写成一模一样（零时长那一档）、要么自己编一个
#: 更晚的——两种写法都留着自由度，所以这个旋钮不与任何判据打架。池子里的值由
#: `test_features` 按 `TIMECODE_SHAPE` 逐条验过形状（形状写错的话，`S3` 的正则就认不出来了）。
TIMECODE_HINTS: Final[tuple[str, ...]] = (
    "00:00:03,480",
    "00:00:41,915",
    "00:01:12,067",
    "00:02:38,504",
    "00:04:19,773",
    "00:06:07,221",
    "00:08:55,608",
    "00:11:26,039",
    "00:15:44,382",
    "00:21:03,917",
    "00:29:37,156",
    "00:43:18,600",
)

#: 时间码的合法形状（与 `PATTERNS["S3"]` 认的那一串同一个形状）。
TIMECODE_SHAPE: Final[re.Pattern[str]] = re.compile(r"^\d{1,2}:\d{2}:\d{2},\d{3}$")


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
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {"hits": HIT_SETS, "track": TRACKS}


def hit_set(text: str) -> tuple[str, ...]:
    """输入：一条字幕对。输出：命中的判据编号（按 `TAG_IDS` 顺序）。"""
    return tuple(tag for tag in TAG_IDS if PATTERNS[tag].search(text))


def hits_of(text: str) -> str:
    """输入：一条字幕对。输出：命中组合的规范串（`"S1+S4"` / `""`）。

    预期行为：与 `hit_set` 是同一件事的两种表示；`hits_of` 给比较用（`render.check_draft` 拿它
    与计划逐字比），`hit_set` 给取元素用。
    """
    return "+".join(hit_set(text))


def missing_keys(text: str) -> tuple[str, ...]:
    """输入：一条字幕对。输出：缺掉的**结构键**（四个键都在就给空元组）。

    预期行为：键必须**单独占一行**、后面跟冒号（`^\\s*键\\s*:`）。为什么要单占一行：判据是钉在
    行上的（见 `PATTERNS`），键不单独成行的话那几条正则连定位都定不到。渲染期用它拦「少写一行」，
    审计期 `FrameJudge` 再用同一份清单复核一遍。
    """
    lowered = text.lower()
    return tuple(key for key in KEY_LINES if not re.search(rf"(?m)^\s*{key}\s*:", lowered))


def line_value(text: str, key: str) -> str:
    """输入：一条字幕对、结构键。输出：那一行冒号后面的内容（没有那一行就给空串）。

    预期行为：只取**第一个**匹配到的键行；行内的空白原样保留（取值里可能有空格）。
    """
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*:(.*)$", text)
    return match.group(1).strip() if match else ""


def act_level(count: int) -> str:
    """输入：命中条数。输出：处置档位名。

    预期行为：0→ship、1→spot_check、2→review、3 及以上→reject（最后一档吃掉所有更大的数）。
    """
    return ACTS[min(count, len(ACTS) - 1)]


@dataclass(frozen=True)
class SubQaCase:
    """一个注入特征组合：该命中哪几条判据 + 这条字幕对来自哪种素材。

    字段：`hits` 命中组合的规范串；`track` 素材类型（`TRACKS` 里的一个）。
    """

    hits: str
    track: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"hits": self.hits, "track": self.track}

    def fingerprint(self) -> str:
        """输入：无。输出：case 指纹。预期行为：同一 case 任何进程都得到同一串。"""
        return "|".join(f"{key}={value}" for key, value in self.as_dict().items())

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.hits or '无命中'}-{self.track}"

    def tags(self) -> tuple[str, ...]:
        """输入：无。输出：该命中的判据编号元组。"""
        return parse_hits(self.hits)


@dataclass(frozen=True)
class SubQaPlan:
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


def plan_for(case: SubQaCase, phrasing: int) -> SubQaPlan:
    """输入：case、问法编号。输出：判定计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里都
    得到同一份计划；两种问法会抽到不同的干扰组合，不会渲染出结构相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    return SubQaPlan(hits=case.tags(), distractors=near_miss_combos(TAG_IDS, case.tags(), random.Random(seed)))


@dataclass(frozen=True)
class SubQaLabels:
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


def label_for(plan: SubQaPlan) -> SubQaLabels:
    """输入：判定计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`render.check_draft` 负责保证
    文本真的兑现这份计划（按字面算出来的命中组合与计划逐条相同）。
    """
    return SubQaLabels(hits=plan.hits)


def observed_labels(text: str) -> SubQaLabels:
    """输入：一条字幕对。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认文本、只认字面。两者对不上就说明渲染没兑现
    计划，审计当场就能看见。
    """
    return SubQaLabels(hits=hit_set(text))


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[SubQaCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`SubQaCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [SubQaCase(hits=str(row["hits"]), track=str(row["track"])) for row in rows]
