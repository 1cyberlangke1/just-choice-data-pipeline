"""检索结果判读（search）的特征层：注入维度 + 标签规则（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一次「该不该信这条搜索结果」的判读由 5 个**注入特征**描述，
**标签先行**——先定这 5 维取值，再由本模块的纯函数算出全部金标签。

五维分成两拨（这条分法是设计的第一步，见范式 §9.13）：
- **由文本承载**（提示词给、状态行不给）：查询意图 `query_kind`、结果是否切题
  `topicality`、结果是不是垃圾内容 `spam`——这三件事真实结果页的标题与摘要本来就会
  体现出来；
- **由代码渲染的「来源与时效」那一行承载**（提示词不给）：来源类型 `authority`、
  时效 `freshness`——这两件事在真实结果页上是引擎标注的元数据，不是摘要里的措辞。

六个题族里三个由文本承载（`query_kind` / `topicality` / `spam`），交给
`judges.TraceJudge` 反解校验；另外三个（`rule` / `verdict` / `trust`）由这五维经纯函数
算出，`audit.py` 拿产物里记下的答案做交叉校验。

输入/输出：只有常量与纯函数；不碰网络、不碰文件、**绝不发任何请求**。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jc.cover import cover_dims, row_allowed

# ---------- 维度表 ----------

#: 查询的意图。**只由那句查询承载**。
QUERY_KINDS: Final[tuple[str, ...]] = (
    "navigational",  # 找某个具体的页面 / 站点
    "informational",  # 想弄懂一件事
    "transactional",  # 想办成一件事（下单 / 下载 / 登录）
)

#: 结果与查询切不切题。**只由标题与摘要承载**。
TOPICALITIES: Final[tuple[str, ...]] = (
    "on_topic",  # 切题
    "off_topic",  # 跑题
)

#: 结果是不是垃圾内容（关键词堆砌 / 导流软文 / 纯广告）。**只由标题与摘要承载**。
SPAMS: Final[tuple[str, ...]] = (
    "clean",  # 干净
    "spam",  # 垃圾
)

#: 来源类型。**只由「来源与时效」那一行承载**，标题与摘要里不许写。
AUTHORITIES: Final[tuple[str, ...]] = (
    "official",  # 官方来源
    "community",  # 社区来源（论坛 / 博客 / 个人站）
    "unknown",  # 看不出出处的转载站
)

#: 时效。**只由「来源与时效」那一行承载**。
FRESHNESSES: Final[tuple[str, ...]] = (
    "fresh",  # 刚发布
    "recent",  # 半年内
    "stale",  # 好几年没动了
)

DIMS: Final[dict[str, tuple[str, ...]]] = {
    "query_kind": QUERY_KINDS,
    "topicality": TOPICALITIES,
    "spam": SPAMS,
    "authority": AUTHORITIES,
    "freshness": FRESHNESSES,
}

#: 判读规则表。顺序就是**从上往下第一条命中**的顺序，改顺序等于改语义——
#: `tests/workflows/search/test_features.py` 拿规则表的文本行顺序钉住它。
RULES: Final[tuple[str, ...]] = (
    "spam_result",
    "off_topic",
    "navigational_non_official",
    "unverified_source",
    "stale_source",
    "community_source",
    "none",
)

#: 每条规则给出的**原始**处置（不含任何后续抬升——本 workflow 没有审批那一层）。
RAW_VERDICTS: Final[dict[str, str]] = {
    "spam_result": "drop",
    "off_topic": "drop",
    "navigational_non_official": "drop",
    "unverified_source": "cite_with_caveat",
    "stale_source": "cite_with_caveat",
    "community_source": "cite_with_caveat",
    "none": "cite",
}

#: 最终处置。`cite` = 直接引用；`cite_with_caveat` = 引用但要标注不确定性；`drop` = 弃用。
VERDICTS: Final[tuple[str, ...]] = ("cite", "cite_with_caveat", "drop")

#: 可信度档数（0 ~ 3）。
TRUST_LEVELS: Final[int] = 4

#: 「来源类型 × 时效」的底分表。表里缺格直接抛 KeyError（见 `trust_for`）。
BASE_TRUST: Final[dict[tuple[str, str], int]] = {
    ("official", "fresh"): 3,
    ("official", "recent"): 3,
    ("official", "stale"): 2,
    ("community", "fresh"): 2,
    ("community", "recent"): 2,
    ("community", "stale"): 1,
    ("unknown", "fresh"): 1,
    ("unknown", "recent"): 1,
    ("unknown", "stale"): 0,
}

#: 天生不能同时出现的取值对（四元组 `(左维, 左值, 右维, 右值)`）：
#: - **官方来源 × 垃圾内容**：官方页面不可能是关键词堆砌，真写出来这道题的规则表会同时
#:   命中前两条，语义自相矛盾；
#: - **跑题 × 垃圾内容**：这两维互相打架——垃圾摘要几乎总是「就在讲这个话题」，
#:   读的人没法从文本里分清该答 `off_topic` 还是 `on_topic`。这不是模型不行，是这一格
#:   本身没有唯一答案：真机第一批就有一条卡在这里（注入 `off_topic`、反解读成 `on_topic`，
#:   而那段摘要确实既跑题又像广告）。判据自相矛盾的格子就不该生成。
FORBIDDEN: Final[frozenset[tuple[str, str, str, str]]] = frozenset(
    {
        ("authority", "official", "spam", "spam"),
        ("spam", "spam", "topicality", "off_topic"),
    }
)


@dataclass(frozen=True)
class Labels:
    """一条样本的全部金标签。字段名与题族名一一对应。"""

    query_kind: str
    topicality: str
    spam: str
    rule: str
    verdict: str
    trust: int


@dataclass(frozen=True)
class TagSet:
    """一组注入特征。字段名与 `DIMS` 的键一一对应。"""

    query_kind: str
    topicality: str
    spam: str
    authority: str
    freshness: str

    def as_dict(self) -> dict[str, str]:
        """输入：无。输出：`{维度名: 取值}`（顺序按 `DIMS`）。"""
        return {name: getattr(self, name) for name in DIMS}

    def label(self) -> Labels:
        """输入：无。输出：这组特征对应的全部金标签。"""
        return label_for(self)


def allow_pair(left: str, left_value: str, right: str, right_value: str) -> bool:
    """输入：两个维度名与各自取值。输出：这一对写得出来吗。

    预期行为：与顺序无关——`(a, x, b, y)` 与 `(b, y, a, x)` 都查同一张禁用表。
    """
    if left > right:
        left, left_value, right, right_value = right, right_value, left, left_value
    return (left, left_value, right, right_value) not in FORBIDDEN


def rule_for(tags: TagSet) -> str:
    """输入：一组注入特征。输出：命中的规则名（第一条命中者）。

    预期行为：严格按 `RULES` 的顺序判。规则表要一眼可复核——这张表就是 state 里
    渲染给考生的那张表，所以这里的顺序必须与 locale 里 `rules` 的书写顺序一致。
    """
    if tags.spam == "spam":
        return "spam_result"
    if tags.topicality == "off_topic":
        return "off_topic"
    if tags.query_kind == "navigational" and tags.authority != "official":
        return "navigational_non_official"
    if tags.authority == "unknown":
        return "unverified_source"
    if tags.freshness == "stale":
        return "stale_source"
    if tags.authority == "community":
        return "community_source"
    return "none"


def verdict_for(tags: TagSet) -> str:
    """输入：一组注入特征。输出：最终处置（本 workflow 没有抬升层，等于原始处置）。"""
    return RAW_VERDICTS[rule_for(tags)]


def trust_for(tags: TagSet) -> int:
    """输入：一组注入特征。输出：可信度档（0~3）。

    预期行为：先查 `BASE_TRUST`（来源类型 × 时效），**垃圾内容或跑题一律压到 0 档**——
    一条再官方的页面，只要摘要里是堆砌的关键词，就不该给考生任何可信度。
    表里缺这一格直接抛 KeyError：缺格意味着有取值组合压根没定义，那这条样本就不该生成，
    静默兜底会把「没定义」伪装成「有定义」。
    """
    if tags.spam == "spam" or tags.topicality == "off_topic":
        return 0
    return BASE_TRUST[(tags.authority, tags.freshness)]


def label_for(tags: TagSet) -> Labels:
    """输入：一组注入特征。输出：全部金标签。

    预期行为：**纯函数、只看特征、不看文本**——文本是后写的，标签先定。

    全空间（3×2×2×3×3 = 108 组，剔除 1 对禁用对后 96 组）上的分布实测如下：

    - `rule`：`none` 与 `stale_source` / `community_source` 最多，`spam_result` 最少；
    - `verdict`：三档都到得了；
    - `trust`：0~3 四档都到得了（0 档要「垃圾或跑题」——这正是它必须依赖文本两维的原因）；
    - `query_kind` / `topicality` / `spam`：各档都到得了。

    偏斜来自判据本身（规则表就是这么写的），照实标出不掩盖；要的是**每一档都到得了**，
    不是每档一样多。同分布侧（留出 `freshness=recent`）每族每档都还在。
    """
    return Labels(
        query_kind=tags.query_kind,
        topicality=tags.topicality,
        spam=tags.spam,
        rule=rule_for(tags),
        verdict=verdict_for(tags),
        trust=trust_for(tags),
    )


def build_cases(max_full: int, dims: dict[str, tuple[str, ...]] | None = None) -> list[TagSet]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖；LLM workflow 要把它压到几十）、
    可选的维度表覆盖（留出某几个取值时用，例如 OOD 的「值留出」）。
    输出：`TagSet` 列表，顺序稳定。
    预期行为：同一入参必得同一输出；成对覆盖保证任意两维的取值组合至少出现一次；
    被 `FORBIDDEN` 否掉的取值对既不出现、也不算没覆盖。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full, allow_pair=allow_pair)
    return [TagSet(**row) for row in rows]


def compatible(tags: TagSet) -> bool:
    """输入：一组特征。输出：这组特征写得出来吗（没有任何被禁的取值对）。"""
    return row_allowed(list(DIMS), tags.as_dict(), allow_pair)


#: choice 题族允许出现的选项名（题面只收本题允许的，别的取值留给反解提示词用）。
CHOICE_OPTIONS: Final[dict[str, tuple[str, ...]]] = {
    "query_kind": QUERY_KINDS,
    "topicality": TOPICALITIES,
    "rule": RULES,
    "verdict": VERDICTS,
}
