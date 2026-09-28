"""技能选择 workflow 的真值规格：一张技能表 + 一条请求 + 四张判据表。

输入/输出：不可变数据类与纯函数；这里只有**键**（技能条目 / 标签 / 权限范围 / 资源 /
授权 / 排布 / 占用配对），中英文案全在 `locales/`。
预期行为：答案只从渲染出来的那屏选择单反推——适用场景要**完全一致**才算匹配、权限范围要
不高于授权、两个技能碰同一类资源就冲突，这些都是已知的**纯函数**，构造侧与判别侧共用
同一套，所以「表写错」与「渲染写歪」会变成同一类错误，不会两边各错一半还互相印证。

覆盖的决策形状（社区用例地图：技能选择与建议，25 个第三方项目，含 `skillranker` /
`jev-skill-suggester` / `pi-jev-skill-picker` / `building-with-jev-skill` / `ask-jev-skill`）：
- **有没有能用的**（`fits`，noul，总是出）：技能表里有没有一个技能的适用场景与请求标签
  完全一致；
- **选哪个**（`pick`，choice 6，只在有时）：该用第几个技能；
- **要什么权限**（`scope`，choice 3，只在有时）：那个技能的权限范围是哪一档；
- **给不给用**（`allow`，noul，只在有时）：按这份授权，这条请求能不能用它；
- **撞不撞**（`conflict`，noul，只在有时）：那个技能与已占用的技能是不是碰同一类资源。

技能表固定六个条目，**按资源成对相邻**排列（`repo`/`patch` 都是代码、`crawl`/`deploy`
都是网络、`sheet`/`audit` 都是数据）。于是「已占用技能」的两种挑法（表格里紧挨着的
上一行 / 下一行）恰好各有一半落在同资源上——成对占据偶数位 / 奇数位，所以偶数行往下、
奇数行往上才是同资源，`conflict` 的 50/50 正是这么构造出来的，不靠事后过滤。

请求标签十二个：六个是技能表里的适用场景、六个是**近似但不等**的干扰标签（每个都把某个
技能标签当子串含在里面），所以「完全一致才算匹配」这条判据不是摆设——按子串匹配的读者会
在干扰标签上选错技能。`fits` 的 50/50 也由此而来。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import hashlib
import random
from dataclasses import dataclass
from itertools import product
from typing import Final

from jc.knobs import require_pool

# ---------- 技能表（六个条目） ----------

#: 技能表的条目顺序。**同一资源的两个条目必须相邻**（成对占据偶数位 / 奇数位）：于是
#: 「上一行」与「下一行」里恰好有一个同资源、另一个异资源（`conflict` 的构造就靠这条）。
CATALOG: Final[tuple[str, ...]] = ("repo", "patch", "crawl", "deploy", "sheet", "audit")

#: 每个条目适用的请求标签（标签就是技能的名字）。
TAG_OF: Final[dict[str, str]] = {
    "repo": "read_code",
    "patch": "edit_code",
    "crawl": "fetch_web",
    "deploy": "ship_release",
    "sheet": "tidy_sheet",
    "audit": "check_data",
}
#: 每个条目的权限范围（三档，六个条目各两档）。
SCOPE_OF: Final[dict[str, str]] = {
    "repo": "read",
    "patch": "write",
    "crawl": "read",
    "deploy": "admin",
    "sheet": "write",
    "audit": "admin",
}
#: 每个条目要碰的资源（三类，六个条目各两类、且同类相邻）。
RESOURCE_OF: Final[dict[str, str]] = {
    "repo": "code",
    "patch": "code",
    "crawl": "net",
    "deploy": "net",
    "sheet": "data",
    "audit": "data",
}

#: 干扰标签：**每个都把某个技能标签当子串含在里面**，但自己一个技能都不匹配。
ORPHANS: Final[tuple[str, ...]] = (
    "read_code_notes",
    "edit_code_style",
    "fetch_web_shot",
    "ship_release_note",
    "tidy_sheet_tpl",
    "check_data_src",
)

#: 请求标签的全集（前六个是技能表里的适用场景，后六个是干扰标签）。
TAGS: Final[tuple[str, ...]] = tuple(TAG_OF[entry] for entry in CATALOG) + ORPHANS

# ---------- 三张判据表 ----------

#: 权限范围的三档（可比较大小）。
SCOPES: Final[tuple[str, ...]] = ("read", "write", "admin")
#: 授权只有两档：管理员级技能要单独审批，只读 / 可写两种授权都不够。
GRANTS: Final[tuple[str, ...]] = ("read", "write")
#: 档位序：权限范围不高于授权才能用。
SCOPE_RANK: Final[dict[str, int]] = {"read": 0, "write": 1, "admin": 2}
GRANT_RANK: Final[dict[str, int]] = {"read": 0, "write": 1}
#: 资源三类。
RESOURCES: Final[tuple[str, ...]] = ("code", "net", "data")

# ---------- 排布与占用配对 ----------

#: 六种排布：`l<i>` 把条目 j 摆到第 (j + i) mod 6 个位置上——每个条目在每个位置上恰好一次。
LAYOUTS: Final[tuple[str, ...]] = ("l0", "l1", "l2", "l3", "l4", "l5")
#: 已占用技能的两种挑法：表格里紧挨着的上一行（`above`）/ 下一行（`below`）。
INUSE_OFFSET: Final[dict[str, int]] = {"above": -1, "below": 1}
INUSE_KINDS: Final[tuple[str, ...]] = tuple(INUSE_OFFSET)
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


def entry_of(tag: str) -> str | None:
    """输入：请求标签。输出：匹配的技能条目（干扰标签给 None）。预期行为：认不出直接报错。

    预期行为：**完全一致**才算匹配——干扰标签把技能标签当子串含在里面，按子串匹配会选错。
    """
    _require("tag", tag, TAGS)
    for entry in CATALOG:
        if TAG_OF[entry] == tag:
            return entry
    return None


def layout_of(layout: str) -> int:
    """输入：排布名。输出：位移量。预期行为：认不出直接报错。"""
    _require("layout", layout, LAYOUTS)
    return LAYOUTS.index(layout)


def entry_at(layout: str, slot: int) -> str:
    """输入：排布名、位置（从 1 起）。输出：摆在那里的技能条目。

    预期行为：环形移位——每个条目在六个位置上各出现一次，`pick` 的答案位置分布因此均匀。
    """
    shift = layout_of(layout)
    size = len(CATALOG)
    if not 1 <= slot <= size:
        raise ValueError(f"位置要在 1..{size} 之间，收到 {slot}")
    return CATALOG[(slot - 1 - shift) % size]


def slot_of(layout: str, entry: str) -> int:
    """输入：排布名、技能条目。输出：它摆在第几个位置（从 1 起）。"""
    _require("entry", entry, CATALOG)
    size = len(CATALOG)
    return (CATALOG.index(entry) + layout_of(layout)) % size + 1


def inuse_entry(tag: str, inuse: str) -> str:
    """输入：请求标签、占用配对的取值。输出：已占用的那个技能条目。

    预期行为：以匹配到的条目为基准在表里上下挪一行（干扰标签没有基准，就以第 0 个条目为
    基准——这一维在干扰样本上也真的改变 state，不会造出两条一模一样的样本）。条目成对
    相邻，所以「上一行 / 下一行」里恰有一个与目标同资源，`conflict` 的 50/50 由此而来。
    """
    _require("inuse", inuse, INUSE_KINDS)
    target = entry_of(tag)
    base = 0 if target is None else CATALOG.index(target)
    return CATALOG[(base + INUSE_OFFSET[inuse]) % len(CATALOG)]


def scope_of(entry: str) -> str:
    """输入：技能条目。输出：它的权限范围。预期行为：认不出直接报错。"""
    _require("entry", entry, CATALOG)
    return SCOPE_OF[entry]


def resource_of(entry: str) -> str:
    """输入：技能条目。输出：它要碰的资源。预期行为：认不出直接报错。"""
    _require("entry", entry, CATALOG)
    return RESOURCE_OF[entry]


def allowed(scope: str, grant: str) -> bool:
    """输入：技能的权限范围、这份授权。输出：授权够不够。

    预期行为：比档位——权限范围不高于授权才够；管理员级技能要单独审批，两种授权都不够，
    所以它在这张表下**永远**不给用（`allow` 的 50/50 正是这么算出来的）。
    取的是**范围那一格的键**而不是技能条目：这样判别器拿从文本里读回来的那一格重算，
    「渲染把权限格写歪」才会被抓住。
    """
    _require("scope", scope, SCOPES)
    _require("grant", grant, GRANTS)
    return SCOPE_RANK[scope] <= GRANT_RANK[grant]


def share_resource(first: str, second: str) -> bool:
    """输入：两个资源的键。输出：是不是同一类资源（同一类就冲突）。

    预期行为：同样只吃**资源那一格**——判别器从文本里读回两行的资源格再比，
    所以「渲染把资源格写歪」也会被抓住。
    """
    _require("resource", first, RESOURCES)
    _require("resource", second, RESOURCES)
    return first == second


# ---------- 技能表的一行与一屏选择单 ----------


@dataclass(frozen=True)
class CatalogRow:
    """技能表的一行：条目 / 适用场景 / 权限范围 / 资源（前三样都是键）。"""

    entry: str
    tag: str
    scope: str
    resource: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：四格（统计与调试用）。"""
        return {
            "entry": self.entry,
            "tag": self.tag,
            "scope": self.scope,
            "resource": self.resource,
        }

    def describe(self) -> str:
        """输入：无。输出：一行描述（报告与调试用）。"""
        return f"{self.tag}/{self.scope}/{self.resource}"


def rows_of(layout: str) -> tuple[CatalogRow, ...]:
    """输入：排布名。输出：位置 1..6 上依次是哪一行技能（顺序稳定）。

    预期行为：先按固定顺序拼条目，再按排布摆位——同一资源的两个条目相邻这件事是
    `CATALOG` 的性质，不随排布变化（环形移位保持相邻关系）。
    """
    return tuple(
        CatalogRow(entry, TAG_OF[entry], SCOPE_OF[entry], RESOURCE_OF[entry])
        for entry in (entry_at(layout, slot) for slot in range(1, len(CATALOG) + 1))
    )


# ---------- case ----------


@dataclass(frozen=True)
class SkillCase:
    """一条请求的全部自由度（五维）。"""

    tag: str
    layout: str
    grant: str
    inuse: str
    phrasing: int

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：任一维取值不合法直接报错。"""
        _require("tag", self.tag, TAGS)
        layout_of(self.layout)
        _require("grant", self.grant, GRANTS)
        _require("inuse", self.inuse, INUSE_KINDS)
        if self.phrasing not in PHRASINGS:
            raise KeyError(f"没有句面 {self.phrasing!r}（可用：{PHRASINGS}）")

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：五维取值（统计与均衡报告用）。"""
        return {
            "tag": self.tag,
            "layout": self.layout,
            "grant": self.grant,
            "inuse": self.inuse,
            "phrasing": self.phrasing,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述。预期行为：与素材指纹同形（报告与文件名用）。"""
        return material_of(self)

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这一条里训练侧没见过的维度名（排序后）。

        预期行为：只认两个留出维——排布 `l5`、第二套句面。其余三维在训练侧全开。
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


def material_of(case: SkillCase) -> str:
    """输入：case。输出：素材指纹（切分按它哈希分桶，同一条请求整组不跨切分）。

    预期行为：五维全进指纹，句面也进——同一条请求的两套问法是两条不同的 state，
    OOD 里才不会出现重复素材。
    """
    return f"{case.tag}|{case.layout}|{case.grant}|{case.inuse}|{case.phrasing}"


# ---------- 采样参数 ----------


@dataclass(frozen=True)
class SkillConfig:
    """一个切分的采样参数：五张取值表 + 是否只留 OOD。"""

    tags: tuple[str, ...] = TAGS
    layouts: tuple[str, ...] = LAYOUTS
    grants: tuple[str, ...] = GRANTS
    inuse_modes: tuple[str, ...] = INUSE_KINDS
    phrasings: tuple[int, ...] = PHRASINGS
    split: str = "train"
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：任一取值表为空就报人话（点名是哪个字段）。"""
        require_pool("tags", self.tags)
        require_pool("layouts", self.layouts)
        require_pool("grants", self.grants)
        require_pool("inuse_modes", self.inuse_modes)
        require_pool("phrasings", self.phrasings)


def build_cases(config: SkillConfig) -> list[SkillCase]:
    """输入：采样参数。输出：case 矩阵（顺序稳定）。

    预期行为：五张表取全组合（**不做成立性过滤**——五个维度两两不矛盾）；`ood_only` 时
    只留「至少含一个训练侧没见过的取值」的组合，所以 OOD 与同分布侧按构造就不重叠。
    """
    out: list[SkillCase] = []
    for tag, layout, grant, inuse, phrasing in product(
        config.tags, config.layouts, config.grants, config.inuse_modes, config.phrasings
    ):
        case = SkillCase(tag, layout, grant, inuse, phrasing)
        if config.ood_only and not case.unseen_dims():
            continue
        out.append(case)
    return out


# ---------- 真值规格 ----------

REQUEST_PREFIX: Final[str] = "Q-"


@dataclass(frozen=True)
class SkillSpec:
    """一屏选择单的完整真值：case + 由规则算出来的一切。"""

    case: SkillCase
    split: str
    seq: int
    phrasing: int
    request_id: str
    rows: tuple[CatalogRow, ...]
    target_slot: int | None
    target_entry: str | None
    #: 已占用技能那一行的**适用场景标签**（渲染、出题、往返校验都按标签找那一行）。
    inuse_tag: str
    allowed: bool
    conflicts: bool


def request_id_of(case: SkillCase) -> str:
    """输入：case。输出：请求号（形如 `Q-48213`）。

    预期行为：由素材指纹的 sha256 派生——确定性，而且**不泄露切分**（用序号或切分名拼
    请求号会让同一条请求在不同切分里拿到不同请求号，等于把切分信息写进 state）。
    """
    digest = hashlib.sha256(material_of(case).encode("utf-8")).hexdigest()
    return f"{REQUEST_PREFIX}{10000 + int(digest[:6], 16) % 90000}"


def sample_spec(case: SkillCase, split: str, seq: int, phrasing: int, rng: random.Random) -> SkillSpec:
    """输入：case、切分、序号、句面、RNG。输出：真值规格。

    预期行为：技能表、匹配结果与三条判据都由 case 决定（**不掷骰子**）——这个 workflow 的
    每个自由度都已经在 case 里，`rng` 只为与其它 workflow 的签名一致，收下就丢。
    """
    del rng
    target = entry_of(case.tag)
    occupied = inuse_entry(case.tag, case.inuse)
    inuse_tag = TAG_OF[occupied]
    rows = rows_of(case.layout)
    target_slot = None if target is None else slot_of(case.layout, target)
    if target is None:
        allow = clash = False
    else:
        target_row = rows[target_slot - 1]
        inuse_row = next(row for row in rows if row.tag == inuse_tag)
        allow = allowed(target_row.scope, case.grant)
        clash = share_resource(target_row.resource, inuse_row.resource)
    return SkillSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        request_id=request_id_of(case),
        rows=rows,
        target_slot=target_slot,
        target_entry=target,
        inuse_tag=inuse_tag,
        allowed=allow,
        conflicts=clash,
    )


def spec_to_dict(spec: SkillSpec) -> dict[str, JsonValue]:
    """输入：规格。输出：可 JSON 序列化的摘要（写进 manifest 与调试，不进样本行）。"""
    return {
        "case": spec.case.describe(),
        "unseen_dims": list(spec.case.unseen_dims()),
        "request_id": spec.request_id,
        "target_slot": spec.target_slot,
        "target_entry": spec.target_entry,
        "inuse_tag": spec.inuse_tag,
        "allowed": spec.allowed,
        "conflicts": spec.conflicts,
        "rows": [row.describe() for row in spec.rows],
    }
