"""界面元素定位 workflow 的真值规格：一屏元素清单 + 一个动作标签 + 四条判据。

输入/输出：不可变数据类与纯函数；这里只有**键**（角色 / 可见性 / 可点性 / 文本 / 排布 /
命中形状 / 多出来的同名行 / 近似元素 / 句面），中英文案全在 `locales/`。
预期行为：答案只从渲染出来的那屏清单反推——「文本与动作标签**完全一致**」「可见**且**可点」
「近似元素把标签当子串含在里面但不相等」都是已知的**纯函数**，构造侧与判别侧共用同一套，
所以「表写错」与「渲染写歪」会变成同一类错误，不会两边各错一半还互相印证。

与 `browser` 的区别：`browser` 问「下一步做什么动作」（要读一串历史、判动作序列与完成度），
这家问「**这个动作落在哪个元素上**」——纯查表，没有历史、没有动作序列、没有完成度。

覆盖的决策形状（社区用例地图：浏览器与计算机操作，83 个第三方项目，含 `Jev Ultrafast`
★9291「在一次请求中选择操作和 DOM 元素」/ `agent-desktop` ★1441 / `surf-cli` ★624 /
`Jev-cu` ★584 / `typesafe-computer-use` ★537 / `mobile-jev` ★240）：
- **有没有能落点的元素**（`found`，noul，总是出）：屏幕上有没有一个元素的文本与动作标签
  完全一致、而且可见且可点；
- **有没有名字只是近似的元素**（`shadow`，noul，总是出）：有没有元素的文本把动作标签
  **当子串含在里面**但不完全一样；
- **落在第几个元素上**（`target`，choice 6，只在有时）：落点是清单里的第几行；
- **落在哪一类元素上**（`role`，choice 3，只在有时）：落点的角色属于哪一类；
- **有几个同名元素被挡住了**（`decoy`，score 3 档，只在有时）：同名元素里可见且可点的
  只有一个，剩下的都被挡住了。

六行清单的版式：每行「角色 · 可见性 · 可点性 · 文本」。**同名元素里恰好一行可见且可点**
（构造保证），所以「落在第几个」不依赖任何「谁更靠上」的约定——判据里说得清、审计也重推得
出来。`hit` 形状多出来的同名行、`miss` 形状的同名行都是被挡住的干扰行，两种挡法（隐藏 /
不可点）按奇偶交替出现，判据里那两种说法都不是摆设。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from typing import Final, Protocol

from jc.knobs import require_pool

# ---------- 元素清单 ----------

#: 元素表的行数。**固定六行**——行数随 case 变会让 `target` 的选项空间随样本变，
#: 均衡闸的参考值（出现过的最大选项数）就会把分布压歪（范式 §9.77 第四条）。
SIZE: Final[int] = 6

#: 六个角色，顺序就是它们在第 1..6 行上的归属（排布只动位置，不动角色）。
ROLES: Final[tuple[str, ...]] = ("button", "toggle", "link", "icon", "input", "select")
#: 角色三分类，每类两个角色（相邻成对，所以三档各占三分之一）。
ROLE_CLASSES: Final[tuple[str, ...]] = ("button_like", "link_like", "input_like")
ROLE_CLASS_OF: Final[dict[str, str]] = {
    "button": "button_like",
    "toggle": "button_like",
    "link": "link_like",
    "icon": "link_like",
    "input": "input_like",
    "select": "input_like",
}

#: 六个标签：动作标签与元素文本共用同一张表（动作标签就是它要找的那段文本）。
LABELS: Final[tuple[str, ...]] = ("save", "cancel", "delete", "edit", "refresh", "submit")
#: 近似文本的键前缀：`shadow:<标签>` 表示「把那个标签当子串含在里面但不相等」的文本。
SHADOW_PREFIX: Final[str] = "shadow:"

#: 可见性两档。
VISIBILITIES: Final[tuple[str, ...]] = ("visible", "hidden")
#: 可点性两档。
CLICKABILITIES: Final[tuple[str, ...]] = ("yes", "no")

# ---------- 排布与命中形状 ----------

#: 六种排布：`l<i>` 让目标那一行落在第 i+1 个位置。
LAYOUTS: Final[tuple[str, ...]] = ("l0", "l1", "l2", "l3", "l4", "l5")
#: 命中形状：`hit` 有落点；`miss` 一个都没有（同名元素全被挡住）。
SHAPES: Final[tuple[str, ...]] = ("hit", "miss")
#: 除第一行同名元素之外还有几条同名行（0~2）：`hit` 时它们是被挡住的干扰行，
#: `miss` 时再加一行（于是 `miss` 侧的同名行数在 1~3 之间变，状态不会重复）。
EXTRA_COUNTS: Final[tuple[int, ...]] = (0, 1, 2)
#: 近似元素在不在屏幕上。
SHADOWS: Final[tuple[bool, ...]] = (False, True)
#: 句面（只影响措辞，不影响答案）。
PHRASINGS: Final[tuple[int, ...]] = (0, 1)


# ---------- 规则函数（构造与判别共用） ----------


def _require(field: str, value: object, pool: tuple[object, ...]) -> None:
    """输入：字段名、取值、取值表。输出：无。

    预期行为：认不出的键直接报错——静默落到某个默认分支会把「表改了、读法忘了改」
    变成一堆答案对不上的样本，而不是一条能定位的错误。
    """
    if value not in pool:
        raise KeyError(f"没有 {field}={value!r}（可用：{', '.join(str(item) for item in pool)}）")


def position_of(layout: str, label: str) -> int:
    """输入：排布名、动作标签。输出：目标那一行的位置（从 1 起）。

    预期行为：**环形移位**——标签自己也参与定位，所以每个标签在六个位置上各出现一次，
    `target` 的位置分布因此均匀；更要紧的是，**留出某一种排布时位置不会塌到某一个值**
    （只按排布定位的话，留出 `l5` 会让留出侧那一半样本全落在第 6 行，`target` 与 `role`
    当场失衡——见范式 §9.79）。认不出直接报错。
    """
    _require("layout", layout, LAYOUTS)
    _require("label", label, LABELS)
    return (LABELS.index(label) + LAYOUTS.index(layout)) % SIZE + 1


def role_at(position: int) -> str:
    """输入：行号（从 1 起）。输出：那一行的角色。

    预期行为：角色按行号循环——三分类相邻成对，所以目标位置在 1..6 上均匀时，
    `role` 的三档也各占三分之一。
    """
    if not 1 <= position <= SIZE:
        raise ValueError(f"行号要在 1..{SIZE} 之间，收到 {position}")
    return ROLES[(position - 1) % SIZE]


def role_class_of(role: str) -> str:
    """输入：角色键。输出：它属于哪一类。预期行为：认不出直接报错。"""
    _require("role", role, ROLES)
    return ROLE_CLASS_OF[role]


def is_shadow_text(text: str) -> bool:
    """输入：文本键。输出：是不是近似文本（`shadow:` 前缀）。"""
    return text.startswith(SHADOW_PREFIX)


def text_matches(text: str, label: str) -> bool:
    """输入：文本键、动作标签。输出：是不是**完全一致**。

    预期行为：近似文本（`shadow:<标签>`）在这里一律不算——判据里「差一个字不算」
    说的就是这条。
    """
    return text == label


def text_is_near(text: str, label: str) -> bool:
    """输入：文本键、动作标签。输出：是不是**近似但不相等**。

    预期行为：只认 `shadow:<标签>`——渲染侧保证那段文本真的把标签原文当子串含在里面，
    反解侧按同一条规则剥后缀（`render.text_of` 与 `parse.decode_text` 是镜像）。
    """
    return text == f"{SHADOW_PREFIX}{label}"


def blocked_style(index: int) -> tuple[str, str]:
    """输入：被挡住的行在那一组里的序号（从 0 起）。输出：(可见性, 可点性)。

    预期行为：奇偶交替——偶数号「隐藏」（可见性不达标）、奇数号「可见但不可点」
    （可点性不达标）。两种挡法在同一批数据里都出现，判据里那两条都不是摆设。
    """
    if index % 2 == 0:
        return ("hidden", "yes")
    return ("visible", "no")


# ---------- 元素清单的一行 ----------


@dataclass(frozen=True)
class ElementRow:
    """元素清单的一行：角色 / 可见性 / 可点性 / 文本（四样都是键）。"""

    role: str
    visibility: str
    clickable: str
    text: str

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：四格认不出直接报错（渲染与反解都靠它）。"""
        _require("role", self.role, ROLES)
        _require("visibility", self.visibility, VISIBILITIES)
        _require("clickable", self.clickable, CLICKABILITIES)
        if not is_shadow_text(self.text):
            _require("text", self.text, LABELS)

    def usable(self) -> bool:
        """输入：无。输出：这一行可见**且**可点吗。预期行为：两格都要达标。"""
        return self.visibility == "visible" and self.clickable == "yes"

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：四格（统计与调试用）。"""
        return {
            "role": self.role,
            "visibility": self.visibility,
            "clickable": self.clickable,
            "text": self.text,
        }

    def describe(self) -> str:
        """输入：无。输出：一行描述（报告与调试用）。"""
        return f"{self.role}/{self.visibility}/{self.clickable}/{self.text}"


def rows_of(case: ElementCase) -> tuple[ElementRow, ...]:
    """输入：case。输出：六行元素清单（顺序 = 屏幕从上到下）。

    预期行为：先摆同名行（从目标位置起环形铺开），再摆近似行，剩下的位置按标签顺序填
    无关文本。`hit` 时第一行同名元素可见且可点（落点），其余同名行被挡住；`miss` 时
    全部被挡住。挡法按奇偶交替（`blocked_style`）。
    """
    start = position_of(case.layout, case.label)
    exact = case.extras + 1
    placed: dict[int, ElementRow] = {}
    blocked = 0
    for order in range(exact):
        position = ((start - 1 + order) % SIZE) + 1
        if case.shape == "hit" and order == 0:
            visibility, clickable = "visible", "yes"
        else:
            visibility, clickable = blocked_style(blocked)
            blocked += 1
        placed[position] = ElementRow(role_at(position), visibility, clickable, case.label)
    if case.shadow:
        position = ((start - 1 + exact) % SIZE) + 1
        placed[position] = ElementRow(
            role_at(position), "visible", "yes", f"{SHADOW_PREFIX}{case.label}"
        )
    others = [label for label in LABELS if label != case.label]
    cursor = 0
    for position in range(1, SIZE + 1):
        if position in placed:
            continue
        placed[position] = ElementRow(role_at(position), "visible", "yes", others[cursor])
        cursor += 1
    return tuple(placed[position] for position in range(1, SIZE + 1))


class RowLike(Protocol):
    """元素清单的一行（`spec.ElementRow` 与 `parse.ParsedRow` 都满足它）。

    输入/输出：只声明四格与一条 `usable()`。预期行为：构造侧与判别侧共用同一批纯函数，
    所以两边不可能对「哪一行算合格」有不同理解。
    """

    role: str
    visibility: str
    clickable: str
    text: str

    def usable(self) -> bool:
        """输入：无。输出：这一行可见且可点吗。"""


def target_slot(rows: Sequence[RowLike], label: str) -> int | None:
    """输入：元素清单、动作标签。输出：落点的行号（没有给 None）。

    预期行为：先按**完全一致**筛出同名元素，再按**可见且可点**筛出落点。构造上同名元素里
    恰好一行可见且可点，所以不用比先后——答案不依赖任何「谁更靠上」的约定。
    """
    for index, row in enumerate(rows, start=1):
        if text_matches(row.text, label) and row.usable():
            return index
    return None


def blocked_count(rows: Sequence[RowLike], label: str) -> int:
    """输入：元素清单、动作标签。输出：被挡住的同名元素有几条。"""
    return sum(
        1 for row in rows if text_matches(row.text, label) and not row.usable()
    )


def shadow_present(rows: Sequence[RowLike], label: str) -> bool:
    """输入：元素清单、动作标签。输出：屏幕上有没有名字只是近似的元素。"""
    return any(text_is_near(row.text, label) for row in rows)


# ---------- case ----------

#: 训练侧见过的排布：最后一种整个留出。
SEEN_LAYOUTS: Final[frozenset[str]] = frozenset(LAYOUTS[:-1])
#: 训练侧见过的句面：第二套句面整个留出。它**不影响任何答案**，所以留出它不会把哪一族
#: 的分布压歪。
SEEN_PHRASINGS: Final[frozenset[int]] = frozenset({PHRASINGS[0]})


@dataclass(frozen=True)
class ElementCase:
    """一屏清单的全部自由度（六维）。"""

    label: str
    layout: str
    shape: str
    extras: int
    shadow: bool
    phrasing: int

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：任一维取值不合法直接报错。"""
        _require("label", self.label, LABELS)
        position_of(self.layout, self.label)
        _require("shape", self.shape, SHAPES)
        _require("extras", self.extras, EXTRA_COUNTS)
        _require("shadow", self.shadow, SHADOWS)
        _require("phrasing", self.phrasing, PHRASINGS)

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：六维取值（统计与均衡报告用）。"""
        return {
            "label": self.label,
            "layout": self.layout,
            "shape": self.shape,
            "extras": self.extras,
            "shadow": self.shadow,
            "phrasing": self.phrasing,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return (
            f"{self.label}/{self.layout}/{self.shape}"
            f"/e{self.extras}/{'near' if self.shadow else 'plain'}"
        )

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：含留出取值的维度名（升序）。

        预期行为：只看那两个**被留出的**维度；`test_ood` 侧靠它筛出「至少含一个没见过
        的取值」的组合。
        """
        out: list[str] = []
        if self.layout not in SEEN_LAYOUTS:
            out.append("layout")
        if self.phrasing not in SEEN_PHRASINGS:
            out.append("phrasing")
        return tuple(out)


def material_of(case: ElementCase) -> str:
    """输入：case。输出：素材指纹（切分按它哈希分桶，同一屏整组不跨切分）。

    预期行为：六维全进指纹，句面也进——同一屏的两套问法是两条不同的 state，
    OOD 里才不会出现重复素材。
    """
    return (
        f"{case.label}|{case.layout}|{case.shape}|{case.extras}"
        f"|{case.shadow}|{case.phrasing}"
    )


# ---------- 采样参数 ----------


@dataclass(frozen=True)
class ElementConfig:
    """一个切分的采样参数：六张取值表 + 是否只留 OOD。"""

    labels: tuple[str, ...] = LABELS
    layouts: tuple[str, ...] = LAYOUTS
    shapes: tuple[str, ...] = SHAPES
    extras: tuple[int, ...] = EXTRA_COUNTS
    shadows: tuple[bool, ...] = SHADOWS
    phrasings: tuple[int, ...] = PHRASINGS
    split: str = "train"
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：任一取值表为空就报人话（点名是哪个字段）。"""
        require_pool("labels", self.labels)
        require_pool("layouts", self.layouts)
        require_pool("shapes", self.shapes)
        require_pool("extras", self.extras)
        require_pool("shadows", self.shadows)
        require_pool("phrasings", self.phrasings)


def build_cases(config: ElementConfig) -> list[ElementCase]:
    """输入：采样参数。输出：case 矩阵（顺序稳定）。

    预期行为：六张表取全组合（**不做成立性过滤**——六个维度两两不矛盾）；`ood_only` 时
    只留「至少含一个训练侧没见过的取值」的组合，所以 OOD 与同分布侧按构造就不重叠。
    """
    out: list[ElementCase] = []
    for label, layout, shape, extras, shadow, phrasing in product(
        config.labels,
        config.layouts,
        config.shapes,
        config.extras,
        config.shadows,
        config.phrasings,
    ):
        case = ElementCase(label, layout, shape, extras, shadow, phrasing)
        if config.ood_only and not case.unseen_dims():
            continue
        out.append(case)
    return out


# ---------- 真值规格 ----------

SCREEN_PREFIX: Final[str] = "S-"


@dataclass(frozen=True)
class ElementSpec:
    """一屏清单的完整真值：case + 由规则算出来的一切。"""

    case: ElementCase
    split: str
    seq: int
    phrasing: int
    screen_id: str
    rows: tuple[ElementRow, ...]
    target_slot: int | None
    target_class: str | None
    blocked: int
    shadow_present: bool


def screen_id_of(case: ElementCase) -> str:
    """输入：case。输出：屏号（形如 `S-48213`）。

    预期行为：由素材指纹的 sha256 派生——确定性，而且**不泄露切分**（用序号或切分名拼
    屏号会让同一屏在不同切分里拿到不同屏号，等于把切分信息写进 state）。
    """
    digest = hashlib.sha256(material_of(case).encode("utf-8")).hexdigest()
    return f"{SCREEN_PREFIX}{10000 + int(digest[:6], 16) % 90000}"


def sample_spec(
    case: ElementCase, split: str, seq: int, phrasing: int, rng: random.Random
) -> ElementSpec:
    """输入：case、切分、序号、句面、RNG。输出：真值规格。

    预期行为：清单与四条判据都由 case 决定（**不掷骰子**）——这个 workflow 的每个自由度
    都已经在 case 里，`rng` 只为与其它 workflow 的签名一致，收下就丢。
    """
    del rng
    rows = rows_of(case)
    slot = target_slot(rows, case.label)
    return ElementSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        screen_id=screen_id_of(case),
        rows=rows,
        target_slot=slot,
        target_class=None if slot is None else role_class_of(rows[slot - 1].role),
        blocked=blocked_count(rows, case.label),
        shadow_present=shadow_present(rows, case.label),
    )
