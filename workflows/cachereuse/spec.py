"""缓存复用判定（cachereuse）workflow 的真值规格：两条操作轨迹逐步对齐。

输入/输出：不可变数据类与纯函数；**不含任何领域词表**（动作名、元素名、抬头、题面与选项
描述全在 `locales/`）。
预期行为：两条轨迹各是一串「动作 + 元素」的步骤，逐步比对只有两种关系——**等价**（动作与
元素归一化后都相同，写法可能不同）与**不同**（至少一处不同）。四个题族的答案全部由这两串
步骤算出来，零 token。

**这一族要训的是「前缀」而不是「计数」**：缓存能复用到第几步，取决于**从头连续等价**的步数；
分叉之后再对上的步**一步都不能用**（操作序列不能跳步执行）。所以「等价的总步数」与
「可复用的步数」是两道不同的题——这正是社区里 `computer-use-cache`（★42）做的事：
Jev 判「缓存的计算机使用工作流在语义上是否可重用于当前请求」。

与 `match` 的区别：match 是**无序的字段对齐**（两条记录五栏各比一次，答案与顺序无关），
这里是**有序的序列对齐**（答案取决于「第一个不同出现在第几步」）。同一个「等价步数」在
两条轨迹里分布不同时，可复用长度可以一个是 0、一个是 3。

**档位是从答案的分布推出来的**（范式 §9.70 一）：最初还打算加一档「整段可复用」，可它
**没有分叉**，于是那一档在「第一个不同在第几步」上永远取同一个值、在「分叉的原因」上根本
出不了题，把两个布尔题族的分布分别拽到 0.111 / 0.556。删掉它之后四档各占 1/4，四个题族
全部按构造均衡。判据：**加一档之前先算它会把哪些题族拽偏**。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

import hashlib
import random
from dataclasses import dataclass
from typing import Final

# ---------- 步骤池（键与数值；写法在 locales/） ----------

#: 动作键（顺序稳定；语言文件必须为每一个给出至少两种写法）。
ACTIONS: Final[tuple[str, ...]] = ("open", "click", "type", "choose", "scroll", "confirm")

#: 元素键。动作与元素是两个独立维度，组合出来的步骤有些别扭（「输入搜索框」），
#: 但每一步都**可判定**，而且这一族考的是序列对齐，不是动作的自然度。
TARGETS: Final[tuple[str, ...]] = (
    "search_box",
    "result_list",
    "filter_menu",
    "sort_menu",
    "cart_button",
    "address_field",
    "pay_button",
    "help_link",
)

#: 每一步的两个键：动作 + 元素。等价 = 两个键都相同。
Step = tuple[str, str]

# ---------- 两条轨迹的关系 ----------

#: 等价步的两种写法：逐字相同（verbatim）与「写法不同但归一化后相同」（synonym）。
#: **只改文本、不改任何标签**——它是渲染维，不是判据维。
VERBATIM: Final[str] = "verbatim"
SYNONYM: Final[str] = "synonym"
SPELLINGS: Final[tuple[str, ...]] = (VERBATIM, SYNONYM)

#: 缓存来源。同样只改抬头那一行，不影响标签，所以是天然的留出维（`kiosk` 只在 OOD 侧出现）。
FLAVORS: Final[tuple[str, ...]] = ("web", "desktop", "kiosk")

# ---------- 尺寸与档位 ----------

#: 轨迹步数：太短则「前缀」与「计数」分不开，太长则一屏塞不下。
LENGTHS: Final[tuple[int, ...]] = (4, 5, 6)

#: 可复用程度四档，**每一档都真的有分叉**（见模块文档：没有分叉的那一档会把布尔题族拽偏）。
BANDS: Final[tuple[str, ...]] = ("most", "half", "few", "none")

#: 每一档分叉之后又对上的步数。**这是「计数 ≠ 前缀」的来源**：只数等价步数的人会把这段
#: 也算进可复用长度。`most` 的分叉落在最后一步，后面没有位置，所以只能是 0。
BAND_TAIL: Final[dict[str, int]] = {"most": 0, "half": 1, "few": 1, "none": 2}

#: 分叉那一步的差异出在哪：只差动作，还是只差元素。做成 case 维，答案才按构造各半。
DIFF_KINDS: Final[tuple[str, ...]] = ("action", "target")

#: 问法池（题面措辞不是判据，换说法不影响答案）。
PHRASINGS: Final[tuple[int, ...]] = (0, 1, 2)


def band_prefix(band: str, length: int) -> int:
    """输入：档名、步数。输出：这一档对应的可复用步数（= 从头连续等价的步数）。

    预期行为：`most` 只差最后一步、`half` 约一半（上取整，保证不贴边）、`few` 只前一步、
    `none` 第一步就分叉。
    """
    if band == "most":
        return length - 1
    if band == "half":
        return -(-length // 2)
    if band == "few":
        return 1
    if band == "none":
        return 0
    raise ValueError(f"未知档位：{band!r}（可用：{BANDS}）")


@dataclass(frozen=True)
class ReuseCase:
    """一个 case：轨迹的形状（步数 / 档位 / 分叉原因）与两个只改文本的维。

    字段：`length` 步数；`band` 可复用程度档；`diff_kind` 分叉那一步差在哪；`spelling`
    等价步的写法；`flavor` 缓存来源。
    """

    length: int
    band: str
    diff_kind: str
    spelling: str
    flavor: str

    @property
    def prefix(self) -> int:
        """输入：无。输出：可复用步数（= 从头连续等价的步数）。"""
        return band_prefix(self.band, self.length)

    @property
    def tail(self) -> int:
        """输入：无。输出：分叉之后又对上的步数。"""
        return BAND_TAIL[self.band]

    @property
    def first_diff(self) -> int:
        """输入：无。输出：第一个不同的步号（从 1 起）。四档都有分叉，所以恒有值。"""
        return self.prefix + 1

    @property
    def equivalent_total(self) -> int:
        """输入：无。输出：两条轨迹**等价的总步数**（含分叉之后再对上的那些步）。"""
        return self.prefix + self.tail

    @property
    def reuse_half(self) -> bool:
        """输入：无。输出：可复用步数是否达到全长的一半以上（`half` 档含等号）。"""
        return self.prefix * 2 >= self.length

    def diverge_band(self) -> str:
        """输入：无。输出：第一个不同出现在哪一档（题面里的四个选项之一）。

        预期行为：档位与**步号**挂钩、与 `length` 无关——「第 1 步 / 第 2 步 / 中间某一步 /
        最后一步」。这样同一份选项表在三种步数下都成立，也不会出现取不到的档。
        """
        step = self.first_diff
        if step == 1:
            return "first"
        if step == 2:
            return "second"
        if step == self.length:
            return "last"
        return "middle"

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这一条里训练侧没见过的维（`flavor == kiosk` 时非空）。"""
        return () if self.flavor in SEEN_FLAVORS else (f"flavor={self.flavor}",)

    def material(self) -> str:
        """输入：无。输出：素材指纹（切分按它分桶，同一份请求永远落在同一档）。

        预期行为：**不含 `spelling` 与 `flavor`**——这两个只改文本，同一条请求的几种写法
        必须整组落在同一个切分里，否则「同素材不跨切分」这条一致性校验会失效。
        """
        return f"{self.length}|{self.band}|{self.diff_kind}"

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：{维度名: 取值}（五个维度全摊平，供统计与均衡报告）。"""
        return {
            "length": self.length,
            "band": self.band,
            "diff_kind": self.diff_kind,
            "spelling": self.spelling,
            "flavor": self.flavor,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return (
            f"len={self.length} band={self.band} diff={self.diff_kind} "
            f"spell={self.spelling} flavor={self.flavor}"
        )


@dataclass(frozen=True)
class ReuseConfig:
    """采样参数：跑哪一档切分、每个维取哪些值。"""

    split: str
    lengths: tuple[int, ...] = LENGTHS
    bands: tuple[str, ...] = BANDS
    diff_kinds: tuple[str, ...] = DIFF_KINDS
    spellings: tuple[str, ...] = SPELLINGS
    flavors: tuple[str, ...] = FLAVORS
    phrasings: tuple[int, ...] = PHRASINGS
    ood_only: bool = False


def feasible(case: ReuseCase) -> bool:
    """输入：一个 case。输出：它在构造上成不成立。

    预期行为：分叉之后剩下的位置要装得下 `tail`。四档都是「有分叉」的档，所以这条恒成立；
    留着它是为了**加档时当场报出来**——没有分叉的档在这里就被挡掉，不会渲染出两条一模一样的
    轨迹（那是重复样本，不是难题）。
    """
    return case.tail <= case.length - case.prefix - 1


def build_cases(config: ReuseConfig) -> list[ReuseCase]:
    """输入：采样参数。输出：case 列表（顺序稳定、可复算）。

    预期行为：五个维全枚举（**先过 `feasible` 再收**）；`ood_only` 时只留「至少含一个训练侧
    没见过的取值」的 case，所以 OOD 与同分布侧**按构造就不重叠**。
    """
    out: list[ReuseCase] = []
    for length in config.lengths:
        for band in config.bands:
            for diff_kind in config.diff_kinds:
                for spelling in config.spellings:
                    for flavor in config.flavors:
                        case = ReuseCase(length, band, diff_kind, spelling, flavor)
                        if not feasible(case):
                            continue
                        if config.ood_only and not case.unseen_dims():
                            continue
                        out.append(case)
    return out


# ---------- 两条轨迹 ----------


def _seed_of(case: ReuseCase) -> int:
    """输入：一个 case。输出：它专属的种子。

    预期行为：用 sha256 而不是内置 `hash()`（后者对字符串每进程随机化），否则同一份 case
    在不同进程里会生成不同的轨迹，「同输入同标签」当场失效。
    """
    text = f"cachereuse|{case.length}|{case.band}|{case.diff_kind}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def _other(rng: random.Random, step: Step, keep_action: bool, keep_target: bool) -> Step:
    """输入：随机源、原步骤、要不要保留动作、要不要保留元素。输出：另一个步骤键。

    预期行为：重抽到「与原来的步骤在**该变的那一处**不同」为止——`keep_action` 为真时动作
    原样保留、只换元素，反之只换动作。这是「分叉原因」这一类题唯一的构造口径。
    """
    for _ in range(64):
        action = step[0] if keep_action else rng.choice(ACTIONS)
        target = step[1] if keep_target else rng.choice(TARGETS)
        if (action, target) != step:
            return (action, target)
    raise AssertionError("步骤池抽不出不同的步骤，池子是不是被改小了")


def traces_of(case: ReuseCase) -> tuple[tuple[Step, ...], tuple[Step, ...]]:
    """输入：一个 case。输出：(缓存轨迹, 当前轨迹)，两条都是步骤键的序列。

    预期行为：**确定性**——同一 case 必得同一对轨迹（种子由 case 指纹派生）。当前轨迹的形状
    是「前 `prefix` 步等价 → 第 `prefix+1` 步按 `diff_kind` 差一处 → 紧接着 `tail` 步又等价
    → 剩下都不同」。两处「不同」都用 `_other` 造，所以差异**只落在动作或元素其中一处**。
    """
    rng = random.Random(_seed_of(case))
    old: list[Step] = [(rng.choice(ACTIONS), rng.choice(TARGETS)) for _ in range(case.length)]
    keep_action = case.diff_kind == "target"
    new: list[Step] = []
    for index, step in enumerate(old):
        if index < case.prefix or case.prefix < index <= case.prefix + case.tail:
            new.append(step)
        elif index == case.prefix:
            new.append(_other(rng, step, keep_action=keep_action, keep_target=not keep_action))
        else:
            new.append(_other(rng, step, keep_action=rng.random() < 0.5, keep_target=False))
    return tuple(old), tuple(new)


def observed_labels(old: tuple[Step, ...], new: tuple[Step, ...]) -> dict[str, JsonValue]:
    """输入：两条轨迹。输出：从轨迹**重算**出来的四个答案。

    预期行为：**审计侧的独立路径**——只看渲染出来的两条轨迹，不看 case。构造侧走
    `ReuseCase` 的属性，两条路径对不上就说明渲染没兑现计划。
    """
    if len(old) != len(new):
        raise ValueError(f"两条轨迹步数不同：{len(old)} vs {len(new)}")
    prefix = 0
    while prefix < len(old) and old[prefix] == new[prefix]:
        prefix += 1
    equivalent_total = sum(1 for a, b in zip(old, new, strict=True) if a == b)
    length = len(old)
    band = "none"
    for name in BANDS:
        if band_prefix(name, length) == prefix:
            band = name
    first_diff = prefix + 1
    if first_diff == 1:
        diverge = "first"
    elif first_diff == 2:
        diverge = "second"
    elif first_diff == length:
        diverge = "last"
    else:
        diverge = "middle"
    old_step, new_step = old[prefix], new[prefix]
    diff_kind = "action" if old_step[0] != new_step[0] else "target"
    return {
        "band": band,
        "diverge": diverge,
        "diff_kind": diff_kind,
        "reuse_half": prefix * 2 >= length,
        "prefix": prefix,
        "equivalent_total": equivalent_total,
        "first_diff": first_diff,
    }


# ---------- 真值规格 ----------


@dataclass(frozen=True)
class ReuseSpec:
    """一条缓存复用请求的完整真值：case + 由规则算出来的四个答案。"""

    case: ReuseCase
    split: str
    seq: int
    phrasing: int
    query_id: str
    prefix: int
    equivalent_total: int
    first_diff: int
    diverge: str
    diff_kind: str
    reuse_half: bool


def query_id_of(case: ReuseCase) -> str:
    """输入：一个 case。输出：请求号（由素材指纹派生，稳定可复算）。

    预期行为：同一份素材在任何切分、任何语言下都是同一个请求号——「同输入同标签」这条
    一致性校验要靠它认人。
    """
    digest = hashlib.sha256(case.material().encode("utf-8")).hexdigest()[:6].upper()
    return f"C-{digest}"


def sample_spec(
    rng: random.Random, case: ReuseCase, split: str, seq: int, phrasing: int
) -> ReuseSpec:
    """输入：RNG（**不用**）、case、切分、序号、问法。输出：真值规格。

    预期行为：不掷骰子——同一 case 必得同一份规格，这样「同素材同题必须同答」才成立。
    """
    del rng  # 规格完全由 case 决定；签名里的 RNG 只为与协议一致
    return ReuseSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        query_id=query_id_of(case),
        prefix=case.prefix,
        equivalent_total=case.equivalent_total,
        first_diff=case.first_diff,
        diverge=case.diverge_band(),
        diff_kind=case.diff_kind,
        reuse_half=case.reuse_half,
    )


# ---------- 值留出（训练侧见过哪些取值） ----------

#: 训练侧见过的缓存来源：最后一个（`kiosk`）只在 OOD 侧出现。
#: 来源只改抬头那一行、不动任何规则分支，所以留出它是安全的。
SEEN_FLAVORS: Final[frozenset[str]] = frozenset(
    flavor for flavor in FLAVORS if flavor != FLAVORS[-1]
)

#: 同分布侧的取值表（顺序按原表，不用 `sorted`——原表顺序本身是「哪个留出」的依据）。
ID_FLAVORS: Final[tuple[str, ...]] = tuple(
    flavor for flavor in FLAVORS if flavor in SEEN_FLAVORS
)
