"""来源选择（sourceselect）的特征层：注入维度 + 候选表计划 + 选源规则（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一次「这条查询该用哪个来源」的判读由 4 个**注入特征**描述，
**标签先行**——先定这 4 维取值，再由本模块的纯函数算出全部金标签。

四维分成两拨（这条分法是设计的第一步）：

- **由文本承载**（提示词给、状态行不给）：查询对来源的**首要要求** `need`——真实请求的语气
  是长尾的，模板拼出来的「我要权威资料」不像人话；这句要求交给模型写，再由
  `judges.TraceJudge` 反解校验；
- **由代码渲染的候选表承载**（提示词不给）：表长 `size`、表的形状 `shape`（在首要要求那一项
  上达标的行有几条）、有没有一条「只差首要要求那一项」的诱饵行 `near_miss`——这三件事是
  结构化事实，让模型去写只会写歪。

候选表**每一行**由 `plan_for` 从 case 指纹播种派生（不是一张写死的目录）：同一 case 在任何
进程、任何切分里都得到同一张表，而不同 case 的表长得不一样。

选源口径（印在 state 里，考生只读 state 与题面就能自己算，范式 §9.8）：

1. **先过滤**：查询要求哪一项（权威 / 时效 / 覆盖），就只保留在这一项上达标的行；
2. **再取最优**：幸存的行不止一条时，依次比 类型（official > community > unknown）→
   时效（fresh > recent > stale）→ 覆盖度（full > partial > none）→ 表里序号靠前的；
3. **都没有就弃权**：一条幸存的行都没有时，答案是「都不合适」，不许将就。

输入/输出：只有常量与纯函数；不碰网络、不碰文件、**绝不发任何请求**。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Final

from jc.cover import Fingerprinted, cover_dims
from jc.jsontypes import JsonValue

# ---------- 取值表 ----------

#: 查询对来源的**首要要求**。**只由那句查询承载**：候选表与选源口径里一个字都不写它。
NEEDS: Final[tuple[str, ...]] = (
    "authority",  # 要权威：想要一手、官方的口径
    "freshness",  # 要最新：想要最近的进展或版本
    "coverage",  # 要覆盖最全：想要把这件事讲全的汇总
)

#: 候选表条数。
SIZES: Final[tuple[int, ...]] = (3, 4, 5)

#: 表的形状：在首要要求那一项上**达标**的行有几条。
SHAPES: Final[tuple[str, ...]] = (
    "unique",  # 恰好一条（用不上第二条规则）
    "tied",  # 两条（必须按第二条规则分出胜负）
    "none",  # 零条（答案只能是「都不合适」）
)

#: 表里有没有一条「只差首要要求那一项、其余两项都取到最高档」的诱饵行。
NEAR_MISSES: Final[tuple[int, ...]] = (0, 1)

#: 候选行的三个属性。都是 ASCII 词元（渲染进 state，反解不用跨语言试）。
TYPES: Final[tuple[str, ...]] = ("official", "community", "unknown")
FRESHNESSES: Final[tuple[str, ...]] = ("fresh", "recent", "stale")
COVERAGES: Final[tuple[str, ...]] = ("full", "partial", "none")

#: 属性名 → 它的取值表。
AXIS_VALUES: Final[dict[str, tuple[str, ...]]] = {
    "type": TYPES,
    "freshness": FRESHNESSES,
    "coverage": COVERAGES,
}

#: 属性名 → 取值 → 优劣档（越大越好）。第二条规则就是按这个档比。
AXIS_RANK: Final[dict[str, dict[str, int]]] = {
    "type": {"official": 2, "community": 1, "unknown": 0},
    "freshness": {"fresh": 2, "recent": 1, "stale": 0},
    "coverage": {"full": 2, "partial": 1, "none": 0},
}

#: 候选表的**固定列序**。第二条规则的比较顺序就是它。
AXIS_ORDER: Final[tuple[str, ...]] = ("type", "freshness", "coverage")

#: 首要要求 → (它落在哪个属性上, 达标取值)。
REQUIREMENT: Final[dict[str, tuple[str, str]]] = {
    "authority": ("type", "official"),
    "freshness": ("freshness", "fresh"),
    "coverage": ("coverage", "full"),
}

#: 选源口径的三条（渲染进 state 的规则段；顺序即语义，`test_features.py` 钉住它）。
RULES: Final[tuple[str, ...]] = ("filter", "rank", "abstain")

#: 弃权选项名（`pick` 那一族的选项名随表长变化，由 `questions` 现场拼）。
NO_PICK: Final[str] = "none"

#: 注入维度表（`cover_dims` 的入参）。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {
    "need": NEEDS,
    "size": SIZES,
    "shape": SHAPES,
    "near_miss": NEAR_MISSES,
}


@dataclass(frozen=True)
class SourceRow:
    """候选来源表里的一行。三个属性都是 ASCII 词元。"""

    type: str
    freshness: str
    coverage: str

    def get(self, axis: str) -> str:
        """输入：属性名。输出：这一行在该属性上的取值。预期行为：属性名写错直接报错。"""
        if axis == "type":
            return self.type
        if axis == "freshness":
            return self.freshness
        if axis == "coverage":
            return self.coverage
        raise KeyError(f"未知属性：{axis!r}（可用：{', '.join(AXIS_ORDER)}）")

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序按 `AXIS_ORDER`。"""
        return {"type": self.type, "freshness": self.freshness, "coverage": self.coverage}

    def rank_key(self) -> tuple[int, int, int]:
        """输入：无。输出：三个属性的优劣档（按 `AXIS_ORDER`）。第二条规则比它。"""
        return (
            AXIS_RANK["type"][self.type],
            AXIS_RANK["freshness"][self.freshness],
            AXIS_RANK["coverage"][self.coverage],
        )


@dataclass(frozen=True)
class SourceCase(Fingerprinted):
    """一条样本的注入特征：首要要求、表长、表的形状、有没有诱饵行。"""

    need: str
    size: int
    shape: str
    near_miss: int

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {
            "need": self.need,
            "size": self.size,
            "shape": self.shape,
            "near_miss": self.near_miss,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.need}-{self.size}行-{self.shape}-诱饵{self.near_miss}"


@dataclass(frozen=True)
class SourcePlan:
    """一条样本的候选表计划：行由 case 指纹播种派生，与语言、后端无关。

    字段：`need` 首要要求；`shape` 表的形状；`near_miss` 有没有诱饵行；
    `rows` 候选表（顺序就是 state 里的展示顺序，序号从 1 起）。
    """

    need: str
    shape: str
    near_miss: int
    rows: tuple[SourceRow, ...]

    @property
    def size(self) -> int:
        """输入：无。输出：候选表条数。"""
        return len(self.rows)


@dataclass(frozen=True)
class SourceLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`need` 首要要求（文本承载）；`pick` 选中的行号（1 起，`None` = 都不合适）；
    `enough` 有没有达标行；`cover` 选中行的覆盖档（0/1/2，弃权时 0）；
    `near_miss` 表里有没有诱饵行。
    """

    need: str
    pick: int | None
    enough: bool
    cover: int
    near_miss: bool

    @property
    def pick_key(self) -> str:
        """输入：无。输出：`pick` 那一族的选项名（`s<行号>` 或 `none`）。"""
        return pick_key(self.pick)


def pick_key(index: int | None) -> str:
    """输入：行号（1 起）或 `None`。输出：`pick` 那一族的选项名。"""
    return NO_PICK if index is None else f"s{index}"


def value_at_rank(axis: str, rank: int) -> str:
    """输入：属性名、优劣档。输出：该档对应的取值。

    预期行为：三个属性各有三个取值、档位两两不同，所以这是个一一映射；档位越界直接报错。
    """
    for value in AXIS_VALUES[axis]:
        if AXIS_RANK[axis][value] == rank:
            return value
    raise KeyError(f"{axis} 没有档位 {rank}")


def qualifies(row: SourceRow, need: str) -> bool:
    """输入：一行、首要要求。输出：这一行在首要要求那一项上达标吗。"""
    axis, good = REQUIREMENT[need]
    return row.get(axis) == good


def is_near_miss(row: SourceRow, need: str) -> bool:
    """输入：一行、首要要求。输出：这一行是不是「只差首要要求那一项」的诱饵行。

    预期行为：在首要要求那一项上**不**达标，而其余两项都取到最高档——只扫三列的话，
    它看起来最像能用的那一条。
    """
    axis, good = REQUIREMENT[need]
    if row.get(axis) == good:
        return False
    return all(
        AXIS_RANK[other][row.get(other)] == max(AXIS_RANK[other].values())
        for other in AXIS_ORDER
        if other != axis
    )


def survivor_indices(rows: tuple[SourceRow, ...], need: str) -> tuple[int, ...]:
    """输入：候选表、首要要求。输出：达标行的下标（0 起，按表序）。"""
    return tuple(index for index, row in enumerate(rows) if qualifies(row, need))


def winner_index(rows: tuple[SourceRow, ...], need: str) -> int | None:
    """输入：候选表、首要要求。输出：该用哪一行（下标，0 起）；没人达标就是 `None`。

    预期行为：只在达标行里挑，取优劣档最大、并列取表里序号靠前的那个——这就是印在 state
    里的第二条规则，一个字都不多。
    """
    alive = survivor_indices(rows, need)
    if not alive:
        return None
    return max(alive, key=lambda index: (rows[index].rank_key(), -index))


def _seed(case: SourceCase, phrasing: int) -> int:
    """输入：case、问法编号。输出：种子。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里
    都得到同一张表；问法编号也进种子，是为了让同一个 case 的不同问法拿到不同的表
    （否则这几种问法会渲染出结构完全相同的样本）。
    """
    payload = f"{case.fingerprint()}|{phrasing}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


#: 除首要要求那一项外，另外两项的档位组合（九格）。诱饵行取最高的 (2, 2)。
_OTHER_COMBOS: Final[tuple[tuple[int, int], ...]] = tuple(
    (left, right) for left in range(3) for right in range(3)
)

#: 非诱饵行能取的档位组合。**必须避开 (2, 2)**——让别的行也凑成「其余两项都最高」，
#: `near_miss` 那一维就不是注入的那个值了。
_NON_DECOY_COMBOS: Final[tuple[tuple[int, int], ...]] = tuple(
    pair for pair in _OTHER_COMBOS if pair != (2, 2)
)


def plan_for(case: SourceCase, phrasing: int) -> SourcePlan:
    """输入：case、问法编号。输出：候选表计划（与语言、后端无关的随机）。

    预期行为：四种形状都**按构造**兑现注入的那一维：
    - `unique` 恰好一条达标行；
    - `tied` 恰好两条达标行，而且**靠后的那条优劣档严格更大**——所以「取第一条达标的」
      这个捷径必然答错，第二条规则真的在承重；
    - `none` 一条达标行都没有；
    - `near_miss=1` 时恰好一条诱饵行（`near_miss=0` 时一条都没有）。
    """
    rng = random.Random(_seed(case, phrasing))
    axis, good = REQUIREMENT[case.need]
    others = [name for name in AXIS_ORDER if name != axis]
    bad_values = tuple(value for value in AXIS_VALUES[axis] if value != good)

    count = {"unique": 1, "tied": 2, "none": 0}[case.shape]
    positions = list(range(case.size))
    rng.shuffle(positions)
    survivors = sorted(positions[:count])
    decoys = sorted(positions[count : count + case.near_miss])

    rows: list[SourceRow | None] = [None] * case.size

    if case.shape == "tied":
        # 两条达标行：靠后的那条在另外两项上严格更好。档位对从 `_OTHER_COMBOS` 里挑两个
        # 递增的下标——(0,0) 到 (2,2) 一共九格，挑法固定，所以「谁赢」是构造出来的。
        low = rng.randrange(len(_OTHER_COMBOS) - 1)
        high = rng.randrange(low + 1, len(_OTHER_COMBOS))
        for index, combo in zip(survivors, (_OTHER_COMBOS[low], _OTHER_COMBOS[high])):
            values = {axis: good}
            for name, rank in zip(others, combo):
                values[name] = value_at_rank(name, rank)
            rows[index] = SourceRow(**values)
    else:
        for index in survivors:
            values = {axis: good}
            for name in others:
                values[name] = rng.choice(AXIS_VALUES[name])
            rows[index] = SourceRow(**values)

    for index in decoys:
        values = {axis: rng.choice(bad_values)}
        for name in others:
            values[name] = value_at_rank(name, 2)
        rows[index] = SourceRow(**values)

    for index in range(case.size):
        if rows[index] is not None:
            continue
        values = {axis: rng.choice(bad_values)}
        combo = rng.choice(_NON_DECOY_COMBOS)
        for name, rank in zip(others, combo):
            values[name] = value_at_rank(name, rank)
        rows[index] = SourceRow(**values)

    return SourcePlan(
        need=case.need,
        shape=case.shape,
        near_miss=case.near_miss,
        rows=tuple(row for row in rows if row is not None),
    )


def label_for(plan: SourcePlan) -> SourceLabels:
    """输入：候选表计划。输出：全部金标签。

    预期行为：**纯函数、只看表、不看文本**——表先定，标签由表算。四件事：
    - `pick` / `enough`：按印在 state 里的两条规则在表上筛一遍；
    - `cover`：选中那一行的覆盖档（弃权时 0 档）；
    - `near_miss`：表里有没有「只差首要要求那一项」的行——**从表本身重算**，不读注入值，
      这样审计拿产物里的表也能算出同一个答案。
    """
    winner = winner_index(plan.rows, plan.need)
    cover = 0 if winner is None else AXIS_RANK["coverage"][plan.rows[winner].coverage]
    return SourceLabels(
        need=plan.need,
        pick=None if winner is None else winner + 1,
        enough=winner is not None,
        cover=cover,
        near_miss=any(is_near_miss(row, plan.need) for row in plan.rows),
    )


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[SourceCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`SourceCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [
        SourceCase(
            need=str(row["need"]),
            size=int(row["size"]),
            shape=str(row["shape"]),
            near_miss=int(row["near_miss"]),
        )
        for row in rows
    ]


#: choice 题族允许出现的固定选项名（`pick` 的选项名随表长变化，由 `questions` 现场拼）。
CHOICE_OPTIONS: Final[dict[str, tuple[str, ...]]] = {"need": NEEDS}
