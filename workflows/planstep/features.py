"""有界任务推进（planstep）的特征层：依赖图形状、已完成集合与四条推进规则。

社区依据：`foreman`（★406）用一个监督器让编程 agent 始终保持任务方向、`stanley-code`（★113）
把 agent 的工作流收进有界决策、`glowbom-oss`（★163）让 Jev 判断「这一步该不该继续」、
`jev-agent-design-with-topk-logits-choices`（★20）在若干候选提案里挑下一步。它们要解决的是
同一件事：**手里有一张有依赖关系的任务表，下一步该推进哪一项**。

本 workflow 把它做成**可判定的算术**，规则只有四条，全印在题面里：

1. 一项**可开工**，当且仅当它的状态是「待开始」、且它的**所有**前置项都已完成；
2. 可开工的项里，先做**依赖层数最浅**的；层数相同就做**编号最小**的。依赖层数 = 从最源头
   到这一项的最长前置链长度（没有前置就是 0）；
3. **一轮** = 把当时所有可开工的项一起标记为已完成；
4. 反复推进，直到没有可开工的项为止。

**为什么这是本仓库还没有的那个轴**：`clipcut` 的「顺序折叠」是**线性前缀**（一段的去留只看
它前面那几段），`cachereuse` 是两条轨迹的**逐步对齐**，`maze` / `invoice` 是**逐条独立判断**；
这里每一项的处境取决于它的**前置集合**是否全部完成——是**偏序**上的贪心推进，「前面」不再
是一条线而是一个下闭集。`elementpick` 那种「唯一解」也在这里出现：下一步永远是唯一的一项。

**两条独立路径**：`label_for` 从注入特征推标签（构造侧），`observed_labels` 从**反解出来的**
任务表重推标签（审计侧）。两者共用同一批纯函数，输入一个是 case、一个是 state，所以
「渲染把任务表写歪了」与「生成器把标签算错了」会被分开抓到。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jc.cover import Fingerprinted, cover_dims
from jc.jsontypes import JsonValue

#: 一张任务表固定 5 项（编号 0~4，渲染成 1~5）。
TASK_COUNT: Final[int] = 5

#: 依赖图形状表。每一项是它的前置项编号（空元组 = 没有前置，可以立刻开工）。
#
# 为什么是这 10 + 2 张图：`next` / `remaining` 两族的答案分布由「完成顺序是不是 0~4 的一个
# 排列」保证**天然均匀**（每张图的完成顺序都是全排列，每个编号在每张图里恰好出现一次），
# 而 `rounds` / `unlocks` 靠图与图的差别铺开——链式图每轮只推进一项（`rounds` 取满 0~5），
# 扇出图一轮能推进四项（`unlocks` 立刻翻面）。这 10 张同分布图逐族算过偏差与噪声地板，
# 四族全绿（口径见 `tests/workflows/planstep/test_pipeline.py` 的分布断言）。
#
# 后两张（下标 10 / 11）只出现在 `test_ood`：它们把「两个源头最后才合流」与「中途分叉再
# 汇合」这两种形状留出来，同分布侧一次都见不到。
GRAPHS: Final[tuple[tuple[tuple[int, ...], ...], ...]] = (
    ((), (0,), (1,), (2,), (3,)),           # 0 链式：一轮一项，`rounds` 取满 0~5
    ((1,), (2,), (3,), (4,), ()),           # 1 链式（倒着编号）：完成顺序是 4,3,2,1,0
    ((), (3,), (0,), (2,), (1,)),           # 2 链式（编号打乱）：完成顺序是 0,2,3,1,4
    ((), (0,), (0,), (1, 2), (3,)),         # 3 菱形：一轮能同时推进两项
    ((), (), (0,), (1,), (2, 3)),           # 4 双源扇入
    ((), (), (0, 1), (1,), (2, 3)),         # 5 交叉
    ((), (0,), (0,), (0,), (1, 2)),         # 6 扇出：一轮推进四项
    ((), (0,), (1,), (0, 2), (3,)),         # 7 混合
    ((), (0,), (0,), (1, 2), (1, 3)),       # 8 宽扇出
    ((), (0,), (1,), (0, 1), (2, 3)),       # 9 合流
    ((), (), (0,), (0, 1), (2, 3)),         # 10 迟合流（留出）
    ((), (0,), (1,), (0, 2), (1,)),         # 11 分叉再汇（留出）
)

#: 同分布侧的图形状下标。
ID_GRAPHS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
#: 留出侧的图形状下标（`test_ood` 用）。
OOD_GRAPHS: Final[tuple[int, ...]] = (10, 11)

#: 已完成项数的取值范围（0~5）：完成顺序的前 k 项已完成。
DONE_COUNTS: Final[tuple[int, ...]] = tuple(range(TASK_COUNT + 1))

#: 采样维度表：图形状 × 已完成项数 = 10 × 6 = 60 组（同分布侧）。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {
    "graph": ID_GRAPHS,
    "done": DONE_COUNTS,
}


def deps_of(graph: int) -> tuple[tuple[int, ...], ...]:
    """输入：图形状下标。输出：这张图的依赖表。预期行为：越界当场报错，不静默回绕。"""
    if not 0 <= graph < len(GRAPHS):
        raise ValueError(f"图形状下标越界：{graph}（可用 0~{len(GRAPHS) - 1}）")
    return GRAPHS[graph]


def depths_of(deps: tuple[tuple[int, ...], ...]) -> dict[int, int]:
    """输入：依赖表。输出：{项编号: 依赖层数}。

    预期行为：层数 = 从最源头到这一项的最长前置链长度（没有前置就是 0）。**有环直接报错**
    ——本表里每一张图都是无环的（`test_features.py` 逐张验过），有环说明表被改坏了，
    而「有环时下一步该怎么走」这个问题本 workflow 没定义，不能悄悄给一个数糊过去。
    """
    out: dict[int, int] = {}

    def visit(index: int, stack: set[int]) -> int:
        """输入：项编号、当前递归栈。输出：这一项的层数。预期行为：环上报人话。"""
        if index in out:
            return out[index]
        if index in stack:
            raise ValueError(f"依赖表里有环：{sorted(stack)} 又回到 {index}")
        stack.add(index)
        value = 0 if not deps[index] else 1 + max(visit(parent, stack) for parent in deps[index])
        stack.discard(index)
        out[index] = value
        return value

    for index in range(len(deps)):
        visit(index, set())
    return out


def ready_of(deps: tuple[tuple[int, ...], ...], done: frozenset[int]) -> tuple[int, ...]:
    """输入：依赖表、已完成集合。输出：现在可开工的项（升序）。

    预期行为：规则 1 的唯一实现——状态是「待开始」（不在 `done` 里）**且**所有前置项都已完成。
    """
    return tuple(
        index
        for index in range(len(deps))
        if index not in done and set(deps[index]) <= done
    )


def next_of(
    deps: tuple[tuple[int, ...], ...], done: frozenset[int], depths: dict[int, int]
) -> int | None:
    """输入：依赖表、已完成集合、层数表。输出：下一步该推进的项；一项都没有给 None。

    预期行为：规则 2 的唯一实现——可开工项里先比层数、再比编号，取最小的那一个。
    """
    ready = ready_of(deps, done)
    if not ready:
        return None
    return min(ready, key=lambda index: (depths[index], index))


def rounds_of(deps: tuple[tuple[int, ...], ...], done: frozenset[int]) -> int:
    """输入：依赖表、已完成集合。输出：还能推进几轮。

    预期行为：规则 3 + 4 的唯一实现——一轮把**当时**所有可开工的项一起做完，反复到没有可
    开工的项为止，数一共做了几轮。一轮都推不动就是 0。
    """
    current = set(done)
    count = 0
    while True:
        ready = ready_of(deps, frozenset(current))
        if not ready:
            return count
        current.update(ready)
        count += 1


def completion_order(deps: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """输入：依赖表。输出：从空计划开始、按规则 3 推进时的完成顺序。

    预期行为：一轮内先按编号升序排列（同一轮内的先后不影响任何答案，只是让顺序确定）。
    同一入参必得同一输出——case 矩阵按「完成顺序的前 k 项」定义已完成集合，靠它复现。
    """
    current: set[int] = set()
    order: list[int] = []
    while True:
        ready = ready_of(deps, frozenset(current))
        if not ready:
            return tuple(order)
        order.extend(ready)
        current.update(ready)


def unlocks_of(
    deps: tuple[tuple[int, ...], ...], done: frozenset[int], nxt: int | None
) -> bool:
    """输入：依赖表、已完成集合、下一步那一项。输出：做完它会不会**立刻**解锁别的任务。

    预期行为：存在另一项 `j`，它现在开不了工、但只差「下一步那一项」就齐了——即 `j` 的前置项
    里含 `nxt`、且除 `nxt` 之外的前置项都已做完。这一步做完能立刻多推进一项，就是「解锁」。
    没有下一步（计划已推不动）时给 False。
    """
    if nxt is None:
        return False
    after = done | {nxt}
    return any(
        index not in done
        and index != nxt
        and nxt in deps[index]
        and set(deps[index]) <= after
        for index in range(len(deps))
    )


@dataclass(frozen=True)
class PlanCase(Fingerprinted):
    """一条样本的注入特征：用哪一张依赖图 + 已经完成了几项（完成顺序的前 k 项）。

    字段：`graph` 图形状下标；`done` 已完成项数（0~5）。
    """

    graph: int
    done: int

    @property
    def deps(self) -> tuple[tuple[int, ...], ...]:
        """输入：无。输出：这一条的依赖表。"""
        return deps_of(self.graph)

    @property
    def done_set(self) -> frozenset[int]:
        """输入：无。输出：已完成项的集合（完成顺序的前 `done` 项）。"""
        return frozenset(completion_order(self.deps)[: self.done])

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"graph": self.graph, "done": self.done}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"G{self.graph}-D{self.done}"


@dataclass(frozen=True)
class PlanLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`remaining` 还剩几项没做完；`nxt` 下一步该推进的项（None = 推不动了）；
    `rounds` 还能推进几轮；`unlocks` 做完下一步会不会立刻解锁别的任务。
    """

    remaining: int
    nxt: int | None
    rounds: int
    unlocks: bool

    @property
    def next_key(self) -> str:
        """输入：无。输出：下一步那一项的选项名（`t1`~`t5`；没有就是 `none`）。"""
        return "none" if self.nxt is None else f"t{self.nxt + 1}"


def label_for(case: PlanCase) -> PlanLabels:
    """输入：注入特征。输出：全部金标签（**构造侧**路径）。"""
    return observed_labels(case.deps, case.done_set)


def observed_labels(
    deps: tuple[tuple[int, ...], ...], done: frozenset[int]
) -> PlanLabels:
    """输入：依赖表、已完成集合。输出：全部金标签（**审计侧**路径）。

    预期行为：构造侧与审计侧走的就是这一个函数——构造式 workflow 里 state 就是规格，
    没有「模型写的正文」可歪，所以两条路径的差别只在输入从哪来（case 还是反解结果）。
    """
    depths = depths_of(deps)
    nxt = next_of(deps, done, depths)
    return PlanLabels(
        remaining=len(deps) - len(done),
        nxt=nxt,
        rounds=rounds_of(deps, done),
        unlocks=unlocks_of(deps, done, nxt),
    )


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[PlanCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`PlanCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [PlanCase(graph=int(row["graph"]), done=int(row["done"])) for row in rows]
