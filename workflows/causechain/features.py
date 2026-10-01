"""根因定位（causechain）的特征层：依赖图 + 现场日志 + 纯函数算出来的金标签。

对应 PLAN 3.2「特征注入式」：一条「这次故障的根因是谁」的样本由三个**注入特征**描述——
`graph` 依赖图形状、`root` 故障注入在哪个服务、`probe` 这次问的是哪个服务。
**标签先行**：先定这三个取值，再由本模块的纯函数算出结论（根因 / 受影响集合 / 跳数）。

**为什么依赖图写成固定拓扑表**：真实微服务的调用关系是长尾的，但**判据**（错误沿上游传播、
根因 = 出错服务里没有出错下游的那个）与拓扑长什么样无关。拓扑由代码写死成四张图，
「谁调用谁」就成了一份可逐字核对的结构事实；模型只负责写**每个服务那一行日志**
（超时 / 报错 / 正常三种形状），三种形状由 `PATTERNS` 从文本**算**出来，模型碰不到标签。

**为什么要这一族**：社区里 `Stratum`（★51）把「日志异常检测与根因分析」明写成 Jev 的使用场景，
`claude-code-trace`（★370）从会话日志里判代理进度与恢复，`jevlogs` 在昂贵分析之前先做日志分诊。
`logtriage` 判的是**单行**日志该不该进告警通道，本 workflow 判的是**一整条时间线**上
「错误是从哪儿开始往上冒的」——判据空间（依赖图 + 传播规则）与素材形态（多服务时间线）
都不一样，不重复。

**`graph` 是这一族的留出维**：训练侧只见过 G1 / G2 / G3，留出侧是 G4。四张图都能把
根因、跳数档、受影响集合三族的**每一档**取到，所以留出一张图不会让任何档位消失——
留出验的是「读法不靠拓扑先验」。

**根因那条是超时还是报错，由 case 指纹派生**：它**只影响渲染、不影响标签**（标签只看
「谁出错」与依赖图）。一半的 case 里根因也是报错，所以「唯一的超时那条就是根因」这条捷径
只在另一半上碰巧成立，读法必须回到依赖图上。

本模块有**两条独立路径**：`label_for` 从计划推标签（构造侧），`observed_labels` 从文本重推
标签（审计侧，走 `events_of` 的正则）。两者对不上就说明渲染没兑现计划。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Final

from jc.cover import cover_dims, near_miss_combos
from jc.jsontypes import JsonValue

#: 六个服务位（**结构键**，不是文案：角色描述与选项文案全在 `locales/`）。
SERVICES: Final[tuple[str, ...]] = ("S1", "S2", "S3", "S4", "S5", "S6")

#: 四张固定的依赖图。每条边 `(上游, 下游)` 读作「上游调用下游」。
GRAPHS: Final[dict[str, tuple[tuple[str, str], ...]]] = {
    "G1": (("S1", "S2"), ("S2", "S3"), ("S3", "S4"), ("S4", "S5"), ("S5", "S6")),
    "G2": (("S1", "S2"), ("S1", "S3"), ("S1", "S4"), ("S2", "S5"), ("S3", "S6")),
    "G3": (("S1", "S2"), ("S1", "S3"), ("S2", "S4"), ("S3", "S4"), ("S4", "S5"), ("S4", "S6")),
    "G4": (("S1", "S2"), ("S1", "S3"), ("S2", "S4"), ("S2", "S5"), ("S3", "S6")),
}

#: 三种事件类型（**结构键**，文案在 `locales/` 的 `[event.*]`）。
EVENT_TIMEOUT: Final[str] = "timeout"
EVENT_ERROR: Final[str] = "error"
EVENT_OK: Final[str] = "ok"
EVENTS: Final[tuple[str, ...]] = (EVENT_TIMEOUT, EVENT_ERROR, EVENT_OK)

#: 事件形状。**这是全 workflow 唯一的判据实现**：构造侧、渲染期、判别器、审计共用。
#: 三条正则在**写好的日志上**互斥（提示词里逐条写明一行只许踩中一种）；正则本身并不互斥——
#: 一行里混着写就会同时命中两条，那时 `event_of` 给 None，管线据此重渲染。
#:
#: **这里列出的英文标记必须与 `locales/*.toml` 的 `[event.ok]` 一字不差地对上**：提示词里给
#: 模型看的唯一具体文字就是判据原文，语言文件承诺了哪个词、正则就得认哪个词，否则模型照承诺
#: 写出来的行会被判成「认不出的形状」白白丢掉（`handled fine` 与 `successfully` 就是这么漏的，
#: 见范式 §9.91）。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    EVENT_TIMEOUT: re.compile(
        r"(超时|等待超时|响应超时|连接超时|读超时|写超时)"
        r"|\b(timed?[ -]?out|timeout|deadline exceeded)\b",
        re.IGNORECASE,
    ),
    EVENT_ERROR: re.compile(
        r"(报错|错误|失败|异常|崩溃|挂了|返回 500|返回500)"
        r"|\b(error|errors|exception|failed|failure|panic|crash|traceback)\b"
        r"|returned 500",
        re.IGNORECASE,
    ),
    EVENT_OK: re.compile(
        r"(正常|成功|处理完成|已完成|健康|已恢复)"
        r"|\b(ok|success|successful|successfully|succeeded|healthy|completed"
        r"|handled|served|processed|finished|resolved|passed|normal)\b",
        re.IGNORECASE,
    ),
}

#: 服务在依赖图里的三种角色（**结构键**，文案在 `locales/` 的 `[role.*]`）。
ROLE_ENTRY: Final[str] = "entry"
ROLE_MIDDLE: Final[str] = "middle"
ROLE_LEAF: Final[str] = "leaf"
ROLES: Final[tuple[str, ...]] = (ROLE_ENTRY, ROLE_MIDDLE, ROLE_LEAF)

#: 探针取值：这次问的是哪个服务。
#:
#: - `h0` / `h1` / `h2`：问一个出错服务「离根因几跳」，分别取 0 跳 / 1 跳 / 2 跳以上；
#: - `ryes` / `rno`：问「这个服务是根因吗」，前者挑根因（答案 True）、后者挑非根因的出错服务。
#:
#: 五个探针的**可用次数由拓扑定死**：`h1` / `rno` 要求根因有调用方（4 张图 × 5 个非入口
#: 服务 = 20），`h2` 还要求根因的上游有两层（只有 12 组满足），`ryes` 每组都可行（24）。
#: 挑 case 的那一步把这些上限都用满，剩下的名额只能给 `h0`，所以全空间里是
#: `h0` 24 / `h1` 20 / `h2` 12 / `ryes` 24 / `rno` 20——跳数三档**不是**等分，
#: 布尔两边接近一半一半。这张表由 `tests/workflows/causechain/test_features.py` 钉住。
PROBES: Final[tuple[str, ...]] = ("h0", "h1", "h2", "ryes", "rno")

#: 跳数档位（**结构键**，文案在 `locales/` 的 `[hop.*]`）。
HOP_ZERO: Final[str] = "zero"
HOP_ONE: Final[str] = "one"
HOP_MORE: Final[str] = "more"
HOP_CLASSES: Final[tuple[str, ...]] = (HOP_ZERO, HOP_ONE, HOP_MORE)

#: 现场日志的正文长度上下限。太短读不出「像不像一行真日志」，太长就只是灌水。
MIN_CONTENT_CHARS: Final[int] = 12
MAX_CONTENT_CHARS: Final[int] = 90

#: 三维取值表（`probe` 的可行性随 `graph` / `root` 变，见 `build_cases`）。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {
    "graph": tuple(GRAPHS),
    "root": SERVICES,
    "probe": PROBES,
}


def callees(graph: str) -> dict[str, tuple[str, ...]]:
    """输入：图名。输出：{服务: 它直接调用的服务}（按 `SERVICES` 顺序，没有就是空元组）。"""
    table: dict[str, list[str]] = {service: [] for service in SERVICES}
    for upstream, downstream in GRAPHS[graph]:
        table[upstream].append(downstream)
    return {service: tuple(peers) for service, peers in table.items()}


def callers(graph: str) -> dict[str, tuple[str, ...]]:
    """输入：图名。输出：{服务: 直接调用它的服务}（按 `SERVICES` 顺序，没有就是空元组）。"""
    table: dict[str, list[str]] = {service: [] for service in SERVICES}
    for upstream, downstream in GRAPHS[graph]:
        table[downstream].append(upstream)
    return {service: tuple(peers) for service, peers in table.items()}


def ancestors(graph: str, node: str) -> tuple[str, ...]:
    """输入：图名、服务。输出：**能（沿调用方向）走到它**的服务（不含它自己，按 `SERVICES` 顺序）。

    预期行为：走的是「谁调用我」的反向边——错误沿调用链向上传播，所以根因的所有调用方
    （以及调用方的调用方）都会跟着出错。
    """
    incoming = callers(graph)
    seen: set[str] = set()
    stack = list(incoming[node])
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(incoming[current])
    return tuple(service for service in SERVICES if service in seen)


def hops_to(graph: str, node: str) -> dict[str, int]:
    """输入：图名、根因服务。输出：{服务: 从它沿调用方向走到根因的最少步数}。

    预期行为：广度优先，走不到的（根因的下游）不进表。根因自己记 0 跳。
    """
    incoming = callers(graph)
    distance = {node: 0}
    frontier = [node]
    while frontier:
        following: list[str] = []
        for current in frontier:
            for upstream in incoming[current]:
                if upstream not in distance:
                    distance[upstream] = distance[current] + 1
                    following.append(upstream)
        frontier = following
    return distance


def role_of(graph: str, service: str) -> str:
    """输入：图名、服务。输出：角色（`ROLES` 里的一个）。预期行为：没人调用=入口，不调用别人=末端。"""
    if not callers(graph)[service]:
        return ROLE_ENTRY
    if not callees(graph)[service]:
        return ROLE_LEAF
    return ROLE_MIDDLE


def graph_of(edges: tuple[tuple[str, str], ...]) -> str | None:
    """输入：从 state 里还原出来的边表。输出：它是四张图里的哪一张（都不是就给 None）。

    预期行为：按**集合**比，不比书写顺序——审计要的是「这张图是不是我们印的那张」，
    行序由渲染器保证，这里只判内容。
    """
    want = set(edges)
    for name, table in GRAPHS.items():
        if set(table) == want:
            return name
    return None


def hop_class(hop: int) -> str:
    """输入：跳数。输出：档位名。预期行为：0→zero、1→one、2 及以上→more（最后一档吃掉更大的数）。"""
    if hop <= 0:
        return HOP_ZERO
    return HOP_ONE if hop == 1 else HOP_MORE


#: `count` 题族的档位数：受影响服务数最少 1（只有根因自己出错）、最多 6（整张图都受影响）。
COUNT_LEVELS: Final[int] = len(SERVICES)


def count_index(errored: tuple[str, ...]) -> int:
    """输入：出错服务集合。输出：`count` 题族的答案下标（受影响服务数 − 1）。

    预期行为：纯函数，下标 0 起——「1 个服务」是下标 0。档位表在 `locales/*.toml` 的
    `[fam.count].levels`，长度必须等于 `COUNT_LEVELS`（有测试钉住）。
    """
    return len(errored) - 1


def events_of(text: str) -> tuple[str, ...]:
    """输入：一行日志。输出：按字面命中的事件类型（按 `EVENTS` 顺序，可能多个、可能为空）。"""
    return tuple(event for event in EVENTS if PATTERNS[event].search(text))


def event_of(text: str) -> str | None:
    """输入：一行日志。输出：**唯一**命中的事件类型；一个都没中、或中了两条以上都给 None。

    预期行为：三条形状**在写好的日志上**互斥（提示词里明令一行只许踩中一种），所以正常产物
    只会中一条；正则本身并不互斥——一行里混着写就会同时命中两条，那种样本按「认不出来」处理，
    渲染期会重渲染。
    """
    hits = events_of(text)
    return hits[0] if len(hits) == 1 else None


@dataclass(frozen=True)
class CauseCase:
    """一个注入特征组合：哪张依赖图 + 故障注入在哪个服务 + 这次问谁。

    字段：`graph` 图名；`root` 根因服务；`probe` 探针（`PROBES` 里的一个）。
    """

    graph: str
    root: str
    probe: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"graph": self.graph, "root": self.root, "probe": self.probe}

    def fingerprint(self) -> str:
        """输入：无。输出：case 指纹。预期行为：同一 case 任何进程都得到同一串。"""
        return "|".join(f"{key}={value}" for key, value in self.as_dict().items())

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.graph}-{self.root}-{self.probe}"


def errored_services(graph: str, root: str) -> tuple[str, ...]:
    """输入：图名、根因。输出：这次出错的服务（根因 + 它的全部调用方，按 `SERVICES` 顺序）。"""
    upstream = ancestors(graph, root)
    return tuple(service for service in SERVICES if service == root or service in upstream)


def probe_target(case: CauseCase, seed: int) -> str | None:
    """输入：case、派生随机种子。输出：这次要问的服务；该探针在这张图上不可行时给 None。

    预期行为：

    - `h0` 取 0 跳的那个（就是根因自己）、`h1` 取 1 跳的、`h2` 取 2 跳及以上的；
    - `ryes` 取根因、`rno` 取一个**非根因**的出错服务；
    - 候选集为空就给 None——调用方据此把这条 case 判为不可行（见 `build_cases`）。

    候选多于一个时由种子决定取哪个，所以同一条 case 在任何进程里都取到同一个服务。
    """
    distances = hops_to(case.graph, case.root)
    errored = errored_services(case.graph, case.root)
    if case.probe == "ryes":
        candidates = (case.root,)
    elif case.probe == "rno":
        candidates = tuple(service for service in errored if service != case.root)
    else:
        want = {"h0": 0, "h1": 1, "h2": 2}[case.probe]
        candidates = tuple(
            service
            for service in errored
            if (distances.get(service, 0) >= 2 if want == 2 else distances.get(service) == want)
        )
    if not candidates:
        return None
    return candidates[random.Random(seed).randrange(len(candidates))]


@dataclass(frozen=True)
class CausePlan:
    """一条样本的现场计划：每个服务是什么事件 + 日志怎么排 + 这次问谁 + 集合题的干扰项。

    字段：`graph` 图名；`root` 根因；`probe` 探针；`events` 服务→事件类型（六个服务都有）；
    `order` 现场日志的排列顺序（时刻序）；`probed` 这次问的服务（不可行时为 None，不会出现在产物里）；
    `distractors` 集合题的三个干扰集合。
    """

    graph: str
    root: str
    probe: str
    events: dict[str, str]
    order: tuple[str, ...]
    probed: str | None
    distractors: tuple[tuple[str, ...], ...]

    @property
    def errored(self) -> tuple[str, ...]:
        """输入：无。输出：这次出错的服务（按 `SERVICES` 顺序）。"""
        return errored_services(self.graph, self.root)

    @property
    def distances(self) -> dict[str, int]:
        """输入：无。输出：{出错服务: 离根因几跳}。"""
        distances = hops_to(self.graph, self.root)
        return {service: distances[service] for service in self.errored}

    def times(self) -> dict[str, str]:
        """输入：无。输出：{服务: 现场时刻标签}（`T1`、`T2`…… 按 `order` 排）。

        预期行为：时刻**只表示现场先后，与因果无关**——规则表里明写了这一点，所以「最早那条
        就是根因」是错的读法，必须回到依赖图上判。
        """
        return {service: f"T{index + 1}" for index, service in enumerate(self.order)}


def plan_for(case: CauseCase, phrasing: int) -> CausePlan:
    """输入：case、问法编号。输出：现场计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里都
    得到同一份计划（包括现场日志的排列、探针取哪个服务、集合题的三个干扰项）。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    rng = random.Random(seed)
    errored = errored_services(case.graph, case.root)
    # 根因那条是超时还是报错由指纹定（只影响渲染、不影响标签）：一半的 case 里根因也是报错，
    # 于是「唯一的超时那条就是根因」这条捷径只在另一半上碰巧成立。
    root_event = EVENT_TIMEOUT if rng.random() < 0.5 else EVENT_ERROR
    events = {
        service: (root_event if service == case.root else EVENT_ERROR)
        if service in errored
        else EVENT_OK
        for service in SERVICES
    }
    order = list(SERVICES)
    rng.shuffle(order)
    return CausePlan(
        graph=case.graph,
        root=case.root,
        probe=case.probe,
        events=events,
        order=tuple(order),
        probed=probe_target(case, seed),
        distractors=near_miss_combos(SERVICES, errored, rng),
    )


@dataclass(frozen=True)
class CauseLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`root` 根因；`errored` 出错服务；`distances` 出错服务→跳数；`probe` 探针；
    `probed` 这次问的服务；`distractors` 集合题的干扰项。
    """

    root: str
    errored: tuple[str, ...]
    distances: dict[str, int]
    probe: str
    probed: str | None
    distractors: tuple[tuple[str, ...], ...]

    def is_root(self, service: str) -> bool:
        """输入：服务。输出：它是不是这次故障的根因。"""
        return service == self.root

    @property
    def probed_is_root(self) -> bool:
        """输入：无。输出：这次问的那个服务是不是根因。预期行为：探针没落地时给 False。"""
        return self.probed is not None and self.probed == self.root

    @property
    def probed_hop_class(self) -> str | None:
        """输入：无。输出：这次问的那个服务落在哪一档跳数；探针没落地时给 None。"""
        if self.probed is None:
            return None
        return hop_class(self.distances[self.probed])

    @property
    def count_index(self) -> int:
        """输入：无。输出：`count` 题族的答案下标。预期行为：直接转发 `count_index(errored)`。"""
        return count_index(self.errored)


def label_for(plan: CausePlan) -> CauseLabels:
    """输入：现场计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`render.check_draft`
    负责保证文本真的兑现这份计划（每个服务那行按字面算出来的事件类型与计划逐条相同）。
    """
    return CauseLabels(
        root=plan.root,
        errored=plan.errored,
        distances=plan.distances,
        probe=plan.probe,
        probed=plan.probed,
        distractors=plan.distractors,
    )


def observed_labels(graph: str, logs: dict[str, str]) -> CauseLabels | None:
    """输入：依赖图名、{服务: 日志正文}。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认文本、只认字面：先按 `event_of` 把每个
    服务的正文翻成事件类型（认不出来的那条直接给 None），再由「出错集合」在图上重推根因。
    两个服务同时「没有出错下游」时说明文本兑现不了依赖图，给 None。

    探针与干扰项在这条路上重推不出来（它们不是从文本算的），所以留 None / 空——
    审计只比这几条能重算的：根因、出错集合、跳数。
    """
    events: dict[str, str] = {}
    for service in SERVICES:
        text = logs.get(service)
        if text is None:
            return None
        event = event_of(text)
        if event is None:
            return None
        events[service] = event
    errored = tuple(service for service in SERVICES if events[service] != EVENT_OK)
    if not errored:
        return None
    outgoing = callees(graph)
    roots = [
        service
        for service in errored
        if not any(peer in errored for peer in outgoing[service])
    ]
    if len(roots) != 1:
        return None
    root = roots[0]
    if errored != errored_services(graph, root):
        return None
    distances = hops_to(graph, root)
    return CauseLabels(
        root=root,
        errored=errored,
        distances={service: distances[service] for service in errored},
        probe="",
        probed=None,
        distractors=(),
    )


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[CauseCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某张图时用）。
    输出：`CauseCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。

    **可行性过滤**：探针要问的那个服务在这张图 + 这个根因下可能根本不存在（例如根因没有
    调用方时就问不出「1 跳」）。这类组合当场剔掉，绝不留给采样阶段去炸。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    cases: list[CauseCase] = []
    for row in rows:
        case = CauseCase(graph=str(row["graph"]), root=str(row["root"]), probe=str(row["probe"]))
        # 可行性只看候选集空不空，与随机种子无关，所以任何问法下都成立。
        if probe_target(case, 0) is None:
            continue
        cases.append(case)
    return cases
