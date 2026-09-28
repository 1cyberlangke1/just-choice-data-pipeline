"""游戏与模拟控制 workflow 的真值规格：一屏局面 + 一张动作规则表。

输入/输出：不可变数据类与纯函数；这里只有**键**（动作 / 耗 / 收益 / 冷却 / 回合 / 资源 /
战利品 / 排布），中英文案全在 `locales/`。
预期行为：答案只从渲染出来的那屏局面反推——能不能用按「资源不低于耗、且冷却为 0」判，
结没结束按「战利品到 6 或回合到 5」判，该用哪个按「能用的动作里收益最大的那个」定，
收益与耗的档就是数值本身。这些都是已知的**纯函数**，构造侧与判别侧共用同一套，
所以「表写错」与「渲染写歪」会变成同一类错误，不会两边各错一半还互相印证。

覆盖的决策形状（社区用例地图：游戏与模拟控制，50 个第三方项目，含 `system1-agents` /
`live-jev` / `jevarena` / `Jev-Trades` / `quackd` / `embodied-jev` / `jev-drone`）：
- **这个动作现在能不能用**（`can_move`，noul，逐位置 6 道，总是出）；
- **这一局结束了吗**（`finished`，noul，总是出）；
- **这个动作的收益是第几档**（`gain`，score 6 档，逐位置 6 道，总是出）；
- **该用第几个动作**（`best`，choice 6，只在还没结束时出）；
- **该用的那个动作要耗多少**（`cost`，score 4 档，只在还没结束时出）。

二十四个局面是从（资源 × 冷却组 × 回合 × 战利品）= 5 × 3 × 3 × 3 张里**按构造挑**的：
合法动作数合计恰好 72（占 144 的一半）、结束的恰好 12 张、没结束的 12 张都有合法动作、
该用的那个动作的耗四档各 3 次——`can_move` / `finished` 的 50/50、`gain` 的六档均匀、
`cost` 的四档均匀都是这么**构造**出来的，不靠事后过滤。六种排布是环形移位，每个动作在
每个位置上恰好出现一次，所以逐位置的合法率也恰好 50%、`best` 的答案位置也恰好均匀。
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

# ---------- 动作与两张规则表 ----------

#: 六个动作的固定顺序（位置就是它们在局面里的行序，排布会整体环移）。
ACTIONS: Final[tuple[str, ...]] = (
    "advance",
    "probe",
    "harvest",
    "guard",
    "strike",
    "idle",
)
#: 每个动作的耗（0 到 3 四档：待机 0、前进与探测 1、采集与戒备 2、突袭 3）。
COST_VALUE: Final[dict[str, int]] = {
    "advance": 1,
    "probe": 1,
    "harvest": 2,
    "guard": 2,
    "strike": 3,
    "idle": 0,
}
#: 每个动作的收益（0 到 5 六档，**六个值互不相同**，所以「该用哪个」不用掷骰子也有唯一解）。
GAIN_VALUE: Final[dict[str, int]] = {
    "advance": 2,
    "probe": 0,
    "harvest": 4,
    "guard": 1,
    "strike": 5,
    "idle": 3,
}
#: 战利品门槛：到了这一局就结束。
TARGET: Final[int] = 6
#: 回合上限：到了这一局就结束。
ROUND_CAP: Final[int] = 5

#: 耗四档（键 → 值），收益六档、冷却三档同理。
COSTS: Final[tuple[str, ...]] = ("c0", "c1", "c2", "c3")
GAINS: Final[tuple[str, ...]] = ("g0", "g1", "g2", "g3", "g4", "g5")
COOLDOWNS: Final[tuple[str, ...]] = ("k0", "k1", "k2")
#: 冷却组三档：只有 `c0` 一条都不限制，另外两组各让两个动作处在冷却里。
CD_SETS: Final[dict[str, tuple[int, ...]]] = {
    "c0": (0, 0, 0, 0, 0, 0),
    "c1": (0, 1, 2, 0, 1, 2),
    "c2": (0, 1, 2, 2, 0, 1),
}
CD_GROUPS: Final[tuple[str, ...]] = tuple(CD_SETS)

# ---------- 二十四个局面 ----------

#: 局面 = (资源 0..4, 冷却组, 回合 3..5, 战利品 4..6)。前六张留给测试的小规格。
BOARD_TABLE: Final[dict[str, tuple[int, str, int, int]]] = {
    "b0": (0, "c0", 4, 4),
    "b1": (3, "c1", 3, 4),
    "b2": (3, "c2", 3, 5),
    "b3": (2, "c2", 5, 5),
    "b4": (3, "c0", 5, 4),
    "b5": (3, "c0", 4, 6),
    "b6": (2, "c0", 3, 4),
    "b7": (2, "c0", 3, 5),
    "b8": (2, "c0", 4, 4),
    "b9": (1, "c0", 3, 4),
    "b10": (1, "c0", 3, 5),
    "b11": (3, "c1", 3, 5),
    "b12": (2, "c2", 4, 5),
    "b13": (3, "c2", 4, 4),
    "b14": (4, "c0", 3, 4),
    "b15": (2, "c0", 5, 4),
    "b16": (2, "c0", 5, 5),
    "b17": (3, "c1", 3, 6),
    "b18": (3, "c1", 5, 6),
    "b19": (3, "c1", 5, 5),
    "b20": (1, "c1", 5, 5),
    "b21": (3, "c1", 4, 6),
    "b22": (0, "c0", 5, 5),
    "b23": (4, "c1", 4, 6),
}
#: 全部局面名（顺序稳定）。
BOARDS: Final[tuple[str, ...]] = tuple(BOARD_TABLE)
#: 三张取值表的取值域（反解与校验按它认范围）。
TURNS: Final[tuple[int, ...]] = (3, 4, 5)
RESOURCES: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
HAULS: Final[tuple[int, ...]] = (4, 5, 6)

# ---------- 排布与句面 ----------

#: 六种排布：`l<i>` 把动作 j 摆到第 (j + i) mod 6 个位置上——每个动作在每个位置上恰好
#: 出现一次，逐位置的合法率与 `best` 的位置分布因此都是均匀的。
LAYOUTS: Final[tuple[str, ...]] = ("l0", "l1", "l2", "l3", "l4", "l5")
#: 两套句面：只改问法措辞，不改任何答案。
PHRASINGS: Final[tuple[int, ...]] = (0, 1)


def _require(field: str, value: object, pool: Sequence[JsonValue]) -> None:
    """输入：字段名、取值、取值域。输出：无。

    预期行为：认不出的键直接报错——静默落到某个默认分支会把「表改了、读法忘了改」
    变成一堆答案对不上的样本，而不是一条能定位的错误。
    """
    if value not in pool:
        raise KeyError(f"没有 {field}={value!r}（可用：{', '.join(map(str, pool))}）")


def board_of(name: str) -> tuple[int, str, int, int]:
    """输入：局面名。输出：(资源, 冷却组, 回合, 战利品)。预期行为：认不出直接报错。"""
    _require("board", name, BOARDS)
    return BOARD_TABLE[name]


def cost_key(action: str) -> str:
    """输入：动作键。输出：它的耗档键（`c<值>`）。预期行为：认不出直接报错。"""
    _require("action", action, ACTIONS)
    return f"c{COST_VALUE[action]}"


def gain_key(action: str) -> str:
    """输入：动作键。输出：它的收益档键（`g<值>`）。预期行为：认不出直接报错。"""
    _require("action", action, ACTIONS)
    return f"g{GAIN_VALUE[action]}"


def cooldown_key(board: tuple[int, str, int, int], index: int) -> str:
    """输入：局面、动作下标。输出：这个动作的冷却档键（`k<值>`）。

    预期行为：冷却只由局面的冷却组决定（资源/回合/战利品都不影响它）。
    """
    if not 0 <= index < len(ACTIONS):
        raise ValueError(f"动作下标要在 0..{len(ACTIONS) - 1} 之间，收到 {index}")
    return f"k{CD_SETS[board[1]][index]}"


def is_legal(board: tuple[int, str, int, int], index: int) -> bool:
    """输入：局面、动作下标。输出：这个动作现在能不能用。

    预期行为：**资源不低于它的耗、且它的冷却为 0**——两个条件缺一不可。资源那一条读的是
    局面抬头的资源格，冷却那一条读的是这一行的冷却格。
    """
    resources, cds, _turn, _haul = board
    return resources >= COST_VALUE[ACTIONS[index]] and CD_SETS[cds][index] == 0


def is_done(board: tuple[int, str, int, int]) -> bool:
    """输入：局面。输出：这一局结束了吗。

    预期行为：**战利品到 6、或者回合到 5** 就结束——两个信号指向同一个答案，是故意的：
    只要读得出抬头那两格，就能判，不用去数合法动作。
    """
    _resources, _cds, turn, haul = board
    return haul >= TARGET or turn >= ROUND_CAP


def best_index(board: tuple[int, str, int, int]) -> int | None:
    """输入：局面。输出：该用哪个动作的下标；一个能用的都没有给 None。

    预期行为：能用的动作里**收益最大的那个**。六个收益互不相同，所以不需要并列规则；
    一个能用的都没有时「该用哪个」没有答案，返回 None（调用方据此不出题）。
    """
    usable = [index for index in range(len(ACTIONS)) if is_legal(board, index)]
    if not usable:
        return None
    return max(usable, key=lambda index: GAIN_VALUE[ACTIONS[index]])


def cost_level(board: tuple[int, str, int, int]) -> int | None:
    """输入：局面。输出：该用的那个动作的耗是第几档；一个能用的都没有给 None。"""
    index = best_index(board)
    return None if index is None else COST_VALUE[ACTIONS[index]]


def gain_levels(board: tuple[int, str, int, int]) -> tuple[int, ...]:
    """输入：局面。输出：六个动作按**固定动作序**的收益档（位置序由排布再换算）。"""
    return tuple(GAIN_VALUE[action] for action in ACTIONS)


# ---------- 排布 ----------


def layout_of(layout: str) -> int:
    """输入：排布名。输出：位移量。预期行为：认不出直接报错。"""
    _require("layout", layout, LAYOUTS)
    return LAYOUTS.index(layout)


def action_at(layout: str, slot: int) -> int:
    """输入：排布名、位置（从 1 起）。输出：摆在那里的动作下标。

    预期行为：环形移位——每个动作在六个位置上各出现一次，逐位置的答案分布因此均匀。
    """
    shift = layout_of(layout)
    size = len(ACTIONS)
    if not 1 <= slot <= size:
        raise ValueError(f"位置要在 1..{size} 之间，收到 {slot}")
    return (slot - 1 - shift) % size


def slot_of(layout: str, index: int) -> int:
    """输入：排布名、动作下标。输出：它摆在第几个位置（从 1 起）。"""
    if not 0 <= index < len(ACTIONS):
        raise ValueError(f"动作下标要在 0..{len(ACTIONS) - 1} 之间，收到 {index}")
    size = len(ACTIONS)
    return (index + layout_of(layout)) % size + 1


# ---------- 局面的一行与整屏 ----------


@dataclass(frozen=True)
class SimRow:
    """局面的一行：动作 / 耗 / 收益 / 冷却（四样都是键）。"""

    action: str
    cost: str
    gain: str
    cooldown: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：四格（统计与调试用）。"""
        return {
            "action": self.action,
            "cost": self.cost,
            "gain": self.gain,
            "cooldown": self.cooldown,
        }

    def describe(self) -> str:
        """输入：无。输出：一行描述（报告与调试用）。"""
        return f"{self.action}/{self.cost}/{self.gain}/{self.cooldown}"


def rows_of(board: tuple[int, str, int, int], layout: str) -> tuple[SimRow, ...]:
    """输入：局面、排布。输出：位置 1..6 上依次是哪一行（顺序稳定）。

    预期行为：行序由**排布**定（不是动作的固定序），所以同一张局面换个排布就是另一屏局面。
    """
    out: list[SimRow] = []
    for slot in range(1, len(ACTIONS) + 1):
        index = action_at(layout, slot)
        action = ACTIONS[index]
        out.append(
            SimRow(
                action=action,
                cost=cost_key(action),
                gain=gain_key(action),
                cooldown=cooldown_key(board, index),
            )
        )
    return tuple(out)


def legal_slots(board: tuple[int, str, int, int], layout: str) -> tuple[bool, ...]:
    """输入：局面、排布。输出：位置 1..6 依次能不能用（逐位置的合法表）。"""
    return tuple(is_legal(board, action_at(layout, slot)) for slot in range(1, len(ACTIONS) + 1))


def best_slot(board: tuple[int, str, int, int], layout: str) -> int | None:
    """输入：局面、排布。输出：该用第几个动作（从 1 起）；一个能用的都没有给 None。"""
    index = best_index(board)
    return None if index is None else slot_of(layout, index)


# ---------- 只吃键的重算（判别器与审计走这条） ----------

#: 档位键 → 数值。判别器与审计把文本读回**键**之后，靠这三张表比大小。
COST_OF_KEY: Final[dict[str, int]] = {key: index for index, key in enumerate(COSTS)}
GAIN_OF_KEY: Final[dict[str, int]] = {key: index for index, key in enumerate(GAINS)}
COOLDOWN_OF_KEY: Final[dict[str, int]] = {key: index for index, key in enumerate(COOLDOWNS)}


class RowLike(Protocol):
    """判别器与审计眼里的一行：四格都是键（真值规格与反解结果都满足）。"""

    action: str
    cost: str
    gain: str
    cooldown: str


class HeadLike(Protocol):
    """判别器与审计眼里的抬头三格。"""

    turn: int
    resources: int
    haul: int


def legal_of_row(row: RowLike, resources: int) -> bool:
    """输入：一行（键）、抬头的资源。输出：这一行的动作现在能不能用。

    预期行为：只吃这一行的耗与冷却两格、外加抬头的资源格——判别器把文本读回来重算时
    走的就是这条，所以「渲染把耗那一格写歪」会直接变成答案对不上。
    """
    return resources >= COST_OF_KEY[row.cost] and COOLDOWN_OF_KEY[row.cooldown] == 0


def best_slot_of_rows(rows: Sequence[RowLike], resources: int) -> int | None:
    """输入：六行（键）、抬头的资源。输出：该用第几个动作（从 1 起）；一个能用的都没有给 None。

    预期行为：能用的行里收益最大的那个；并列取位置最靠前的（六个收益互不相同，实际不会并列）。
    """
    usable = [slot for slot, row in enumerate(rows, start=1) if legal_of_row(row, resources)]
    if not usable:
        return None
    return max(usable, key=lambda slot: GAIN_OF_KEY[rows[slot - 1].gain])


def cost_level_of_rows(rows: Sequence[RowLike], resources: int) -> int | None:
    """输入：六行（键）、抬头的资源。输出：该用的那一行的耗档；一个能用的都没有给 None。"""
    slot = best_slot_of_rows(rows, resources)
    return None if slot is None else COST_OF_KEY[rows[slot - 1].cost]


def gain_level_of_row(row: RowLike) -> int:
    """输入：一行（键）。输出：它的收益档。"""
    return GAIN_OF_KEY[row.gain]


def done_of_head(head: HeadLike) -> bool:
    """输入：抬头（回合与战利品）。输出：这一局结束了吗。"""
    return head.haul >= TARGET or head.turn >= ROUND_CAP


# ---------- 值留出（训练侧见过哪些取值） ----------

#: 训练侧见过的排布：最后一种整个留出。
SEEN_LAYOUTS: Final[frozenset[str]] = frozenset(LAYOUTS[:-1])
#: 训练侧见过的句面：第二套句面整个留出。它**不影响任何答案**，所以留出它不会把哪一族的
#: 分布压歪。
SEEN_PHRASINGS: Final[frozenset[int]] = frozenset({PHRASINGS[0]})


# ---------- 采样参数 ----------


@dataclass(frozen=True)
class SimCase:
    """一条样本的素材：局面 + 排布 + 句面。"""

    board: str
    layout: str
    phrasing: int

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：三个维度认不出就当场报人话。"""
        _require("board", self.board, BOARDS)
        layout_of(self.layout)
        if self.phrasing not in PHRASINGS:
            raise KeyError(f"没有句面 {self.phrasing!r}（可用：{PHRASINGS}）")

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这条素材里训练侧没见过的维度名（排序后）。"""
        out: list[str] = []
        if self.layout not in SEEN_LAYOUTS:
            out.append("layout")
        if self.phrasing not in SEEN_PHRASINGS:
            out.append("phrasing")
        return tuple(sorted(out))

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：{维度名: 取值}（三个维度全摊平）。"""
        return {"board": self.board, "layout": self.layout, "phrasing": self.phrasing}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.board}/{self.layout}/p{self.phrasing}"


def material_of(case: SimCase) -> str:
    """输入：case。输出：素材指纹（切分按它哈希分桶，同一屏局面整组不跨切分）。"""
    return f"{case.board}|{case.layout}|{case.phrasing}"


@dataclass(frozen=True)
class SimConfig:
    """一个切分的采样参数：三张取值表 + 是否只留 OOD。"""

    boards: tuple[str, ...] = BOARDS
    layouts: tuple[str, ...] = LAYOUTS
    phrasings: tuple[int, ...] = PHRASINGS
    split: str = "train"
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：任一取值表为空就报人话（点名是哪个字段）。"""
        require_pool("boards", self.boards)
        require_pool("layouts", self.layouts)
        require_pool("phrasings", self.phrasings)


def build_cases(config: SimConfig) -> list[SimCase]:
    """输入：采样参数。输出：case 矩阵（顺序稳定）。

    预期行为：三张表取全组合（**不做成立性过滤**——三个维度两两不矛盾）；`ood_only` 时
    只留「至少含一个训练侧没见过的取值」的组合，所以 OOD 与同分布侧按构造就不重叠。
    """
    out: list[SimCase] = []
    for board, layout, phrasing in product(config.boards, config.layouts, config.phrasings):
        case = SimCase(board, layout, phrasing)
        if config.ood_only and not case.unseen_dims():
            continue
        out.append(case)
    return out


# ---------- 真值规格 ----------

REQUEST_PREFIX: Final[str] = "S-"


@dataclass(frozen=True)
class SimSpec:
    """一屏局面的完整真值：case + 由规则算出来的一切。"""

    case: SimCase
    split: str
    seq: int
    phrasing: int
    request_id: str
    resources: int
    turn: int
    haul: int
    rows: tuple[SimRow, ...]
    legal: tuple[bool, ...]
    gains: tuple[int, ...]
    done: bool
    best_slot: int | None
    cost_level: int | None


def request_id_of(case: SimCase) -> str:
    """输入：case。输出：请求号（形如 `S-48213`）。

    预期行为：由素材指纹的 sha256 派生——确定性，而且**不泄露切分**（用序号或切分名拼
    请求号会让同一屏局面在不同切分里拿到不同请求号，等于把切分信息写进 state）。
    """
    digest = hashlib.sha256(material_of(case).encode("utf-8")).hexdigest()
    return f"{REQUEST_PREFIX}{10000 + int(digest[:6], 16) % 90000}"


def sample_spec(case: SimCase, split: str, seq: int, phrasing: int, rng: random.Random) -> SimSpec:
    """输入：case、切分、序号、句面、RNG。输出：真值规格。

    预期行为：局面、合法表与答案都由 case 决定（**不掷骰子**）——这个 workflow 的每个
    自由度都已经在 case 里，`rng` 只为与其它 workflow 的签名一致，收下就丢。
    """
    del rng
    board = board_of(case.board)
    return SimSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        request_id=request_id_of(case),
        resources=board[0],
        turn=board[2],
        haul=board[3],
        rows=rows_of(board, case.layout),
        legal=legal_slots(board, case.layout),
        gains=tuple(
            GAIN_VALUE[ACTIONS[action_at(case.layout, slot)]]
            for slot in range(1, len(ACTIONS) + 1)
        ),
        done=is_done(board),
        best_slot=best_slot(board, case.layout),
        cost_level=cost_level(board),
    )


def spec_to_dict(spec: SimSpec) -> dict[str, JsonValue]:
    """输入：规格。输出：可 JSON 序列化的摘要（写进 manifest 与调试，不进样本行）。"""
    return {
        "case": spec.case.describe(),
        "unseen_dims": list(spec.case.unseen_dims()),
        "request_id": spec.request_id,
        "resources": spec.resources,
        "turn": spec.turn,
        "haul": spec.haul,
        "done": spec.done,
        "best_slot": spec.best_slot,
        "cost_level": spec.cost_level,
        "legal": list(spec.legal),
        "gains": list(spec.gains),
        "rows": [row.describe() for row in spec.rows],
    }
