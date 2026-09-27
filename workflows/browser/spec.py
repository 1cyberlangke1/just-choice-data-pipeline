"""界面操作 workflow 的真值规格：一张元素表 + 一句任务 + 一张动作表。

输入/输出：不可变数据类与纯函数；这里只有**键**（角色 / 状态 / 任务意图 / 动作 / 对象），
中英文案全在 `locales/`。
预期行为：答案只从渲染出来的那屏界面反推——动作表、可用判据、完成判据、敏感清单都是已知的
**纯函数**，构造侧与判别侧共用同一套，所以「表写错」与「渲染写歪」会变成同一类错误，
不会两边各错一半还互相印证。

覆盖的决策形状（社区用例地图：浏览器与计算机操作，39 个第三方项目，含 `jev-browser` /
`Jev-cu` / `typesafe-computer-use` / `arc-cua` / `jev-use` / `unclutter`）：
从一屏界面元素里选**下一步操作哪个元素、做什么动作、这一步要不要人工确认、任务是不是
已经完成**。
- **完成度**（`done`，noul）：任务是不是已经完成（总是出）；
- **定位**（`target`，choice 5）：下一步该操作第几个元素（只在任务未完成时出）；
- **动作**（`action`，choice 4）：这一步做什么动作（只在任务未完成时出）；
- **风险**（`risk`，noul）：这一步要不要先人工确认（只在任务未完成时出）。

元素表固定 5 行：一个**目标槽**（任务要操作的那个对象）、一个**干扰槽**（同角色、别的对象）、
三个**陪衬槽**（另外三个角色各一个）。目标槽的状态在「可用 / 已是当前」之间切换——这正是
`done` 的标签来源；排布用拉丁方（环形移位），所以 `target` 的答案位置在五个位置上均匀
（范式 §5.7）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import product
from typing import Any, Final, Protocol

from jc.knobs import require_pool

# ---------- 元素的三格（键；文案在 locales/） ----------

#: 四种角色。
ROLES: Final[tuple[str, ...]] = ("field", "link", "tab", "button")
#: 四种状态：可用 / 已是当前 / 置灰不可点 / 不可见。
STATES: Final[tuple[str, ...]] = ("ready", "current", "disabled", "hidden")

# ---------- 任务意图与动作 ----------

#: 四种任务意图。
TASKS: Final[tuple[str, ...]] = ("fill", "open", "switch", "submit")
#: 四种动作。
ACTIONS: Final[tuple[str, ...]] = ("type", "open", "switch", "click")

#: 任务意图 → (要操作的角色, 该做的动作)。这张表整段印在题面里（范式 §9.8）。
INTENT: Final[dict[str, tuple[str, str]]] = {
    "fill": ("field", "type"),
    "open": ("link", "open"),
    "switch": ("tab", "switch"),
    "submit": ("button", "click"),
}

# ---------- 对象（键；文案在 locales/） ----------

#: 任务 → (普通对象, 命中敏感清单的对象)。两个都归这个任务要操作的角色。
OBJECTS: Final[dict[str, tuple[str, str]]] = {
    "fill": ("note", "pay_password"),
    "open": ("monthly_report", "unsubscribe"),
    "switch": ("export_view", "pay_history"),
    "submit": ("draft", "order"),
}
#: 干扰槽的对象：与目标槽**同角色、不同对象**（「只按角色找」这条捷径因此不够）。
DECOYS: Final[dict[str, str]] = {
    "fill": "phone",
    "open": "help",
    "switch": "overview",
    "submit": "cancel",
}
#: 陪衬槽的对象：该角色不是任务要操作的角色时用它。
NEUTRALS: Final[dict[str, str]] = {
    "field": "address",
    "link": "manual",
    "tab": "history",
    "button": "refresh",
}
#: 命中敏感清单的对象（键）。它们渲染出来的标签里都含清单里的词，由 `test_locales` 锁住。
SENSITIVE_OBJECTS: Final[frozenset[str]] = frozenset(
    {"pay_password", "unsubscribe", "pay_history", "order"}
)

# ---------- 六个自由度（键） ----------

#: 对象档：普通 / 命中敏感清单。
FORMS: Final[tuple[str, ...]] = ("plain", "sensitive")
#: 完成档：目标槽的状态是「可用」（没做完）还是「已是当前」（做完了）。
DONES: Final[tuple[bool, ...]] = (False, True)
#: 干扰槽的状态：三种都不可用，只是形态不同（其中「已是当前」专门压「看到当前就算完成」）。
DECOY_STATES: Final[tuple[str, ...]] = ("disabled", "hidden", "current")

#: 五种排布。`p<i>` 把第 j 个槽放到第 (j+i) mod 5 个位置——每种槽在每个位置上恰好一次。
ORDERS: Final[tuple[str, ...]] = ("p0", "p1", "p2", "p3", "p4")
#: 句面（只影响措辞，不影响答案）。
PHRASINGS: Final[tuple[int, ...]] = (0, 1)

#: 五个槽：目标槽、干扰槽，加三个陪衬槽（角色由任务意图补出来）。
SLOTS: Final[tuple[str, ...]] = ("tgt", "dec", "aux0", "aux1", "aux2")


# ---------- 规则函数（构造与判别共用） ----------


def _require(field: str, value: str, pool: tuple[str, ...]) -> None:
    """输入：字段名、取值、取值表。输出：无。

    预期行为：认不出的键直接报错——静默落到某个默认分支会把「表改了、读法忘了改」
    变成一堆答案对不上的样本，而不是一条能定位的错误。
    """
    if value not in pool:
        raise KeyError(f"没有 {field}={value!r}（可用：{', '.join(pool)}）")


def target_role(task: str) -> str:
    """输入：任务意图。输出：要操作的角色。预期行为：认不出的意图直接报错。"""
    _require("task", task, TASKS)
    return INTENT[task][0]


def action_of(task: str) -> str:
    """输入：任务意图。输出：该做的动作。预期行为：认不出的意图直接报错。"""
    _require("task", task, TASKS)
    return INTENT[task][1]


def object_key(task: str, form: str) -> str:
    """输入：任务意图、对象档。输出：目标槽的对象键。预期行为：任一个认不出都报错。"""
    _require("task", task, TASKS)
    _require("form", form, FORMS)
    plain, sensitive = OBJECTS[task]
    return sensitive if form == "sensitive" else plain


def aux_roles(task: str) -> tuple[str, ...]:
    """输入：任务意图。输出：另外三个角色（按 `ROLES` 的顺序，稳定）。

    预期行为：三个陪衬槽的角色 = 四个角色去掉目标角色，顺序固定，所以同一 case 的
    元素表唯一确定。
    """
    role = target_role(task)
    return tuple(other for other in ROLES if other != role)


def is_sensitive(object_: str) -> bool:
    """输入：对象键。输出：这个对象是不是命中敏感清单。预期行为：纯查表。"""
    return object_ in SENSITIVE_OBJECTS


def order_of(order: str) -> tuple[str, ...]:
    """输入：排布名。输出：位置 1..5 上依次是哪个槽。

    预期行为：环形移位（拉丁方），所以任一槽在五个位置上都恰好出现一次——
    `target` 的答案位置分布因此是构造出来的均匀，不靠事后过滤。
    """
    if order not in ORDERS:
        raise KeyError(f"没有排布 {order!r}（可用：{', '.join(ORDERS)}）")
    shift = ORDERS.index(order)
    size = len(SLOTS)
    return tuple(SLOTS[(index - shift) % size] for index in range(size))


# ---------- 元素与一屏界面 ----------


@dataclass(frozen=True)
class Element:
    """界面上的一个元素：角色 / 对象 / 状态。三格就是判据的全部输入。"""

    role: str
    object: str
    state: str

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：{角色, 对象, 状态}（统计与调试用）。"""
        return {"role": self.role, "object": self.object, "state": self.state}

    def describe(self) -> str:
        """输入：无。输出：一行描述（报告与调试用）。"""
        return f"{self.role}/{self.object}/{self.state}"


class RowLike(Protocol):
    """一行元素的最小接口：三格。

    预期行为：`Element` 与反解出来的 `ParsedElement` 都满足它，所以规则函数既能吃
    构造侧的元素、也能吃从文本读回来的行。
    """

    role: str
    object: str
    state: str


def rows_of(case: ScreenCase) -> tuple[Element, ...]:
    """输入：case。输出：位置 1..5 上依次是哪个元素（顺序稳定）。

    预期行为：先把五个槽拼好，再按排布摆位。目标槽的对象由 `form` 定、状态由 `done` 定；
    干扰槽与目标槽同角色、不同对象；三个陪衬槽是另外三个角色各一个、状态一律可用。
    """
    role = target_role(case.task)
    others = aux_roles(case.task)
    slots = {
        "tgt": Element(
            role, object_key(case.task, case.form), "current" if case.done else "ready"
        ),
        "dec": Element(role, DECOYS[case.task], case.decoy_state),
        "aux0": Element(others[0], NEUTRALS[others[0]], "ready"),
        "aux1": Element(others[1], NEUTRALS[others[1]], "ready"),
        "aux2": Element(others[2], NEUTRALS[others[2]], "ready"),
    }
    return tuple(slots[slot] for slot in order_of(case.order))


def object_row(rows: tuple[RowLike, ...], task: str, object_: str) -> RowLike | None:
    """输入：五行元素、任务意图、对象键。输出：目标那一行；不是恰好一行就给 None。

    预期行为：按「角色符合任务意图表 + 对象就是要操作的那个」两格找，不是恰好一行说明
    表坏了（同角色的干扰槽不会撞上，因为对象不同），这时给 None 而不是猜一行。
    """
    role = target_role(task)
    found = [row for row in rows if row.role == role and row.object == object_]
    return found[0] if len(found) == 1 else None


def is_done(rows: tuple[RowLike, ...], task: str, object_: str) -> bool:
    """输入：五行元素、任务意图、对象键。输出：任务是不是已经完成。

    预期行为：判据是「要操作的那个对象，表里已经有一行状态是『已是当前』、角色也符合
    任务意图表」。干扰槽也常常是「已是当前」，但它挂的是**别的**对象，所以不算完成。
    """
    row = object_row(rows, task, object_)
    return row is not None and row.state == "current"


def target_position(rows: tuple[RowLike, ...], task: str, object_: str) -> int | None:
    """输入：五行元素、任务意图、对象键。输出：该操作的那一行位置（从 1 起）；不可操作时 None。

    预期行为：只有状态是「可用」的那一行能操作；状态不是可用（已是当前 / 置灰 / 不可见）
    就返回 None——这时定位、动作、风险三族都不该出（两阶段依赖）。
    """
    row = object_row(rows, task, object_)
    if row is None or row.state != "ready":
        return None
    for index, candidate in enumerate(rows, start=1):
        if candidate is row:
            return index
    return None  # pragma: no cover  （`row` 来自 `rows`，这一支走不到）


# ---------- case ----------


@dataclass(frozen=True)
class ScreenCase:
    """一屏界面 + 一句任务的全部自由度（六维）。"""

    task: str
    form: str
    done: bool
    decoy_state: str
    order: str
    phrasing: int

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：任一维取值不合法直接报错。"""
        _require("task", self.task, TASKS)
        _require("form", self.form, FORMS)
        _require("decoy_state", self.decoy_state, DECOY_STATES)
        _require("order", self.order, ORDERS)
        if self.phrasing not in PHRASINGS:
            raise KeyError(f"没有句面 {self.phrasing!r}（可用：{PHRASINGS}）")
        if self.done not in DONES:
            raise KeyError(f"完成档必须是布尔：{self.done!r}")

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：六维取值（统计与均衡报告用）。"""
        return {
            "task": self.task,
            "form": self.form,
            "done": self.done,
            "decoy_state": self.decoy_state,
            "order": self.order,
            "phrasing": self.phrasing,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述。预期行为：与素材指纹同形（报告与文件名用）。"""
        return material_of(self)

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这一条里训练侧没见过的维度名（排序后）。

        预期行为：只认两个留出维——任务意图 `switch`、第二套句面。其余四维在训练侧全开。
        """
        out: list[str] = []
        if self.task not in SEEN_TASKS:
            out.append("task")
        if self.phrasing not in SEEN_PHRASINGS:
            out.append("phrasing")
        return tuple(sorted(out))


# ---------- 值留出（训练侧见过哪些取值） ----------

#: 训练侧见过的任务意图：`switch`（切换页签）整个留出。
SEEN_TASKS: Final[frozenset[str]] = frozenset(task for task in TASKS if task != "switch")
#: 训练侧见过的句面：第二套句面整个留出。它**不影响任何答案**，所以留出它不会把哪一族
#: 的分布压歪。
SEEN_PHRASINGS: Final[frozenset[int]] = frozenset({PHRASINGS[0]})


def material_of(case: ScreenCase) -> str:
    """输入：case。输出：素材指纹（切分按它哈希分桶，同一屏界面整组不跨切分）。

    预期行为：六维全进指纹，句面也进——同一屏界面的两套问法是两条不同的 state，
    OOD 里才不会出现重复素材。
    """
    return (
        f"{case.task}|{case.form}|{case.done}|{case.decoy_state}"
        f"|{case.order}|{case.phrasing}"
    )


# ---------- 采样参数 ----------


@dataclass(frozen=True)
class ScreenConfig:
    """一个切分的采样参数：六张取值表 + 是否只留 OOD。"""

    tasks: tuple[str, ...] = TASKS
    forms: tuple[str, ...] = FORMS
    dones: tuple[bool, ...] = DONES
    decoy_states: tuple[str, ...] = DECOY_STATES
    orders: tuple[str, ...] = ORDERS
    phrasings: tuple[int, ...] = PHRASINGS
    split: str = "train"
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：任一取值表为空就报人话（点名是哪个字段）。"""
        require_pool("tasks", self.tasks)
        require_pool("forms", self.forms)
        require_pool("dones", self.dones)
        require_pool("decoy_states", self.decoy_states)
        require_pool("orders", self.orders)
        require_pool("phrasings", self.phrasings)


def build_cases(config: ScreenConfig) -> list[ScreenCase]:
    """输入：采样参数。输出：case 矩阵（顺序稳定）。

    预期行为：六张表取全组合（**不做成立性过滤**——六个维度两两不矛盾）；`ood_only` 时
    只留「至少含一个训练侧没见过的取值」的组合，所以 OOD 与同分布侧按构造就不重叠。
    """
    out: list[ScreenCase] = []
    for task, form, done, decoy_state, order, phrasing in product(
        config.tasks,
        config.forms,
        config.dones,
        config.decoy_states,
        config.orders,
        config.phrasings,
    ):
        case = ScreenCase(task, form, done, decoy_state, order, phrasing)
        if config.ood_only and not case.unseen_dims():
            continue
        out.append(case)
    return out


# ---------- 真值规格 ----------

SCREEN_PREFIX: Final[str] = "W-"


@dataclass(frozen=True)
class ScreenSpec:
    """一屏界面的完整真值：case + 由规则算出来的一切。"""

    case: ScreenCase
    split: str
    seq: int
    phrasing: int
    screen_id: str
    rows: tuple[Element, ...]
    task: str
    object: str
    done: bool
    target_position: int | None


def screen_id_of(case: ScreenCase) -> str:
    """输入：case。输出：窗口号（形如 `W-48213`）。

    预期行为：由素材指纹的 sha256 派生——确定性，而且**不泄露切分**（用序号或切分名拼
    窗口号会让同一屏界面在不同切分里拿到不同窗口号，等于把切分信息写进 state）。
    """
    digest = hashlib.sha256(material_of(case).encode("utf-8")).hexdigest()
    return f"{SCREEN_PREFIX}{10000 + int(digest[:6], 16) % 90000}"


def sample_spec(
    case: ScreenCase, split: str, seq: int, phrasing: int, rng: Any
) -> ScreenSpec:
    """输入：case、切分、序号、句面、RNG。输出：真值规格。

    预期行为：元素表与全部判断都由 case 决定（**不掷骰子**）——这个 workflow 的每个
    自由度都已经在 case 里，`rng` 只为与其它 workflow 的签名一致，收下就丢。
    """
    del rng
    rows = rows_of(case)
    object_ = object_key(case.task, case.form)
    return ScreenSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        screen_id=screen_id_of(case),
        rows=rows,
        task=case.task,
        object=object_,
        done=is_done(rows, case.task, object_),
        target_position=target_position(rows, case.task, object_),
    )


def spec_to_dict(spec: ScreenSpec) -> dict[str, Any]:
    """输入：规格。输出：可 JSON 序列化的摘要（写进 manifest 与调试，不进样本行）。"""
    return {
        "case": spec.case.describe(),
        "unseen_dims": list(spec.case.unseen_dims()),
        "screen_id": spec.screen_id,
        "task": spec.task,
        "object": spec.object,
        "done": spec.done,
        "target_position": spec.target_position,
        "rows": [row.describe() for row in spec.rows],
    }
