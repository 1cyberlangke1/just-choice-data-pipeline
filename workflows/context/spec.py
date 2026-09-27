"""会话上下文压缩 workflow 的真值规格：一张处置表 + 一段五块会话。

输入/输出：不可变数据类与纯函数；这里只有**键**（块类型 / 大小 / 年龄 / 主题 / 动作），
中英文案全在 `locales/`。
预期行为：答案只从渲染出来的那屏会话反推——处置表、尺寸表、动作开销、预算比较都是已知的
**纯函数**，构造侧与判别侧共用同一套，所以「表写错」与「渲染写歪」会变成同一类错误，
不会两边各错一半还互相印证。

覆盖的决策形状（社区用例地图：上下文压缩与保留，38 个第三方项目，含 ★4.4k 的
`fast-jev-compaction` / `jev-pruner` / `winnow` / `compact-adviser`）：
- **逐块处置**（`action`，choice 4）：第 k 块原样保留 / 截断 / 压成摘要 / 丢弃；
- **门控**（`compact`，noul）：按处置表算完，这次会话还要不要再压缩；
- **定位**（`spot`，choice 5）：被丢弃的那一块是第几块（只在恰好一块被丢时出）；
- **极值**（`top`，choice 5）：处置后占用最多的是第几块（只在最大值唯一时出）。

会话固定 5 块：四个**基本角色**（保留 / 截断 / 摘要 / 丢弃各一块）加一个**重复角色**
（`extra`，四种动作均匀）。于是 `action` 的答案分布是构造出来的 1/4（每块一次），
`spot` / `top` 的答案位置在五种排布上均匀（拉丁方），`compact` 靠**先定标签再挑预算**
实现（范式 §5.2、§5.7）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import product
from typing import Any, Final

from jc.knobs import require_pool

# ---------- 块的四维属性（键；文案在 locales/） ----------

KINDS: Final[tuple[str, ...]] = ("ask", "read", "run", "err")
#: 非报错的三个类型（重复角色的类型从这里挑，避开与它重复的那一块撞成同一行）。
NON_ERROR_KINDS: Final[tuple[str, ...]] = ("ask", "read", "run")
SIZES: Final[tuple[str, ...]] = ("small", "big")
AGES: Final[tuple[str, ...]] = ("recent", "old")
TOPICS: Final[tuple[str, ...]] = ("same", "other")

#: 两种大小的具体 token 数。题面里写明这张表，考生要按它算总账。
TOKENS: Final[dict[str, int]] = {"small": 180, "big": 1600}

# ---------- 四种处置动作（键；文案在 locales/） ----------

KEEP: Final[str] = "keep"
TRIM: Final[str] = "trim"
DIGEST: Final[str] = "digest"
DROP: Final[str] = "drop"
ACTIONS: Final[tuple[str, ...]] = (KEEP, TRIM, DIGEST, DROP)

#: 截断之后保留的 token 数。
TRIM_COST: Final[int] = 800
#: 压成摘要之后记的 token 数。
DIGEST_COST: Final[int] = 60

#: 预算池（token）。四个取值刻意跨过全部可能的总账，让「超 / 不超」两种标签都落得下。
BUDGETS: Final[tuple[int, ...]] = (512, 1024, 2048, 4096)

# ---------- 规则函数（构造与判别共用） ----------


def _require(field: str, value: str, pool: tuple[str, ...]) -> None:
    """输入：字段名、取值、取值表。输出：无。

    预期行为：认不出的键直接报错——静默落到某个默认分支会把「表改了、读法忘了改」
    变成一堆答案对不上的样本，而不是一条能定位的错误。
    """
    if value not in pool:
        raise KeyError(f"没有 {field}={value!r}（可用：{', '.join(pool)}）")


def action_of(kind: str, size: str, age: str, topic: str) -> str:
    """输入：块类型、大小、年龄、主题。输出：处置动作。

    预期行为：三条规则**按顺序**判——报错优先（不管年龄、大小、主题），其次主题
    （与当前任务无关的内容一律丢），最后才是年龄与大小。顺序本身是判据的一部分，
    题面里写明了。
    """
    _require("kind", kind, KINDS)
    _require("size", size, SIZES)
    _require("age", age, AGES)
    _require("topic", topic, TOPICS)
    if kind == "err":
        return KEEP
    if topic == "other":
        return DROP
    if age == "recent":
        return KEEP if size == "small" else TRIM
    return DIGEST if size == "small" else DROP


def cost_of(action: str, size: str) -> int:
    """输入：处置动作、块大小。输出：处置后这一块占用的 token 数。

    预期行为：原样保留按原大小；截断到 800；摘要记 60；丢弃记 0。
    """
    _require("size", size, SIZES)
    if action == KEEP:
        return TOKENS[size]
    if action == TRIM:
        return TRIM_COST
    if action == DIGEST:
        return DIGEST_COST
    if action == DROP:
        return 0
    raise KeyError(f"没有动作 {action!r}（可用：{', '.join(ACTIONS)}）")


def compact_of(total: int, budget: int) -> bool:
    """输入：处置后总账、预算。输出：这次会话还需不需要压缩。

    预期行为：总账**严格超过**预算才算需要压缩（刚好等于预算时不必再压）。
    """
    return total > budget


def budgets_for(total: int, target: bool) -> tuple[int, ...]:
    """输入：处置后总账、目标标签（要不要压缩）。输出：能落地这个标签的预算（非空）。

    预期行为：一个能落地的都没有就直接报错——静默退回某个预算会让标签与事实相反，
    那种样本比直接报错危险得多（范式 §5.2：先定标签，再挑素材）。
    """
    pool = tuple(budget for budget in BUDGETS if compact_of(total, budget) == target)
    if not pool:
        raise ValueError(f"总账 {total} 在预算池 {BUDGETS} 里落不下标签 {target}")
    return pool


# ---------- 块 ----------


@dataclass(frozen=True)
class BlockCase:
    """会话里的一个块：类型 / 大小 / 年龄 / 主题。"""

    kind: str
    size: str
    age: str
    topic: str

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：{属性名: 取值}。"""
        return {"kind": self.kind, "size": self.size, "age": self.age, "topic": self.topic}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（`类型/大小/年龄/主题`）。"""
        return f"{self.kind}/{self.size}/{self.age}/{self.topic}"

    def action(self) -> str:
        """输入：无。输出：这个块该走的处置动作（查处置表）。"""
        return action_of(self.kind, self.size, self.age, self.topic)

    def cost(self) -> int:
        """输入：无。输出：处置后这个块占用的 token 数。"""
        return cost_of(self.action(), self.size)


# ---------- 会话的构造（槽 → 块） ----------

#: 四个基本角色：四种动作各一块。
BASE_ROLES: Final[tuple[str, ...]] = (KEEP, TRIM, DIGEST, DROP)
#: 会话里的五个槽：四个基本角色 + 一个重复角色（动作由 case 的 `extra` 指定）。
SLOTS: Final[tuple[str, ...]] = (*BASE_ROLES, "extra")

#: 保留块的三种实现：报错小块 / 报错大块 / 最近的小块。
KEEP_FORMS: Final[tuple[str, ...]] = ("err_small", "err_big", "fresh")
#: 丢弃块的三种实现：无关的小块 / 无关的大块 / 同主题但过期的大块。
DROP_FORMS: Final[tuple[str, ...]] = ("off_small", "off_big", "stale")
#: 截断块 / 摘要块 / 丢弃块的类型搭配（三个非报错类型各用一次）。
KIND_SETS: Final[dict[str, tuple[str, str, str]]] = {
    "read_run": ("read", "run", "ask"),
    "run_ask": ("run", "ask", "read"),
    "ask_read": ("ask", "read", "run"),
}
#: 五种排布。`p<i>` 把第 j 个槽放到第 (j+i) mod 5 个位置——每种槽在每个位置上恰好一次。
ORDERS: Final[tuple[str, ...]] = ("p0", "p1", "p2", "p3", "p4")
#: 句面（只影响措辞，不影响答案）。
PHRASINGS: Final[tuple[int, ...]] = (0, 1)


def order_of(order: str) -> tuple[str, ...]:
    """输入：排布名。输出：位置 1..5 上依次是哪个槽。

    预期行为：环形移位（拉丁方），所以任一槽在五个位置上都恰好出现一次——
    `spot` / `top` 的答案位置分布因此是构造出来的均匀，不靠事后过滤。
    """
    if order not in ORDERS:
        raise KeyError(f"没有排布 {order!r}（可用：{', '.join(ORDERS)}）")
    shift = ORDERS.index(order)
    size = len(SLOTS)
    return tuple(SLOTS[(index - shift) % size] for index in range(size))


def _other_kind(kind: str) -> str:
    """输入：一个非报错类型。输出：另一个非报错类型（确定性，取第一个不同的）。

    预期行为：让重复角色那一块与它重复的块**类型不同**，两块的四元组因此不会撞车
    （撞了就会渲染出两行一模一样的正文）。
    """
    if kind not in NON_ERROR_KINDS:
        raise KeyError(f"没有非报错类型 {kind!r}（可用：{', '.join(NON_ERROR_KINDS)}）")
    return next(candidate for candidate in NON_ERROR_KINDS if candidate != kind)


def _base_blocks(case: SessionCase) -> dict[str, BlockCase]:
    """输入：case。输出：{槽名: 块}（`extra` 之外的四块）。"""
    trim_kind, digest_kind, drop_kind = KIND_SETS[case.kind_set]
    keep = {
        "err_small": BlockCase("err", "small", "recent", "same"),
        "err_big": BlockCase("err", "big", "old", "other"),
        "fresh": BlockCase("ask", "small", "recent", "same"),
    }[case.keep_form]
    drop = {
        "off_small": BlockCase(drop_kind, "small", "recent", "other"),
        "off_big": BlockCase(drop_kind, "big", "old", "other"),
        "stale": BlockCase(drop_kind, "big", "old", "same"),
    }[case.drop_form]
    return {
        KEEP: keep,
        TRIM: BlockCase(trim_kind, "big", "recent", "same"),
        DIGEST: BlockCase(digest_kind, "small", "old", "same"),
        DROP: drop,
    }


def _extra_block(case: SessionCase, base: dict[str, BlockCase]) -> BlockCase:
    """输入：case、四个基本块。输出：重复角色的块。

    预期行为：按 `extra` 指定的动作，从该动作的**原像**里挑一块，并与已经用掉的四块
    两两不同。两处是给 `top` 铺路的：两块保留块的大小**故意不同**（最大值因此唯一）；
    重复的截断 / 摘要块换一个类型（不与同名块撞成同一行）。
    """
    trim_kind, digest_kind, drop_kind = KIND_SETS[case.kind_set]
    if case.extra == KEEP:
        # 基本保留块小 → 重复保留块取大的；基本保留块大 → 取小的。两块大小必定不同。
        if base[KEEP].size == "small":
            return BlockCase("err", "big", "old", "other")
        return BlockCase("err", "small", "recent", "same")
    if case.extra == TRIM:
        return BlockCase(_other_kind(trim_kind), "big", "recent", "same")
    if case.extra == DIGEST:
        return BlockCase(_other_kind(digest_kind), "small", "old", "same")
    if case.extra == DROP:
        return {
            "off_small": BlockCase(_other_kind(drop_kind), "big", "old", "other"),
            "off_big": BlockCase(_other_kind(drop_kind), "small", "recent", "other"),
            "stale": BlockCase(_other_kind(drop_kind), "big", "old", "other"),
        }[case.drop_form]
    raise KeyError(f"没有重复动作 {case.extra!r}（可用：{', '.join(ACTIONS)}）")


def blocks_of(case: SessionCase) -> tuple[BlockCase, ...]:
    """输入：case。输出：位置 1..5 上依次是哪个块（顺序稳定）。

    预期行为：五个块的 `类型/大小/年龄/主题` 四元组两两不同，所以每块的正文行也互不相同。
    """
    slots = _base_blocks(case)
    slots["extra"] = _extra_block(case, slots)
    return tuple(slots[slot] for slot in order_of(case.order))


def total_of(blocks: tuple[BlockCase, ...]) -> int:
    """输入：五个块。输出：按处置表算完之后这次会话的总账（token）。"""
    return sum(block.cost() for block in blocks)


def drop_position(blocks: tuple[BlockCase, ...]) -> int | None:
    """输入：五个块。输出：被丢弃的那一块的位置（从 1 起）；不止一块时给 None。

    预期行为：不止一块被丢时返回 None——「被丢的是哪一块」这时没有唯一解，出这种题就是
    坏数据（`spot` 那一族因此是两阶段依赖）。
    """
    dropped = [
        index for index, block in enumerate(blocks, start=1) if block.action() == DROP
    ]
    return dropped[0] if len(dropped) == 1 else None


def top_position(blocks: tuple[BlockCase, ...]) -> int | None:
    """输入：五个块。输出：处置后占用最多的那一块的位置（从 1 起）；并列时给 None。

    预期行为：并列时返回 None，理由同上（`top` 也是两阶段依赖）。
    """
    costs = [block.cost() for block in blocks]
    best = max(costs)
    return costs.index(best) + 1 if costs.count(best) == 1 else None


# ---------- case ----------


@dataclass(frozen=True)
class SessionCase:
    """一段会话的构造参数：五个槽各自怎么实现 + 排布 + 句面。"""

    keep_form: str
    drop_form: str
    kind_set: str
    extra: str
    order: str
    phrasing: int

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：{维度名: 取值}（六个维度全摊平）。"""
        return {
            "keep_form": self.keep_form,
            "drop_form": self.drop_form,
            "kind_set": self.kind_set,
            "extra": self.extra,
            "order": self.order,
            "phrasing": self.phrasing,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return (
            f"{self.keep_form}/{self.drop_form}/{self.kind_set}"
            f"/{self.extra}/{self.order}/{self.phrasing}"
        )

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这个组合里「训练侧没见过」的维度名（按固定顺序）。

        预期行为：留出两处——`keep_form` 的 `err_big`（答案相关的值留出：训练侧从没见过
        「报错 + 大 + 过期 + 无关主题」这一格，`compact` 的总账也因此多出 2460 / 2520 / 3260
        这几档）与 `phrasing` 的第二套句面（措辞轴）。另外几个维度**不能**留出：
        `extra` 留出一个取值会让 `action` 的四档分布塌掉，`order` 留出会让 `spot` / `top`
        的位置分布不再是均匀的拉丁方。
        """
        out: list[str] = []
        if self.keep_form not in SEEN_KEEP_FORMS:
            out.append("keep_form")
        if self.phrasing not in SEEN_PHRASINGS:
            out.append("phrasing")
        return tuple(out)


def material_of(case: SessionCase) -> str:
    """输入：case。输出：素材指纹（确定性；切分分桶与内容种子都由它派生）。"""
    return (
        f"{case.keep_form}|{case.drop_form}|{case.kind_set}"
        f"|{case.extra}|{case.order}|{case.phrasing}"
    )


# ---------- 值留出（训练侧见过哪些取值） ----------

#: 训练侧见过的保留实现：`err_big` 整个留出。
SEEN_KEEP_FORMS: Final[frozenset[str]] = frozenset(
    form for form in KEEP_FORMS if form != "err_big"
)
#: 训练侧见过的句面：第二套句面整个留出。它**不影响任何答案**，所以留出它不会把哪一族
#: 的分布压歪。
SEEN_PHRASINGS: Final[frozenset[int]] = frozenset({PHRASINGS[0]})


# ---------- 采样参数 ----------


@dataclass(frozen=True)
class SessionConfig:
    """一个切分的采样参数：六张取值表 + 是否只留 OOD。"""

    keep_forms: tuple[str, ...] = KEEP_FORMS
    drop_forms: tuple[str, ...] = DROP_FORMS
    kind_sets: tuple[str, ...] = tuple(KIND_SETS)
    extras: tuple[str, ...] = ACTIONS
    orders: tuple[str, ...] = ORDERS
    phrasings: tuple[int, ...] = PHRASINGS
    split: str = "train"
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：任一取值表为空就报人话（点名是哪个字段）。"""
        require_pool("keep_forms", self.keep_forms)
        require_pool("drop_forms", self.drop_forms)
        require_pool("kind_sets", self.kind_sets)
        require_pool("extras", self.extras)
        require_pool("orders", self.orders)
        require_pool("phrasings", self.phrasings)


def build_cases(config: SessionConfig) -> list[SessionCase]:
    """输入：采样参数。输出：case 矩阵（顺序稳定）。

    预期行为：六张表取全组合（**不做成立性过滤**——六个维度两两不矛盾）；
    `ood_only` 时只留「至少含一个训练侧没见过的取值」的组合，所以 OOD 与同分布侧
    按构造就不重叠。
    """
    out: list[SessionCase] = []
    for keep_form, drop_form, kind_set, extra, order, phrasing in product(
        config.keep_forms,
        config.drop_forms,
        config.kind_sets,
        config.extras,
        config.orders,
        config.phrasings,
    ):
        case = SessionCase(keep_form, drop_form, kind_set, extra, order, phrasing)
        if config.ood_only and not case.unseen_dims():
            continue
        out.append(case)
    return out


# ---------- 真值规格 ----------

SESSION_PREFIX: Final[str] = "S-"


@dataclass(frozen=True)
class SessionSpec:
    """一段会话的完整真值：case + 由规则算出来的一切。"""

    case: SessionCase
    split: str
    seq: int
    phrasing: int
    session_id: str
    blocks: tuple[BlockCase, ...]
    budget: int
    total: int
    compact: bool
    drop_position: int | None
    top_position: int | None


def session_id_of(case: SessionCase) -> str:
    """输入：case。输出：会话号（形如 `S-48213`）。

    预期行为：由素材指纹的 sha256 派生——确定性，而且**不泄露切分**（用序号或切分名拼
    会话号会让同一段会话在不同切分里拿到不同会话号，等于把切分信息写进 state）。
    句面也进指纹：同一段会话的两套问法是两条不同的 state，OOD 里才不会出现重复素材。
    """
    digest = hashlib.sha256(material_of(case).encode("utf-8")).hexdigest()
    return f"{SESSION_PREFIX}{10000 + int(digest[:6], 16) % 90000}"


def sample_spec(
    case: SessionCase, split: str, seq: int, phrasing: int, rng: Any
) -> SessionSpec:
    """输入：case、切分、序号、句面、RNG。输出：真值规格。

    预期行为：块与全部判断都由 case 决定（不掷骰子）；**只有预算**由 RNG 定——
    先按均匀抽一个「要不要压缩」的目标标签，再从能落地这个标签的预算里挑一个
    （范式 §5.2）。于是 `compact` 的分布是构造出来的 50/50，不靠事后过滤。
    """
    blocks = blocks_of(case)
    total = total_of(blocks)
    budget = rng.choice(budgets_for(total, rng.random() < 0.5))
    return SessionSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        session_id=session_id_of(case),
        blocks=blocks,
        budget=budget,
        total=total,
        compact=compact_of(total, budget),
        drop_position=drop_position(blocks),
        top_position=top_position(blocks),
    )


def spec_to_dict(spec: SessionSpec) -> dict[str, Any]:
    """输入：规格。输出：可 JSON 序列化的摘要（写进 manifest 与调试，不进样本行）。"""
    return {
        "case": spec.case.describe(),
        "unseen_dims": list(spec.case.unseen_dims()),
        "split": spec.split,
        "seq": spec.seq,
        "session_id": spec.session_id,
        "blocks": [block.describe() for block in spec.blocks],
        "budget": spec.budget,
        "total": spec.total,
        "compact": spec.compact,
        "drop_position": spec.drop_position,
        "top_position": spec.top_position,
    }
