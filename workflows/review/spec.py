"""改动审查 workflow 的真值规格：一张规则表 + 一张「该不该提」的二维表 + 一份改动。

输入/输出：不可变数据类与纯函数；这里只有**键**（规则 / 文件 / 变更 / 规模 / 门控），
中英文案全在 `locales/`。
预期行为：答案只从渲染出来的那屏改动反推——规则判据（正则 + 规模格）、该不该提的二维表、
严重度表、门控表都是已知的**纯函数**，构造侧与判别侧共用同一套，所以「表写错」与
「渲染写歪」会变成同一类错误，不会两边各错一半还互相印证。

覆盖的决策形状（社区用例地图：代码审查与 diff 门控，29 个第三方项目，含 `jev-review` /
`clean-code-review` / `hermes-jev-approvals` / `jev-test-filter` / `supercov`）：
逐处判**这一处该不该提意见**（命中规则 ≠ 该提——同样的模式在不同文件里权重不同）、
**严重度几档**、**这一处走哪条门控**，外加**最严重的是哪一处**。
- **该不该提**（`flag`，noul，逐处 6 道）：这一处该提意见吗；
- **严重度**（`severity`，score 3 档，只对该提的处）：这一处严重到几档；
- **门控**（`gate`，choice 3，只对该提的处）：这一处该怎么处置；
- **极值**（`top`，choice 6，只在最严重的那一处唯一时出）：最严重的是第几处。

改动固定 6 处（槽）：三个**该提**的（严重度恰好 3 / 2 / 1 各一个）、两个**命中但不提**
的（噪声）、一个**没命中**的（干净）。于是 `flag` 的 True 占比是构造出来的 3/6 = 0.5、
`severity` 三档各 1/3；排布用拉丁方（六槽环形移位），所以 `top` 的答案位置在六个位置上
均匀（范式 §5.7）。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from itertools import product
from typing import Any, Final

from jc.knobs import require_pool

# ---------- 三条审查规则（键；文案在 locales/） ----------

RULE_SECRET: Final[str] = "secret"  # 正文里有硬编码的凭据
RULE_DEBUG: Final[str] = "debug"  # 正文里有调试输出
RULE_BULK: Final[str] = "bulk"  # 规模格是「大」（一次改了太多）

RULES: Final[tuple[str, ...]] = (RULE_SECRET, RULE_DEBUG, RULE_BULK)

#: 规则的文本判据（纯函数，构造侧与读法侧共用）。`bulk` 不吃正文，吃抬头里的规模格。
#: `secret` 要求「凭据名 + 赋值号 + 带引号的字面量」——走环境变量的写法不算命中。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    RULE_SECRET: re.compile(
        r"(?i)\b\w*(?:passwd|password|pwd|secret|token|api[_-]?key|access[_-]?key"
        r"|private[_-]?key|credential)\w*\s*[:=]\s*[\"'][^\"']{4,}[\"']"
    ),
    RULE_DEBUG: re.compile(
        r"(?i)\b(?:print|console\.log|debugger|var_dump|logger\.debug|fmt\.Print)\s*\("
    ),
}

# ---------- 四格属性（键；文案在 locales/） ----------

#: 改动落在哪种文件里。它决定「该不该提」那一列。
FILES: Final[tuple[str, ...]] = ("src", "test", "doc", "config")
#: 变更类型。
CHANGES: Final[tuple[str, ...]] = ("add", "edit", "remove")
#: 规模（只有两档：`bulk` 规则看的就是它）。
SIZES: Final[tuple[str, ...]] = ("small", "big")

# ---------- 三张表（判据的核心） ----------

#: 规则的严重度档（1 轻 / 2 中 / 3 重）。
SEVERITY: Final[dict[str, int]] = {RULE_SECRET: 3, RULE_DEBUG: 2, RULE_BULK: 1}

#: **该不该提**：规则 × 文件。命中规则**不等于**该提意见——同样的模式在不同文件里
#: 权重不同（文档里写个示例凭据不算问题、测试里打印一行是常态、文档一次改很多也正常）。
RAISE: Final[dict[str, frozenset[str]]] = {
    RULE_SECRET: frozenset({"src", "test", "config"}),
    RULE_DEBUG: frozenset({"src", "config"}),
    RULE_BULK: frozenset({"src"}),
}

#: 门控三档。
GATES: Final[tuple[str, ...]] = ("pass", "request", "block")

#: **门控**：严重度 × 变更类型。它比严重度更细一档——重的一律拦下，中的只在「删东西」
#: 时拦下，轻的只有「改东西」才要请求修改，其余放行。
GATE: Final[dict[tuple[int, str], str]] = {
    (3, "add"): "block",
    (3, "edit"): "block",
    (3, "remove"): "block",
    (2, "add"): "request",
    (2, "edit"): "request",
    (2, "remove"): "block",
    (1, "add"): "pass",
    (1, "edit"): "request",
    (1, "remove"): "pass",
}

# ---------- 六个槽 ----------

#: 六个槽：三个该提的（严重度 3 / 2 / 1 各一）+ 两个命中但不提的 + 一个没命中的。
SLOTS: Final[tuple[str, ...]] = ("heavy", "medium", "light", "noise0", "noise1", "clean")

#: 六个排布。`o<i>` 把第 j 个槽放到第 (j+i) mod 6 个位置——每种槽在每个位置上恰好一次。
ORDERS: Final[tuple[str, ...]] = ("o0", "o1", "o2", "o3", "o4", "o5")

#: 三个该提的槽的文件搭配。**全部合法组合**都收进来——重槽的文件可以是源码 / 测试 / 配置、
#: 中槽是源码 / 配置、轻槽只有源码，于是 3 × 2 × 1 = 6 种；少一种就等于把「同样的模式在
#: 不同文件里权重不同」这条判据的某个分支从数据里抹掉。枚举全了，同分布切分也够大
#: （每种搭配 2 × 2 × 6 = 24 条），不至于让小切分里的位置分布压不住噪声地板。
TRIPLES: Final[dict[str, tuple[str, str, str]]] = {
    "t0": ("src", "src", "src"),
    "t1": ("src", "config", "src"),
    "t2": ("test", "src", "src"),
    "t3": ("test", "config", "src"),
    "t4": ("config", "src", "src"),
    "t5": ("config", "config", "src"),
}
#: 两个噪声槽的（规则, 文件）搭配（都要**命中**且**不该提**）。
NOISE_PAIRS: Final[dict[str, tuple[tuple[str, str], tuple[str, str]]]] = {
    "n0": ((RULE_SECRET, "doc"), (RULE_DEBUG, "test")),
    "n1": ((RULE_DEBUG, "doc"), (RULE_BULK, "test")),
    "n2": ((RULE_SECRET, "doc"), (RULE_BULK, "config")),
}
#: 三个该提的槽的变更类型（门控表靠它分档）。两套搭配是**挑过的**：重槽一律「拦下」，
#: 中槽落在「新增 / 修改」（都是请求修改），轻槽落在「新增 / 删除」（都是放行）——
#: 于是门控三档在整个网格上各出现两次，`gate` 那一族的答案分布是构造出来的均匀。
CHANGE_SETS: Final[dict[str, tuple[str, str, str]]] = {
    "c0": ("add", "edit", "add"),
    "c1": ("remove", "add", "remove"),
}
#: 干净槽的文件 / 变更（它没命中规则，这两格只是语境）。文件挑**三个该提槽都没用到**的那一种，
#: 这样「这一处什么都没命中」在抬头三格上也看得出来，不至于和某条命中规则的处混成一样。
CLEAN_FILES: Final[dict[str, str]] = {
    "t0": "doc",
    "t1": "test",
    "t2": "doc",
    "t3": "doc",
    "t4": "test",
    "t5": "doc",
}
CLEAN_CHANGES: Final[dict[str, str]] = {"c0": "remove", "c1": "add"}

#: 六张取值表。
TRIPLE_KEYS: Final[tuple[str, ...]] = tuple(TRIPLES)
NOISE_KEYS: Final[tuple[str, ...]] = tuple(NOISE_PAIRS)
CHANGE_KEYS: Final[tuple[str, ...]] = tuple(CHANGE_SETS)
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


def matches(rule: str, size: str, body: str) -> bool:
    """输入：规则、规模格、正文。输出：这条规则命不命中。

    预期行为：`bulk` 只看规模格是不是「大」，其余两条按 `PATTERNS` 里的正则搜正文；
    规则名不认识直接报错。构造侧与判别侧共用它，所以「规则写歪」两边同时歪，查得出。
    """
    if rule == RULE_BULK:
        return size == "big"
    pattern = PATTERNS.get(rule)
    if pattern is None:
        raise KeyError(f"没有规则 {rule!r}（可用：{', '.join(RULES)}）")
    return pattern.search(body) is not None


def rule_from(size: str, body: str) -> str | None:
    """输入：规模格、正文。输出：命中的规则（没命中给 None）。

    预期行为：命中两条以上直接报错——一条正文同时踩两条规则说明素材造重了，
    静默取第一条会让「该不该提」跟着掷骰子。
    """
    hits = [rule for rule in RULES if matches(rule, size, body)]
    if len(hits) > 1:
        raise ValueError(f"一条正文同时命中多条规则：{hits}（正文 {body!r}）")
    return hits[0] if hits else None


def order_of(order: str) -> tuple[str, ...]:
    """输入：排布名。输出：位置 1..6 上依次是哪个槽。

    预期行为：环形移位（拉丁方），所以任一槽在六个位置上都恰好出现一次——
    `top` 的答案位置分布因此是构造出来的均匀，不靠事后过滤。
    """
    if order not in ORDERS:
        raise KeyError(f"没有排布 {order!r}（可用：{', '.join(ORDERS)}）")
    shift = ORDERS.index(order)
    size = len(SLOTS)
    return tuple(SLOTS[(index - shift) % size] for index in range(size))


def raises_(finding: Finding) -> bool:
    """输入：一处改动。输出：这一处该不该提意见。

    预期行为：没命中规则 → 不提；命中了还要查「规则 × 文件」那张二维表——这是本 workflow
    的核心判据（命中 ≠ 该提）。
    """
    if finding.rule is None:
        return False
    return finding.file in RAISE[finding.rule]


def severity_of(finding: Finding) -> int | None:
    """输入：一处改动。输出：严重度档；没命中规则时给 None。"""
    return None if finding.rule is None else SEVERITY[finding.rule]


def gate_of(finding: Finding) -> str | None:
    """输入：一处改动。输出：门控档；没命中规则时给 None。

    预期行为：查「严重度 × 变更类型」那张表——所以门控**不是**严重度的简单改名，
    同一条规则在「新增」和「删除」下可能走不同的门控。
    """
    severity = severity_of(finding)
    if severity is None:
        return None
    return GATE[(severity, finding.change)]


# ---------- 一处改动与一屏改动 ----------


@dataclass(frozen=True)
class Finding:
    """一处改动：规则 / 文件 / 变更 / 规模 / 正文键。抬头三格 + 正文就是判据的全部输入。"""

    rule: str | None
    file: str
    change: str
    size: str
    body: str

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：四格与正文键（统计与调试用）。"""
        return {
            "rule": self.rule,
            "file": self.file,
            "change": self.change,
            "size": self.size,
            "body": self.body,
        }

    def describe(self) -> str:
        """输入：无。输出：一行描述（报告与调试用）。"""
        return f"{self.rule or 'clean'}/{self.file}/{self.change}/{self.size}"


def _finding(slot: str, rule: str | None, file: str, change: str) -> Finding:
    """输入：槽名、规则、文件、变更。输出：一处改动。

    预期行为：规模格与正文键都由规则派生——`bulk` 的规模一定是「大」、正文键是
    `{规则}_{槽}`（干净槽是 `clean_{槽}`），所以正文与规则判据天然对得上。
    """
    size = "big" if rule == RULE_BULK else "small"
    body = f"clean_{slot}" if rule is None else f"{rule}_{slot}"
    return Finding(rule, file, change, size, body)


def rows_of(case: ChangeCase) -> tuple[Finding, ...]:
    """输入：case。输出：位置 1..6 上依次是哪一处改动（顺序稳定）。

    预期行为：先把六个槽拼好，再按排布摆位。三个该提的槽严重度恰好 3 / 2 / 1 各一个，
    两个噪声槽命中但不提，一个干净槽什么都没命中。
    """
    files = TRIPLES[case.triple]
    changes = CHANGE_SETS[case.change]
    noise = NOISE_PAIRS[case.noise]
    slots = {
        "heavy": _finding("heavy", RULE_SECRET, files[0], changes[0]),
        "medium": _finding("medium", RULE_DEBUG, files[1], changes[1]),
        "light": _finding("light", RULE_BULK, files[2], changes[2]),
        "noise0": _finding("noise0", noise[0][0], noise[0][1], "edit"),
        "noise1": _finding("noise1", noise[1][0], noise[1][1], "add"),
        "clean": _finding("clean", None, CLEAN_FILES[case.triple], CLEAN_CHANGES[case.change]),
    }
    return tuple(slots[slot] for slot in order_of(case.order))


def raised_positions(rows: tuple[Finding, ...]) -> tuple[int, ...]:
    """输入：六处改动。输出：该提意见的那几处的位置（从 1 起，升序）。"""
    return tuple(index for index, row in enumerate(rows, start=1) if raises_(row))


def top_position(rows: tuple[Finding, ...]) -> int | None:
    """输入：六处改动。输出：最严重的那一处的位置（从 1 起）；并列或一处都没提时给 None。

    预期行为：并列时返回 None——「最严重的是哪一处」这时没有唯一解，出这种题就是坏数据
    （`top` 因此是两阶段依赖）。
    """
    raised = [(index, severity_of(row)) for index, row in enumerate(rows, start=1) if raises_(row)]
    if not raised:
        return None
    best = max(severity for _, severity in raised)
    winners = [index for index, severity in raised if severity == best]
    return winners[0] if len(winners) == 1 else None


# ---------- case ----------


@dataclass(frozen=True)
class ChangeCase:
    """一屏改动的全部自由度（五维）。"""

    triple: str
    noise: str
    change: str
    order: str
    phrasing: int

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：任一维取值不合法直接报错。"""
        _require("triple", self.triple, TRIPLE_KEYS)
        _require("noise", self.noise, NOISE_KEYS)
        _require("change", self.change, CHANGE_KEYS)
        _require("order", self.order, ORDERS)
        if self.phrasing not in PHRASINGS:
            raise KeyError(f"没有句面 {self.phrasing!r}（可用：{PHRASINGS}）")

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：五维取值（统计与均衡报告用）。"""
        return {
            "triple": self.triple,
            "noise": self.noise,
            "change": self.change,
            "order": self.order,
            "phrasing": self.phrasing,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述。预期行为：与素材指纹同形（报告与文件名用）。"""
        return material_of(self)

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这一条里训练侧没见过的维度名（排序后）。

        预期行为：只认两个留出维——噪声搭配 `n2`、第二套句面。其余三维在训练侧全开。
        """
        out: list[str] = []
        if self.noise not in SEEN_NOISES:
            out.append("noise")
        if self.phrasing not in SEEN_PHRASINGS:
            out.append("phrasing")
        return tuple(sorted(out))


# ---------- 值留出（训练侧见过哪些取值） ----------

#: 训练侧见过的噪声搭配：`n2`（两条都是「同一规则落在文档/配置上」）整个留出。
SEEN_NOISES: Final[frozenset[str]] = frozenset(key for key in NOISE_KEYS if key != "n2")
#: 训练侧见过的句面：第二套句面整个留出。它**不影响任何答案**，所以留出它不会把哪一族
#: 的分布压歪。
SEEN_PHRASINGS: Final[frozenset[int]] = frozenset({PHRASINGS[0]})


def material_of(case: ChangeCase) -> str:
    """输入：case。输出：素材指纹（切分按它哈希分桶，同一屏改动整组不跨切分）。

    预期行为：五维全进指纹，句面也进——同一屏改动的两套问法是两条不同的 state，
    OOD 里才不会出现重复素材。
    """
    return (
        f"{case.triple}|{case.noise}|{case.change}|{case.order}|{case.phrasing}"
    )


# ---------- 采样参数 ----------


@dataclass(frozen=True)
class ReviewConfig:
    """一个切分的采样参数：五张取值表 + 是否只留 OOD。"""

    triples: tuple[str, ...] = TRIPLE_KEYS
    noises: tuple[str, ...] = NOISE_KEYS
    changes: tuple[str, ...] = CHANGE_KEYS
    orders: tuple[str, ...] = ORDERS
    phrasings: tuple[int, ...] = PHRASINGS
    split: str = "train"
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：任一取值表为空就报人话（点名是哪个字段）。"""
        require_pool("triples", self.triples)
        require_pool("noises", self.noises)
        require_pool("changes", self.changes)
        require_pool("orders", self.orders)
        require_pool("phrasings", self.phrasings)


def build_cases(config: ReviewConfig) -> list[ChangeCase]:
    """输入：采样参数。输出：case 矩阵（顺序稳定）。

    预期行为：五张表取全组合（**不做成立性过滤**——五个维度两两不矛盾）；`ood_only` 时
    只留「至少含一个训练侧没见过的取值」的组合，所以 OOD 与同分布侧按构造就不重叠。
    """
    out: list[ChangeCase] = []
    for triple, noise, change, order, phrasing in product(
        config.triples, config.noises, config.changes, config.orders, config.phrasings
    ):
        case = ChangeCase(triple, noise, change, order, phrasing)
        if config.ood_only and not case.unseen_dims():
            continue
        out.append(case)
    return out


# ---------- 真值规格 ----------

CHANGE_PREFIX: Final[str] = "R-"


@dataclass(frozen=True)
class ChangeSpec:
    """一屏改动的完整真值：case + 由规则算出来的一切。"""

    case: ChangeCase
    split: str
    seq: int
    phrasing: int
    change_id: str
    rows: tuple[Finding, ...]
    raised: tuple[int, ...]
    top_position: int | None


def change_id_of(case: ChangeCase) -> str:
    """输入：case。输出：改动号（形如 `R-48213`）。

    预期行为：由素材指纹的 sha256 派生——确定性，而且**不泄露切分**（用序号或切分名拼
    改动号会让同一屏改动在不同切分里拿到不同改动号，等于把切分信息写进 state）。
    """
    digest = hashlib.sha256(material_of(case).encode("utf-8")).hexdigest()
    return f"{CHANGE_PREFIX}{10000 + int(digest[:6], 16) % 90000}"


def sample_spec(
    case: ChangeCase, split: str, seq: int, phrasing: int, rng: Any
) -> ChangeSpec:
    """输入：case、切分、序号、句面、RNG。输出：真值规格。

    预期行为：改动与全部判断都由 case 决定（**不掷骰子**）——这个 workflow 的每个自由度
    都已经在 case 里，`rng` 只为与其它 workflow 的签名一致，收下就丢。
    """
    del rng
    rows = rows_of(case)
    return ChangeSpec(
        case=case,
        split=split,
        seq=seq,
        phrasing=phrasing,
        change_id=change_id_of(case),
        rows=rows,
        raised=raised_positions(rows),
        top_position=top_position(rows),
    )


def spec_to_dict(spec: ChangeSpec) -> dict[str, Any]:
    """输入：规格。输出：可 JSON 序列化的摘要（写进 manifest 与调试，不进样本行）。"""
    return {
        "case": spec.case.describe(),
        "unseen_dims": list(spec.case.unseen_dims()),
        "change_id": spec.change_id,
        "raised": list(spec.raised),
        "top_position": spec.top_position,
        "rows": [row.describe() for row in spec.rows],
    }
