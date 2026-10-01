"""改动引入性判定（findtriage）的特征层：注入维度 + 改动记录计划 + 判定口径（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一次「这条审查发现该怎么处置」的判断由 5 个**注入特征**描述，
**标签先行**——先定这 5 维取值，再由本模块的纯函数算出全部金标签。

五维分成两拨：

- **由文本承载**（提示词给、记录表不给）：`cite` 这条发现指向哪一行、`fix` 它的修法点不点名
  具体符号——真实审查意见的措辞是长尾的，模板拼出来的「第 N 行有问题」不像人话；
  这两维交给 `judges.TraceJudge` 反解校验，答不回来就说明文本与标签脱节了。
- **由代码渲染的记录表承载**（提示词不给）：`evidence` 它引用的证据符号落在哪、表长 `size`、
  有没有一行近形符号的诱饵 `decoy`——这三件事是结构化事实，让模型去写只会写歪。

改动记录**每一行**由 `plan_for` 从 case 指纹播种派生（不是一张写死的目录）：同一 case 在任何
进程、任何切分里都得到同一张表，而不同 case 的表长得不一样。

判定口径（印在 state 里，考生只读 state 与题面就能自己算，范式 §9.8），**从上往下第一条命中**：

1. **空口**：引用的证据符号在记录里根本查不到 → 这条发现没有依据，处置「丢弃」；
2. **归因**：指向的那一行本次没被碰过 → 不归这次改动，处置「只记录」；
3. **未知**：指向的行号在记录里查不到 → 引入性说不清，处置「只记录」；
4. **修法**：建议修法没有点名具体符号 → 落不了地，处置「只记录」；
5. **兜底**：以上都不沾 → 处置「必须修」。

口径为什么这么排：三档处置要在同分布侧都**站得住**。把「不归这次改动」直接判丢弃会让
「丢弃」占掉三分之一、把「必须修」压到 6%（实测），而「证据查不到」本来就是无依据的
空口发现——让它先判丢弃，三档才落在 37.5% / 50% / 12.5% 上（`test_features.py` 钉住）。

输入/输出：只有常量与纯函数；不碰网络、不碰文件、**绝不执行任何被合成的代码**。
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Final

from jc.cover import cover_dims
from jc.jsontypes import JsonValue

# ---------- 取值表 ----------

#: 这条发现指向的那一行，在本次改动里是什么处境。**只由那条发现承载**（它得把行号写出来）。
CITES: Final[tuple[str, ...]] = (
    "touched",  # 本次改动碰过它（新增或改过）
    "untouched",  # 本次改动没碰它（改动之前就在）
    "missing",  # 记录里根本没有这一行
)

#: 发现引用的证据符号落在哪里。
EVIDENCES: Final[tuple[str, ...]] = (
    "in_row",  # 就在它指向的那一行里
    "elsewhere",  # 落在记录里别的行上
    "none",  # 记录里查不到
)

#: 建议修法有没有点名具体符号。**只由那句修法承载**。
FIXES: Final[tuple[str, ...]] = (
    "concrete",  # 点名了要改的那个符号
    "vague",  # 只有泛泛的话，一个符号都没点
)

#: 改动记录的条数。
SIZES: Final[tuple[int, ...]] = (3, 4, 5)

#: 记录里有没有一行「符号与证据符号只差一个字母」的诱饵行。
DECOYS: Final[tuple[int, ...]] = (0, 1)

#: 记录里「本次改动」那一列能取的三个值。前两个是「碰过」，后一个是「没碰过」。
TOUCH_KINDS: Final[tuple[str, ...]] = ("added", "modified", "untouched")
TOUCHED_KINDS: Final[tuple[str, ...]] = ("added", "modified")

#: 引入性三档。`unknown` 是「说不清」，不是「没有」。
INTRODUCED: Final[tuple[str, ...]] = ("yes", "no", "unknown")

#: 证据三档。
SUPPORTED: Final[tuple[str, ...]] = ("strong", "weak", "none")

#: 处置三档。
TRIAGES: Final[tuple[str, ...]] = ("must_fix", "log_only", "drop")

#: 判定口径的规则名（渲染进 state 的口径段；顺序即语义，`test_features.py` 钉住它）。
RULES: Final[tuple[str, ...]] = ("baseless", "attribution", "unknown", "fix", "else")

#: 归属四档（`owner` 那一族的选项名，也就是「这条发现该进谁的队列」的四个去处）。
OWNER_RECORD: Final[str] = "record"
OWNER_RULE: Final[str] = "rule"
OWNER_AUTHOR: Final[str] = "author"
OWNER_ARCH: Final[str] = "arch"
OWNERS: Final[tuple[str, ...]] = (OWNER_RECORD, OWNER_RULE, OWNER_AUTHOR, OWNER_ARCH)

#: 归属口径的规则名（渲染进 state 的归属段；**顺序即优先级**，从上往下第一条命中的说了算）。
OWNER_RULES: Final[tuple[str, ...]] = ("missing", "baseless", "touched", "else")

#: 弃权选项名（`cite` 那一族的选项名随表长变化，由 `questions` 现场拼）。
NO_CITE: Final[str] = "none"

#: 记录表的**固定列序**（行号 / 本次改动 / 符号）。
RECORD_COLUMNS: Final[tuple[str, ...]] = ("line", "touch", "symbol")

#: 注入维度表（`cover_dims` 的入参）。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {
    "cite": CITES,
    "evidence": EVIDENCES,
    "fix": FIXES,
    "size": SIZES,
    "decoy": DECOYS,
}

#: 被禁的取值对：指向的行号都不在记录里了，就不可能有「指向那一行的符号」。
#: 由 `tests/workflows/findtriage/test_features.py` 钉住。
FORBIDDEN: Final[frozenset[tuple[str, str, str, str]]] = frozenset(
    {("cite", "missing", "evidence", "in_row")}
)

#: 维度名 → 声明下标。`allow_pair` 用它把取值对归一化到同一个顺序上。
_ORDER: Final[dict[str, int]] = {name: index for index, name in enumerate(DIMS)}

#: 符号池。**每个名字都不以 `s` 结尾**——`near_form` 靠末尾补 `s` 造近形符号，
#: 这条不变量保证近形符号永远不撞池子里的名字（`test_features.py` 钉住）。
SYMBOLS: Final[tuple[str, ...]] = (
    "parse_timeout",
    "retry_budget",
    "load_config",
    "flush_queue",
    "merge_shard",
    "parse_header",
    "read_chunk",
    "write_index",
    "scan_dir",
    "open_socket",
    "close_handle",
    "check_digest",
    "build_query",
    "escape_html",
    "trim_path",
    "lock_row",
    "evict_cache",
    "sort_batch",
    "split_frame",
    "encode_token",
    "decode_token",
    "resolve_name",
    "emit_event",
    "drain_pool",
)


def allow_pair(left: str, left_value: JsonValue, right: str, right_value: JsonValue) -> bool:
    """输入：两个维度名与各自取值（顺序任意）。输出：这一对取值允许出现吗。

    预期行为：先把顺序归一化到 `DIMS` 的声明序上再查表，所以 `(a, x, b, y)` 与 `(b, y, a, x)`
    落到同一个四元组上。
    """
    if _ORDER[left] > _ORDER[right]:
        left, left_value, right, right_value = right, right_value, left, left_value
    return (left, str(left_value), right, str(right_value)) not in FORBIDDEN


def near_form(symbol: str) -> str:
    """输入：符号名。输出：与它只差一个字母的近形符号（末尾补一个 `s`）。

    预期行为：`SYMBOLS` 里没有一个名字以 `s` 结尾，所以补出来的名字既不在池子里，
    也不会等于任何一个别的池子成员——诱饵行正是靠这条不变量才「像而不同」。
    """
    return f"{symbol}s"


@dataclass(frozen=True)
class RecordRow:
    """改动记录里的一行。`line` 是行号，`touch` 是本次改动的种类，`symbol` 是 ASCII 符号名。"""

    line: int
    touch: str
    symbol: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序按 `RECORD_COLUMNS`。"""
        return {"line": self.line, "touch": self.touch, "symbol": self.symbol}


@dataclass(frozen=True)
class RecordCase:
    """一条样本的注入特征：指向何处、证据在哪、修法点不点名、表长、有没有诱饵行。"""

    cite: str
    evidence: str
    fix: str
    size: int
    decoy: int

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {
            "cite": self.cite,
            "evidence": self.evidence,
            "fix": self.fix,
            "size": self.size,
            "decoy": self.decoy,
        }

    def fingerprint(self) -> str:
        """输入：无。输出：case 指纹。预期行为：同一 case 任何进程都得到同一串。"""
        return "|".join(f"{key}={value}" for key, value in self.as_dict().items())

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.cite}-{self.evidence}-{self.fix}-{self.size}行-诱饵{self.decoy}"


@dataclass(frozen=True)
class RecordPlan:
    """一条样本的改动记录计划：行由 case 指纹播种派生，与语言、后端无关。

    字段：`case` 注入特征；`rows` 记录表（顺序即 state 里的展示顺序，序号从 1 起）；
    `cited_index` 发现指向的行下标（0 起，指向记录外时 `None`）；`cited_line` 指向的行号；
    `evidence_symbol` 发现引用的证据符号；`fix_symbol` 修法要点名的符号。
    """

    case: RecordCase
    rows: tuple[RecordRow, ...]
    cited_index: int | None
    cited_line: int
    evidence_symbol: str
    fix_symbol: str

    @property
    def size(self) -> int:
        """输入：无。输出：改动记录条数。"""
        return len(self.rows)


@dataclass(frozen=True)
class RecordLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`cite` 指向第几行（1 起，`None` = 记录里没有这一行）；`introduced` 引入性；
    `supported` 证据强度；`actionable` 修法落不落地；`triage` 处置档；`decoy` 有没有诱饵行；
    `owner` 这条发现该进谁的队列（`OWNERS` 里的一个）。
    """

    cite: int | None
    introduced: str
    supported: str
    actionable: bool
    triage: str
    decoy: bool
    owner: str

    @property
    def cite_key(self) -> str:
        """输入：无。输出：`cite` 那一族的选项名（`s<行号>` 或 `none`）。"""
        return cite_key(self.cite)


def cite_key(index: int | None) -> str:
    """输入：行号（1 起）或 `None`。输出：`cite` 那一族的选项名。"""
    return NO_CITE if index is None else f"s{index}"


def triage_for(introduced: str, supported: str, actionable: bool) -> str:
    """输入：引入性、证据强度、修法落不落地。输出：处置档。

    预期行为：就是印在 state 里的那五条口径，**从上往下第一条命中**——顺序改了语义就变了，
    所以这条函数与 locale 里的口径文本必须逐字对齐（`test_locales.py` 钉住条数与顺序）。
    """
    if supported == "none":
        return "drop"
    if introduced == "no":
        return "log_only"
    if introduced == "unknown":
        return "log_only"
    if not actionable:
        return "log_only"
    return "must_fix"


def _seed(case: RecordCase, phrasing: int) -> int:
    """输入：case、问法编号。输出：播种用的整数。预期行为：只由入参决定，与进程无关。"""
    payload = f"{case.fingerprint()}|{phrasing}"
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _line_numbers(rng: random.Random, size: int) -> tuple[int, ...]:
    """输入：RNG、表长。输出：严格递增的 `size` 个行号（从两位数起，避免与序号混淆）。"""
    out: list[int] = []
    current = rng.randrange(10, 40)
    for _ in range(size):
        out.append(current)
        current += rng.randrange(1, 9)
    return tuple(out)


def _missing_line(rng: random.Random, lines: tuple[int, ...]) -> int:
    """输入：RNG、表里已有的行号。输出：一个**不在**表里的行号。

    预期行为：优先挑两条记录之间的空档（这样「查不到」才像是真查过），
    没有空档就退到表首之前那一行。
    """
    gaps = [lines[index] + 1 for index in range(len(lines) - 1) if lines[index] + 1 < lines[index + 1]]
    if gaps and rng.random() < 0.7:
        return rng.choice(gaps)
    return lines[0] - 1


def plan_for(case: RecordCase, phrasing: int) -> RecordPlan:
    """输入：case、问法编号。输出：改动记录计划（与语言、后端无关的随机）。

    预期行为：五个注入维度都**按构造**兑现：
    - `cite`：指向那一行的「本次改动」列，`touched` 时是 added/modified、`untouched` 时是
      untouched、`missing` 时那一行压根不在表里；
    - `evidence`：引用的符号落在指向那一行 / 记录里别的行 / 表外；
    - `fix`：修法点名的符号**就是发现引用的那个证据符号**（修法针对的是发现点出来的东西，
      两者必须是同一个符号；`fix=vague` 时这一格只写进计划、不进提示词）；
    - `decoy=1` 时表里恰好一行，其符号是证据符号的近形（补一个 `s`）。
    """
    rng = random.Random(_seed(case, phrasing))
    size = case.size
    pool = list(SYMBOLS)
    rng.shuffle(pool)
    # 池子够用：表里 size 个 + 表外证据符号 1 个 + 兜底 1 个。
    spare = pool[size:]

    slots = list(range(size))
    rng.shuffle(slots)
    cited_index = slots.pop() if case.cite != "missing" else None
    decoy_index = slots.pop() if case.decoy == 1 else None

    symbols: list[str] = pool[:size]
    if case.evidence == "in_row":
        evidence_symbol = symbols[cited_index]
    elif case.evidence == "elsewhere":
        others = [index for index in range(size) if index != cited_index and index != decoy_index]
        evidence_symbol = symbols[rng.choice(others)]
    else:
        evidence_symbol = spare[0]
    # 修法点名的符号**就是证据符号**：审查意见的修法针对的是它自己引用的那个符号。
    # 早先写成「指向那一行的符号」时，真机批次里反解读出了「修法点的符号与发现引用的
    # 不是同一个 → 这条修法落不了地」，把 actionable 判反（见范式 §9.78 第三条）。
    fix_symbol = evidence_symbol
    if decoy_index is not None:
        symbols[decoy_index] = near_form(evidence_symbol)

    touches = [rng.choice(TOUCH_KINDS) for _ in range(size)]
    if cited_index is not None:
        touches[cited_index] = (
            rng.choice(TOUCHED_KINDS) if case.cite == "touched" else "untouched"
        )
    if decoy_index is not None:
        touches[decoy_index] = "untouched"
    if not any(touch in TOUCHED_KINDS for touch in touches):
        forced = next(
            index
            for index in range(size)
            if index != cited_index and index != decoy_index
        )
        touches[forced] = "added"

    lines = _line_numbers(rng, size)
    cited_line = lines[cited_index] if cited_index is not None else _missing_line(rng, lines)
    rows = tuple(
        RecordRow(line=lines[index], touch=touches[index], symbol=symbols[index])
        for index in range(size)
    )
    return RecordPlan(
        case=case,
        rows=rows,
        cited_index=cited_index,
        cited_line=cited_line,
        evidence_symbol=evidence_symbol,
        fix_symbol=fix_symbol,
    )


def introduced_for(rows: tuple[RecordRow, ...], cited_line: int) -> str:
    """输入：记录表、指向的行号。输出：引入性三档之一。

    预期行为：**只看那一行的「本次改动」列**——查得到且是 added/modified 就是 `yes`，
    查得到但没碰过就是 `no`，查不到就是 `unknown`。这条与 locale 里的口径逐字对齐。
    """
    for row in rows:
        if row.line == cited_line:
            return "yes" if row.touch in TOUCHED_KINDS else "no"
    return "unknown"


def supported_for(rows: tuple[RecordRow, ...], cited_line: int, symbol: str) -> str:
    """输入：记录表、指向的行号、引用的证据符号。输出：证据强度三档之一。

    预期行为：符号出现在指向那一行里 → `strong`；只出现在别的行上 → `weak`；表里没有 → `none`。
    """
    for row in rows:
        if row.line == cited_line and row.symbol == symbol:
            return "strong"
    return "weak" if any(row.symbol == symbol for row in rows) else "none"


def owner_for(rows: tuple[RecordRow, ...], cited_line: int, evidence_symbol: str) -> str:
    """输入：记录表、发现指向的行号、发现引用的符号。输出：归属四档之一（`OWNERS` 里的一个）。

    预期行为：**从上往下第一条命中的说了算**，与印在 state 里的归属口径逐条对齐：

    1. 行号在表里查不到 → 记录维护者（记录本身缺行，先补记录）；
    2. 引用的符号在表里查不到 → 规则维护者（发现站不住，先修规则）；
    3. 指向的那一行本次改动碰过（added / modified）→ 本次改动的作者；
    4. 其余 → 架构组。

    构造侧与审计侧**共用这一个实现**（一边喂计划、一边喂从文本解析出来的表），两边不可能漂移。
    """
    index = next((i for i, row in enumerate(rows) if row.line == cited_line), None)
    if index is None:
        return OWNER_RECORD
    if not any(row.symbol == evidence_symbol for row in rows):
        return OWNER_RULE
    return OWNER_AUTHOR if rows[index].touch in TOUCHED_KINDS else OWNER_ARCH


def label_for(plan: RecordPlan) -> RecordLabels:
    """输入：记录计划。输出：全部金标签。

    预期行为：**纯函数、只看计划**——计划先定，标签由计划算。四件事都按印在 state 里的口径
    重推一遍，所以审计拿产物里的表也能算出同一个答案。
    """
    rows = plan.rows
    introduced = introduced_for(rows, plan.cited_line)
    supported = supported_for(rows, plan.cited_line, plan.evidence_symbol)
    actionable = plan.case.fix == "concrete"
    return RecordLabels(
        cite=None if plan.cited_index is None else plan.cited_index + 1,
        introduced=introduced,
        supported=supported,
        actionable=actionable,
        triage=triage_for(introduced, supported, actionable),
        decoy=any(row.symbol == near_form(plan.evidence_symbol) for row in rows),
        owner=owner_for(rows, plan.cited_line, plan.evidence_symbol),
    )


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[RecordCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`RecordCase` 列表，顺序稳定。
    预期行为：同一入参必得同一输出；成对覆盖保证任意两维的取值组合至少出现一次；
    被 `FORBIDDEN` 否掉的取值对既不出现、也不算没覆盖。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full, allow_pair=allow_pair)
    return [
        RecordCase(
            cite=str(row["cite"]),
            evidence=str(row["evidence"]),
            fix=str(row["fix"]),
            size=int(row["size"]),
            decoy=int(row["decoy"]),
        )
        for row in rows
    ]


def compatible(case: RecordCase) -> bool:
    """输入：一组特征。输出：这组特征写得出来吗（没有任何被禁的取值对）。"""
    values = case.as_dict()
    names = list(DIMS)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if not allow_pair(left, values[left], right, values[right]):
                return False
    return True


#: choice 题族允许出现的固定选项名（`cite` 的选项名随表长变化，由 `questions` 现场拼）。
CHOICE_OPTIONS: Final[dict[str, tuple[str, ...]]] = {
    "introduced": INTRODUCED,
    "supported": SUPPORTED,
    "triage": TRIAGES,
    "owner": OWNERS,
}
