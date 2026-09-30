"""改动审查（review）的特征层：注入维度 + 审查规则表 / 严重度表 / 门控表（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一次「这一处改动该怎么审」的判断由 5 个**注入特征**描述，
**标签先行**——先定这 5 维取值，再由本模块的纯函数算出全部金标签。

五维分成两拨：

- **改动正文承载**（读者只能从那段改动里看出来）：`secret` 改动的**新增行**里有没有凭据
  字面量、`debug` 改动的**新增行**里有没有调试输出。这两维交给 `judges.TraceJudge` 反解
  校验——答不回来就说明那段代码与标签脱节了。
  **只算新增行**：一行写死的凭据被删掉时，那行字面量出现在删除行上，这一格是 `no`
  （否则「有凭据」会被读成「刚写进去的」与「刚删掉的」两种意思，没有唯一答案——同一条口径
  也是 `secret=yes × deleted` / `debug=yes × deleted` 两对禁用对的来由）；
- **改动清单承载**（读者从「改动清单」那一行字段里读）：`file_kind` 文件类型、
  `change_kind` 变更类型、`size` 改动规模。这三维由 `audit.py` 拿产物里记下的答案交叉校验。

提示词**五维都给**——这一点与 routing 不同，理由在这里说清楚：那段正文是**一段代码改动**，
它的形状（是不是测试文件、是新增还是删除、几行）本来就必须与清单那一行对得上，不给就写不出
相称的改动。泄底风险由另一条口径挡：正文里不许出现表名（见 `render.check_draft` 与 locale 的
`banned_intent_phrases`），而且清单那一行是**权威**——正文与它冲突时以它为准，读者不会歧义。

三条审查规则**只看正文能证明的东西**（凭据字面量 / 调试输出）与清单里的一格（规模）。
三条规则的**适用文件类型**互不相同，这正是「同一处改动落在不同文件里权重不同」的落地：

- 凭据规则：改动的**新增行**里有「凭据名 = 带引号的字面量」——源码 / 测试 / 配置该提，文档不该提；
- 调试规则：改动的**新增行**里有调试输出——源码 / 配置该提，测试与文档不该提（测试里打日志很正常）；
- 规模规则：改动规模是「大」——只有源码该提（文档与配置本来就长）。

输入/输出：只有常量与纯函数；不碰网络、不碰文件、**绝不执行任何代码**。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from jc.cover import cover_dims, row_allowed

# ---------- 维度表 ----------

#: 改动的新增行里有没有凭据字面量。**只由改动正文承载**。
SECRETS: Final[tuple[str, ...]] = ("no", "yes")

#: 改动的新增行里有没有调试输出。**只由改动正文承载**。
DEBUGS: Final[tuple[str, ...]] = ("no", "yes")

#: 被改的文件类型。**只由「改动清单」那一行承载**。
FILE_KINDS: Final[tuple[str, ...]] = ("source", "test", "config", "docs")

#: 这次改动是哪一种。**只由「改动清单」那一行承载**。
CHANGE_KINDS: Final[tuple[str, ...]] = ("added", "modified", "deleted")

#: 改动规模。**只由「改动清单」那一行承载**。
SIZES: Final[tuple[str, ...]] = ("small", "medium", "large")

DIMS: Final[dict[str, tuple[str, ...]]] = {
    "secret": SECRETS,
    "debug": DEBUGS,
    "file_kind": FILE_KINDS,
    "change_kind": CHANGE_KINDS,
    "size": SIZES,
}

# ---------- 三张判据表 ----------

#: 审查规则名。顺序就是**从上往下第一条命中**的顺序，改顺序等于改语义——
#: `tests/workflows/review/test_features.py` 拿审查规则表的文本行顺序钉住它。
REVIEW_RULES: Final[tuple[str, ...]] = ("credential", "debug", "size", "none")

#: 每条规则**适用**的文件类型。不在适用范围里就一律不算命中（文档里写个示例凭据不该按
#: 凭据规则提），兜底规则 `none` 对任何文件类型都「适用」。
RULE_SCOPES: Final[dict[str, frozenset[str]]] = {
    "credential": frozenset({"source", "test", "config"}),
    "debug": frozenset({"source", "config"}),
    "size": frozenset({"source"}),
    "none": frozenset(FILE_KINDS),
}

#: 严重度档数：0 不提 / 1 轻 / 2 中 / 3 重。
SEVERITY_LEVELS: Final[int] = 4

#: 「不提」那一档的下标。
NOT_RAISED: Final[int] = 0

#: 严重度表：(规则, 文件类型) → 档位。**查表，不是加权求和**——缺格直接抛 KeyError。
#: 表里只放真正会命中的格子（不适用的文件类型由 `RULE_SCOPES` 挡在外面，渲染时写成 0 档）。
SEVERITY_TABLE: Final[dict[tuple[str, str], int]] = {
    ("credential", "source"): 3,
    ("credential", "test"): 2,
    ("credential", "config"): 3,
    ("debug", "source"): 2,
    ("debug", "config"): 2,
    ("size", "source"): 1,
}

#: 门控动作（也是 `gate` 题族的选项顺序）。
GATES: Final[tuple[str, ...]] = ("allow", "request_changes", "block")

#: 门控表：(严重度, 变更类型) → 动作。四档严重度 × 三种变更类型全定义。
#:
#: 口径：重的**一律拦下**；中的在新增 / 修改时**请求修改**、删除时**拦下**（删掉的是别人
#: 正在用的东西，得有人看一眼）；轻的在修改时**请求修改**、新增 / 删除时**放行**。
GATE_TABLE: Final[dict[tuple[int, str], str]] = {
    (3, "added"): "block",
    (3, "modified"): "block",
    (3, "deleted"): "block",
    (2, "added"): "request_changes",
    (2, "modified"): "request_changes",
    (2, "deleted"): "block",
    (1, "added"): "allow",
    (1, "modified"): "request_changes",
    (1, "deleted"): "allow",
    (0, "added"): "allow",
    (0, "modified"): "allow",
    (0, "deleted"): "allow",
}

# ---------- 正文的形状 ----------

#: 改动行数的期望区间：规模 → (最少行, 最多行)。渲染期（`render.check_draft`）与产物期
#: （`audit.py`）共用同一份口径。
#:
#: 口径刻意**宽松且重叠**：这一维进标签的唯一去处是规模规则（大 + 源码 → 轻），所以真正要
#: 保证的是「小的看着小、大的看着大」，而不是精确到行。真机第一批（`review-llm-10a-zh`）
#: 里就有一条把 7 行标成「中」——收窄区间只会白拒好数据（范式 §9.15 第二条教训：
#: 能禁就禁，不能禁才改提示词；这里是「能放宽就放宽，不能放宽才拒」）。
#:
#: 「小」为什么要到 4 行：`修改` 这一档要求**同时**有新增行与删除行，而两维特征都写在
#: 新增行上（有凭据字面量 + 有调试输出 = 两行），再加一行删除行就是 3 行。
SIZE_LINES: Final[dict[str, tuple[int, int]]] = {
    "small": (1, 4),
    "medium": (3, 8),
    "large": (6, 12),
}

#: 验收时上界的宽容倍数：`SIZE_LINES` 既是**给模型的写作指引**（提示词里那句「小 1~4 行…」
#: 就是它），也是题面的口径；但**清单那一行才是权威**——正文比指引长几行，读者照样按清单
#: 回答规模，标签一点没错，按上界硬拒是白拒好数据（英文批真机上「大 6~12」被写到 14 行，
#: 连撞两次；见范式 §9.19）。所以验收把上界放大到这个倍数，下界不放——正文短于下界才是
#: 真与清单打架（「大」改动只有 3 行）。
SIZE_LINES_SLACK: Final[int] = 2

#: 改动行的三种前缀：新增 / 删除 / 上下文。
LINE_PREFIXES: Final[tuple[str, ...]] = ("+", "-", " ")


def size_line_bounds(size: str) -> tuple[int, int]:
    """输入：规模取值。输出：**验收**用的 (下界, 上界)。

    预期行为：下界照 `SIZE_LINES`，上界乘以 `SIZE_LINES_SLACK`。渲染期、判别器与产物审计
    都用这一份，免得三处各写各的。
    """
    low, high = SIZE_LINES[size]
    return low, high * SIZE_LINES_SLACK


def diff_mark(line: str) -> str:
    """输入：改动正文的一行。输出：这一行的前缀（`+` / `-` / 空格）。

    预期行为：先剥掉行首空白再认标记——真机上模型常把整段缩进两格（Markdown 列表那种），
    标记落在缩进后面（`  - foo`），那与 `- foo` 是同一件事。标记后面必须是空格或行尾，
    免得把 `-1`、`-> int` 这类真实代码读成删除行；认不出标记的一律当上下文行。
    """
    stripped = line.lstrip()
    if stripped[:1] in ("+", "-") and stripped[1:2] in ("", " "):
        return stripped[0]
    return " "

#: 天生不能同时出现的取值对（四元组，按 `DIMS` 的声明顺序写）。两条都是「这一格没有唯一
#: 答案」，不是「物理上不可能」：
#:
#: - `有凭据字面量 × 删除`：把一行写死的凭据删掉是**好事**，按「正文里有凭据」去提它，
#:   读的人分不出这是「刚写进去的」还是「刚删掉的」；
#: - `有调试输出 × 删除`：同上，删掉一处调试输出本来就该放行。
#:
#: 这两格由 `tests/workflows/review/test_features.py` 钉住；真机批次若暴露别的格子，
#: 按同样口径补进来（范式 §9.15 第二条教训：能禁就禁，不能禁才改提示词）。
FORBIDDEN: Final[frozenset[tuple[str, str, str, str]]] = frozenset(
    {
        ("secret", "yes", "change_kind", "deleted"),
        ("debug", "yes", "change_kind", "deleted"),
    }
)

#: 维度名 → 声明下标。`allow_pair` 用它把取值对归一化到同一个顺序上。
_ORDER: Final[dict[str, int]] = {name: index for index, name in enumerate(DIMS)}


@dataclass(frozen=True)
class Labels:
    """一条样本的全部金标签。字段名与题族名一一对应。"""

    secret: str
    debug: str
    rule: str
    severity: int
    gate: str


@dataclass(frozen=True)
class TagSet:
    """一组注入特征。字段名与 `DIMS` 的键一一对应。"""

    secret: str
    debug: str
    file_kind: str
    change_kind: str
    size: str

    def as_dict(self) -> dict[str, str]:
        """输入：无。输出：`{维度名: 取值}`（顺序按 `DIMS`）。"""
        return {name: getattr(self, name) for name in DIMS}

    def label(self) -> Labels:
        """输入：无。输出：这组特征对应的全部金标签。"""
        return label_for(self)


def allow_pair(left: str, left_value: str, right: str, right_value: str) -> bool:
    """输入：两个维度名与各自取值。输出：这一对写得出来吗。

    预期行为：与调用顺序无关——按 `DIMS` 的声明顺序归一化后再查禁用表，
    所以 `(a, x, b, y)` 与 `(b, y, a, x)` 落到同一个四元组上。
    """
    if _ORDER[left] > _ORDER[right]:
        left, left_value, right, right_value = right, right_value, left, left_value
    return (left, left_value, right, right_value) not in FORBIDDEN


def rule_hits(name: str, tags: TagSet) -> bool:
    """输入：规则名（不含兜底规则）、一组注入特征。输出：这条规则命中吗。

    预期行为：**先看适用范围**——文件类型不在 `RULE_SCOPES` 里就一律不命中，
    哪怕正文里真有凭据字面量。兜底规则 `none` 不是「命中」，传进来直接报错。
    """
    if name not in RULE_SCOPES or name == "none":
        raise KeyError(f"没有这条规则：{name!r}（兜底规则不算命中）")
    if tags.file_kind not in RULE_SCOPES[name]:
        return False
    if name == "credential":
        return tags.secret == "yes"
    if name == "debug":
        return tags.debug == "yes"
    return tags.size == "large"


def rule_for(tags: TagSet) -> str:
    """输入：一组注入特征。输出：命中的审查规则名（第一条命中者，都不中给 `none`）。

    预期行为：严格按 `REVIEW_RULES` 的顺序判。这张表就是 state 里渲染给考生的那张表，
    所以这里的顺序必须与 locale 里 `rules` 的书写顺序一致。
    """
    for name in REVIEW_RULES[:-1]:
        if rule_hits(name, tags):
            return name
    return "none"


def severity_for(tags: TagSet) -> int:
    """输入：一组注入特征。输出：严重度档（0 不提 / 1 轻 / 2 中 / 3 重）。

    预期行为：命中兜底规则就是「不提」（0 档）；否则查 `SEVERITY_TABLE`——缺格直接抛
    KeyError：缺格意味着有取值组合压根没定义，静默兜底会把「没定义」伪装成「有定义」。
    """
    rule = rule_for(tags)
    if rule == "none":
        return NOT_RAISED
    return SEVERITY_TABLE[(rule, tags.file_kind)]


def gate_for(tags: TagSet) -> str:
    """输入：一组注入特征。输出：门控动作（allow / request_changes / block）。"""
    return GATE_TABLE[(severity_for(tags), tags.change_kind)]


def label_for(tags: TagSet) -> Labels:
    """输入：一组注入特征。输出：全部金标签。

    预期行为：**纯函数、只看特征、不看文本**——文本是后写的，标签先定。

    全空间（2×2×4×3×3 = 144 组，剔除 2 对禁用对后 108 组）上的分布：`secret` / `debug`
    各少掉「删除」那一档（禁用对砍的），其余各维均匀；`rule` 四条都到得了；
    `severity` 0~3 四档都到得了；`gate` 三种动作都到得了。要的是**每一档都到得了**，
    不是每档一样多。
    """
    return Labels(
        secret=tags.secret,
        debug=tags.debug,
        rule=rule_for(tags),
        severity=severity_for(tags),
        gate=gate_for(tags),
    )


def build_cases(max_full: int, dims: dict[str, tuple[str, ...]] | None = None) -> list[TagSet]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖；LLM workflow 要把它压到几十）、
    可选的维度表覆盖（留出某几个取值时用，例如 OOD 的「值留出」）。
    输出：`TagSet` 列表，顺序稳定。
    预期行为：同一入参必得同一输出；成对覆盖保证任意两维的取值组合至少出现一次；
    被 `FORBIDDEN` 否掉的取值对既不出现、也不算没覆盖。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full, allow_pair=allow_pair)
    return [TagSet(**row) for row in rows]


def compatible(tags: TagSet) -> bool:
    """输入：一组特征。输出：这组特征写得出来吗（没有任何被禁的取值对）。"""
    return row_allowed(list(DIMS), tags.as_dict(), allow_pair)


#: choice 题族允许出现的选项名（题面只收本题允许的，别的取值留给反解提示词用）。
CHOICE_OPTIONS: Final[dict[str, tuple[str, ...]]] = {
    "rule": REVIEW_RULES,
    "gate": GATES,
}
