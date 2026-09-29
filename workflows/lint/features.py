"""行级规则检查（lint）的特征层：规则表 + 行计划（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条「这段片段里有几行命中这条规则」的样本由三个**注入
特征**描述——`rule` 查哪条规则、`genre` 片段是什么体裁、`hits` 有几行命中。**标签先行**：
先定这三个取值，再由本模块的纯函数算出全部金标签。

分三层，每层只干一件事：

- **判据层**（`matches`）：一行命不命中某条规则由**代码**算，构造侧与读法侧共用这一份
  实现——所以「答案对不对」是机械可判的，不靠自然语言约定；
- **计划层**（`plan_for`）：哪几行命中由 case 指纹派生（`sha256` 播种），同一 case 在任何
  进程、任何切分里都得到同一份计划（「同素材同题必须同答」靠这条成立）；
- **文本层**（`render`）：行原文由后端写，代码只校验它兑现没兑现计划——模型写歪了就拒掉
  重来，而不是把标签改得跟文本一致（那才是真正的坏数据）。

`hits` 取 0~8 均匀（片段固定 8 行），四条题族的答案分布因此都由构造保证均衡：`line` 的
True 占比 = E[hits] / 8 = 0.5；`count` 的 9 个取值各 1/9；`gate` 三档各 1/3；`spot` 只在
恰好一行命中时出，条件分布仍是 8 行里均匀。

输入/输出：只有常量与纯函数；不碰网络、不碰文件、**绝不执行任何代码**——片段始终当文本读。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Final

from jc.cover import cover_dims
from jc.jsontypes import JsonValue

# ---------- 规则表 ----------

#: 硬编码的凭据：凭据名 + 赋值号 + 引号字面量。走环境变量的写法**不**命中（见 `PATTERNS`）。
RULE_SECRET: Final[str] = "secret"
#: 遗留的待办标记：TODO / FIXME / XXX / HACK。
RULE_TODO: Final[str] = "todo"
#: 调试残留的输出：print / console.log / debugger 这类调用。
#: Go 那一家按 `fmt.Print` / `Printf` / `Println` 三个都收——`fmt.Println` 才是 Go 里最
#: 常用的调试输出，真机 lint 批次里模型写它被判「不命中」而拒稿（2026-09-29）。
RULE_DEBUG: Final[str] = "debug"
#: 硬编码的绝对路径：引号包起来的 /Users/ /home/ /var/ 这类前缀。
RULE_PATH: Final[str] = "path"
#: 超长的行：整行超过 `LONG_LIMIT` 个字符。
RULE_LONG: Final[str] = "long"

RULES: Final[tuple[str, ...]] = (RULE_SECRET, RULE_TODO, RULE_DEBUG, RULE_PATH, RULE_LONG)

#: 片段体裁：源码 / 服务日志 / 文档。它同时决定提示词怎么写与片段抬头怎么念。
GENRES: Final[tuple[str, ...]] = ("code", "log", "doc")

#: 片段行数（固定 8）。定成 8 是为了让 `count` 的 9 个取值均匀、`gate` 的三档各占 1/3：
#: 命中行数取 0~8 均匀 → E[hits] = 4 → `line` 的 True 占比恰好 4/8 = 0.5。
SIZE: Final[int] = 8

#: 命中行数（case 维度，均匀取 0~8）。
HITS: Final[tuple[int, ...]] = tuple(range(SIZE + 1))

#: 超长行的阈值：60 个字符（中文一行超过 60 字已经明显超长）。
LONG_LIMIT: Final[int] = 60

#: 处置办法的三档边界：命中 ≤2 行放行、3~5 行人工复核、≥6 行拦下。
GATE_PASS_MAX: Final[int] = 2
GATE_REVIEW_MAX: Final[int] = 5

#: 三种处置。
GATES: Final[tuple[str, ...]] = ("pass", "review", "block")

#: 注入维度表（`cover_dims` 的入参；`hits` 的取值是整数，JSON 里也是整数）。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {
    "rule": RULES,
    "genre": GENRES,
    "hits": HITS,
}

#: 规则的文本判据（纯函数，构造侧与读法侧共用）。
#: `secret` 要求「凭据名 + 赋值号 + 带引号的字面量」——`API_KEY = os.environ["API_KEY"]`
#: 这种走环境变量的写法因此**不**算命中，这是最容易被看走眼的一格。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    RULE_SECRET: re.compile(
        r"(?i)\b\w*(?:passwd|password|pwd|secret|token|api[_-]?key|access[_-]?key"
        r"|private[_-]?key|credential)\w*\s*[:=]\s*[\"'][^\"']{4,}[\"']"
    ),
    RULE_TODO: re.compile(r"(?i)\b(?:TODO|FIXME|XXX|HACK)\b"),
    RULE_DEBUG: re.compile(
        r"(?i)\b(?:print|console\.log|var_dump|logger\.debug|fmt\.Print(?:f|ln)?)\s*\(|\bdebugger\b"
    ),
    RULE_PATH: re.compile(
        r"[\"'](?:/Users/|/home/|/var/|/etc/|/opt/|/tmp/|C:\\|D:\\|E:\\)[^\"']*[\"']"
    ),
}


def matches(rule: str, line: str) -> bool:
    """输入：规则名、一行文本。输出：这一行是否命中该规则。

    预期行为：`long` 按字符数判，其余四条按 `PATTERNS` 里的正则判；规则名不认识直接报错
    （不静默返回 False——那会把「规则表改了、读法忘了改」变成一堆答案对不上的样本）。
    """
    if rule == RULE_LONG:
        return len(line) > LONG_LIMIT
    pattern = PATTERNS.get(rule)
    if pattern is None:
        raise KeyError(f"没有规则 {rule!r}（可用：{', '.join(RULES)}）")
    return pattern.search(line) is not None


def match_any(line: str) -> str | None:
    """输入：一行文本。输出：命中的第一条规则名（一条都不命中给 None）。

    预期行为：按 `RULES` 的固定顺序取第一条命中——`render.check_draft` 用它保证「干净行一条
    都不命中」，报错时也能指出是撞上了哪一条。
    """
    for rule in RULES:
        if matches(rule, line):
            return rule
    return None


def gate_of(hits: int) -> str:
    """输入：命中行数。输出：处置档位（pass / review / block）。

    预期行为：按 `GATE_PASS_MAX` / `GATE_REVIEW_MAX` 分段，越界（负数或超过片段行数）报错。
    """
    if not 0 <= hits <= SIZE:
        raise ValueError(f"命中行数必须在 0~{SIZE} 之间：{hits}")
    if hits <= GATE_PASS_MAX:
        return "pass"
    if hits <= GATE_REVIEW_MAX:
        return "review"
    return "block"


# ---------- 注入特征与行计划 ----------


@dataclass(frozen=True)
class LintCase:
    """一条样本的注入特征：查哪条规则、什么体裁、几行命中。"""

    rule: str
    genre: str
    hits: int

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：{维度名: 取值}（统计与指纹共用）。"""
        return {"rule": self.rule, "genre": self.genre, "hits": self.hits}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.rule}/{self.genre}/{self.hits}"

    def fingerprint(self) -> str:
        """输入：无。输出：素材指纹（确定性；行计划的种子由它派生）。"""
        return f"{self.rule}|{self.genre}|{self.hits}"


@dataclass(frozen=True)
class LintPlan:
    """一份片段的行计划：查哪条规则、什么体裁、问法编号、哪几行命中（0 起下标，升序）。

    为什么不直接存文本：`sample` 阶段拿不到语言文件、也拿不到后端，所以这里只定「哪几行
    命中」，真正的文本在 `render` 阶段由后端写。同一份计划在任何语言下都能渲染。
    """

    rule: str
    genre: str
    phrasing: int
    hit_rows: tuple[int, ...]

    @property
    def hits(self) -> int:
        """输入：无。输出：命中行数。预期行为：就是 `hit_rows` 的长度。"""
        return len(self.hit_rows)

    def matched(self) -> tuple[bool, ...]:
        """输入：无。输出：逐行的命中标记（按片段顺序）。"""
        return tuple(row in self.hit_rows for row in range(SIZE))


def plan_for(case: LintCase, phrasing: int) -> LintPlan:
    """输入：case、问法编号。输出：行计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里
    都得到同一份计划；命中行的**位置**由指纹掷出，`spot` 题的答案因此在 8 行里均匀分布。
    问法编号也进种子，是为了让同一个 (规则, 体裁, 命中数) 的不同问法拿到不同的行计划——
    否则这几种问法会渲染出逐字节相同的片段，样本就成了重复数据。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    rng = random.Random(seed)
    return LintPlan(
        rule=case.rule,
        genre=case.genre,
        phrasing=phrasing,
        hit_rows=tuple(sorted(rng.sample(range(SIZE), case.hits))),
    )


@dataclass(frozen=True)
class Labels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`matched` 逐行命中标记；`hits` 命中行数；`gate` 处置档位；
    `spot` 唯一命中行的**行号**（1 起；不是恰好一行命中时为 None）。
    """

    matched: tuple[bool, ...]
    hits: int
    gate: str
    spot: int | None


def label_for(plan: LintPlan) -> Labels:
    """输入：行计划。输出：全部金标签。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`spot` 只在恰好一行
    命中时才有值（多行或零行命中时「是哪一行」没有唯一解，`questions.families_for` 据此
    不出这一族）。
    """
    matched = plan.matched()
    rows = [index + 1 for index, hit in enumerate(matched) if hit]
    return Labels(
        matched=matched,
        hits=plan.hits,
        gate=gate_of(plan.hits),
        spot=rows[0] if len(rows) == 1 else None,
    )


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[LintCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`LintCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [
        LintCase(rule=str(row["rule"]), genre=str(row["genre"]), hits=int(row["hits"]))
        for row in rows
    ]
