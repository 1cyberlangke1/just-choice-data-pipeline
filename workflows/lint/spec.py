"""行级规则检查（lint）workflow 的真值规格：一张规则表 + 一段片段的行模式。

输入/输出：不可变数据类与纯函数；这里只有**键**（规则名 / 体裁名 / 槽号），
中文英文的文案全在 `locales/`。
预期行为：答案只从渲染出来的片段文本反推——规则靠**规则短语**认，某一行命中与否靠
`matches(规则, 行)` 这个**纯文本判据**算，所以构造侧与判别侧共用同一套读法
（`parse.read_state`）。构造写歪与渲染写歪会变成同一类错误，不会两边各错一半还互相印证。

覆盖的决策形状（社区用例地图缺口：语义 grep / 语义 lint 家族，20+ 个第三方项目）：
- **逐行判定**（`line`）：第 k 行命中这条规则吗——语义 grep 的本体；
- **计数**（`count`）：这段片段里有几行命中；
- **门控**（`gate`）：按题面写明的处置办法（≤2 行放行 / 3~5 行人工复核 / ≥6 行拦下）走哪一档；
- **定位**（`spot`）：恰好只有一行命中时，是哪一行（高基数 Choice）。

片段固定 8 行、命中行数 `hits` 取 0~8 均匀——四条题族的答案分布因此都由构造保证均衡：
`line` 的 True 占比 = E[hits]/8 = 0.5；`count` 的 9 个取值各 1/9；`gate` 三档各 1/3；
`spot` 在 8 行里均匀（它只在「恰好一行命中」时出，条件分布仍是均匀的）。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from itertools import product
from typing import Any, Final

from jc.knobs import require_pool

# ---------- 规则表（键；文案在 locales/） ----------

RULE_SECRET: Final[str] = "secret"  # 硬编码的凭据
RULE_TODO: Final[str] = "todo"  # 遗留的待办标记
RULE_DEBUG: Final[str] = "debug"  # 调试残留的输出
RULE_PATH: Final[str] = "path"  # 硬编码的绝对路径
RULE_LONG: Final[str] = "long"  # 超长的行

RULES: Final[tuple[str, ...]] = (RULE_SECRET, RULE_TODO, RULE_DEBUG, RULE_PATH, RULE_LONG)

#: 片段体裁：源码 / 服务日志 / 文档。它同时决定背景行长什么样与片段抬头怎么写。
GENRES: Final[tuple[str, ...]] = ("code", "log", "doc")

#: 片段行数（固定 8）。定成 8 是为了让 `count` 的 9 个取值均匀、`gate` 的三档各占 1/3：
#: 命中行数取 0~8 均匀 → E[hits] = 4 → `line` 的 True 占比恰好 4/8 = 0.5。
SIZE: Final[int] = 8

#: 命中行数（case 维度，均匀取 0~8）。
HITS: Final[tuple[int, ...]] = tuple(range(SIZE + 1))

#: 超长行的阈值（`long` 规则的判据）。60 个字符：中文一行超过 60 字已经明显超长。
LONG_LIMIT: Final[int] = 60

#: 门控的三档边界：命中 ≤2 行放行、3~5 行人工复核、≥6 行拦下。
#: 9 个命中档恰好三等分，所以三档答案各占 1/3（由测试钉住）。
GATE_PASS_MAX: Final[int] = 2
GATE_REVIEW_MAX: Final[int] = 5

#: 规则的文本判据（纯函数，构造侧与读法侧共用）。
#: `secret` 要求「凭据名 + 赋值号 + 带引号的字面量」——`API_KEY = os.environ["API_KEY"]`
#: 这种走环境变量的写法因此**不**算命中（背景行正是靠这条规矩躲开的）。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    RULE_SECRET: re.compile(
        r"(?i)\b\w*(?:passwd|password|pwd|secret|token|api[_-]?key|access[_-]?key"
        r"|private[_-]?key|credential)\w*\s*[:=]\s*[\"'][^\"']{4,}[\"']"
    ),
    RULE_TODO: re.compile(r"(?i)\b(?:TODO|FIXME|XXX|HACK)\b"),
    RULE_DEBUG: re.compile(
        r"(?i)\b(?:print|console\.log|debugger|var_dump|logger\.debug|fmt\.Print)\s*\("
    ),
    RULE_PATH: re.compile(
        r"[\"'](?:/Users/|/home/|/var/|/etc/|/opt/|/tmp/|C:\\|D:\\|E:\\)[^\"']*[\"']"
    ),
}

#: 片段里每一行的行号前缀。**这是 state 格式的一部分，不是文案**，所以不进语言文件：
#: 渲染与读法共用这一个常量，两边不可能漂移。
LINE_TEMPLATE: Final[str] = "{n}. {line}"
LINE_PREFIX: Final[re.Pattern[str]] = re.compile(r"^\s*(\d+)\.\s?(.*)$")


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


def gate_of(hits: int) -> str:
    """输入：命中行数。输出：处置档位（pass / review / block）。

    预期行为：按 `GATE_PASS_MAX` / `GATE_REVIEW_MAX` 分段，越界（负数或超过片段行数）报错。
    """
    if not 0 <= hits <= SIZE:
        raise ValueError(f"命中行数必须落在 0~{SIZE}，收到 {hits!r}")
    if hits <= GATE_PASS_MAX:
        return "pass"
    if hits <= GATE_REVIEW_MAX:
        return "review"
    return "block"


# ---------- 值留出（训练侧见过哪些取值） ----------

#: 训练侧见过的规则：`long`（按长度判的那条）整个留出。
SEEN_RULES: Final[frozenset[str]] = frozenset({RULE_SECRET, RULE_TODO, RULE_DEBUG, RULE_PATH})
#: 训练侧见过的体裁：`doc` 留出（文档片段的背景行与抬头在训练侧一个都不出现）。
SEEN_GENRES: Final[frozenset[str]] = frozenset({"code", "log"})
#: 训练侧见过的问法：第二套句面留出。
SEEN_PHRASINGS: Final[frozenset[int]] = frozenset({0})

#: 问法（句面模板）编号。
PHRASINGS: Final[tuple[int, ...]] = (0, 1)


@dataclass(frozen=True)
class LintCase:
    """一个 case：规则 × 体裁 × 命中行数 × 问法。"""

    rule: str
    genre: str
    hits: int
    phrasing: int

    def as_dict(self) -> dict[str, Any]:
        """输入：无。输出：{维度名: 取值}（四个维度全摊平）。"""
        return {
            "rule": self.rule,
            "genre": self.genre,
            "hits": self.hits,
            "phrasing": self.phrasing,
        }

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.rule}/{self.genre}/{self.hits}/{self.phrasing}"

    def unseen_dims(self) -> tuple[str, ...]:
        """输入：无。输出：这个组合里「训练侧没见过」的维度名（按固定顺序）。"""
        out: list[str] = []
        if self.rule not in SEEN_RULES:
            out.append("rule")
        if self.genre not in SEEN_GENRES:
            out.append("genre")
        if self.phrasing not in SEEN_PHRASINGS:
            out.append("phrasing")
        return tuple(out)


def material_of(case: LintCase) -> str:
    """输入：case。输出：素材指纹（确定性；切分分桶与内容种子都由它派生）。"""
    return f"{case.rule}|{case.genre}|{case.hits}|{case.phrasing}"


@dataclass(frozen=True)
class LintConfig:
    """一个切分的采样参数：四张取值表 + 是否只留 OOD。"""

    rules: tuple[str, ...] = RULES
    genres: tuple[str, ...] = GENRES
    hits: tuple[int, ...] = HITS
    phrasings: tuple[int, ...] = PHRASINGS
    split: str = "train"
    ood_only: bool = False

    def __post_init__(self) -> None:
        """输入：无。输出：无。预期行为：任一取值表为空就报人话（点名是哪个字段）。"""
        require_pool("rules", self.rules)
        require_pool("genres", self.genres)
        require_pool("hits", self.hits)
        require_pool("phrasings", self.phrasings)


def build_cases(config: LintConfig) -> list[LintCase]:
    """输入：采样参数。输出：case 矩阵（顺序稳定）。

    预期行为：四张表取全组合；`ood_only` 时只留「至少含一个训练侧没见过的取值」的组合，
    所以 OOD 与同分布侧**按构造就不重叠**。
    """
    out: list[LintCase] = []
    for rule, genre, hits, phrasing in product(
        config.rules, config.genres, config.hits, config.phrasings
    ):
        case = LintCase(rule, genre, hits, phrasing)
        if config.ood_only and not case.unseen_dims():
            continue
        out.append(case)
    return out


# ---------- 行计划（还没填文案） ----------

#: 行的两种来源：命中槽（`[hit.<规则>.<体裁>]`）与背景槽（`[bg.<体裁>]`）。
KIND_HIT: Final[str] = "hit"
KIND_BG: Final[str] = "bg"


@dataclass(frozen=True)
class Entry:
    """片段里的一行：它从哪个槽来。

    预期行为：`kind == KIND_HIT` 的行**必然**命中当前规则、`KIND_BG` 的行**必然**不命中
    任何规则——这两条是构造不变量，由测试与 `build_plan` 的断言一起钉住。
    """

    kind: str
    slot: int


@dataclass(frozen=True)
class LintPlan:
    """一份片段的行计划：规则 + 体裁 + 每行来自哪个槽 + 问法。

    为什么不直接存文本：`sample` 阶段拿不到语言文件（协议里没有 locale），所以这里只定
    「哪几行命中、来自哪个槽」，真正的文本在 `render` 阶段填。同一份计划在任何语言下都能
    渲染，题目与答案两边用的是同一份计划。
    """

    rule: str
    genre: str
    phrasing: int
    entries: tuple[Entry, ...]

    @property
    def hits(self) -> int:
        """输入：无。输出：命中行数。预期行为：就是 `KIND_HIT` 的行数。"""
        return sum(1 for entry in self.entries if entry.kind == KIND_HIT)

    def matched(self) -> tuple[bool, ...]:
        """输入：无。输出：逐行的命中标记（按片段顺序）。"""
        return tuple(entry.kind == KIND_HIT for entry in self.entries)


@dataclass(frozen=True)
class LintSpec:
    """一条样本的真值：case + 行计划（文案在渲染阶段才填）。"""

    case: LintCase
    split: str
    seq: int
    plan: LintPlan


def build_plan(case: LintCase) -> LintPlan:
    """输入：case。输出：行计划（与语言无关的随机）。

    预期行为：随机种子由**素材指纹**派生，所以同一个 case 在任何切分、任何进程里都得到
    同一份计划（「同素材同题必须同答」这条一致性校验才成立）。命中行数就是 case 的 `hits`，
    命中行在片段里的**位置**由指纹掷出——`spot` 题的答案因此在 8 行里均匀分布。
    """
    seed = int(hashlib.sha256(material_of(case).encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    hit_slots = rng.sample(range(SIZE), case.hits)
    bg_slots = rng.sample(range(SIZE), SIZE - case.hits)
    entries = [Entry(KIND_HIT, slot) for slot in hit_slots]
    entries += [Entry(KIND_BG, slot) for slot in bg_slots]
    rng.shuffle(entries)
    return LintPlan(case.rule, case.genre, case.phrasing, tuple(entries))


def sample_spec(case: LintCase, split: str, seq: int) -> LintSpec:
    """输入：case、切分、序号。输出：真值规格。

    预期行为：**不掷骰子**——计划完全由 case 指纹决定，这样「同素材同题必须同答」成立。
    """
    return LintSpec(case, split, seq, build_plan(case))
