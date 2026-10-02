"""测试影响选择（testimpact）的特征层：覆盖表 + 改动计划 + 该跑哪些套件（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条样本由两个**注入特征**描述——`touch`（这次改动触及了哪几条
路径）与 `flavor`（这是一次什么性质的改动）。**标签先行**：先定这两个取值，再由本模块的纯函数
算出结论（这次改动需要跑哪几套测试）。

**为什么要有这一族**：社区里 `jev-test-filter`（★15）拿 Jev 给 git diff 里的每条测试打分、吐出
`vitest` / `node:test` / `Playwright` / `cargo test` / `go test` 的过滤参数；`jev-flash-router`
把它写成「文件选择、任务路径与**测试影响判断**」；`semdecide` 与 `jev-cli` 把同一种判断接到 CI
与 shell 的退出码上。共同点是一份**改动**进来、要一个**立刻能落地**的跑法（只跑冒烟 / 跑一套 /
跑两套 / 全套），而不是让人通读整份 diff 再猜。

**与 `review` 的区别**：`review` 判的是「这段改动本身有没有毛病」（清单行 + 三张表，读的是代码
写得对不对）；本 workflow 判的是「这段改动**该让 CI 跑什么**」（覆盖表 + 跑法档位表，读的是改动
落在哪些路径上）。前者读内容，后者读**落点**；判据空间与结论语义都不一样，不重复。

**为什么结论是「并集」而不是「相加」**：该跑哪些套件是**把触及路径的覆盖列并起来**算出来的。
本 workflow 的表里每条代码路径恰好挂一套（真实仓库里一条路径挂好几套、一套测试覆盖好几条路径
都常见，但那样一来某一套的出现频率会顶到 13/16，`flag` 那一族按构造就失衡了，见范式 §9.25），
所以并集这一步看着简单，它仍然是真的：改到 `P1` 与 `P2` 算两套、改到 `P1` 与 `P5` 只算一套
（文档路径不挂套件）。

**盲区与文档路径的覆盖列都是空的，结论却相反**：`src/legacy/parser.py`（代码，一条测试都没有）
要**跑全套**（影响面未知），`docs/guide.md`（文档，本来就不挂代码套件）**只跑冒烟**。分得开它们
的是表里的**类型列**（代码 / 文档），不是覆盖列——这一条是本 workflow 的核心读法。

**`flavor` 是这一族的留出维**：它**只影响渲染、不影响标签**——同一份触及组合写成修缺陷、加功能
还是重构，答案一个字都不差。训练侧只见过前三种，留出侧是 `revert`（回滚），用来验「读法不靠
改动性质的先验」（范式 §9.25 那条「只影响渲染的那一维」）。

本模块有**两条独立路径**：`label_for` 从计划推标签（构造侧），`observed_labels` 从 diff 文本重推
标签（审计侧，走 `touched_in` 的文件头解析）。两者对不上就说明渲染没兑现计划。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Final

from jc.cover import Fingerprinted, cover_dims
from jc.jsontypes import JsonValue

#: 三套代码套件（**结构键**：套件的显示名与说明全在 `locales/` 的 `[suite.*]`）。
SUITES: Final[tuple[str, ...]] = ("unit", "integration", "e2e")

#: 覆盖表里的路径编号（**结构键**）。文件路径也写在这里：它们是代码标识符、不是自然语言，
#: 所以不进语言文件——渲染与解析共用同一份，两边不可能漂移。
PATH_IDS: Final[tuple[str, ...]] = ("P1", "P2", "P3", "P4", "P5")

#: 路径编号 → 仓库里的文件路径。
PATH_FILES: Final[dict[str, str]] = {
    "P1": "src/api/handlers.py",
    "P2": "src/db/schema.py",
    "P3": "src/ui/app.js",
    "P4": "src/legacy/parser.py",
    "P5": "docs/guide.md",
}

#: 路径类型：`code` 参与代码套件，`docs` 不参与（改了它只需过冒烟）。
KIND_CODE: Final[str] = "code"
KIND_DOCS: Final[str] = "docs"
PATH_KINDS: Final[dict[str, str]] = {
    "P1": KIND_CODE,
    "P2": KIND_CODE,
    "P3": KIND_CODE,
    "P4": KIND_CODE,
    "P5": KIND_DOCS,
}

#: 路径编号 → 覆盖它的套件。三条代码路径各挂一套，两条 `∅` 分别是盲区（代码、没人写测试）与
#: 文档（本来就不挂代码套件）——两者的覆盖列一模一样，结论却相反，靠类型列分。
COVERAGE: Final[dict[str, tuple[str, ...]]] = {
    "P1": ("unit",),
    "P2": ("integration",),
    "P3": ("e2e",),
    "P4": (),
    "P5": (),
}

#: 盲区：代码路径，却一条测试都不覆盖。触及它 = 影响面未知 = 跑全套。
BLIND_PATH: Final[str] = "P4"

#: 改动性质（**结构键**：文案在 `locales/` 的 `[flavor.*]`）。这一维**只影响渲染、不影响
#: 标签**——它是 `test_ood` 的留出维：训练侧只见过前三种，留出侧是 `revert`。
FLAVORS: Final[tuple[str, ...]] = ("fix", "feat", "refactor")
OOD_FLAVOR: Final[str] = "revert"
ALL_FLAVORS: Final[tuple[str, ...]] = (*FLAVORS, OOD_FLAVOR)

#: 跑法档位（**结构键**：文案在 `locales/` 的 `[fam.route.opt]`）。四档由「需要几套」算出来，
#: 盲区单独占满档（影响面未知，只能全套跑）。
LEVELS: Final[tuple[str, ...]] = ("smoke", "targeted", "pair", "full")

#: 提交标题与改动的长度上下限。太短读不出「像不像一次真改动」，太长就只是灌水。
MIN_TITLE_CHARS: Final[int] = 6
MAX_TITLE_CHARS: Final[int] = 60
MIN_DIFF_CHARS: Final[int] = 200
MAX_DIFF_CHARS: Final[int] = 2000

#: 触及组合的连接符（**结构键**：case 指纹与维度表都用它）。
TOUCH_JOINER: Final[str] = "+"

#: diff 里「改了哪个文件」的那一行。**这是 state 格式的一部分，不是文案**，所以不进语言文件：
#: 渲染期的 `check_draft`、判别器与审计都按同一份正则读它。
HEADER_RE: Final[re.Pattern[str]] = re.compile(r"^diff --git a/(\S+) b/(\S+)$")

#: hunk 块头：`@@ -旧起点,旧行数 +新起点,新行数 @@`（行数可省）。同样是格式的一部分。
#: **光写 `@@` 也认**——模型很爱这么写（2026-10-01 小批量：`testimpact` 的 8 条拒稿全是它，
#: 而且光秃秃的 `@@` 一不被认，整块 `+` / `-` 行就跟着落进「块外」，一条样本会连报三条错）。
#: 行号是算术、题面又用不到，`render.normalize_hunks` 会把这种块头补成规范形态，所以这里放行：
#: 格式简写不该把整条样本判死。
HUNK_RE: Final[re.Pattern[str]] = re.compile(r"^@@(?: -\d+(?:,\d+)? \+\d+(?:,\d+)? @@.*)?$")

#: 光秃秃的 `@@`（没给行号）。`normalize_hunks` 单独认它，才好把行号补上。
BARE_HUNK_RE: Final[re.Pattern[str]] = re.compile(r"^@@$")

#: 完整的块头：起点行号 + 尾部上下文（git 会补一句 ` def f():` 这样的定位串）。
#: **行数那两段故意不捕获**——它们是纯算术，`render.normalize_hunks` 一律按块内实际行数重算，
#: 起点与尾部上下文才原样保留。
HUNK_SPAN_RE: Final[re.Pattern[str]] = re.compile(
    r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)$"
)

#: 文件路径 → 路径编号（解析 diff 用）。
FILE_TO_ID: Final[dict[str, str]] = {path: pid for pid, path in PATH_FILES.items()}


def touch_key(ids: Iterable[str]) -> str:
    """输入：路径编号集合。输出：规范顺序的连接串。

    预期行为：按 `PATH_IDS` 的顺序排，认不出的编号排在后面并保持原顺序；空集合给空串。
    同一个集合在任何调用点都得到同一串（哈希分桶与指纹都靠它）。
    """
    seen = set(ids)
    known = [pid for pid in PATH_IDS if pid in seen]
    extra = [pid for pid in ids if pid not in PATH_IDS]
    return TOUCH_JOINER.join([*known, *dict.fromkeys(extra)])


def parse_touch(value: str) -> tuple[str, ...]:
    """输入：触及组合串。输出：路径编号元组。

    预期行为：空串给空元组；认不出的编号原样留在结果里（审计要靠它报「改了覆盖表之外的路径」）。
    """
    if not value:
        return ()
    return tuple(part for part in value.split(TOUCH_JOINER) if part)


def need_suites(touched: Iterable[str]) -> tuple[str, ...]:
    """输入：触及的路径编号。输出：这次改动需要跑的套件（按 `SUITES` 顺序）。

    预期行为：**并集**——把触及路径的覆盖列并起来（多条路径挂同一套时只算一次）；触及盲区时
    直接给全部三套（影响面未知，只能全套跑）。认不出的编号不参与并集（它们不是覆盖表里的路径）。
    """
    ids = set(touched)
    if BLIND_PATH in ids:
        return SUITES
    covered = {suite for pid in ids for suite in COVERAGE.get(pid, ())}
    return tuple(suite for suite in SUITES if suite in covered)


def level_for(need: Sequence[str]) -> str:
    """输入：需要跑的套件。输出：跑法档位名。

    预期行为：0 套 → 只跑冒烟；1 套 → 跑一套；2 套 → 跑两套；3 套 → 跑全套。四档与
    `LEVELS` 的顺序一一对应，改档位表就改这里一处。
    """
    return LEVELS[min(len(need), len(LEVELS) - 1)]


def touched_in(diff: str) -> tuple[str, ...]:
    """输入：一段改动正文。输出：它触及的路径编号（按 `PATH_IDS` 顺序）。

    预期行为：只认 `diff --git a/<路径> b/<路径>` 那一行的**新路径**——这是「改到了哪个文件」
    唯一无歧义的字面依据。认不出的文件路径原样排在后面（审计要靠它报「改了覆盖表之外的
    路径」）；同一路径出现多次只算一次。
    """
    known: list[str] = []
    unknown: list[str] = []
    for raw in diff.splitlines():
        match = HEADER_RE.match(raw.strip())
        if match is None:
            continue
        target = match.group(2)
        pid = FILE_TO_ID.get(target)
        if pid is None:
            if target not in unknown:
                unknown.append(target)
        elif pid not in known:
            known.append(pid)
    ordered = tuple(pid for pid in PATH_IDS if pid in set(known))
    return (*ordered, *sorted(unknown))


@dataclass(frozen=True)
class ImpactCase(Fingerprinted):
    """一条样本的注入特征：这次改动触及了哪几条路径、以及它是什么性质的改动。

    字段：`touch` 触及组合（`P1+P3` 这种，空串 = 一条都没触及）；`flavor` 改动性质。
    """

    touch: str
    flavor: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"touch": self.touch, "flavor": self.flavor}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.touch or '无路径'}-{self.flavor}"

    def touched(self) -> tuple[str, ...]:
        """输入：无。输出：触及的路径编号。"""
        return parse_touch(self.touch)

    def need(self) -> tuple[str, ...]:
        """输入：无。输出：需要跑的套件。"""
        return need_suites(self.touched())

    def level(self) -> str:
        """输入：无。输出：跑法档位。"""
        return level_for(self.need())

    def blind(self) -> bool:
        """输入：无。输出：这次改动有没有触及盲区。"""
        return BLIND_PATH in self.touched()


@dataclass(frozen=True)
class ImpactPlan:
    """一条样本的改动计划：触及哪几条路径 + 组合题的三个干扰项。

    字段：`touched` 触及的路径编号；`distractors` 组合题的三个干扰套件集合（与真值不同、彼此
    也不同，且都取「与真值只差一两套」的邻近集合，题目才有区分度）。
    """

    touched: tuple[str, ...]
    distractors: tuple[tuple[str, ...], ...]

    @property
    def need(self) -> tuple[str, ...]:
        """输入：无。输出：需要跑的套件（并集，盲区给全套）。"""
        return need_suites(self.touched)

    @property
    def level(self) -> str:
        """输入：无。输出：跑法档位。"""
        return level_for(self.need)


def _distractors(truth: tuple[str, ...], rng: random.Random) -> tuple[tuple[str, ...], ...]:
    """输入：真值套件集合、随机源。输出：三个干扰集合。

    预期行为：只取**与真值相差不超过两套**的集合（差满三套 = 与真值完全不相交，等于把答案摆到
    另一个角落，反而好猜），按「差得越少越靠前」排序后抽三个；抽完按 (套数, 套件名) 排好，
    保证同一入参必得同一结果。

    为什么这个池子恒有 6 个：三套件一共 8 种集合，去掉真值自己剩 7 种，其中只有 1 种与真值差满
    三套（真值大小为 1 或 2 时是补集方向的极端项，为 0 时是全集，为 3 时是空集），所以池子
    恒为 6 个，抽三个不会撞车。
    """
    truth_set = set(truth)
    near = sorted(
        (
            combo
            for size in range(len(SUITES) + 1)
            for combo in combinations(SUITES, size)
            if set(combo) != truth_set and len(set(combo) ^ truth_set) <= 2
        ),
        key=lambda combo: (len(set(combo) ^ truth_set), combo),
    )
    picked = rng.sample(near, 3)
    return tuple(sorted(picked, key=lambda combo: (len(combo), combo)))


def plan_for(case: ImpactCase, phrasing: int) -> ImpactPlan:
    """输入：case、问法编号。输出：改动计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里都
    得到同一份计划；两种问法会抽到不同的干扰集合，不会渲染出结构相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    return ImpactPlan(
        touched=case.touched(),
        distractors=_distractors(need_suites(case.touched()), random.Random(seed)),
    )


@dataclass(frozen=True)
class ImpactLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`touched` 触及的路径编号（按 `PATH_IDS` 顺序）。
    """

    touched: tuple[str, ...]

    @property
    def need(self) -> tuple[str, ...]:
        """输入：无。输出：需要跑的套件（并集，盲区给全套）。"""
        return need_suites(self.touched)

    @property
    def level(self) -> str:
        """输入：无。输出：跑法档位。"""
        return level_for(self.need)

    @property
    def blind(self) -> bool:
        """输入：无。输出：这次改动有没有触及盲区。"""
        return BLIND_PATH in self.touched

    def needs(self, suite: str) -> bool:
        """输入：套件名。输出：这一条需不需要跑它。"""
        return suite in self.need


def label_for(plan: ImpactPlan) -> ImpactLabels:
    """输入：改动计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`render.check_draft`
    负责保证文本真的兑现这份计划（按字面读出来的落点与计划逐条相同）。
    """
    return ImpactLabels(touched=plan.touched)


def observed_labels(diff: str) -> ImpactLabels:
    """输入：改动正文。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认文本、只认 `diff --git` 那一行。两者对不上
    就说明渲染没兑现计划，审计当场就能看见。
    """
    return ImpactLabels(touched=touched_in(diff))


def _touch_sets() -> tuple[str, ...]:
    """输入：无。输出：允许的触及组合（16 组）。

    预期行为：`P1`~`P3`、`P5` 的**全部非空子集**（15 组），加上**单独触及盲区**（1 组）。

    为什么「盲区 + 别的路径」这一族不进枚举：它的结论恒为「跑全套」，16 组都放进来会让 `full`
    占掉 68%，`route` 族退化成单值主导（范式 §9.25：档位不能按构造占大头）；而「单独改了没有
    测试覆盖的文件」正是现实里最常见的形态，保留它。16 组也让每一族的取值都摊得开。
    """
    others = [pid for pid in PATH_IDS if pid != BLIND_PATH]
    keys = [
        touch_key(combo)
        for size in range(1, len(others) + 1)
        for combo in combinations(others, size)
    ]
    return (*keys, BLIND_PATH)


#: 触及维度：16 组允许的组合（见 `_touch_sets`）。
TOUCH_SETS: Final[tuple[str, ...]] = _touch_sets()

#: 两个注入维度。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {"touch": TOUCH_SETS, "flavor": ALL_FLAVORS}


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[ImpactCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`ImpactCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [
        ImpactCase(touch=str(row["touch"]), flavor=str(row["flavor"])) for row in rows
    ]
