"""依赖与许可证风险门控（depcheck）的特征层：四个风险标记 + 定级规则 + 处置档位（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条「这个依赖能不能进我们的依赖树」的样本由两个**注入特征**描述
——`flags` 哪几个风险标记命中了、`ecosystem` 这个包来自哪个生态（npm / PyPI / crates.io）。
**标签先行**：先定这两个取值，再由本模块的纯函数算出结论。

**为什么风险标记写成机械短语表**：供应链准入在真实流水线里也是固定形状——许可证是不是传染性的、
仓库是不是已经归档、有没有已知漏洞、版本到没到 1.0。把四条形状写成正则之后，「命中了哪几个标记」
就是**算**出来的，模型碰不到；它只负责写一屏读得通的依赖记录。

**为什么等级是加权和而不是数条数**：四个标记的后果本来就不等重——传染性许可证（`D1`）与已知漏洞
（`D3`）各值 2 分，仓库归档（`D2`）与没到 1.0（`D4`）各值 1 分。等级 = `1 + 权重和`、封顶 5 级，
所以「两个轻标记」与「一个重标记」同档（都是 3 级）。这条规则**印在 state 里**（见 `render`），
考生只读 state 就能自己算出来（范式 §9.8）。

**为什么要这一族**：社区里 `jev-scout` 明写「Jev 判断许可证与依赖风险」，`dependabot` 一族
（GitHub 的依赖升级门控）、`pip-audit` / `npm audit` 的告警分诊是同一形状的另外几个落地：机器
把清单摆出来，**要不要放进依赖树**这个判断交给类型化决策。与 `sqlreview` / `alertrisk` 不重复：
那两家判的是一屏变更记录与一条安全警报，本家判的是**一个依赖的准入**，判据是许可证类型、维护
状态、漏洞计数与版本成熟度。

**`ecosystem` 是这一族的留出维**：它**只影响渲染、不影响标签**——同一个包写在 npm、PyPI 还是
crates.io 的清单里，四个标记算出来的等级一个字都不差。训练侧只见过 `npm` / `pypi`，留出侧是
`cargo`。

本模块有**两条独立路径**：`label_for` 从计划推标签（构造侧），`observed_labels` 从文本重推标签
（审计侧，走 `hit_set` 的正则）。两者对不上就说明渲染没兑现计划。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Final

from jc.contract import parse_hits
from jc.cover import Fingerprinted, all_hit_sets, cover_dims, near_miss_combos
from jc.jsontypes import JsonValue

#: 四个风险标记的编号（**结构键**，不是文案：标记名与判据原文全在 `locales/`）。
TAG_IDS: Final[tuple[str, ...]] = ("D1", "D2", "D3", "D4")

#: 依赖记录的**结构键**：这六个英文键名与它们的冒号是 state 格式的一部分，**不进语言文件**——
#: 渲染、校验、解析、判别器共用这一份清单，四边不可能漂移。键名用英文是因为它们本来就是格式的
#: 一部分（包名、版本号、SPDX 标识符、状态、计数都是机器写的），只有 `note` 那一行是自然语言。
KEY_LINES: Final[tuple[str, ...]] = (
    "package",
    "version",
    "license",
    "status",
    "advisories",
    "note",
)

#: 标记编号 → 它钉在哪一个结构键那一行上。四条判据**各占一整行**（范式 §9.51）：同一段文字上
#: 挂两条判据时，写手与读者会一起丢掉内层那一条，这里从设计上避开。
LINE_KEYS: Final[dict[str, str]] = {
    "D1": "license",
    "D2": "status",
    "D3": "advisories",
    "D4": "version",
}

#: 标记编号 → 判据。**这是全 workflow 唯一的判据实现**：构造侧、渲染期、判别器、审计共用。
#: 每条都只认字面、只认自己那一行：
#: - `D1` `license:` 行里出现**传染性许可证**的 SPDX 标识（`AGPL` / `GPL` / `SSPL` 整词）；
#: - `D2` `status:` 行里写着 `archived`（仓库已归档，不再维护）；
#: - `D3` `advisories:` 行里 `count=` 后面是**非零**的十进制整数（至少一条已知漏洞）；
#: - `D4` `version:` 行的版本号**以 `0.` 开头**（还没到 1.0）。
#:
#: 三处细节都不是拍出来的：
#: - `\b` 让 `LGPL` 不误判成 `GPL`（`GPL` 前面是词字符 `L`，词边界不成立），而 `AGPL-3.0-only`
#:   照样命中（`AGPL` 整词在前）；
#: - `count=[1-9]\d*` 而不是 `count=\d+`：`count=0` 是「没有已知漏洞」，不该命中；
#: - `D4` 用 `(?=\s*0\.\d)`（冒号后紧跟着版本号）而不是「行里某处有 `0.`」：`1.0.3` 里也有
#:   `0.`，按后者会把它误判成「没到 1.0」。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "D1": re.compile(
        r"^[^\n]*\blicense\s*:(?=[^\n]*\b(?:AGPL|GPL|SSPL)\b)",
        re.IGNORECASE | re.MULTILINE,
    ),
    "D2": re.compile(
        r"^[^\n]*\bstatus\s*:(?=[^\n]*\barchived\b)",
        re.IGNORECASE | re.MULTILINE,
    ),
    "D3": re.compile(
        r"^[^\n]*\badvisories\s*:(?=[^\n]*\bcount\s*=\s*[1-9]\d*)",
        re.IGNORECASE | re.MULTILINE,
    ),
    "D4": re.compile(
        r"^[^\n]*\bversion\s*:(?=\s*0\.\d)",
        re.IGNORECASE | re.MULTILINE,
    ),
}

#: 四个标记的权重：等级 = `1 + 命中标记的权重和`，封顶 `MAX_LEVEL` 级。
#: 权重不等重是有意的（许可证与漏洞各 2 分、归档与未到 1.0 各 1 分），所以「一个重标记」与
#: 「两个轻标记」同档；**每一维单调**（多命中一个标记，等级不会降）由这个加法形式保证。
LEVEL_WEIGHTS: Final[dict[str, int]] = {"D1": 2, "D2": 1, "D3": 2, "D4": 1}

#: 等级上限（封顶）。四个标记全命中时权重和是 6，按公式算出来是 7 级——封顶到 5。
MAX_LEVEL: Final[int] = 5

#: 生态（**结构键**：包名与版本号的写法在 `locales/` 的描述里，取值本身是格式）。
ECO_NPM: Final[str] = "npm"
ECO_PYPI: Final[str] = "pypi"
ECO_CARGO: Final[str] = "cargo"
ECOSYSTEMS: Final[tuple[str, ...]] = (ECO_NPM, ECO_PYPI, ECO_CARGO)

#: 全部风险等级（1~5，**结构键**：等级名与处置动作的文案在 `locales/`）。
LEVELS: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)

#: 处置档位（**结构键**，文案全在 `locales/`）。四档由等级经 `act_for` 映射：1 级直接放行、
#: 2 级登记观察、3 级引入前复核、4 级及以上换成别的依赖。四档而不是五档的理由与 `subqa` 相同：
#: 档位太多会让小批量的答案分布被噪声吞掉（`pipeline.balance_report` 的噪声地板是 3 倍标准误）。
ACTS: Final[tuple[str, ...]] = ("approve", "watch", "review", "replace")

#: 依赖记录的字符数上下限。太短装不下六行，太长就不是「一屏清单里的那一条」了。
MIN_CONTENT_CHARS: Final[int] = 100

#: 上限。六行材料加各自的取值，再长就不是一条依赖记录了。
MAX_CONTENT_CHARS: Final[int] = 700

#: 写文本时的「依赖类别」池：spec 只存**下标**（与问法编号同一套约定），每种语言各有一份等长的
#: 词表（`test_locales.py` 锁住）。
#:
#: **来由**（范式 §9.57 / §9.58 的同一类坑）：一屏记录里唯一的自由文字是 `note:` 那一行，而提示词
#: 里只有「哪个生态」一个变量——不加样本级变量的话，同一生态的每条记录都会塌成同一句（`alertrisk`
#: 与 `citation` 都真机踩过）。给每条样本抽一个依赖类别，提示词就不再逐字相同了。
#:
#: 类别是**跨生态通用**的（日志库 / 序列化 / 数据库驱动…哪个生态都有），所以不会与 `ecosystem` 打架；
#: 它也只影响 `note:` 怎么写与包名怎么起，一个字符都不参与判据。
CATEGORY_POOL: Final[int] = 10
CATEGORY_IDS: Final[tuple[int, ...]] = tuple(range(CATEGORY_POOL))

#: 写文本时的「版本号尾段」池：spec 里存的是**取值本身**（数字，与语言无关，所以不像类别词表那样
#: 需要各语言等长）。
#:
#: **来由**（范式 §9.60）：`version:` 是这一族**最后一个**没有样本级变量的自由字段，真机上它塌得
#: 最狠（28 条产物里 `1.4.2` 出现 7 次、留出批 zh 四条全是 `1.4.2`）。给每条样本钉一个尾段数字，
#: 提示词就不再逐字相同了。
#:
#: 为什么钉**最后一个**数字段：`D4` 的判据只看第一段（是不是 `0.` 开头），所以这个旋钮与任何标签
#: 都不打架；`x.y.47` 这种形状在三个生态里都成立（点分数字）。
VERSION_HINTS: Final[tuple[int, ...]] = (11, 17, 23, 29, 34, 41, 47, 53, 61, 73, 82, 97)


#: 写文本时的「许可证标识符」池：spec 里存的是**取值本身**（SPDX 标识符，与语言无关，所以不像
#: 类别词表那样需要各语言等长）。
#:
#: **来由**（范式 §9.61）：`license:` 是这一族**最后一个**没有样本级变量的自由字段——`dep-10b`
#: 的 40 条产物里 `MIT` 占了七成，模型自己的先验（「许可证就写 MIT」）把整批压平。
#:
#: 池子**按 `D1` 分两半**：`D1` 的判据是「`license:` 那一行里有 `AGPL` / `GPL` / `SSPL` 整词」，
#: 所以命中侧只能从那三个词族的标识符里取、不命中侧一个都不能带（见 `license_pool`）。两侧都
#: **不含** `LGPL`：`\b` 让 `LGPL-3.0` 不算命中（`GPL` 前面是词字符 `L`），把它放进任何一侧都
#: 只会让读者以为判据认它。两半各由一条测试对着 `PATTERNS["D1"]` 钉住，改池子时不会悄悄跑偏。
LICENSE_COPYLEFT: Final[tuple[str, ...]] = (
    "AGPL-3.0-only",
    "AGPL-3.0-or-later",
    "GPL-2.0-only",
    "GPL-3.0-only",
    "GPL-3.0-or-later",
    "SSPL-1.0",
)

LICENSE_PERMISSIVE: Final[tuple[str, ...]] = (
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "MPL-2.0",
    "Zlib",
    "Unlicense",
    "0BSD",
    "PostgreSQL",
)


def license_pool(tags: tuple[str, ...]) -> tuple[str, ...]:
    """输入：命中组合。输出：这条样本可用的许可证池。

    预期行为：`D1` 在命中组合里就给传染性那一半，否则给宽松那一半——两半都**不与判据打架**，
    所以旋钮抽到的值一定兑现得了计划（兑现不了的样本会被 `render.check_draft` 当场打回重渲染）。
    """
    return LICENSE_COPYLEFT if "D1" in tags else LICENSE_PERMISSIVE


def pick_license(rng: random.Random, pool: tuple[str, ...], tags: tuple[str, ...]) -> str:
    """输入：随机源、配置的许可证池、命中组合。输出：这一条要写的 SPDX 标识符。

    预期行为：只从池子里**与判据相容的那一半**取（`D1` 命中取传染性、否则取宽松）；相容的那一半
    空着就报人话——配置层已经拦过一道（`DepCheckConfig.__post_init__`），这里是采样期的兜底，
    免得调用方自己塞一个只含单侧的池子时静默产出一批标签对不上的样本。
    """
    half = license_pool(tags)
    allowed = tuple(item for item in pool if item in half)
    if not allowed:
        side = "命中" if "D1" in tags else "不命中"
        raise ValueError(
            f"许可证池里没有与这条 case 相容的标识符（D1 {side}，要从 {half} 里取）：{pool!r}"
        )
    return rng.choice(allowed)


#: 全部命中组合（`cover_dims` 的取值表）。
HIT_SETS: Final[tuple[str, ...]] = all_hit_sets(TAG_IDS)

#: 维度表。全空间 16 × 3 = 48 组。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {"flags": HIT_SETS, "ecosystem": ECOSYSTEMS}


def hit_set(text: str) -> tuple[str, ...]:
    """输入：一屏依赖记录。输出：命中的标记编号（按 `TAG_IDS` 顺序）。"""
    return tuple(tag for tag in TAG_IDS if PATTERNS[tag].search(text))


def hits_of(text: str) -> str:
    """输入：一屏依赖记录。输出：命中组合的规范串（`D1+D3` / 空串）。

    预期行为：与 `hit_set` 是同一件事的两种表示；`hits_of` 给比较用（`render.check_draft` 拿它
    与计划逐字比），`hit_set` 给取元素用。
    """
    return "+".join(hit_set(text))


def missing_keys(text: str) -> tuple[str, ...]:
    """输入：一屏依赖记录。输出：缺掉的**结构键**（六个键都在就给空元组）。

    预期行为：键必须**单独占一行**、后面跟冒号。为什么要单占一行：判据是钉在行上的
    （见 `PATTERNS`），键不单独成行的话那四条正则连定位都定不到。渲染期用它拦「少写一行」，
    审计期 `FrameJudge` 再用同一份清单复核一遍。
    """
    lowered = text.lower()
    return tuple(key for key in KEY_LINES if not re.search(rf"(?m)^\s*{key}\s*:", lowered))


def line_value(text: str, key: str) -> str:
    """输入：一屏依赖记录、结构键。输出：那一行冒号后面的内容（没有那一行就给空串）。

    预期行为：只取**第一个**匹配到的键行；行内的空白原样保留（取值里可能有空格）。
    """
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*:(.*)$", text)
    return match.group(1).strip() if match else ""


def level_of(hits: tuple[str, ...]) -> int:
    """输入：命中组合。输出：风险等级（1~5）。

    预期行为：`1 + 命中标记的权重和`，封顶 `MAX_LEVEL`。加法形式让**单调性**是构造出来的（每个
    权重都是正数），`test_features` 里那条单调性测试就是钉这件事。
    """
    total = 1 + sum(LEVEL_WEIGHTS[tag] for tag in TAG_IDS if tag in hits)
    return min(MAX_LEVEL, total)


def act_for(level: int) -> str:
    """输入：风险等级。输出：处置档位名。

    预期行为：1→approve、2→watch、3→review、4 及以上→replace（最后一档吃掉所有更高的级）。
    """
    return ACTS[min(level - 1, len(ACTS) - 1)]


def combo_text_key(hits: tuple[str, ...]) -> str:
    """输入：命中组合。输出：定级表里那一行的标记取值键（如 `D1=hit D2=miss D3=miss D4=miss`）。

    预期行为：**这是 state 格式的一部分**（定级表里示例行的行首），不进语言文件；四个标记按
    `TAG_IDS` 顺序排，命中与否用 ASCII 的 hit / miss 写——它们是格式，不是文案。
    """
    marks = ("hit" if tag in hits else "miss" for tag in TAG_IDS)
    return " ".join(f"{tag}={mark}" for tag, mark in zip(TAG_IDS, marks, strict=True))


@dataclass(frozen=True)
class DepCheckCase(Fingerprinted):
    """一个注入特征组合：哪几个标记命中 + 这个包来自哪个生态。

    字段：`flags` 命中组合的规范串；`ecosystem` 生态（`ECOSYSTEMS` 里的一个）。
    """

    flags: str
    ecosystem: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"flags": self.flags, "ecosystem": self.ecosystem}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.flags or '无风险标记'}-{self.ecosystem}"

    def tags(self) -> tuple[str, ...]:
        """输入：无。输出：命中的标记编号元组。"""
        return parse_hits(self.flags)


@dataclass(frozen=True)
class DepCheckPlan:
    """一条样本的判定计划：命中哪几个标记 + 组合题的三个干扰项。

    字段：`hits` 命中组合；`distractors` 组合题的三个干扰组合（与 `hits` 不同、彼此也不同，
    且都取「与真值只差一两处」的邻近组合，题目才有区分度）。
    """

    hits: tuple[str, ...]
    distractors: tuple[tuple[str, ...], ...]

    @property
    def level(self) -> int:
        """输入：无。输出：按定级规则算出来的风险等级。"""
        return level_of(self.hits)


def plan_for(case: DepCheckCase, phrasing: int) -> DepCheckPlan:
    """输入：case、问法编号。输出：判定计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里都
    得到同一份计划；两种问法会抽到不同的干扰组合，不会渲染出结构相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    return DepCheckPlan(
        hits=case.tags(), distractors=near_miss_combos(TAG_IDS, case.tags(), random.Random(seed))
    )


@dataclass(frozen=True)
class DepCheckLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`hits` 命中的标记编号（按 `TAG_IDS` 顺序）。
    """

    hits: tuple[str, ...]

    @property
    def level(self) -> int:
        """输入：无。输出：风险等级（按定级规则算）。"""
        return level_of(self.hits)

    @property
    def act(self) -> str:
        """输入：无。输出：按等级算出来的处置档位名。"""
        return act_for(self.level)

    def contains(self, tag: str) -> bool:
        """输入：标记编号。输出：这一条是不是命中了它。"""
        return tag in self.hits


def label_for(plan: DepCheckPlan) -> DepCheckLabels:
    """输入：判定计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`render.check_draft` 负责保证
    文本真的兑现这份计划（按字面算出来的命中组合与计划逐条相同）。
    """
    return DepCheckLabels(hits=plan.hits)


def observed_labels(text: str) -> DepCheckLabels:
    """输入：一屏依赖记录。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认文本、只认字面。两者对不上就说明渲染没兑现
    计划，审计当场就能看见。
    """
    return DepCheckLabels(hits=hit_set(text))


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[DepCheckCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`DepCheckCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [DepCheckCase(flags=str(row["flags"]), ecosystem=str(row["ecosystem"])) for row in rows]
