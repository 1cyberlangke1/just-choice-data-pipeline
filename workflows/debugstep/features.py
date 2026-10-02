"""调试器下一步动作（debugstep）的特征层：停下原因的判据 + 运行时 + 命中标签（纯函数，零依赖）。

对应 PLAN 3.2「特征注入式」：一条「这次调试暂停该走哪一步」的样本由两个**注入特征**描述
——`hits` 该命中哪几条判据、`runtime` 这个会话来自哪种运行时（Python / Node / Go）。**标签先行**：
先定这两个取值，再由本模块的纯函数算出结论。

**为什么判据写成机械短语表**：调试器暂停时「这一屏值不值得细看」在真实会话里是有固定形状的
——停在异常上、当前帧里有空值、调用栈不止一帧（说明错可能在外层）、断点重复落在同一处。
把这些形状写成正则之后，「命中哪几条」就是**算**出来的，模型碰不到——它只负责写一屏读得通的
会话记录。判据表由代码写、内容由模型写（范式 §9.22 / §9.23 同一条路子）。

**为什么四条判据各占一整行**（范式 §9.46 的教训）：一屏会话是**分行**写的，四条判据各自钉在
自己的那一行上——`D1` 只看 `Stopped:` 行、`D2` 只看 `Locals:` 行、`D3` 只看 `Stack:` 行、
`D4` 只看 `Breakpoints:` 行。`adblock` 上量过一次「两条判据落在同一段文字上」的代价（写手与
读者一起丢内层那条），这里从设计上避开：**判据之间连一行都不共用**。

**为什么要这一族**：社区里 `llm-debugger-vscode-extension`（★359）把它明写成「在每次调试器
暂停时，Jev 从堆栈、局部变量与断点里判断下一步该做什么」；`jev-superpowers`（★4）把 Jev 决策
接进「规划 / 执行 / 调试 / 完成」四个门控。共同点是**每一次暂停都要一次立刻能落地的动作决策**
（继续跑 / 看一眼变量 / 单步 / 停下叫人），而不是让人把整屏材料通读一遍。

**与 `logtriage` / `shell` / `turntake` 的区别**：`logtriage` 判的是一条机器日志行该不该打扰人、
`shell` 判的是一条命令行能不能放行、`turntake` 判的是一段人话有没有说完——三者判的都是**单向
流入的材料**。本家判的是**一个正在被观察的程序状态**（谁调用了谁、这一帧里有什么、断点打在
哪里），处置语义是「下一步动作」，判据空间与素材形态都不一样。

**`runtime` 是这一族的留出维**：它**只影响渲染、不影响标签**——同一份命中组合写在 Python、
Node 还是 Go 的会话里，判据算出来的答案一个字都不差（四条判据认的全是键名与符号，不是语言）。
训练侧只见过 `python` / `node`，留出侧是 `go`，用来验「读法不靠运行时先验」。

本模块有**两条独立路径**：`label_for` 从计划推标签（构造侧），`observed_labels` 从文本重推标签
（审计侧，走 `hit_set` 的正则）。两者对不上就说明渲染没兑现计划。

输入/输出：只有常量与纯函数；不碰网络、不碰文件。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from itertools import combinations
from typing import Final

from jc.contract import parse_hits
from jc.cover import Fingerprinted, all_hit_sets, cover_dims, near_miss_combos
from jc.jsontypes import JsonValue

#: 四条「这一屏值不值得细看」判据的编号（**结构键**，不是文案：判据名与判据原文全在 `locales/`）。
TAG_IDS: Final[tuple[str, ...]] = ("D1", "D2", "D3", "D4")

#: 一屏会话的**结构键**：这五个英文键名与它们的冒号是 state 格式的一部分，**不进语言文件**——
#: 渲染、校验、解析、判别器共用这一份清单，四边不可能漂移。键名用英文是因为它们本来就是标识符
#: （文件、函数、变量也都是英文），只有 `note` 那一段的值是目标语言的说明文字。
KEY_LINES: Final[tuple[str, ...]] = ("stopped", "stack", "locals", "breakpoints", "note")

#: 判据编号 → 判据。**这是全 workflow 唯一的判据实现**：构造侧、渲染期、判别器、审计共用。
#: 每条都只认字面，而且**各占一整行**（见模块文档）：
#: - `D1` `Stopped:` 行里带着异常 / 崩溃的痕迹（`error` / `exception` / `traceback` / `panic` /
#:   `fatal` / `uncaught` 这类异常类型名——它们在哪种运行时里都是英文）；
#: - `D2` `Locals:` 行里有取值为空的变量（`= None` / `= null` / `= nil` / `= undefined`）；
#: - `D3` `Stack:` 行里不止一帧（出现帧分隔符 `<-`，说明还有更外层没看）；
#: - `D4` `Breakpoints:` 行里同一个位置出现了两次（同一个 `文件:行` 写了两次）。
PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "D1": re.compile(
        r"^[^\n]*\bstopped\s*:[^\n]*?\b\w*(?:error|exception|traceback|panic|fatal|uncaught)\w*\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    "D2": re.compile(
        r"^[^\n]*\blocals\s*:[^\n]*?\b[\w.]+\s*=\s*(?:None|null|nil|undefined)\b",
        re.IGNORECASE | re.MULTILINE,
    ),
    "D3": re.compile(r"^[^\n]*\bstack\s*:[^\n]*<-", re.IGNORECASE | re.MULTILINE),
    "D4": re.compile(
        r"^[^\n]*\bbreakpoints\s*:[^\n]*?(?<![\w./\\-])([\w./\\-]+:\d+)(?![\w])[^\n]*?\1(?![\w])",
        re.IGNORECASE | re.MULTILINE,
    ),
}

#: 运行时（**结构键**，文案在 `locales/` 的 `[runtime.*]`）。
#:
#: 这一维**只影响渲染、不影响标签**：同一份命中组合写在 Python、Node 还是 Go 的会话里，判据算出来
#: 的答案一个字都不差——四条判据认的是键名、赋值号与帧分隔符，与运行时的语法无关。它正是
#: test_ood 的留出维：训练侧只见过 `python` / `node`，留出侧是 `go`（范式 §9.25 那条「只影响
#: 渲染的那一维」）。
RUNTIME_PYTHON: Final[str] = "python"
RUNTIME_NODE: Final[str] = "node"
RUNTIME_GO: Final[str] = "go"
RUNTIMES: Final[tuple[str, ...]] = (RUNTIME_PYTHON, RUNTIME_NODE, RUNTIME_GO)

#: 处置档位（**结构键**，文案全在 `locales/`）。阈值是 0 条 / 1 条 / 2 条 / 3 条及以上——语义是
#: 「可疑的地方越多，下一步要花的力气越大」：0 条继续跑、1 条先看一眼变量、2 条单步走一行、
#: 3 条及以上停下并叫人。四档而不是三档的理由与 `logtriage` 相同：三档会让「停下」一档按构造占
#: 11/16（69%），均衡闸会把整批标成失衡。四档把占比摊成 1/4/6/5。
ACTS: Final[tuple[str, ...]] = ("continue", "inspect", "step_over", "stop")

#: 一屏会话的字符数上下限。太短装不下五行（四个键 + 说明），太长就成了一整份调试日志。
MIN_CONTENT_CHARS: Final[int] = 80

#: 上限。五行会话加各自的取值，再长就不是「一屏」了。
MAX_CONTENT_CHARS: Final[int] = 500


#: 全部命中组合（`cover_dims` 的取值表）。
HIT_SETS: Final[tuple[str, ...]] = all_hit_sets(TAG_IDS)

#: 维度表。全空间 16 × 3 = 48 组。
DIMS: Final[dict[str, tuple[JsonValue, ...]]] = {"hits": HIT_SETS, "runtime": RUNTIMES}


def hit_set(text: str) -> tuple[str, ...]:
    """输入：一屏调试会话。输出：命中的判据编号（按 `TAG_IDS` 顺序）。"""
    return tuple(tag for tag in TAG_IDS if PATTERNS[tag].search(text))


def hits_of(text: str) -> str:
    """输入：一屏调试会话。输出：命中组合的规范串（`"D1+D4"` / `""`）。

    预期行为：与 `hit_set` 是同一件事的两种表示；`hits_of` 给比较用（`render.check_draft` 拿它
    与计划逐字比），`hit_set` 给取元素用。
    """
    return "+".join(hit_set(text))


def missing_keys(text: str) -> tuple[str, ...]:
    """输入：一屏调试会话。输出：缺掉的**结构键**（五个键都在就给空元组）。

    预期行为：键必须**单独占一行**、后面跟冒号（`^\\s*键\\s*:`）。为什么要单占一行：判据是钉在
    行上的（见 `PATTERNS`），键不单独成行的话那几条正则连定位都定不到。渲染期用它拦「少写一行」，
    审计期 `FrameJudge` 再用同一份清单复核一遍。
    """
    lowered = text.lower()
    return tuple(
        key for key in KEY_LINES if not re.search(rf"(?m)^\s*{key}\s*:", lowered)
    )


def line_value(text: str, key: str) -> str:
    """输入：一屏调试会话、结构键。输出：那一行冒号后面的内容（没有那一行就给空串）。

    预期行为：只取**第一个**匹配到的键行；行内的空白原样保留（取值里可能有空格）。
    """
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*:(.*)$", text)
    return match.group(1).strip() if match else ""


def act_level(count: int) -> str:
    """输入：命中条数。输出：处置档位名。

    预期行为：0→continue、1→inspect、2→step_over、3 及以上→stop（最后一档吃掉所有更大的数）。
    """
    return ACTS[min(count, len(ACTS) - 1)]


@dataclass(frozen=True)
class DebugStepCase(Fingerprinted):
    """一个注入特征组合：该命中哪几条判据 + 这个会话来自哪种运行时。

    字段：`hits` 命中组合的规范串；`runtime` 运行时（`RUNTIMES` 里的一个）。
    """

    hits: str
    runtime: str

    def as_dict(self) -> dict[str, JsonValue]:
        """输入：无。输出：可 JSON 序列化的字典。预期行为：键序固定（哈希分桶要用）。"""
        return {"hits": self.hits, "runtime": self.runtime}

    def describe(self) -> str:
        """输入：无。输出：一行可读描述（报告与文件名用）。"""
        return f"{self.hits or '无命中'}-{self.runtime}"

    def tags(self) -> tuple[str, ...]:
        """输入：无。输出：该命中的判据编号元组。"""
        return parse_hits(self.hits)


@dataclass(frozen=True)
class DebugStepPlan:
    """一条样本的判定计划：命中哪几条 + 组合题的三个干扰项。

    字段：`hits` 该命中的判据；`distractors` 组合题的三个干扰组合（与 `hits` 不同、彼此也不同，
    且都取「与真值只差一两处」的邻近组合，题目才有区分度）。
    """

    hits: tuple[str, ...]
    distractors: tuple[tuple[str, ...], ...]

    @property
    def count(self) -> int:
        """输入：无。输出：命中条数。"""
        return len(self.hits)


def plan_for(case: DebugStepCase, phrasing: int) -> DebugStepPlan:
    """输入：case、问法编号。输出：判定计划（与语言、后端无关的随机）。

    预期行为：种子由 **case 指纹 + 问法编号** 派生，所以同一个 case 在任何切分、任何进程里都
    得到同一份计划；两种问法会抽到不同的干扰组合，不会渲染出结构相同的样本。
    """
    seed = int(
        hashlib.sha256(f"{case.fingerprint()}|{phrasing}".encode("utf-8")).hexdigest()[:16], 16
    )
    return DebugStepPlan(
        hits=case.tags(), distractors=near_miss_combos(TAG_IDS, case.tags(), random.Random(seed))
    )


@dataclass(frozen=True)
class DebugStepLabels:
    """一条样本的全部金标签（纯代码推导，模型碰不到）。

    字段：`hits` 命中的判据编号（按 `TAG_IDS` 顺序）。
    """

    hits: tuple[str, ...]

    @property
    def count(self) -> int:
        """输入：无。输出：命中条数。"""
        return len(self.hits)

    def contains(self, tag: str) -> bool:
        """输入：判据编号。输出：这一条是否命中。"""
        return tag in self.hits

    @property
    def act(self) -> str:
        """输入：无。输出：按阈值算出来的档位名（`features.act_level`）。"""
        return act_level(self.count)


def label_for(plan: DebugStepPlan) -> DebugStepLabels:
    """输入：判定计划。输出：全部金标签（**构造侧**路径）。

    预期行为：纯函数、只看计划、不看文本——文本是后写的，标签先定。`render.check_draft` 负责保证
    文本真的兑现这份计划（按字面算出来的命中组合与计划逐条相同）。
    """
    return DebugStepLabels(hits=plan.hits)


def observed_labels(text: str) -> DebugStepLabels:
    """输入：一屏调试会话。输出：全部金标签（**审计侧**路径）。

    预期行为：与 `label_for` 各走一条路——这条路只认文本、只认字面。两者对不上就说明渲染没兑现
    计划，审计当场就能看见。
    """
    return DebugStepLabels(hits=hit_set(text))


def build_cases(
    max_full: int, dims: dict[str, tuple[JsonValue, ...]] | None = None
) -> list[DebugStepCase]:
    """枚举特征组合。

    输入：全枚举的规模上限（超过就换成对覆盖）、可选的维度表覆盖（留出某个取值时用）。
    输出：`DebugStepCase` 列表，顺序稳定。预期行为：同一入参必得同一输出。
    """
    rows = cover_dims(DIMS if dims is None else dims, max_full=max_full)
    return [DebugStepCase(hits=str(row["hits"]), runtime=str(row["runtime"])) for row in rows]
