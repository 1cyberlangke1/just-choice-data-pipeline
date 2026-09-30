"""判别器抽象（通用层）：判断「这条样本对不对」。

设计要点（对应 PLAN 6.3）：
- 判别器不是真理：输出必须带理由与原始结果，不能只给 pass/fail；
- 判别器由 **workflow 提供**（`Workflow.make_judges`），通用层只定义协议与聚合方式；
- 判别器本身可被评估：把人工抽检当金标准即可（`reviews.jsonl`）。

输入/输出：`Verdict` 是结论；`run_judges` 跑一组判别器并返回 (名字, 结论) 列表。
"""

from __future__ import annotations
from jc.jsontypes import JsonValue

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Final, Protocol, runtime_checkable

#: LLM 判别器的名字前缀（`llm:solvable`）。通用层负责装配它们（要后端），
#: 各 workflow 只管在自己的判别器名单里写名字，不必重复接线。
LLM_PREFIX: Final[str] = "llm:"


def squash(text: str) -> str:
    """输入：一段文本。输出：空白折叠成单个空格之后的文本。

    预期行为：只为比对用，不改样本本身——判别器拿它把「换行 / 多空格」的差异抹平，
    免得同一句话因为排版不同被判成两回事。
    """
    return " ".join(text.split())


def leftover_in(state: str, lines: Iterable[str], tags: Iterable[str]) -> str:
    """输入：state 文本、残留行清单、残留标记清单。输出：命中的残留（没有就空串）。

    预期行为：先按**整行**比对（去掉首尾空白后全等，所以围栏单独成行才算残留），再按**子串**
    比对标签；两份清单的顺序就是优先级，先命中的先返回。清单由各 workflow 给（判什么算残留是
    领域知识），比对本身是通用的。
    """
    for line in state.splitlines():
        if line.strip() in lines:
            return line.strip()
    for tag in tags:
        if tag in state:
            return tag
    return ""


def partition_names(names: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """输入：判别器名单。输出：(规则判别器名, LLM 视角名)。

    预期行为：只按前缀切分，顺序与调用方声明的一致，不改写也不去重；
    `llm:` 后面留空（或写了不存在的视角）由装配方报错，不在这里静默丢掉。
    """
    rules = tuple(name for name in names if not name.startswith(LLM_PREFIX))
    llm = tuple(name[len(LLM_PREFIX) :] for name in names if name.startswith(LLM_PREFIX))
    return rules, llm


@dataclass(frozen=True)
class Verdict:
    """一条判别结论。

    字段：ok 是否通过；score 可选分数（0~1）；reason 人类可读理由；raw 原始结果（留档审计）；
    skipped 表示**这条根本没判成**（例如端点故障、视角不适用）。

    `skipped=True` 时 `ok` 不代表数据好坏，消费方必须先看 `skipped`：把「没判成」当成
    「不通过」会把端点抖动记成数据质量问题（实测踩过），当成「通过」又不该悄悄发生——
    所以两者都不是，单独记一笔账。
    """

    ok: bool
    score: float | None = None
    reason: str = ""
    raw: dict[str, JsonValue] = field(default_factory=dict)
    skipped: bool = False


@runtime_checkable
class Judge(Protocol):
    """判别器协议：给样本（与可选真值 spec）出一个结论。"""

    name: str

    def judge(self, sample: dict[str, JsonValue], spec: object = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：只判定、不修改样本。"""


def run_judges(
    sample: dict[str, JsonValue], spec: object, judges: tuple[Judge, ...]
) -> list[tuple[str, Verdict]]:
    """依次跑所有判别器。

    输入：样本、真值 spec、判别器元组。
    输出：[(判别器名, Verdict)]，顺序与入参一致。
    预期行为：任何判别器抛异常都转成「不通过」的 Verdict（不让管线崩），
    **除了标了 `fatal_for_run` 的**——那是「整批该停下」（全部 provider 都挂了），
    不是「这条样本不对」，吞掉它会产出一整个「全被拒」的假数据集。
    """
    out: list[tuple[str, Verdict]] = []
    for judge in judges:
        try:
            out.append((judge.name, judge.judge(sample, spec)))
        except Exception as exc:  # noqa: BLE001 —— 判别器崩了要记下来，不能影响整批
            if getattr(exc, "fatal_for_run", False):
                raise
            out.append((judge.name, Verdict(False, reason=f"判别器异常：{exc}")))
    return out
