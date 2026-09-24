"""判别器抽象（通用层）：判断「这条样本对不对」。

设计要点（对应 PLAN 6.3）：
- 判别器不是真理：输出必须带理由与原始结果，不能只给 pass/fail；
- 判别器由 **workflow 提供**（`Workflow.make_judges`），通用层只定义协议与聚合方式；
- 判别器本身可被评估：把人工抽检当金标准即可（`reviews.jsonl`）。

输入/输出：`Verdict` 是结论；`run_judges` 跑一组判别器并返回 (名字, 结论) 列表。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Verdict:
    """一条判别结论。

    字段：ok 是否通过；score 可选分数（0~1）；reason 人类可读理由；raw 原始结果（留档审计）。
    """

    ok: bool
    score: float | None = None
    reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Judge(Protocol):
    """判别器协议：给样本（与可选真值 spec）出一个结论。"""

    name: str

    def judge(self, sample: dict[str, Any], spec: Any = None) -> Verdict:
        """输入：样本、真值 spec。输出：Verdict。预期行为：只判定、不修改样本。"""


def run_judges(
    sample: dict[str, Any], spec: Any, judges: tuple[Judge, ...]
) -> list[tuple[str, Verdict]]:
    """依次跑所有判别器。

    输入：样本、真值 spec、判别器元组。
    输出：[(判别器名, Verdict)]，顺序与入参一致。
    预期行为：任何判别器抛异常都转成「不通过」的 Verdict（不让管线崩）。
    """
    out: list[tuple[str, Verdict]] = []
    for judge in judges:
        try:
            out.append((judge.name, judge.judge(sample, spec)))
        except Exception as exc:  # noqa: BLE001 —— 判别器崩了要记下来，不能影响整批
            out.append((judge.name, Verdict(False, reason=f"判别器异常：{exc}")))
    return out
