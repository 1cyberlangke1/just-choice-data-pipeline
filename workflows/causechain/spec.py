"""根因定位的真值规格：注入特征 + 现场计划 + 切分信息。

输入/输出：只有不可变数据类与纯函数；不含领域词表（规则文案全在 `locales/`）。
"""

from __future__ import annotations

from dataclasses import dataclass

from jc.jsontypes import JsonValue

from .features import CauseCase, CauseLabels, CausePlan, label_for


@dataclass(frozen=True)
class CauseSpec:
    """一条样本的真值规格。

    字段：`case` 注入特征；`phrasing` 问法编号；`split` 切分名；`seq` 样本序号；`plan` 现场计划。
    """

    case: CauseCase
    phrasing: int
    split: str
    seq: int
    plan: CausePlan

    @property
    def labels(self) -> CauseLabels:
        """输入：无。输出：这一条的全部金标签。"""
        return label_for(self.plan)

    @property
    def render_task_id(self) -> str:
        """输入：无。输出：渲染任务的后端任务 id。

        预期行为：同一次 run 内唯一且可重算（断点续跑按它查进度）；进度日志按 run 目录分文件，
        所以不同 run_seed 之间不会串。
        """
        return f"causechain-{self.split}-{self.seq:07d}-render"

    @property
    def trace_task_id(self) -> str:
        """输入：无。输出：反解（判别器）任务的后端任务 id。"""
        return f"causechain-{self.split}-{self.seq:07d}-trace"


def spec_to_dict(spec: CauseSpec) -> dict[str, JsonValue]:
    """输入：规格。输出：可 JSON 序列化的字典（写进 manifest 与调试，不进样本行）。"""
    return {
        "case": spec.case.as_dict(),
        "phrasing": spec.phrasing,
        "split": spec.split,
        "seq": spec.seq,
        "events": dict(spec.plan.events),
        "order": list(spec.plan.order),
        "probed": spec.plan.probed,
        "distractors": [list(combo) for combo in spec.plan.distractors],
    }
