"""来源选择的真值规格：一组注入特征 + 这条样本的问法编号。

输入/输出：只有不可变数据类与纯函数；不含领域词表（词表在 `locales/`）。
"""

from __future__ import annotations

from dataclasses import dataclass


from .features import SourceCase, SourceLabels, SourcePlan, label_for, plan_for


@dataclass(frozen=True)
class SourceSpec:
    """一条样本的真值规格。

    字段：`case` 注入特征；`phrasing` 问法编号；`split` 切分名；`seq` 样本序号。
    预期行为：候选表由 `plan_for(case, phrasing)` **现场派生**，不存第二份——存下来就会
    出现「规格里的表」与「渲染出来的表」不一致的可能。
    """

    case: SourceCase
    phrasing: int
    split: str
    seq: int

    @property
    def plan(self) -> SourcePlan:
        """输入：无。输出：这一条的候选表计划（由 case 与问法唯一决定）。"""
        return plan_for(self.case, self.phrasing)

    @property
    def labels(self) -> SourceLabels:
        """输入：无。输出：这一条的全部金标签。"""
        return label_for(self.plan)

    @property
    def render_task_id(self) -> str:
        """输入：无。输出：渲染任务的后端任务 id。

        预期行为：同一次 run 内唯一且可重算（断点续跑按它查进度）；
        进度日志按 run 目录分文件，所以不同 run_seed 之间不会串。
        """
        return f"sourceselect-{self.split}-{self.seq:07d}-render"

    @property
    def trace_task_id(self) -> str:
        """输入：无。输出：反解（判别器）任务的后端任务 id。"""
        return f"sourceselect-{self.split}-{self.seq:07d}-trace"
