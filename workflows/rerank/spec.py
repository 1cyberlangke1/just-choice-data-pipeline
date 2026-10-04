"""检索重排的真值规格：注入特征 + 名次计划 + 切分信息。

输入/输出：只有不可变数据类与纯函数；不含领域词表（词表在 `locales/`）。
"""

from __future__ import annotations

from dataclasses import dataclass


from .features import RerankCase, RerankLabels, RerankPlan, label_for


@dataclass(frozen=True)
class RerankSpec:
    """一条样本的真值规格。

    字段：`case` 注入特征；`phrasing` 问法编号；`split` 切分名；`seq` 样本序号；
    `plan` 名次计划。
    """

    case: RerankCase
    phrasing: int
    split: str
    seq: int
    plan: RerankPlan

    @property
    def labels(self) -> RerankLabels:
        """输入：无。输出：这一条的全部金标签。"""
        return label_for(self.plan)

    @property
    def render_task_id(self) -> str:
        """输入：无。输出：渲染任务的后端任务 id。

        预期行为：同一次 run 内唯一且可重算（断点续跑按它查进度）；进度日志按 run 目录
        分文件，所以不同 run_seed 之间不会串。
        """
        return f"rerank-{self.split}-{self.seq:07d}-render"

    @property
    def trace_task_id(self) -> str:
        """输入：无。输出：反解（判别器）任务的后端任务 id。"""
        return f"rerank-{self.split}-{self.seq:07d}-trace"
