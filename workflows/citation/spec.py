"""引文核查的真值规格：注入特征 + 引文计划 + 切分信息。

输入/输出：只有不可变数据类与纯函数；不含领域词表（词表在 `locales/`）。
"""

from __future__ import annotations

from dataclasses import dataclass


from .features import CitationCase, CitationLabels, CitationPlan, label_for


@dataclass(frozen=True)
class CitationSpec:
    """一条样本的真值规格。

    字段：`case` 注入特征；`phrasing` 问法编号；`split` 切分名；`seq` 样本序号；
    `plan` 引文计划。
    `topic` 是**写短文时的样本级旋钮**（题材方向下标，见 `features.TOPIC_IDS`）：它只改短文与标题
    怎么写，与 case、标签、切分都无关——所以它不算维度，也不进 case 空间。
    """

    case: CitationCase
    phrasing: int
    topic: int
    split: str
    seq: int
    plan: CitationPlan

    @property
    def labels(self) -> CitationLabels:
        """输入：无。输出：这一条的全部金标签。"""
        return label_for(self.plan)

    @property
    def render_task_id(self) -> str:
        """输入：无。输出：渲染任务的后端任务 id。

        预期行为：同一次 run 内唯一且可重算（断点续跑按它查进度）；进度日志按 run 目录
        分文件，所以不同 run_seed 之间不会串。
        """
        return f"citation-{self.split}-{self.seq:07d}-render"

    @property
    def trace_task_id(self) -> str:
        """输入：无。输出：反解（判别器）任务的后端任务 id。"""
        return f"citation-{self.split}-{self.seq:07d}-trace"
