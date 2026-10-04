"""字幕对质检的真值规格：注入特征 + 判定计划 + 切分信息。

输入/输出：只有不可变数据类与纯函数；不含领域词表（判据文案全在 `locales/`）。
"""

from __future__ import annotations

from dataclasses import dataclass


from .features import SubQaCase, SubQaLabels, SubQaPlan, label_for


@dataclass(frozen=True)
class SubQaSpec:
    """一条样本的真值规格。

    字段：`case` 注入特征；`phrasing` 问法编号；`split` 切分名；`seq` 样本序号；`plan` 判定计划。
    `timecode` 是**写文本时的样本级旋钮**（`Timing:` 那一行的开始时间码，见
    `features.TIMECODE_HINTS`）：它只改这一行写成哪一串数字，与 case、标签、切分都无关——所以它
    不算维度，也不进 `stats_dimensions`。
    """

    case: SubQaCase
    phrasing: int
    timecode: str
    split: str
    seq: int
    plan: SubQaPlan

    @property
    def labels(self) -> SubQaLabels:
        """输入：无。输出：这一条的全部金标签。"""
        return label_for(self.plan)

    @property
    def render_task_id(self) -> str:
        """输入：无。输出：渲染任务的后端任务 id。

        预期行为：同一次 run 内唯一且可重算（断点续跑按它查进度）；进度日志按 run 目录分文件，
        所以不同 run_seed 之间不会串。
        """
        return f"subqa-{self.split}-{self.seq:07d}-render"

    @property
    def trace_task_id(self) -> str:
        """输入：无。输出：反解（判别器）任务的后端任务 id。"""
        return f"subqa-{self.split}-{self.seq:07d}-trace"
