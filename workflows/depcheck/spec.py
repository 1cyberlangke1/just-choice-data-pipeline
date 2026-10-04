"""依赖与许可证风险门控的真值规格：注入特征 + 判定计划 + 切分信息。

输入/输出：只有不可变数据类与纯函数；不含领域词表（标记文案与等级文案全在 `locales/`）。
"""

from __future__ import annotations

from dataclasses import dataclass


from .features import DepCheckCase, DepCheckLabels, DepCheckPlan, label_for


@dataclass(frozen=True)
class DepCheckSpec:
    """一条样本的真值规格。

    字段：`case` 注入特征；`phrasing` 问法编号；`split` 切分名；`seq` 样本序号；`plan` 判定计划。
    `category` 是**写文本时的样本级旋钮**（依赖类别下标，见 `features.CATEGORY_IDS`）：它只改
    `note:` 怎么写与包名怎么起，与 case、标签、切分都无关——所以它不算维度，也不进 case 空间。
    `version` 是**第二个样本级旋钮**（版本号尾段提示值，见 `features.VERSION_HINTS`）：它只改
    `version:` 那一行怎么写（钉住最后一个数字段），同样不参与任何判据。
    `license` 是**第三个样本级旋钮**（`license:` 那一行要写的 SPDX 标识符，见
    `features.LICENSE_COPYLEFT` / `LICENSE_PERMISSIVE`）：它与判据相容（`D1` 命中就从传染性
    那一半取、不命中就只从宽松那一半取），所以只改这一行写成哪个标识符，不改任何标签。
    """

    case: DepCheckCase
    phrasing: int
    category: int
    version: int
    license: str
    split: str
    seq: int
    plan: DepCheckPlan

    @property
    def labels(self) -> DepCheckLabels:
        """输入：无。输出：这一条的全部金标签。"""
        return label_for(self.plan)

    @property
    def render_task_id(self) -> str:
        """输入：无。输出：渲染任务的后端任务 id。

        预期行为：同一次 run 内唯一且可重算（断点续跑按它查进度）；进度日志按 run 目录分文件，
        所以不同 run_seed 之间不会串。
        """
        return f"depcheck-{self.split}-{self.seq:07d}-render"

    @property
    def trace_task_id(self) -> str:
        """输入：无。输出：反解（判别器）任务的后端任务 id。"""
        return f"depcheck-{self.split}-{self.seq:07d}-trace"
