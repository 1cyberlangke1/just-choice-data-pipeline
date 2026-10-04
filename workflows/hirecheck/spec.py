"""招聘初筛门禁的真值规格：注入特征 + 判定计划 + 切分信息。

输入/输出：只有不可变数据类与纯函数；不含领域词表（判据文案与等级文案全在 `locales/`）。
"""

from __future__ import annotations

from dataclasses import dataclass


from .features import HireCheckCase, HireCheckLabels, HireCheckPlan, label_for


@dataclass(frozen=True)
class HireCheckSpec:
    """一条样本的真值规格。

    字段：`case` 注入特征；`phrasing` 问法编号；`split` 切分名；`seq` 样本序号；`plan` 判定计划。
    `role` 是**写文本时的样本级旋钮**（岗位方向下标，见 `features.ROLE_IDS`）：它只改 `skills:` 那一
    行怎么列、`note:` 怎么写，与 case、标签、切分都无关——所以它不算维度，也不进 case 空间。
    `years` 是**第二个样本级旋钮**（`years:` 那一行要写的年限，见 `features.YEAR_HINTS_SHORT` /
    `YEAR_HINTS_LONG`）：它与判据相容（`H1` 命中就从「不足三年」那一半取、不命中就只从三年以上那
    一半取），所以只改这一行写成哪个数字，不改任何标签。
    `name` 是**第三个样本级旋钮**（`candidate:` 那一行的姓名下标，见 `features.NAME_IDS`）：池子按
    语言放在 `locales/*.toml` 的 `prompt.names` 里，所以存的是下标；它只改姓名怎么写，同样不改
    任何标签。
    """

    case: HireCheckCase
    phrasing: int
    role: int
    years: int
    name: int
    split: str
    seq: int
    plan: HireCheckPlan

    @property
    def labels(self) -> HireCheckLabels:
        """输入：无。输出：这一条的全部金标签。"""
        return label_for(self.plan)

    @property
    def render_task_id(self) -> str:
        """输入：无。输出：渲染任务的后端任务 id。

        预期行为：同一次 run 内唯一且可重算（断点续跑按它查进度）；进度日志按 run 目录分文件，
        所以不同 run_seed 之间不会串。
        """
        return f"hirecheck-{self.split}-{self.seq:07d}-render"

    @property
    def trace_task_id(self) -> str:
        """输入：无。输出：反解（判别器）任务的后端任务 id。"""
        return f"hirecheck-{self.split}-{self.seq:07d}-trace"
