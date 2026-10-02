"""改动引入性判定的真值规格：一组注入特征 + 这条样本的问法编号。

输入/输出：只有不可变数据类与纯函数；不含领域词表（词表在 `locales/`）。
"""

from __future__ import annotations

from dataclasses import dataclass

from jc.jsontypes import JsonValue
from jc.spec_dict import case_spec_dict

from .features import RecordCase, RecordLabels, RecordPlan, label_for, plan_for


@dataclass(frozen=True)
class FindTriageSpec:
    """一条样本的真值规格。

    字段：`case` 注入特征；`phrasing` 问法编号；`split` 切分名；`seq` 样本序号。
    预期行为：改动记录由 `plan_for(case, phrasing)` **现场派生**，不存第二份——存下来就会
    出现「规格里的表」与「渲染出来的表」不一致的可能。
    """

    case: RecordCase
    phrasing: int
    split: str
    seq: int

    @property
    def plan(self) -> RecordPlan:
        """输入：无。输出：这一条的改动记录计划（由 case 与问法唯一决定）。"""
        return plan_for(self.case, self.phrasing)

    @property
    def labels(self) -> RecordLabels:
        """输入：无。输出：这一条的全部金标签。"""
        return label_for(self.plan)

    @property
    def render_task_id(self) -> str:
        """输入：无。输出：渲染任务的后端任务 id。

        预期行为：同一次 run 内唯一且可重算（断点续跑按它查进度）；
        进度日志按 run 目录分文件，所以不同 run_seed 之间不会串。
        """
        return f"findtriage-{self.split}-{self.seq:07d}-render"

    @property
    def trace_task_id(self) -> str:
        """输入：无。输出：反解（判别器）任务的后端任务 id。"""
        return f"findtriage-{self.split}-{self.seq:07d}-trace"


def spec_to_dict(spec: FindTriageSpec) -> dict[str, JsonValue]:
    """输入：规格。输出：可 JSON 序列化的字典（写进 manifest 与调试，不进样本行）。"""
    return case_spec_dict(case=spec.case, phrasing=spec.phrasing, split=spec.split, seq=spec.seq)
