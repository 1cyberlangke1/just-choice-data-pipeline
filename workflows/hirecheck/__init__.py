"""招聘初筛门禁（hirecheck）workflow：判据表、定级规则与处置档位表由代码写，一屏候选人材料由后端写。

导入本包即自注册（`jc.workflow.load("hirecheck")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
