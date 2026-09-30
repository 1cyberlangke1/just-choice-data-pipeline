"""社交帖子分诊（posttriage）workflow：判据表与阈值表由代码写，一条帖子由后端写。

导入本包即自注册（`jc.workflow.load("posttriage")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
