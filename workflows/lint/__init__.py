"""行级规则检查（lint）workflow：规则表与处置办法由代码写，片段原文由后端写。

导入本包即自注册（`jc.workflow.load("lint")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
