"""来源选择（sourceselect）workflow：候选表与选源口径由代码写，那句查询由后端写。

导入本包即自注册（`jc.workflow.load("sourceselect")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
