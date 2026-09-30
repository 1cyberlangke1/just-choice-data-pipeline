"""改动引入性判定（findtriage）workflow：改动记录与判定口径由代码写，那条发现与修法由后端写。

导入本包即自注册（`jc.workflow.load("findtriage")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
