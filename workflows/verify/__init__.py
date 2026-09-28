"""完成声明核验 workflow：自报事实与验收门槛由代码写，那句声明由后端写。

导入本包即自注册（`jc.workflow.load("verify")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]