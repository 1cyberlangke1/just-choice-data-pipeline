"""安全警报风险定级（alertrisk）workflow：定级矩阵与处置档位表由代码写，一屏警报记录由后端写。

导入本包即自注册（`jc.workflow.load("alertrisk")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
