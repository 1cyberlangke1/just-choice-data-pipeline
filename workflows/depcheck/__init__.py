"""依赖与许可证风险门控（depcheck）workflow：标记表、定级规则与处置档位表由代码写，一屏依赖记录由后端写。

导入本包即自注册（`jc.workflow.load("depcheck")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
