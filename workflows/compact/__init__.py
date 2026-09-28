"""会话上下文压缩（compact）workflow：处置表与位置行由代码写，那一块正文由后端写。

导入本包即自注册（`jc.workflow.load("compact")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]