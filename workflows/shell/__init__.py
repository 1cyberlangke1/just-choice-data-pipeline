"""命令审批 workflow：特征注入式（标签先行，模型只写命令文本）。

导入本包即自注册（`jc.workflow.load("shell")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]