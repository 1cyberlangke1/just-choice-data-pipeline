"""命令审批 workflow：构造式（命令与审批事实都由表格算出来，零 token）。

导入本包即自注册（`jc.workflow.load("shell")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
