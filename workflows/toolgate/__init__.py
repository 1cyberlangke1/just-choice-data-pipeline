"""工具调用门控（toolgate）workflow：策略表与状态由代码写，那句调用意图由后端写。

导入本包即自注册（`jc.workflow.load("toolgate")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]