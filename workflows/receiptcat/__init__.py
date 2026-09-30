"""报销收据归类（receiptcat）workflow：三张表由代码写，一条收据行由后端写。

导入本包即自注册（`jc.workflow.load("receiptcat")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
