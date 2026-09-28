"""工单分诊（routing）workflow：属性行与三张表由代码写，工单正文由后端写。

导入本包即自注册（`jc.workflow.load("routing")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
