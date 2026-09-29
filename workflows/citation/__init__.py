"""引文核查（citation）workflow：文档原文由后端写，引文由代码从原文里剪、结论由代码算。

导入本包即自注册（`jc.workflow.load("citation")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
