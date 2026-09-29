"""内容审核（guardrail）workflow：政策条款由代码写，待审内容由后端写，命中与否由代码按字面算。

导入本包即自注册（`jc.workflow.load("guardrail")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
