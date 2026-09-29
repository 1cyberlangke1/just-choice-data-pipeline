"""检索重排（rerank）workflow：查询与候选原文由后端写，覆盖率与名次由代码算。

导入本包即自注册（`jc.workflow.load("rerank")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
