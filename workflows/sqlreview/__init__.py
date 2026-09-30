"""数据库变更评审（sqlreview）workflow：判据表由代码写，一屏评审记录由后端写，命中与否按字面算。

导入本包即自注册（`jc.workflow.load("sqlreview")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
