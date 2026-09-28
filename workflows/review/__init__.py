"""改动审查（review）workflow：清单行与三张表由代码写，改动正文由后端写。

导入本包即自注册（`jc.workflow.load("review")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]