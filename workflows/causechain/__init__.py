"""根因定位（causechain）workflow：依赖图与规则表由代码写，现场日志由后端写，标签按字面算。

导入本包即自注册（`jc.workflow.load("causechain")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
