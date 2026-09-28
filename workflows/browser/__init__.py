"""浏览器 / GUI 操作（browser）workflow：元素表与两张判据表由代码写，那句话由后端写。

导入本包即自注册（`jc.workflow.load("browser")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]