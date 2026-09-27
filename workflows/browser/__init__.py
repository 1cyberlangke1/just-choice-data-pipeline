"""界面操作 workflow：构造式（一屏元素表与全部判断都由规则算出来，零 token）。

导入本包即自注册（`jc.workflow.load("browser")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
