"""界面元素定位 workflow：构造式（一屏元素清单与全部判断都由规则算出来，零 token）。

导入本包即自注册（`jc.workflow.load("elementpick")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
