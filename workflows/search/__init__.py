"""检索 workflow：构造式（检索页与全部判断都由表格算出来，零 token）。

导入本包即自注册（`jc.workflow.load("search")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
