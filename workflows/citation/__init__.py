"""引文核查 workflow：构造式（文档与全部判断都由表格算出来，零 token）。

导入本包即自注册（`jc.workflow.load("citation")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
