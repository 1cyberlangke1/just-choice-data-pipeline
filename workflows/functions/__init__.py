"""函数调用（functions）workflow：构造式（工具表与全部答案都由规则算出来，零 token）。

导入本包即自注册（`jc.workflow.load("functions")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
