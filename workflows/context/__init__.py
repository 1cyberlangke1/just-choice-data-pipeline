"""会话上下文压缩 workflow：构造式（会话与全部判断都由处置表算出来，零 token）。

导入本包即自注册（`jc.workflow.load("context")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
