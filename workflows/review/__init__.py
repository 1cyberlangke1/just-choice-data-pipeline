"""改动审查 workflow：构造式（一屏改动与全部判断都由规则算出来，零 token）。

导入本包即自注册（`jc.workflow.load("review")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
