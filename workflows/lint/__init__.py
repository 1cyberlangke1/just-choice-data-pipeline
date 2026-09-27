"""行级规则检查（lint）workflow：构造式（片段与全部答案都由规则算出来，零 token）。

导入本包即自注册（`jc.workflow.load("lint")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
