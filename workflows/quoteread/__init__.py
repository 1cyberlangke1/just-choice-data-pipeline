"""行情读数判读（quoteread）workflow：**构造式**（读数表与四个答案都由代码算出来，零 token）。

导入本包即自注册（`jc.workflow.load("quoteread")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
