"""有界任务推进（planstep）workflow：**构造式**（任务表与四个答案都由代码算出来，零 token）。

导入本包即自注册（`jc.workflow.load("planstep")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
