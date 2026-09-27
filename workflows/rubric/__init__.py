"""打分与评分 workflow：构造式（一张评分单与全部判断都由规则算出来，零 token）。

导入本包即自注册（`jc.workflow.load("rubric")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]

