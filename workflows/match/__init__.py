"""记录匹配（match）workflow：构造式（两条记录与全部判断都由字段关系表算出来，零 token）。

导入本包即自注册（`jc.workflow.load("match")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
