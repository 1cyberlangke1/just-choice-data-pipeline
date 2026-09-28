"""检索结果判读（search）workflow：规则表与「来源与时效」由代码写，查询与结果条目由后端写。

导入本包即自注册（`jc.workflow.load("search")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]