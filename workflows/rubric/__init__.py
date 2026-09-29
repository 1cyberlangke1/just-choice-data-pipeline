"""打分与评分（rubric）workflow：材料清单与三张表由代码写，待评材料由后端写。

导入本包即自注册（`jc.workflow.load("rubric")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
