"""文档包切分（segmentation）workflow：种类表与切分规则由代码写，页面正文由后端写，标签按字面算。

导入本包即自注册（`jc.workflow.load("segmentation")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
