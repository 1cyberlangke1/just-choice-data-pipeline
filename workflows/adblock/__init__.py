"""广告元素判定（adblock）workflow：判据表由代码写，HTML 片段由后端写，命中与否按字面算。

导入本包即自注册（`jc.workflow.load("adblock")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
