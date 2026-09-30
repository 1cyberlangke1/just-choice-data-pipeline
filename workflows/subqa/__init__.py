"""字幕对质检（subqa）workflow：判据表由代码写，一屏双语字幕对由后端写，命中与否按字面算。

导入本包即自注册（`jc.workflow.load("subqa")` 可用）。
"""

from . import workflow as workflow  # noqa: F401  （导入即注册）

__all__ = ["workflow"]
